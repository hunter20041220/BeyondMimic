#!/usr/bin/env python3
"""Decode guided official-csv-loop latents through the local VAE action decoder.

This is an offline bridge from the state-latent guidance stage toward a future
closed-loop rollout gate. It recomputes best-scale guided denoiser outputs over
the full validation/test windows, decodes the latent part with the local
conditional action VAE, and compares decoded actions with teacher actions.
It is not IsaacLab closed-loop control.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/official_csv_loop_guidance_vae_action_decode_eval"
RUN_ROOT = ROOT / "res/runs/level_c_official_csv_loop_guidance_vae_action_decode_eval"
LOG_DIR = ROOT / "logs/level_c_official_csv_loop_guidance_vae_action_decode_eval"
FAILED_DIR = ROOT / "res/failed_runs/level_c_official_csv_loop_guidance_vae_action_decode_eval"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_state_latent_guidance_eval/"
    "level_c_official_csv_loop_state_latent_guidance_eval.json"
)
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_state_latent_diffusion_training/"
    "level_c_official_csv_loop_state_latent_diffusion_training.json"
)
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
    "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
)
VAE_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
    "level_c_official_csv_loop_teacher_rollout_vae_training.json"
)
CANDIDATE_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_GUIDED_DECODE_SEED", "20260636"))
BATCH_WINDOWS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_GUIDED_DECODE_BATCH_WINDOWS", "512"))
TASKS = ["velocity_command", "latent_smoothness", "latent_magnitude", "composed"]


WORKER_CODE = r"""
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GUIDANCE_JSON = Path(os.environ["BM_GUIDANCE_JSON"])
DIFFUSION_JSON = Path(os.environ["BM_DIFFUSION_JSON"])
STATE_LATENT_JSON = Path(os.environ["BM_STATE_LATENT_JSON"])
VAE_JSON = Path(os.environ["BM_VAE_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
SEED = int(os.environ["BM_OFFICIAL_CSV_LOOP_GUIDED_DECODE_SEED"])
BATCH_WINDOWS = int(os.environ["BM_OFFICIAL_CSV_LOOP_GUIDED_DECODE_BATCH_WINDOWS"])
TASKS = ["velocity_command", "latent_smoothness", "latent_magnitude", "composed"]


class StateLatentDenoiser(nn.Module):
    def __init__(self, token_dim, hidden_dim, steps):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(token_dim + steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, noisy, step_idx):
        onehot = F.one_hot(step_idx, num_classes=self.steps).to(noisy.dtype)
        return self.net(torch.cat([noisy, onehot], dim=-1))


class ConditionalActionVAE(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def decode(self, obs, latent):
        return self.decoder(torch.cat([obs, latent], dim=-1))


def alpha_bars(steps):
    betas = torch.linspace(1e-4, 0.02, steps)
    return torch.cumprod(1.0 - betas, dim=0)


def read_window_index(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] in {"validation", "test"}:
                rows.append(
                    {
                        "rank": int(row["rank"]),
                        "env_index": int(row["env_index"]),
                        "start": int(row["start"]),
                        "end_exclusive": int(row["end_exclusive"]),
                        "split": row["split"],
                    }
                )
    return rows


def load_arrays(dataset_summary):
    obs_by_rank = {}
    latent_by_rank = {}
    action_by_rank = {}
    for shard in dataset_summary["worker_summary"]["shards"]:
        rank = int(shard["rank"])
        with np.load(shard["source_shard"], mmap_mode="r") as data:
            obs_by_rank[rank] = data["policy_obs"].astype(np.float32)
            action_by_rank[rank] = data["actions"].astype(np.float32)
        with np.load(shard["latent_shard"], mmap_mode="r") as data:
            latent_by_rank[rank] = data["latent_mu"].astype(np.float32)
    return obs_by_rank, latent_by_rank, action_by_rank


def make_batch(rows, obs_by_rank, latent_by_rank, action_by_rank, start, end):
    tokens = []
    actions = []
    splits = []
    for row in rows[start:end]:
        rank = row["rank"]
        env = row["env_index"]
        s = row["start"]
        e = row["end_exclusive"]
        obs = obs_by_rank[rank][s:e, env, :]
        latent = latent_by_rank[rank][s:e, env, :]
        action = action_by_rank[rank][s:e, env, :]
        tokens.append(np.concatenate([obs, latent], axis=-1).astype(np.float32))
        actions.append(action.astype(np.float32))
        splits.append(row["split"])
    return torch.from_numpy(np.stack(tokens, axis=0)), torch.from_numpy(np.stack(actions, axis=0)), splits


def task_cost(task, tau, actions):
    obs = tau[..., :160]
    latent = tau[..., 160:]
    root_vel = obs[..., :2]
    command = torch.tensor([0.35, 0.0], dtype=tau.dtype, device=tau.device)
    velocity = torch.mean((root_vel - command) ** 2, dim=(-2, -1))
    latent_smooth = torch.mean((latent[:, 1:, :] - latent[:, :-1, :]) ** 2, dim=(-2, -1))
    latent_mag = torch.mean(latent**2, dim=(-2, -1))
    if task == "velocity_command":
        return velocity
    if task == "latent_smoothness":
        return latent_smooth
    if task == "latent_magnitude":
        return latent_mag
    if task == "composed":
        return velocity + 0.25 * latent_smooth + 0.1 * latent_mag
    raise ValueError(task)


def best_scales(guidance_summary):
    result = {}
    for task, task_summary in guidance_summary["worker_summary"]["task_summaries"].items():
        result[task] = {
            split: float(row["scale"])
            for split, row in task_summary["splits"].items()
        }
    return result


def load_diffusion(summary, device):
    ckpt = Path(summary["worker_summary"]["outputs"]["checkpoint"])
    payload = torch.load(ckpt, map_location="cpu")
    cfg = payload["config"]
    model = StateLatentDenoiser(cfg["token_dim"], cfg["hidden_dim"], cfg["denoising_steps"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def load_vae(summary, device):
    ckpt = Path(summary["worker_summary"]["outputs"]["checkpoint"])
    payload = torch.load(ckpt, map_location="cpu")
    cfg = payload["config"]
    model = ConditionalActionVAE(cfg["obs_dim"], cfg["action_dim"], cfg["latent_dim"], cfg["hidden_dim"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    guidance_summary = json.loads(GUIDANCE_JSON.read_text(encoding="utf-8"))
    diffusion_summary = json.loads(DIFFUSION_JSON.read_text(encoding="utf-8"))
    dataset_summary = json.loads(STATE_LATENT_JSON.read_text(encoding="utf-8"))
    vae_summary = json.loads(VAE_JSON.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion, diffusion_cfg, diffusion_ckpt = load_diffusion(diffusion_summary, device)
    vae, vae_cfg, vae_ckpt = load_vae(vae_summary, device)
    bars = alpha_bars(diffusion_cfg["denoising_steps"]).to(device)
    scales = best_scales(guidance_summary)
    rows = read_window_index(dataset_summary["worker_summary"]["outputs"]["window_index_csv"])
    obs_by_rank, latent_by_rank, action_by_rank = load_arrays(dataset_summary)
    rng = np.random.default_rng(SEED)
    rows = list(rows)
    rng.shuffle(rows)

    aggregates = {}
    sample_npz = {}
    for task in TASKS:
        for split in ["validation", "test"]:
            aggregates[(task, split)] = {
                "window_count": 0,
                "action_count": 0,
                "base_teacher_mse_sum": 0.0,
                "guided_teacher_mse_sum": 0.0,
                "action_delta_l2_sum": 0.0,
                "action_abs_mean_sum": 0.0,
                "guided_action_abs_mean_sum": 0.0,
                "finite": True,
                "scale": scales[task][split],
            }

    for start in range(0, len(rows), BATCH_WINDOWS):
        end = min(start + BATCH_WINDOWS, len(rows))
        clean, teacher_actions, splits = make_batch(rows, obs_by_rank, latent_by_rank, action_by_rank, start, end)
        clean = clean.to(device)
        teacher_actions = teacher_actions.to(device)
        step = torch.full(clean.shape[:2], diffusion_cfg["denoising_steps"] - 1, dtype=torch.long, device=device)
        noise = torch.randn_like(clean)
        alpha = bars[step].unsqueeze(-1)
        noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
        with torch.no_grad():
            pred = diffusion(noisy, step)
        base_obs = pred[..., : vae_cfg["obs_dim"]]
        base_latent = pred[..., vae_cfg["obs_dim"] :]
        with torch.no_grad():
            base_action = vae.decode(base_obs.reshape(-1, vae_cfg["obs_dim"]), base_latent.reshape(-1, vae_cfg["latent_dim"]))
            base_action = base_action.reshape(pred.shape[0], pred.shape[1], vae_cfg["action_dim"])
        for task in TASKS:
            variable = pred.detach().clone().requires_grad_(True)
            cost_vec = task_cost(task, variable, teacher_actions)
            torch.mean(cost_vec).backward()
            grad = variable.grad.detach()
            for split in ["validation", "test"]:
                mask = torch.tensor([item == split for item in splits], device=device)
                if not bool(torch.any(mask)):
                    continue
                guided = (pred.detach() - scales[task][split] * grad).detach()
                guided_obs = guided[..., : vae_cfg["obs_dim"]]
                guided_latent = guided[..., vae_cfg["obs_dim"] :]
                with torch.no_grad():
                    guided_action = vae.decode(
                        guided_obs.reshape(-1, vae_cfg["obs_dim"]),
                        guided_latent.reshape(-1, vae_cfg["latent_dim"]),
                    ).reshape(guided.shape[0], guided.shape[1], vae_cfg["action_dim"])
                base_sel = base_action[mask]
                guided_sel = guided_action[mask]
                teacher_sel = teacher_actions[mask]
                action_count = int(teacher_sel.numel() / vae_cfg["action_dim"])
                item = aggregates[(task, split)]
                item["window_count"] += int(torch.sum(mask).detach().cpu())
                item["action_count"] += action_count
                item["base_teacher_mse_sum"] += float(torch.sum((base_sel - teacher_sel) ** 2).detach().cpu())
                item["guided_teacher_mse_sum"] += float(torch.sum((guided_sel - teacher_sel) ** 2).detach().cpu())
                item["action_delta_l2_sum"] += float(
                    torch.sum(torch.linalg.vector_norm((guided_sel - base_sel).reshape(-1, vae_cfg["action_dim"]), dim=-1)).detach().cpu()
                )
                item["action_abs_mean_sum"] += float(torch.sum(torch.abs(base_sel)).detach().cpu())
                item["guided_action_abs_mean_sum"] += float(torch.sum(torch.abs(guided_sel)).detach().cpu())
                item["finite"] = item["finite"] and bool(torch.isfinite(guided_sel).all().detach().cpu())
                if start == 0 and split == "validation" and task in {"velocity_command", "composed"}:
                    sample_npz[f"base_action_{task}"] = base_sel[: min(8, base_sel.shape[0])].detach().cpu().numpy()
                    sample_npz[f"guided_action_{task}"] = guided_sel[: min(8, guided_sel.shape[0])].detach().cpu().numpy()
                    sample_npz[f"teacher_action_{task}"] = teacher_sel[: min(8, teacher_sel.shape[0])].detach().cpu().numpy()

    aggregate_rows = []
    task_summaries = {}
    for task in TASKS:
        split_rows = []
        for split in ["validation", "test"]:
            item = aggregates[(task, split)]
            denom = max(item["action_count"] * vae_cfg["action_dim"], 1)
            action_denom = max(item["action_count"], 1)
            row = {
                "task": task,
                "split": split,
                "scale": item["scale"],
                "window_count": item["window_count"],
                "action_count": item["action_count"],
                "base_teacher_action_mse": item["base_teacher_mse_sum"] / denom,
                "guided_teacher_action_mse": item["guided_teacher_mse_sum"] / denom,
                "guided_minus_base_teacher_mse": (item["guided_teacher_mse_sum"] - item["base_teacher_mse_sum"]) / denom,
                "guided_base_action_l2_mean": item["action_delta_l2_sum"] / action_denom,
                "base_action_abs_mean": item["action_abs_mean_sum"] / denom,
                "guided_action_abs_mean": item["guided_action_abs_mean_sum"] / denom,
                "finite": item["finite"],
            }
            aggregate_rows.append(row)
            split_rows.append(row)
        task_summaries[task] = {
            "mean_guided_base_action_l2": float(np.mean([r["guided_base_action_l2_mean"] for r in split_rows])),
            "mean_guided_minus_base_teacher_mse": float(np.mean([r["guided_minus_base_teacher_mse"] for r in split_rows])),
            "all_actions_finite": all(r["finite"] for r in split_rows),
            "split_rows": {r["split"]: r for r in split_rows},
        }

    rows_tsv = RUN_DIR / "official_csv_loop_guidance_vae_action_decode_rows.tsv"
    with rows_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    samples_npz = RUN_DIR / "official_csv_loop_guidance_vae_action_decode_samples.npz"
    if sample_npz:
        np.savez_compressed(samples_npz, **sample_npz)

    summary = {
        "status": "ok",
        "duration_seconds": round(time.time() - start_time, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "source_guidance": {
            "json": str(GUIDANCE_JSON),
            "status": guidance_summary.get("status"),
        },
        "source_diffusion": {
            "json": str(DIFFUSION_JSON),
            "checkpoint": str(diffusion_ckpt),
            "status": diffusion_summary.get("status"),
            "official_beyondmimic_diffusion_checkpoint": False,
        },
        "source_vae": {
            "json": str(VAE_JSON),
            "checkpoint": str(vae_ckpt),
            "status": vae_summary.get("status"),
            "official_beyondmimic_vae_checkpoint": False,
        },
        "source_dataset": {
            "json": str(STATE_LATENT_JSON),
            "status": dataset_summary.get("status"),
            "official_dagger_rollout_dataset": False,
        },
        "settings": {
            "seed": SEED,
            "batch_windows": BATCH_WINDOWS,
            "tasks": TASKS,
            "best_scales": scales,
        },
        "metrics": {
            "row_count": len(aggregate_rows),
            "task_count": len(TASKS),
            "split_count": 2,
            "total_windows": sum(row["window_count"] for row in aggregate_rows) // len(TASKS),
            "total_action_steps_per_task": {
                task: sum(row["action_count"] for row in aggregate_rows if row["task"] == task)
                for task in TASKS
            },
            "tasks_with_finite_actions": int(sum(v["all_actions_finite"] for v in task_summaries.values())),
        },
        "task_summaries": task_summaries,
        "rows": aggregate_rows,
        "outputs": {
            "rows_tsv": str(rows_tsv),
            "samples_npz": str(samples_npz) if sample_npz else "",
        },
        "checks": {
            "uses_official_csv_loop_guidance": guidance_summary.get("status") == "ok_official_csv_loop_state_latent_guidance_eval",
            "uses_official_csv_loop_diffusion": diffusion_summary.get("status") == "ok_official_csv_loop_state_latent_diffusion_training",
            "uses_official_csv_loop_vae": vae_summary.get("status") == "ok_official_csv_loop_teacher_rollout_vae_training",
            "uses_full_validation_test_windows": sum(row["window_count"] for row in aggregate_rows) // len(TASKS) == 57140,
            "all_tasks_decoded": sorted(TASKS) == sorted({row["task"] for row in aggregate_rows}),
            "row_count_matches_tasks_splits": len(aggregate_rows) == len(TASKS) * 2,
            "all_decoded_actions_finite": all(row["finite"] for row in aggregate_rows),
            "decoded_action_dim_29": vae_cfg["action_dim"] == 29,
            "does_not_claim_closed_loop_rollout": True,
            "does_not_claim_fig5_fig6_reproduction": True,
            "does_not_claim_paper_level_guidance": True,
        },
    }
    (RUN_DIR / "official_csv_loop_guidance_vae_action_decode_worker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("BM_SENTINEL:summary:" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
"""


def run_command(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def parse_gpu_processes() -> list[dict[str, Any]]:
    proc = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "gpu_bus_id": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mb": int(parts[3]),
            }
        )
    return rows


def gpu_index_to_bus_id() -> dict[int, str]:
    proc = run_command(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"])
    mapping = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            mapping[int(parts[0])] = parts[1]
    return mapping


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def kill_wangjc_on_target_gpus() -> dict[str, Any]:
    guard_dir = ROOT / "res/gpu_guard"
    guard_dir.mkdir(parents=True, exist_ok=True)
    bus = gpu_index_to_bus_id()
    target_bus = {bus[index] for index in CANDIDATE_GPUS if index in bus}
    killed = []
    skipped = []
    for row in parse_gpu_processes():
        if row["gpu_bus_id"] not in target_bus:
            continue
        pid = row["pid"]
        command = cmdline(pid)
        if WANGJC_PATH_MARKER in command:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(row | {"cmdline": command, "signal": "SIGTERM"})
            except ProcessLookupError:
                killed.append(row | {"cmdline": command, "signal": "already_exited"})
        else:
            skipped.append(row | {"cmdline": command})
    if killed:
        time.sleep(8)
        for item in killed:
            pid = item["pid"]
            if Path(f"/proc/{pid}").exists():
                try:
                    os.kill(pid, signal.SIGKILL)
                    item["signal"] = "SIGKILL_after_grace"
                except ProcessLookupError:
                    pass
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_gpus": CANDIDATE_GPUS,
        "killed": killed,
        "skipped_non_wangjc": skipped,
    }
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_guided_decode_guard.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(path)
    return summary


def start_gpu_monitor(path: Path) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "while true; do "
        "date -Is; "
        "nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw "
        "--format=csv,noheader,nounits -i 4,7; "
        "sleep 5; "
        "done"
    )
    f = path.open("w", encoding="utf-8")
    return subprocess.Popen(["bash", "-lc", script], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)


def summarize_gpu_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    per_gpu: dict[str, dict[str, Any]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) != 6 or not parts[0].isdigit():
                continue
            row_count += 1
            index, _, util, mem_used, mem_total, power = parts
            item = per_gpu.setdefault(index, {"util": [], "memory": [], "power": [], "memory_total_mb": int(mem_total)})
            item["util"].append(float(util))
            item["memory"].append(float(mem_used))
            try:
                item["power"].append(float(power))
            except ValueError:
                pass
    return {
        "exists": True,
        "row_count": row_count,
        "per_gpu": {
            index: {
                "samples": len(values["util"]),
                "mean_utilization_gpu_percent": sum(values["util"]) / len(values["util"]) if values["util"] else 0.0,
                "peak_memory_used_mb": max(values["memory"]) if values["memory"] else 0.0,
                "mean_power_w": sum(values["power"]) / len(values["power"]) if values["power"] else 0.0,
                "memory_total_mb": values["memory_total_mb"],
            }
            for index, values in per_gpu.items()
        },
    }


def extract_worker_summary(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith("BM_SENTINEL:summary:"):
            return json.loads(line.split("BM_SENTINEL:summary:", 1)[1])
    return {}


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "status",
        "row_count",
        "total_windows",
        "tasks_with_finite_actions",
        "velocity_command_mean_action_delta_l2",
        "composed_mean_action_delta_l2",
        "does_not_claim_closed_loop_rollout",
        "does_not_claim_paper_level_guidance",
    ]
    task_summaries = summary["task_summaries"]
    row = {
        "status": summary["status"],
        "row_count": summary["metrics"]["row_count"],
        "total_windows": summary["metrics"]["total_windows"],
        "tasks_with_finite_actions": summary["metrics"]["tasks_with_finite_actions"],
        "velocity_command_mean_action_delta_l2": task_summaries["velocity_command"]["mean_guided_base_action_l2"],
        "composed_mean_action_delta_l2": task_summaries["composed"]["mean_guided_base_action_l2"],
        "does_not_claim_closed_loop_rollout": summary["checks"]["does_not_claim_closed_loop_rollout"],
        "does_not_claim_paper_level_guidance": summary["checks"]["does_not_claim_paper_level_guidance"],
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"official_csv_loop_guided_decode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{SEED}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = run_dir / "official_csv_loop_guidance_vae_action_decode_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_target_gpus()
    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    monitor = start_gpu_monitor(gpu_metrics_csv)
    log_path = LOG_DIR / "level_c_official_csv_loop_guidance_vae_action_decode_eval.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "4,7",
            "PYTHONUNBUFFERED": "1",
            "BM_GUIDANCE_JSON": str(GUIDANCE_JSON),
            "BM_DIFFUSION_JSON": str(DIFFUSION_JSON),
            "BM_STATE_LATENT_JSON": str(STATE_LATENT_JSON),
            "BM_VAE_JSON": str(VAE_JSON),
            "BM_RUN_DIR": str(run_dir),
            "BM_OFFICIAL_CSV_LOOP_GUIDED_DECODE_SEED": str(SEED),
            "BM_OFFICIAL_CSV_LOOP_GUIDED_DECODE_BATCH_WINDOWS": str(BATCH_WINDOWS),
        }
    )
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.run(
            [str(BM_DIFFUSION_PY), str(worker)],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        monitor.terminate()
        monitor.wait(timeout=10)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait(timeout=10)
    duration = round(time.time() - start, 3)
    output = log_path.read_text(encoding="utf-8", errors="replace")
    worker_summary = extract_worker_summary(output)
    failed_log_copy = ""
    if proc.returncode != 0 or not worker_summary:
        failed_log = FAILED_DIR / "level_c_official_csv_loop_guidance_vae_action_decode_eval.log"
        failed_log.write_text(output, encoding="utf-8", errors="replace")
        failed_log_copy = str(failed_log)
    gpu_summary = summarize_gpu_metrics(gpu_metrics_csv)
    status = "ok_official_csv_loop_guidance_vae_action_decode_eval" if (
        proc.returncode == 0 and worker_summary and all(worker_summary.get("checks", {}).values())
    ) else "failed"
    summary = {
        "status": status,
        "experiment_type": "official_csv_loop_guidance_vae_action_decode_eval",
        "scope": (
            "Offline VAE action decoding of guided official-csv-loop state-latent denoiser outputs. "
            "This is a bridge toward closed-loop rollout, not closed-loop IsaacLab control."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "settings": {
            "selected_physical_gpus": CANDIDATE_GPUS,
            "cuda_visible_devices": "4,7",
            "seed": SEED,
            "batch_windows": BATCH_WINDOWS,
            "tasks": TASKS,
        },
        "gpu_guard": guard,
        "gpu_metrics_summary": gpu_summary,
        "worker_summary": worker_summary,
        "checks": {
            "bm_diffusion_python_exists": BM_DIFFUSION_PY.is_file(),
            "guidance_summary_exists": GUIDANCE_JSON.is_file(),
            "diffusion_summary_exists": DIFFUSION_JSON.is_file(),
            "state_latent_dataset_exists": STATE_LATENT_JSON.is_file(),
            "vae_summary_exists": VAE_JSON.is_file(),
            "process_returned_zero": proc.returncode == 0,
            "worker_summary_recorded": bool(worker_summary),
            "uses_official_csv_loop_guidance": bool(worker_summary.get("checks", {}).get("uses_official_csv_loop_guidance")),
            "uses_official_csv_loop_diffusion": bool(worker_summary.get("checks", {}).get("uses_official_csv_loop_diffusion")),
            "uses_official_csv_loop_vae": bool(worker_summary.get("checks", {}).get("uses_official_csv_loop_vae")),
            "uses_full_validation_test_windows": bool(worker_summary.get("checks", {}).get("uses_full_validation_test_windows")),
            "all_decoded_actions_finite": bool(worker_summary.get("checks", {}).get("all_decoded_actions_finite")),
            "decoded_action_dim_29": bool(worker_summary.get("checks", {}).get("decoded_action_dim_29")),
            "does_not_claim_closed_loop_rollout": bool(worker_summary.get("checks", {}).get("does_not_claim_closed_loop_rollout")),
            "does_not_claim_fig5_fig6_reproduction": bool(worker_summary.get("checks", {}).get("does_not_claim_fig5_fig6_reproduction")),
            "does_not_claim_paper_level_guidance": bool(worker_summary.get("checks", {}).get("does_not_claim_paper_level_guidance")),
            "gpu_metrics_recorded": gpu_summary.get("exists", False),
        },
        "outputs": {
            "json": str(OUT / "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"),
            "tsv": str(OUT / "level_c_official_csv_loop_guidance_vae_action_decode_eval.tsv"),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "failed_log_copy": failed_log_copy,
            "worker_script": str(worker),
            "gpu_metrics_csv": str(gpu_metrics_csv),
        },
        "interpretation": {
            "closed_loop_rollout": False,
            "paper_level_guidance": False,
            "fig5_fig6_reproduction": False,
            "goal_complete": False,
            "boundary": (
                "This decodes guided latents into local VAE actions offline. It checks action-level feasibility "
                "before closed-loop control, but it does not execute those actions in IsaacLab."
            ),
        },
    }
    json_path = OUT / "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if worker_summary:
        write_tsv(OUT / "level_c_official_csv_loop_guidance_vae_action_decode_eval.tsv", worker_summary)
    else:
        (OUT / "level_c_official_csv_loop_guidance_vae_action_decode_eval.tsv").write_text("status\nfailed\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "json": str(json_path),
                "returncode": proc.returncode,
                "total_windows": worker_summary.get("metrics", {}).get("total_windows"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
