#!/usr/bin/env python3
"""Offline guidance evaluation for the resource-adjusted state-latent denoiser."""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/resource_adjusted_state_latent_guidance_eval"
RUN_ROOT = ROOT / "res/runs/level_c_resource_adjusted_state_latent_guidance_eval"
LOG_DIR = ROOT / "logs/level_c_resource_adjusted_state_latent_guidance_eval"
FAILED_DIR = ROOT / "res/failed_runs/level_c_resource_adjusted_state_latent_guidance_eval"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_state_latent_diffusion_training/"
    "level_c_resource_adjusted_state_latent_diffusion_training.json"
)
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
    "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
)
CANDIDATE_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
SEED = int(os.environ.get("BM_RESOURCE_ADJUSTED_GUIDANCE_SEED", "20260627"))
MAX_WINDOWS_PER_SPLIT = int(os.environ.get("BM_RESOURCE_ADJUSTED_GUIDANCE_MAX_WINDOWS_PER_SPLIT", "4096"))
BATCH_WINDOWS = int(os.environ.get("BM_RESOURCE_ADJUSTED_GUIDANCE_BATCH_WINDOWS", "512"))
SCALES = os.environ.get("BM_RESOURCE_ADJUSTED_GUIDANCE_SCALES", "0,0.0005,0.001,0.002,0.005,0.01")
TASKS = ["velocity_command", "latent_smoothness", "latent_magnitude", "composed"]


def cuda_visible_devices() -> str:
    return ",".join(str(gpu) for gpu in CANDIDATE_GPUS)


WORKER_CODE = r"""
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DIFFUSION_JSON = Path(os.environ["BM_DIFFUSION_JSON"])
STATE_LATENT_JSON = Path(os.environ["BM_STATE_LATENT_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
SEED = int(os.environ["BM_RESOURCE_ADJUSTED_GUIDANCE_SEED"])
MAX_WINDOWS_PER_SPLIT = int(os.environ["BM_RESOURCE_ADJUSTED_GUIDANCE_MAX_WINDOWS_PER_SPLIT"])
BATCH_WINDOWS = int(os.environ["BM_RESOURCE_ADJUSTED_GUIDANCE_BATCH_WINDOWS"])
SCALES = [float(item) for item in os.environ["BM_RESOURCE_ADJUSTED_GUIDANCE_SCALES"].split(",") if item.strip()]
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


def alpha_bars(steps):
    betas = torch.linspace(1e-4, 0.02, steps)
    return torch.cumprod(1.0 - betas, dim=0)


def read_window_index(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def select_rows(rows):
    rng = np.random.default_rng(SEED)
    selected = []
    counts = {}
    for split in ["validation", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if MAX_WINDOWS_PER_SPLIT > 0 and len(split_rows) > MAX_WINDOWS_PER_SPLIT:
            idx = rng.choice(len(split_rows), size=MAX_WINDOWS_PER_SPLIT, replace=False)
            split_rows = [split_rows[int(i)] for i in idx]
        counts[split] = len(split_rows)
        selected.extend(split_rows)
    return selected, counts


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


def direction(task):
    return "lower_is_better"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    start_time = time.time()
    diffusion_summary = json.loads(DIFFUSION_JSON.read_text(encoding="utf-8"))
    dataset_summary = json.loads(STATE_LATENT_JSON.read_text(encoding="utf-8"))
    ckpt = Path(diffusion_summary["worker_summary"]["outputs"]["checkpoint"])
    payload = torch.load(ckpt, map_location="cpu")
    cfg = payload["config"]
    model = StateLatentDenoiser(cfg["token_dim"], cfg["hidden_dim"], cfg["denoising_steps"])
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    bars = alpha_bars(cfg["denoising_steps"]).to(device)
    rows = read_window_index(dataset_summary["worker_summary"]["outputs"]["window_index_csv"])
    selected_rows, split_counts = select_rows(rows)
    obs_by_rank, latent_by_rank, action_by_rank = load_arrays(dataset_summary)
    result_rows = []
    sample_npz: dict[str, np.ndarray] = {}
    for start in range(0, len(selected_rows), BATCH_WINDOWS):
        end = min(start + BATCH_WINDOWS, len(selected_rows))
        clean, actions, splits = make_batch(selected_rows, obs_by_rank, latent_by_rank, action_by_rank, start, end)
        clean = clean.to(device)
        actions = actions.to(device)
        step = torch.full(clean.shape[:2], cfg["denoising_steps"] - 1, dtype=torch.long, device=device)
        noise = torch.randn_like(clean)
        alpha = bars[step].unsqueeze(-1)
        noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
        with torch.no_grad():
            pred = model(noisy, step)
        for task in TASKS:
            variable = pred.detach().clone().requires_grad_(True)
            base_cost_vec = task_cost(task, variable, actions)
            base_cost = torch.mean(base_cost_vec)
            base_cost.backward()
            grad = variable.grad.detach()
            grad_norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=-1)
            for scale in SCALES:
                guided = (pred.detach() - scale * grad).detach()
                guided_cost_vec = task_cost(task, guided, actions)
                delta = base_cost_vec.detach() - guided_cost_vec.detach()
                for split in ["validation", "test"]:
                    mask = torch.tensor([item == split for item in splits], device=device)
                    if not bool(torch.any(mask)):
                        continue
                    result_rows.append(
                        {
                            "task": task,
                            "split": split,
                            "scale": float(scale),
                            "window_count": int(torch.sum(mask).detach().cpu()),
                            "base_cost_mean": float(torch.mean(base_cost_vec.detach()[mask]).cpu()),
                            "guided_cost_mean": float(torch.mean(guided_cost_vec.detach()[mask]).cpu()),
                            "cost_delta_mean": float(torch.mean(delta[mask]).cpu()),
                            "positive_delta_fraction": float(torch.mean((delta[mask] > 0).to(torch.float32)).cpu()),
                            "gradient_norm_mean": float(torch.mean(grad_norm[mask]).cpu()),
                            "finite": bool(torch.isfinite(guided_cost_vec[mask]).all().detach().cpu()),
                        }
                    )
            if start == 0 and task in {"velocity_command", "composed"}:
                sample_npz[f"pred_{task}_batch0"] = pred[: min(8, pred.shape[0])].detach().cpu().numpy()
                sample_npz[f"guided_{task}_max_scale_batch0"] = (
                    pred[: min(8, pred.shape[0])].detach() - max(SCALES) * grad[: min(8, grad.shape[0])]
                ).detach().cpu().numpy()
    # Aggregate duplicate mini-batch rows into task/split/scale rows.
    grouped: dict[tuple[str, str, float], list[dict[str, float]]] = {}
    for row in result_rows:
        grouped.setdefault((row["task"], row["split"], row["scale"]), []).append(row)
    aggregate_rows = []
    for (task, split, scale), items in sorted(grouped.items()):
        total = sum(item["window_count"] for item in items)
        def wavg(key):
            return sum(item[key] * item["window_count"] for item in items) / total
        aggregate_rows.append(
            {
                "task": task,
                "split": split,
                "scale": scale,
                "window_count": total,
                "base_cost_mean": wavg("base_cost_mean"),
                "guided_cost_mean": wavg("guided_cost_mean"),
                "cost_delta_mean": wavg("cost_delta_mean"),
                "positive_delta_fraction": wavg("positive_delta_fraction"),
                "gradient_norm_mean": wavg("gradient_norm_mean"),
                "finite": all(item["finite"] for item in items),
            }
        )
    best_rows = []
    task_summaries = {}
    for task in TASKS:
        best_by_split = []
        for split in ["validation", "test"]:
            candidates = [row for row in aggregate_rows if row["task"] == task and row["split"] == split and row["scale"] > 0]
            best = max(candidates, key=lambda row: row["cost_delta_mean"])
            best_by_split.append(best)
            best_rows.append(best)
        task_summaries[task] = {
            "scale_count": len(SCALES),
            "splits": {row["split"]: row for row in best_by_split},
            "mean_best_cost_delta": float(np.mean([row["cost_delta_mean"] for row in best_by_split])),
            "mean_positive_delta_fraction": float(np.mean([row["positive_delta_fraction"] for row in best_by_split])),
            "all_best_costs_improve": all(row["cost_delta_mean"] > 0.0 for row in best_by_split),
            "all_best_gradients_nonzero": all(row["gradient_norm_mean"] > 0.0 for row in best_by_split),
        }
    rows_tsv = RUN_DIR / "resource_adjusted_state_latent_guidance_rows.tsv"
    with rows_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(aggregate_rows[0].keys()),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)
    npz_path = RUN_DIR / "resource_adjusted_state_latent_guidance_samples.npz"
    if sample_npz:
        np.savez_compressed(npz_path, **sample_npz)
    summary = {
        "status": "ok",
        "duration_seconds": round(time.time() - start_time, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "source_diffusion": {
            "json": str(DIFFUSION_JSON),
            "checkpoint": str(ckpt),
            "status": diffusion_summary.get("status"),
            "official_beyondmimic_diffusion_checkpoint": False,
            "paper_level_diffusion_checkpoint": False,
        },
        "source_dataset": {
            "json": str(STATE_LATENT_JSON),
            "status": dataset_summary.get("status"),
            "official_dagger_rollout_dataset": False,
            "paper_level_state_latent_dataset": False,
        },
        "settings": {
            "seed": SEED,
            "max_windows_per_split": MAX_WINDOWS_PER_SPLIT,
            "batch_windows": BATCH_WINDOWS,
            "scales": SCALES,
            "tasks": TASKS,
            "selected_split_counts": split_counts,
        },
        "metrics": {
            "row_count": len(aggregate_rows),
            "best_row_count": len(best_rows),
            "task_count": len(TASKS),
            "split_count": 2,
            "total_selected_windows": len(selected_rows),
            "tasks_with_all_best_costs_improve": int(sum(v["all_best_costs_improve"] for v in task_summaries.values())),
            "tasks_with_nonzero_best_gradients": int(sum(v["all_best_gradients_nonzero"] for v in task_summaries.values())),
        },
        "task_summaries": task_summaries,
        "rows": aggregate_rows,
        "outputs": {
            "rows_tsv": str(rows_tsv),
            "samples_npz": str(npz_path) if sample_npz else "",
        },
        "checks": {
            "uses_resource_adjusted_diffusion_checkpoint": diffusion_summary.get("status") == "ok",
            "uses_resource_adjusted_state_latent_dataset": dataset_summary.get("status") == "ok",
            "validation_and_test_splits_evaluated": sorted(split_counts) == ["test", "validation"],
            "all_tasks_evaluated": sorted(TASKS) == sorted({row["task"] for row in aggregate_rows}),
            "scale_grid_includes_unguided": 0.0 in SCALES and len(SCALES) >= 5,
            "row_count_matches_tasks_splits_scales": len(aggregate_rows) == len(TASKS) * 2 * len(SCALES),
            "all_rows_finite": all(row["finite"] for row in aggregate_rows),
            "all_best_costs_improve": all(v["all_best_costs_improve"] for v in task_summaries.values()),
            "all_best_guidance_gradients_nonzero": all(v["all_best_gradients_nonzero"] for v in task_summaries.values()),
            "does_not_claim_closed_loop_rollout": True,
            "does_not_claim_fig5_fig6_reproduction": True,
            "does_not_claim_paper_level_guidance": True,
        },
    }
    (RUN_DIR / "resource_adjusted_state_latent_guidance_worker_summary.json").write_text(
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
    gpu_tag = "gpu" + "".join(str(gpu) for gpu in CANDIDATE_GPUS)
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gpu_tag}_wangjc_state_latent_guidance_guard.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(path)
    return summary


def start_gpu_monitor(path: Path) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "while true; do "
        "date -Is; "
        "nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw "
        f"--format=csv,noheader,nounits -i {cuda_visible_devices()}; "
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
        "best_row_count",
        "total_selected_windows",
        "tasks_with_all_best_costs_improve",
        "tasks_with_nonzero_best_gradients",
        "scales",
        "does_not_claim_closed_loop_rollout",
        "does_not_claim_paper_level_guidance",
    ]
    row = {
        "status": summary["status"],
        "row_count": summary["metrics"]["row_count"],
        "best_row_count": summary["metrics"]["best_row_count"],
        "total_selected_windows": summary["metrics"]["total_selected_windows"],
        "tasks_with_all_best_costs_improve": summary["metrics"]["tasks_with_all_best_costs_improve"],
        "tasks_with_nonzero_best_gradients": summary["metrics"]["tasks_with_nonzero_best_gradients"],
        "scales": ",".join(str(x) for x in summary["settings"]["scales"]),
        "does_not_claim_closed_loop_rollout": summary["checks"]["does_not_claim_closed_loop_rollout"],
        "does_not_claim_paper_level_guidance": summary["checks"]["does_not_claim_paper_level_guidance"],
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"resource_adjusted_state_latent_guidance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{SEED}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = run_dir / "resource_adjusted_state_latent_guidance_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_target_gpus()
    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    monitor = start_gpu_monitor(gpu_metrics_csv)
    log_path = LOG_DIR / "level_c_resource_adjusted_state_latent_guidance_eval.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices(),
            "PYTHONUNBUFFERED": "1",
            "BM_DIFFUSION_JSON": str(DIFFUSION_JSON),
            "BM_STATE_LATENT_JSON": str(STATE_LATENT_JSON),
            "BM_RUN_DIR": str(run_dir),
            "BM_RESOURCE_ADJUSTED_GUIDANCE_SEED": str(SEED),
            "BM_RESOURCE_ADJUSTED_GUIDANCE_MAX_WINDOWS_PER_SPLIT": str(MAX_WINDOWS_PER_SPLIT),
            "BM_RESOURCE_ADJUSTED_GUIDANCE_BATCH_WINDOWS": str(BATCH_WINDOWS),
            "BM_RESOURCE_ADJUSTED_GUIDANCE_SCALES": SCALES,
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
        failed_log = FAILED_DIR / "level_c_resource_adjusted_state_latent_guidance_eval.log"
        failed_log.write_text(output, encoding="utf-8", errors="replace")
        failed_log_copy = str(failed_log)
    gpu_summary = summarize_gpu_metrics(gpu_metrics_csv)
    status = "ok" if proc.returncode == 0 and worker_summary and all(worker_summary.get("checks", {}).values()) else "failed"
    summary = {
        "status": status,
        "experiment_type": "resource_adjusted_state_latent_guidance_eval",
        "scope": (
            "Offline task-cost guidance over the resource-adjusted state-latent denoiser outputs. This is not a "
            "closed-loop IsaacLab rollout or paper Fig.5/Fig.6 reproduction."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "settings": {
            "selected_physical_gpus": CANDIDATE_GPUS,
            "cuda_visible_devices": cuda_visible_devices(),
            "seed": SEED,
            "max_windows_per_split": MAX_WINDOWS_PER_SPLIT,
            "batch_windows": BATCH_WINDOWS,
            "scales": SCALES,
            "tasks": TASKS,
        },
        "gpu_guard": guard,
        "gpu_metrics_summary": gpu_summary,
        "worker_summary": worker_summary,
        "checks": {
            "bm_diffusion_python_exists": BM_DIFFUSION_PY.is_file(),
            "diffusion_summary_exists": DIFFUSION_JSON.is_file(),
            "state_latent_dataset_exists": STATE_LATENT_JSON.is_file(),
            "process_returned_zero": proc.returncode == 0,
            "worker_summary_recorded": bool(worker_summary),
            "all_tasks_evaluated": bool(worker_summary.get("checks", {}).get("all_tasks_evaluated")),
            "all_best_costs_improve": bool(worker_summary.get("checks", {}).get("all_best_costs_improve")),
            "all_best_guidance_gradients_nonzero": bool(
                worker_summary.get("checks", {}).get("all_best_guidance_gradients_nonzero")
            ),
            "does_not_claim_closed_loop_rollout": bool(
                worker_summary.get("checks", {}).get("does_not_claim_closed_loop_rollout")
            ),
            "does_not_claim_fig5_fig6_reproduction": bool(
                worker_summary.get("checks", {}).get("does_not_claim_fig5_fig6_reproduction")
            ),
            "does_not_claim_paper_level_guidance": bool(
                worker_summary.get("checks", {}).get("does_not_claim_paper_level_guidance")
            ),
            "gpu_metrics_recorded": gpu_summary.get("exists", False),
        },
        "outputs": {
            "json": str(OUT / "level_c_resource_adjusted_state_latent_guidance_eval.json"),
            "tsv": str(OUT / "level_c_resource_adjusted_state_latent_guidance_eval.tsv"),
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
                "This evaluates task-cost gradients on local resource-adjusted denoiser outputs. It advances the "
                "guidance pipeline but cannot be reported as official BeyondMimic closed-loop guidance."
            ),
        },
    }
    json_path = OUT / "level_c_resource_adjusted_state_latent_guidance_eval.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if worker_summary:
        write_tsv(OUT / "level_c_resource_adjusted_state_latent_guidance_eval.tsv", worker_summary)
    else:
        (OUT / "level_c_resource_adjusted_state_latent_guidance_eval.tsv").write_text("status\nfailed\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "returncode": proc.returncode}, sort_keys=True))


if __name__ == "__main__":
    main()
