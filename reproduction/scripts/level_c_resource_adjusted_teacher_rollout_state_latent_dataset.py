#!/usr/bin/env python3
"""Build a full resource-adjusted teacher-rollout policy-observation/latent dataset."""

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
OUT = ROOT / "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset"
RUN_ROOT = ROOT / "res/runs/level_c_resource_adjusted_teacher_rollout_state_latent_dataset"
LOG_DIR = ROOT / "logs/level_c_resource_adjusted_teacher_rollout_state_latent_dataset"
FAILED_DIR = ROOT / "res/failed_runs/level_c_resource_adjusted_teacher_rollout_state_latent_dataset"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
    "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_teacher_rollout_vae_training/"
    "level_c_resource_adjusted_teacher_rollout_vae_training.json"
)
CANDIDATE_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
SEQUENCE_LENGTH = int(os.environ.get("BM_RESOURCE_ADJUSTED_STATE_LATENT_SEQ_LEN", "21"))
BATCH_SIZE = int(os.environ.get("BM_RESOURCE_ADJUSTED_STATE_LATENT_BATCH_SIZE", "32768"))
SEED = int(os.environ.get("BM_RESOURCE_ADJUSTED_STATE_LATENT_SEED", "20260625"))


def cuda_visible_devices() -> str:
    return ",".join(str(gpu) for gpu in CANDIDATE_GPUS)


WORKER_CODE = r"""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
TEACHER_ROLLOUT_JSON = Path(os.environ["BM_TEACHER_ROLLOUT_JSON"])
VAE_TRAINING_JSON = Path(os.environ["BM_VAE_TRAINING_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
SEQUENCE_LENGTH = int(os.environ["BM_RESOURCE_ADJUSTED_STATE_LATENT_SEQ_LEN"])
BATCH_SIZE = int(os.environ["BM_RESOURCE_ADJUSTED_STATE_LATENT_BATCH_SIZE"])
SEED = int(os.environ["BM_RESOURCE_ADJUSTED_STATE_LATENT_SEED"])


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

    def encode(self, obs, action):
        stats = self.encoder(torch.cat([obs, action], dim=-1))
        return stats.chunk(2, dim=-1)

    def decode_zero(self, obs):
        z = torch.zeros((obs.shape[0], self.decoder[0].in_features - obs.shape[1]), device=obs.device, dtype=obs.dtype)
        return self.decoder(torch.cat([obs, z], dim=-1))


def split_label(start, rank, env_index):
    # Deterministic split at window level. This is local resource-adjusted data, not paper DAgger splitting.
    value = (start * 1000003 + rank * 9176 + env_index * 37 + SEED) % 10
    if value < 8:
        return "train"
    if value == 8:
        return "validation"
    return "test"


def load_model():
    vae_summary = json.loads(VAE_TRAINING_JSON.read_text(encoding="utf-8"))
    checkpoint = Path(vae_summary["worker_summary"]["outputs"]["checkpoint"])
    obj = torch.load(checkpoint, map_location="cpu")
    cfg = obj["config"]
    model = ConditionalActionVAE(
        cfg["obs_dim"],
        cfg["action_dim"],
        cfg["latent_dim"],
        cfg["hidden_dim"],
    )
    model.load_state_dict(obj["model_state_dict"])
    model.eval()
    return vae_summary, checkpoint, cfg, model


def encode_flat(model, obs, actions, device):
    mus = []
    logvars = []
    recon_mses = []
    with torch.inference_mode():
        for start in range(0, obs.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, obs.shape[0])
            obs_b = torch.from_numpy(obs[start:end]).to(device)
            act_b = torch.from_numpy(actions[start:end]).to(device)
            mu, logvar = model.encode(obs_b, act_b)
            pred = model.decoder(torch.cat([obs_b, mu], dim=-1))
            recon_mses.append(torch.mean(torch.square(pred - act_b)).detach().cpu().item())
            mus.append(mu.detach().cpu().numpy().astype(np.float32))
            logvars.append(logvar.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(mus, axis=0), np.concatenate(logvars, axis=0), float(np.mean(recon_mses))


def build_window_rows(rank, env_count, step_count):
    rows = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for env_index in range(env_count):
        for start in range(0, step_count - SEQUENCE_LENGTH + 1):
            split = split_label(start, rank, env_index)
            split_counts[split] += 1
            rows.append(
                {
                    "rank": rank,
                    "env_index": env_index,
                    "start": start,
                    "end_exclusive": start + SEQUENCE_LENGTH,
                    "split": split,
                }
            )
    return rows, split_counts


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    teacher_summary = json.loads(TEACHER_ROLLOUT_JSON.read_text(encoding="utf-8"))
    vae_summary, checkpoint, cfg, model = load_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    source_shards = teacher_summary["run"]["shard_npz_paths"]
    shard_summaries = []
    all_window_rows = []
    aggregate_split_counts = {"train": 0, "validation": 0, "test": 0}
    total_samples = 0
    total_windows = 0
    weighted_recon = []
    for shard_path_str in source_shards:
        shard_path = Path(shard_path_str)
        with np.load(shard_path) as data:
            policy_obs = data["policy_obs"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            rewards = data["rewards"].astype(np.float32)
            dones = data["dones"].astype(np.bool_)
            timeouts = data["timeouts"].astype(np.bool_)
            motion_time_steps = data["motion_time_steps"].astype(np.int32)
            rank = int(data["rank"][0])
            world_size = int(data["world_size"][0])
            seed = int(data["seed"][0])
        step_count, env_count, obs_dim = policy_obs.shape
        action_dim = actions.shape[-1]
        flat_obs = policy_obs.reshape(-1, obs_dim)
        flat_actions = actions.reshape(-1, action_dim)
        mu, logvar, recon_mse = encode_flat(model, flat_obs, flat_actions, device)
        mu = mu.reshape(step_count, env_count, cfg["latent_dim"])
        logvar = logvar.reshape(step_count, env_count, cfg["latent_dim"])
        window_rows, split_counts = build_window_rows(rank, env_count, step_count)
        for key, value in split_counts.items():
            aggregate_split_counts[key] += value
        total_samples += int(flat_obs.shape[0])
        total_windows += len(window_rows)
        weighted_recon.append((recon_mse, int(flat_obs.shape[0])))
        latent_npz = RUN_DIR / f"rank_{rank}_policy_obs_action_latent_sequences.npz"
        np.savez_compressed(
            latent_npz,
            latent_mu=mu,
            latent_logvar=logvar,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            motion_time_steps=motion_time_steps,
            rank=np.asarray([rank], dtype=np.int32),
            world_size=np.asarray([world_size], dtype=np.int32),
            seed=np.asarray([seed], dtype=np.int32),
        )
        shard_summary = {
            "rank": rank,
            "source_shard": str(shard_path),
            "latent_shard": str(latent_npz),
            "latent_shard_size_bytes": latent_npz.stat().st_size,
            "step_count": int(step_count),
            "env_count": int(env_count),
            "sample_count": int(flat_obs.shape[0]),
            "window_count": len(window_rows),
            "split_counts": split_counts,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "latent_dim": int(cfg["latent_dim"]),
            "posterior_reconstruction_mse": recon_mse,
            "latent_mu_abs_mean": float(np.mean(np.abs(mu))),
            "latent_mu_std": float(np.std(mu)),
            "latent_logvar_mean": float(np.mean(logvar)),
            "done_count": int(np.sum(dones)),
            "timeout_count": int(np.sum(timeouts)),
        }
        shard_summaries.append(shard_summary)
        all_window_rows.extend(window_rows)
        print("BM_SENTINEL:shard:" + json.dumps(shard_summary, sort_keys=True), flush=True)
    window_csv = RUN_DIR / "resource_adjusted_state_latent_window_index.csv"
    with window_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "env_index", "start", "end_exclusive", "split"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(all_window_rows)
    weighted_recon_mse = sum(v * n for v, n in weighted_recon) / max(sum(n for _, n in weighted_recon), 1)
    summary = {
        "status": "ok",
        "duration_seconds": round(time.time() - start_time, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "device": str(device),
        "source_teacher_rollout": {
            "json": str(TEACHER_ROLLOUT_JSON),
            "status": teacher_summary.get("status"),
            "official_dagger_rollout_dataset": False,
            "paper_level_teacher_rollout_dataset": False,
            "uses_resource_adjusted_usd": True,
        },
        "source_vae": {
            "json": str(VAE_TRAINING_JSON),
            "checkpoint": str(checkpoint),
            "status": vae_summary.get("status"),
            "official_beyondmimic_vae_checkpoint": False,
            "paper_level_vae_checkpoint": False,
            "config": cfg,
        },
        "dataset": {
            "sequence_length": SEQUENCE_LENGTH,
            "sample_count": total_samples,
            "window_count": total_windows,
            "split_counts": aggregate_split_counts,
            "obs_dim": int(cfg["obs_dim"]),
            "action_dim": int(cfg["action_dim"]),
            "latent_dim": int(cfg["latent_dim"]),
            "token_dim": int(cfg["obs_dim"] + cfg["latent_dim"]),
            "state_source": "policy_obs in original resource-adjusted teacher rollout shards",
            "latent_source": "posterior mean/logvar from local resource-adjusted conditional action VAE",
            "weighted_posterior_reconstruction_mse": float(weighted_recon_mse),
        },
        "shards": shard_summaries,
        "outputs": {
            "run_dir": str(RUN_DIR),
            "window_index_csv": str(window_csv),
            "latent_shards": [row["latent_shard"] for row in shard_summaries],
        },
        "checks": {
            "uses_full_teacher_rollout_samples": total_samples == 306176,
            "uses_two_rollout_shards": len(shard_summaries) == 2,
            "window_count_matches_full_sequences": total_windows == (299 - SEQUENCE_LENGTH + 1) * 1024,
            "has_train_validation_test_splits": all(aggregate_split_counts[k] > 0 for k in ["train", "validation", "test"]),
            "latent_dim_matches_vae": cfg["latent_dim"] == 32,
            "obs_dim_matches_teacher_policy_obs": cfg["obs_dim"] == 160,
            "does_not_claim_official_dagger": True,
            "does_not_claim_paper_level_state_latent_dataset": True,
        },
    }
    (RUN_DIR / "resource_adjusted_teacher_rollout_state_latent_dataset_worker_summary.json").write_text(
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
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gpu_tag}_wangjc_state_latent_dataset_guard.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(path)
    return summary


def start_gpu_monitor(path: Path) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpu_indices = cuda_visible_devices()
    script = (
        "while true; do "
        "date -Is; "
        "nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw "
        f"--format=csv,noheader,nounits -i {gpu_indices}; "
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
        "sample_count",
        "window_count",
        "train_windows",
        "validation_windows",
        "test_windows",
        "sequence_length",
        "obs_dim",
        "latent_dim",
        "token_dim",
        "weighted_posterior_reconstruction_mse",
        "official_dagger_rollout_dataset",
        "paper_level_state_latent_dataset",
    ]
    dataset = summary["dataset"]
    splits = dataset["split_counts"]
    row = {
        "status": summary["status"],
        "sample_count": dataset["sample_count"],
        "window_count": dataset["window_count"],
        "train_windows": splits["train"],
        "validation_windows": splits["validation"],
        "test_windows": splits["test"],
        "sequence_length": dataset["sequence_length"],
        "obs_dim": dataset["obs_dim"],
        "latent_dim": dataset["latent_dim"],
        "token_dim": dataset["token_dim"],
        "weighted_posterior_reconstruction_mse": dataset["weighted_posterior_reconstruction_mse"],
        "official_dagger_rollout_dataset": summary["source_teacher_rollout"]["official_dagger_rollout_dataset"],
        "paper_level_state_latent_dataset": not summary["checks"]["does_not_claim_paper_level_state_latent_dataset"],
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
    run_id = f"resource_adjusted_state_latent_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{SEED}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = run_dir / "resource_adjusted_teacher_rollout_state_latent_dataset_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_target_gpus()
    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    monitor = start_gpu_monitor(gpu_metrics_csv)
    log_path = LOG_DIR / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices(),
            "PYTHONUNBUFFERED": "1",
            "BM_TEACHER_ROLLOUT_JSON": str(TEACHER_ROLLOUT_JSON),
            "BM_VAE_TRAINING_JSON": str(VAE_TRAINING_JSON),
            "BM_RUN_DIR": str(run_dir),
            "BM_RESOURCE_ADJUSTED_STATE_LATENT_SEQ_LEN": str(SEQUENCE_LENGTH),
            "BM_RESOURCE_ADJUSTED_STATE_LATENT_BATCH_SIZE": str(BATCH_SIZE),
            "BM_RESOURCE_ADJUSTED_STATE_LATENT_SEED": str(SEED),
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
        failed_log = FAILED_DIR / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.log"
        failed_log.write_text(output, encoding="utf-8", errors="replace")
        failed_log_copy = str(failed_log)
    gpu_summary = summarize_gpu_metrics(gpu_metrics_csv)
    status = "ok" if proc.returncode == 0 and worker_summary and all(worker_summary.get("checks", {}).values()) else "failed"
    summary = {
        "status": status,
        "experiment_type": "resource_adjusted_teacher_rollout_state_latent_dataset",
        "scope": (
            "Full local resource-adjusted policy-observation/action-latent sequence dataset for downstream "
            "diffusion experiments; not official DAgger and not paper-level state-latent data."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "settings": {
            "selected_physical_gpus": CANDIDATE_GPUS,
            "cuda_visible_devices": cuda_visible_devices(),
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "gpu_guard": guard,
        "gpu_metrics_summary": gpu_summary,
        "worker_summary": worker_summary,
        "checks": {
            "bm_diffusion_python_exists": BM_DIFFUSION_PY.is_file(),
            "teacher_rollout_summary_exists": TEACHER_ROLLOUT_JSON.is_file(),
            "vae_training_summary_exists": VAE_TRAINING_JSON.is_file(),
            "process_returned_zero": proc.returncode == 0,
            "worker_summary_recorded": bool(worker_summary),
            "uses_full_teacher_rollout_samples": bool(
                worker_summary.get("checks", {}).get("uses_full_teacher_rollout_samples")
            ),
            "has_full_window_index": bool(
                worker_summary.get("checks", {}).get("window_count_matches_full_sequences")
            ),
            "has_train_validation_test_splits": bool(
                worker_summary.get("checks", {}).get("has_train_validation_test_splits")
            ),
            "does_not_claim_official_dagger": bool(
                worker_summary.get("checks", {}).get("does_not_claim_official_dagger")
            ),
            "does_not_claim_paper_level_state_latent_dataset": bool(
                worker_summary.get("checks", {}).get("does_not_claim_paper_level_state_latent_dataset")
            ),
            "gpu_metrics_recorded": gpu_summary.get("exists", False),
        },
        "outputs": {
            "json": str(OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"),
            "tsv": str(OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv"),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "failed_log_copy": failed_log_copy,
            "worker_script": str(worker),
            "gpu_metrics_csv": str(gpu_metrics_csv),
        },
        "interpretation": {
            "official_dagger_rollout_dataset": False,
            "paper_level_state_latent_dataset": False,
            "goal_complete": False,
            "boundary": (
                "This dataset is generated from the local resource-adjusted teacher rollout and local action VAE. "
                "It is suitable for resource-adjusted diffusion experiments but cannot be reported as the official "
                "BeyondMimic DAgger/state-latent dataset."
            ),
        },
    }
    json_path = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if worker_summary:
        write_tsv(OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv", worker_summary)
    else:
        (OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv").write_text(
            "status\nfailed\n", encoding="utf-8"
        )
    print(json.dumps({"status": status, "json": str(json_path), "returncode": proc.returncode}, sort_keys=True))


if __name__ == "__main__":
    main()
