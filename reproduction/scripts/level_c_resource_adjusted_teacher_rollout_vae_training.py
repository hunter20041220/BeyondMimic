#!/usr/bin/env python3
"""Train a resource-adjusted conditional action VAE on the full teacher rollout shards."""

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
OUT = ROOT / "res/level_c/resource_adjusted_teacher_rollout_vae_training"
RUN_ROOT = ROOT / "res/runs/level_c_resource_adjusted_teacher_rollout_vae_training"
LOG_DIR = ROOT / "logs/level_c_resource_adjusted_teacher_rollout_vae_training"
FAILED_DIR = ROOT / "res/failed_runs/level_c_resource_adjusted_teacher_rollout_vae_training"
DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
    "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
)
CANDIDATE_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
SEED = int(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_SEED", "20260624"))
EPOCHS = int(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_BATCH_SIZE", "16384"))
LATENT_DIM = int(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_LATENT_DIM", "32"))
HIDDEN_DIM = int(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_HIDDEN_DIM", "512"))
KL_COEF = float(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_KL_COEF", "1e-4"))
LEARNING_RATE = float(os.environ.get("BM_RESOURCE_ADJUSTED_VAE_LR", "3e-4"))


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
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
TEACHER_ROLLOUT_JSON = Path(os.environ["BM_TEACHER_ROLLOUT_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
SEED = int(os.environ["BM_RESOURCE_ADJUSTED_VAE_SEED"])
EPOCHS = int(os.environ["BM_RESOURCE_ADJUSTED_VAE_EPOCHS"])
BATCH_SIZE = int(os.environ["BM_RESOURCE_ADJUSTED_VAE_BATCH_SIZE"])
LATENT_DIM = int(os.environ["BM_RESOURCE_ADJUSTED_VAE_LATENT_DIM"])
HIDDEN_DIM = int(os.environ["BM_RESOURCE_ADJUSTED_VAE_HIDDEN_DIM"])
KL_COEF = float(os.environ["BM_RESOURCE_ADJUSTED_VAE_KL_COEF"])
LEARNING_RATE = float(os.environ["BM_RESOURCE_ADJUSTED_VAE_LR"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    def forward(self, obs, action=None, deterministic=False):
        if action is None:
            raise ValueError("training forward requires action for posterior inference")
        stats = self.encoder(torch.cat([obs, action], dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        if deterministic:
            z = mu
        else:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        pred = self.decoder(torch.cat([obs, z], dim=-1))
        return pred, mu, logvar

    def decode_mean(self, obs):
        z = torch.zeros((obs.shape[0], LATENT_DIM), device=obs.device, dtype=obs.dtype)
        return self.decoder(torch.cat([obs, z], dim=-1))


def load_full_dataset():
    summary = json.loads(TEACHER_ROLLOUT_JSON.read_text(encoding="utf-8"))
    paths = summary["run"]["shard_npz_paths"]
    obs_chunks = []
    action_chunks = []
    reward_chunks = []
    done_chunks = []
    time_chunks = []
    shard_rows = []
    for path_str in paths:
        path = Path(path_str)
        with np.load(path) as data:
            obs = data["policy_obs"].reshape(-1, data["policy_obs"].shape[-1]).astype(np.float32)
            actions = data["actions"].reshape(-1, data["actions"].shape[-1]).astype(np.float32)
            rewards = data["rewards"].reshape(-1).astype(np.float32)
            dones = data["dones"].reshape(-1).astype(np.bool_)
            motion_time_steps = data["motion_time_steps"].reshape(-1).astype(np.int32)
        obs_chunks.append(obs)
        action_chunks.append(actions)
        reward_chunks.append(rewards)
        done_chunks.append(dones)
        time_chunks.append(motion_time_steps)
        shard_rows.append(
            {
                "path": str(path),
                "sample_count": int(obs.shape[0]),
                "obs_dim": int(obs.shape[1]),
                "action_dim": int(actions.shape[1]),
            }
        )
    obs = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    rewards = np.concatenate(reward_chunks, axis=0)
    dones = np.concatenate(done_chunks, axis=0)
    motion_time_steps = np.concatenate(time_chunks, axis=0)
    return summary, shard_rows, obs, actions, rewards, dones, motion_time_steps


def split_indices(count):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(count)
    train_end = int(count * 0.8)
    val_end = int(count * 0.9)
    return {
        "train": perm[:train_end],
        "validation": perm[train_end:val_end],
        "test": perm[val_end:],
    }


def evaluate(model, obs, actions, indices, device):
    model.eval()
    losses = []
    abs_errors = []
    mu_abs = []
    kl_values = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(obs[indices]), torch.from_numpy(actions[indices])),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.inference_mode():
        for obs_b, act_b in loader:
            obs_b = obs_b.to(device, non_blocking=True)
            act_b = act_b.to(device, non_blocking=True)
            pred, mu, logvar = model(obs_b, act_b, deterministic=True)
            err = pred - act_b
            losses.append(F.mse_loss(pred, act_b, reduction="mean").detach().cpu().item())
            abs_errors.append(err.abs().mean().detach().cpu().item())
            mu_abs.append(mu.abs().mean().detach().cpu().item())
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            kl_values.append(kl.detach().cpu().item())
    return {
        "sample_count": int(len(indices)),
        "action_mse": float(np.mean(losses)),
        "action_abs_error_mean": float(np.mean(abs_errors)),
        "latent_mu_abs_mean": float(np.mean(mu_abs)),
        "kl_mean": float(np.mean(kl_values)),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    teacher_summary, shard_rows, obs, actions, rewards, dones, motion_time_steps = load_full_dataset()
    splits = split_indices(obs.shape[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConditionalActionVAE(obs.shape[1], actions.shape[1], LATENT_DIM, HIDDEN_DIM).to(device)
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(obs[splits["train"]]), torch.from_numpy(actions[splits["train"]])),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    epoch_rows = []
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        recon_values = []
        kl_values = []
        total_values = []
        for obs_b, act_b in train_loader:
            obs_b = obs_b.to(device, non_blocking=True)
            act_b = act_b.to(device, non_blocking=True)
            pred, mu, logvar = model(obs_b, act_b, deterministic=False)
            recon = F.mse_loss(pred, act_b)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + KL_COEF * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            recon_values.append(float(recon.detach().cpu()))
            kl_values.append(float(kl.detach().cpu()))
            total_values.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            epoch_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_reconstruction_mse": float(np.mean(recon_values)),
                    "train_kl_mean": float(np.mean(kl_values)),
                    "train_total_loss": float(np.mean(total_values)),
                }
            )
            print("BM_SENTINEL:epoch:" + json.dumps(epoch_rows[-1], sort_keys=True), flush=True)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    evaluations = {
        split: evaluate(base_model, obs, actions, idx, device)
        for split, idx in splits.items()
    }
    checkpoint_path = RUN_DIR / "resource_adjusted_teacher_rollout_action_vae.pt"
    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
            "config": {
                "obs_dim": int(obs.shape[1]),
                "action_dim": int(actions.shape[1]),
                "latent_dim": LATENT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "kl_coef": KL_COEF,
                "learning_rate": LEARNING_RATE,
            },
        },
        checkpoint_path,
    )
    tsv_path = RUN_DIR / "resource_adjusted_teacher_rollout_vae_epochs.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_reconstruction_mse", "train_kl_mean", "train_total_loss"], delimiter="\t")
        writer.writeheader()
        writer.writerows(epoch_rows)
    summary = {
        "status": "ok",
        "duration_seconds": round(time.time() - start, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "data_parallel_used": torch.cuda.device_count() >= 2,
        "dataset": {
            "sample_count": int(obs.shape[0]),
            "obs_dim": int(obs.shape[1]),
            "action_dim": int(actions.shape[1]),
            "shards": shard_rows,
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "done_count": int(np.sum(dones)),
            "motion_time_step_min": int(np.min(motion_time_steps)),
            "motion_time_step_max": int(np.max(motion_time_steps)),
        },
        "splits": {split: int(len(idx)) for split, idx in splits.items()},
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "kl_coef": KL_COEF,
            "learning_rate": LEARNING_RATE,
            "epoch_rows": epoch_rows,
        },
        "evaluation": evaluations,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "epoch_tsv": str(tsv_path),
        },
        "source_teacher_rollout": {
            "json": str(TEACHER_ROLLOUT_JSON),
            "status": teacher_summary.get("status"),
            "official_dagger_rollout_dataset": False,
            "paper_level_teacher_rollout_dataset": False,
            "uses_resource_adjusted_usd": True,
        },
    }
    (RUN_DIR / "resource_adjusted_teacher_rollout_vae_training_worker_summary.json").write_text(
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
    GPU_GUARD = ROOT / "res/gpu_guard"
    GPU_GUARD.mkdir(parents=True, exist_ok=True)
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
    path = GPU_GUARD / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_resource_adjusted_vae_guard.json"
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"resource_adjusted_teacher_rollout_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{SEED}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = run_dir / "resource_adjusted_teacher_rollout_vae_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_target_gpus()
    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    monitor = start_gpu_monitor(gpu_metrics_csv)
    log_path = LOG_DIR / "level_c_resource_adjusted_teacher_rollout_vae_training.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "4,7",
            "PYTHONUNBUFFERED": "1",
            "BM_TEACHER_ROLLOUT_JSON": str(TEACHER_ROLLOUT_JSON),
            "BM_RUN_DIR": str(run_dir),
            "BM_RESOURCE_ADJUSTED_VAE_SEED": str(SEED),
            "BM_RESOURCE_ADJUSTED_VAE_EPOCHS": str(EPOCHS),
            "BM_RESOURCE_ADJUSTED_VAE_BATCH_SIZE": str(BATCH_SIZE),
            "BM_RESOURCE_ADJUSTED_VAE_LATENT_DIM": str(LATENT_DIM),
            "BM_RESOURCE_ADJUSTED_VAE_HIDDEN_DIM": str(HIDDEN_DIM),
            "BM_RESOURCE_ADJUSTED_VAE_KL_COEF": str(KL_COEF),
            "BM_RESOURCE_ADJUSTED_VAE_LR": str(LEARNING_RATE),
        }
    )
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.run(
            [str(DIFFUSION_PY), str(worker)],
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
        failed_log = FAILED_DIR / "level_c_resource_adjusted_teacher_rollout_vae_training.log"
        failed_log.write_text(output, encoding="utf-8", errors="replace")
        failed_log_copy = str(failed_log)
    gpu_summary = summarize_gpu_metrics(gpu_metrics_csv)
    teacher_summary = json.loads(TEACHER_ROLLOUT_JSON.read_text(encoding="utf-8")) if TEACHER_ROLLOUT_JSON.is_file() else {}
    expected_sample_count = int(
        teacher_summary.get("aggregate_metrics", {}).get(
            "total_env_steps",
            worker_summary.get("dataset", {}).get("sample_count", 0),
        )
    )
    expected_train = int(expected_sample_count * 0.8)
    expected_validation = int(expected_sample_count * 0.9) - expected_train
    expected_test = expected_sample_count - expected_train - expected_validation
    expected_splits = {
        "train": expected_train,
        "validation": expected_validation,
        "test": expected_test,
    }
    checks = {
        "bm_diffusion_python_exists": DIFFUSION_PY.is_file(),
        "teacher_rollout_summary_exists": TEACHER_ROLLOUT_JSON.is_file(),
        "process_returned_zero": proc.returncode == 0,
        "worker_summary_recorded": bool(worker_summary),
        "uses_full_teacher_rollout_dataset": worker_summary.get("dataset", {}).get("sample_count")
        == expected_sample_count,
        "uses_two_visible_gpus": worker_summary.get("torch_cuda_device_count", 0) >= 2
        and worker_summary.get("cuda_visible_devices") == "4,7",
        "data_parallel_used": worker_summary.get("data_parallel_used") is True,
        "train_validation_test_splits_present": worker_summary.get("splits") == expected_splits,
        "test_action_mse_finite": bool(worker_summary)
        and worker_summary.get("evaluation", {}).get("test", {}).get("action_mse", float("inf")) < float("inf"),
        "test_action_mse_below_initial_action_variance_proxy": bool(worker_summary)
        and worker_summary.get("evaluation", {}).get("test", {}).get("action_mse", 1e9)
        < 2.0,
        "gpu_metrics_recorded": gpu_summary.get("row_count", 0) > 0,
        "checkpoint_written_to_ignored_runs_dir": bool(worker_summary)
        and str(worker_summary.get("outputs", {}).get("checkpoint", "")).startswith(str(RUN_ROOT)),
        "does_not_claim_official_dagger": True,
        "does_not_claim_paper_level_vae": True,
        "does_not_claim_goal_complete": True,
    }
    status = "ok" if all(checks.values()) else "failed"
    tsv_path = OUT / "level_c_resource_adjusted_teacher_rollout_vae_training.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["split", "sample_count", "action_mse", "action_abs_error_mean", "latent_mu_abs_mean", "kl_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for split, metrics in worker_summary.get("evaluation", {}).items():
            writer.writerow({"split": split, **{key: metrics.get(key) for key in fieldnames if key != "split"}})
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "resource_adjusted_teacher_rollout_conditional_action_vae_training",
        "scope": (
            "Trains a conditional action VAE on the full local resource-adjusted teacher rollout dataset. This uses "
            "all retained rollout shards from the generated USD/resource-adjusted task stack and is not official "
            "BeyondMimic DAgger data, not an official VAE checkpoint, and not paper-level closed-loop VAE evaluation."
        ),
        "duration_seconds": duration,
        "returncode": proc.returncode,
        "settings": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "kl_coef": KL_COEF,
            "learning_rate": LEARNING_RATE,
            "visible_gpus": CANDIDATE_GPUS,
        },
        "worker_summary": worker_summary,
        "gpu_metrics_summary": gpu_summary,
        "gpu_guard": guard,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "resource_adjusted_full_rollout_training_only",
            "why_not_complete": (
                "This advances the downstream VAE path from tiny debug fixtures to a full local resource-adjusted "
                "teacher-rollout dataset. It still cannot be reported as the paper's official DAgger/VAE result "
                "because the official G1 converter/replay gate is blocked and the dataset/checkpoint are produced "
                "from a generated resource-adjusted asset and local PPO teacher candidate."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_resource_adjusted_teacher_rollout_vae_training.json"),
            "tsv": str(tsv_path),
            "log": str(log_path),
            "failed_log_copy": failed_log_copy,
            "run_dir": str(run_dir),
            "worker_script": str(worker),
            "gpu_metrics_csv": str(gpu_metrics_csv),
        },
    }
    (OUT / "level_c_resource_adjusted_teacher_rollout_vae_training.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"], "returncode": proc.returncode}, sort_keys=True))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
