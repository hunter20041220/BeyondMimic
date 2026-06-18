#!/usr/bin/env python3
"""Train a full resource-adjusted state-latent denoiser on teacher-rollout windows."""

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
OUT = ROOT / "res/level_c/resource_adjusted_state_latent_diffusion_training"
RUN_ROOT = ROOT / "res/runs/level_c_resource_adjusted_state_latent_diffusion_training"
LOG_DIR = ROOT / "logs/level_c_resource_adjusted_state_latent_diffusion_training"
FAILED_DIR = ROOT / "res/failed_runs/level_c_resource_adjusted_state_latent_diffusion_training"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
    "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
)
CANDIDATE_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
SEED = int(os.environ.get("BM_RESOURCE_ADJUSTED_DIFFUSION_SEED", "20260626"))
EPOCHS = int(os.environ.get("BM_RESOURCE_ADJUSTED_DIFFUSION_EPOCHS", "30"))
BATCH_WINDOWS = int(os.environ.get("BM_RESOURCE_ADJUSTED_DIFFUSION_BATCH_WINDOWS", "2048"))
HIDDEN_DIM = int(os.environ.get("BM_RESOURCE_ADJUSTED_DIFFUSION_HIDDEN_DIM", "512"))
LEARNING_RATE = float(os.environ.get("BM_RESOURCE_ADJUSTED_DIFFUSION_LR", "3e-4"))
DENOISING_STEPS = int(os.environ.get("BM_RESOURCE_ADJUSTED_DIFFUSION_STEPS", "20"))


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
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
STATE_LATENT_JSON = Path(os.environ["BM_STATE_LATENT_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
SEED = int(os.environ["BM_RESOURCE_ADJUSTED_DIFFUSION_SEED"])
EPOCHS = int(os.environ["BM_RESOURCE_ADJUSTED_DIFFUSION_EPOCHS"])
BATCH_WINDOWS = int(os.environ["BM_RESOURCE_ADJUSTED_DIFFUSION_BATCH_WINDOWS"])
HIDDEN_DIM = int(os.environ["BM_RESOURCE_ADJUSTED_DIFFUSION_HIDDEN_DIM"])
LEARNING_RATE = float(os.environ["BM_RESOURCE_ADJUSTED_DIFFUSION_LR"])
DENOISING_STEPS = int(os.environ["BM_RESOURCE_ADJUSTED_DIFFUSION_STEPS"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def alpha_bars(steps):
    betas = torch.linspace(1e-4, 0.02, steps)
    return torch.cumprod(1.0 - betas, dim=0)


class WindowDataset(Dataset):
    def __init__(self, windows, obs_by_rank, latent_by_rank, split):
        self.rows = [row for row in windows if row["split"] == split]
        self.obs_by_rank = obs_by_rank
        self.latent_by_rank = latent_by_rank

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        rank = row["rank"]
        env = row["env_index"]
        start = row["start"]
        end = row["end_exclusive"]
        obs = self.obs_by_rank[rank][start:end, env, :]
        latent = self.latent_by_rank[rank][start:end, env, :]
        token = np.concatenate([obs, latent], axis=-1).astype(np.float32)
        return torch.from_numpy(token)


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
        x = torch.cat([noisy, onehot], dim=-1)
        return self.net(x)


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


def load_arrays(dataset_summary):
    obs_by_rank = {}
    latent_by_rank = {}
    for shard in dataset_summary["worker_summary"]["shards"]:
        rank = int(shard["rank"])
        with np.load(shard["source_shard"], mmap_mode="r") as data:
            obs_by_rank[rank] = data["policy_obs"].astype(np.float32)
        with np.load(shard["latent_shard"], mmap_mode="r") as data:
            latent_by_rank[rank] = data["latent_mu"].astype(np.float32)
    return obs_by_rank, latent_by_rank


def collate(batch):
    return torch.stack(batch, dim=0)


def train_epoch(model, loader, optimizer, device, bars):
    model.train()
    losses = []
    for clean in loader:
        clean = clean.to(device, non_blocking=True)
        step = torch.randint(0, DENOISING_STEPS, clean.shape[:2], device=device)
        noise = torch.randn_like(clean)
        alpha = bars.to(device)[step].unsqueeze(-1)
        noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
        pred = model(noisy, step)
        loss = F.mse_loss(pred, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate(model, loader, device, bars):
    model.eval()
    pred_losses = []
    noisy_losses = []
    with torch.inference_mode():
        for clean in loader:
            clean = clean.to(device, non_blocking=True)
            step = torch.randint(0, DENOISING_STEPS, clean.shape[:2], device=device)
            noise = torch.randn_like(clean)
            alpha = bars.to(device)[step].unsqueeze(-1)
            noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
            pred = model(noisy, step)
            pred_losses.append(float(F.mse_loss(pred, clean).detach().cpu()))
            noisy_losses.append(float(F.mse_loss(noisy, clean).detach().cpu()))
    return {
        "pred_token_mse": float(np.mean(pred_losses)),
        "noisy_token_mse": float(np.mean(noisy_losses)),
        "denoising_improvement_ratio": float(1.0 - (np.mean(pred_losses) / max(np.mean(noisy_losses), 1e-12))),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    start = time.time()
    dataset_summary = json.loads(STATE_LATENT_JSON.read_text(encoding="utf-8"))
    worker_dataset = dataset_summary["worker_summary"]["dataset"]
    windows = read_window_index(dataset_summary["worker_summary"]["outputs"]["window_index_csv"])
    obs_by_rank, latent_by_rank = load_arrays(dataset_summary)
    train_ds = WindowDataset(windows, obs_by_rank, latent_by_rank, "train")
    val_ds = WindowDataset(windows, obs_by_rank, latent_by_rank, "validation")
    test_ds = WindowDataset(windows, obs_by_rank, latent_by_rank, "test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_WINDOWS, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_WINDOWS, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_WINDOWS, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StateLatentDenoiser(worker_dataset["token_dim"], HIDDEN_DIM, DENOISING_STEPS).to(device)
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    bars = alpha_bars(DENOISING_STEPS)
    epoch_rows = []
    best_val = math.inf
    best_epoch = 0
    checkpoint_path = RUN_DIR / "resource_adjusted_state_latent_denoiser.pt"
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, bars)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            val_metrics = evaluate(base_model, val_loader, device, bars)
            row = {"epoch": epoch + 1, "train_token_mse": train_loss, **{f"validation_{k}": v for k, v in val_metrics.items()}}
            epoch_rows.append(row)
            print("BM_SENTINEL:epoch:" + json.dumps(row, sort_keys=True), flush=True)
            if val_metrics["pred_token_mse"] < best_val:
                best_val = val_metrics["pred_token_mse"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model_state_dict": base_model.state_dict(),
                        "config": {
                            "token_dim": worker_dataset["token_dim"],
                            "obs_dim": worker_dataset["obs_dim"],
                            "latent_dim": worker_dataset["latent_dim"],
                            "sequence_length": worker_dataset["sequence_length"],
                            "hidden_dim": HIDDEN_DIM,
                            "denoising_steps": DENOISING_STEPS,
                            "seed": SEED,
                            "epochs": EPOCHS,
                            "batch_windows": BATCH_WINDOWS,
                            "learning_rate": LEARNING_RATE,
                        },
                    },
                    checkpoint_path,
                )
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    final_val = evaluate(base_model, val_loader, device, bars)
    final_test = evaluate(base_model, test_loader, device, bars)
    tsv_path = RUN_DIR / "resource_adjusted_state_latent_diffusion_epochs.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["epoch", "train_token_mse", "validation_pred_token_mse", "validation_noisy_token_mse", "validation_denoising_improvement_ratio"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(epoch_rows)
    summary = {
        "status": "ok",
        "duration_seconds": round(time.time() - start, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "data_parallel_used": torch.cuda.device_count() >= 2,
        "source_dataset": {
            "json": str(STATE_LATENT_JSON),
            "status": dataset_summary.get("status"),
            "official_dagger_rollout_dataset": False,
            "paper_level_state_latent_dataset": False,
        },
        "dataset": {
            "sample_count": worker_dataset["sample_count"],
            "window_count": worker_dataset["window_count"],
            "split_counts": worker_dataset["split_counts"],
            "sequence_length": worker_dataset["sequence_length"],
            "obs_dim": worker_dataset["obs_dim"],
            "latent_dim": worker_dataset["latent_dim"],
            "token_dim": worker_dataset["token_dim"],
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_windows": BATCH_WINDOWS,
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "denoising_steps": DENOISING_STEPS,
            "epoch_rows": epoch_rows,
            "best_validation_epoch": best_epoch,
            "best_validation_pred_token_mse": best_val,
        },
        "evaluation": {
            "validation": final_val,
            "test": final_test,
        },
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "epoch_tsv": str(tsv_path),
        },
        "checks": {
            "uses_full_window_dataset": worker_dataset["window_count"] == 285696,
            "has_train_validation_test_splits": all(worker_dataset["split_counts"][k] > 0 for k in ["train", "validation", "test"]),
            "uses_two_visible_gpus": torch.cuda.device_count() >= 2,
            "data_parallel_used": torch.cuda.device_count() >= 2,
            "test_denoising_improves_over_noisy": final_test["denoising_improvement_ratio"] > 0.0,
            "checkpoint_written_to_ignored_runs_dir": checkpoint_path.is_file() and "/res/runs/" in str(checkpoint_path),
            "does_not_claim_official_dagger": True,
            "does_not_claim_paper_level_diffusion": True,
        },
    }
    (RUN_DIR / "resource_adjusted_state_latent_diffusion_training_worker_summary.json").write_text(
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
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_state_latent_diffusion_guard.json"
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
        "window_count",
        "train_windows",
        "validation_windows",
        "test_windows",
        "epochs",
        "batch_windows",
        "validation_pred_token_mse",
        "test_pred_token_mse",
        "test_noisy_token_mse",
        "test_denoising_improvement_ratio",
        "data_parallel_used",
        "official_dagger_rollout_dataset",
        "paper_level_diffusion",
    ]
    dataset = summary["dataset"]
    splits = dataset["split_counts"]
    row = {
        "status": summary["status"],
        "window_count": dataset["window_count"],
        "train_windows": splits["train"],
        "validation_windows": splits["validation"],
        "test_windows": splits["test"],
        "epochs": summary["training"]["epochs"],
        "batch_windows": summary["training"]["batch_windows"],
        "validation_pred_token_mse": summary["evaluation"]["validation"]["pred_token_mse"],
        "test_pred_token_mse": summary["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": summary["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": summary["evaluation"]["test"]["denoising_improvement_ratio"],
        "data_parallel_used": summary["data_parallel_used"],
        "official_dagger_rollout_dataset": summary["source_dataset"]["official_dagger_rollout_dataset"],
        "paper_level_diffusion": not summary["checks"]["does_not_claim_paper_level_diffusion"],
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
    run_id = f"resource_adjusted_state_latent_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{SEED}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = run_dir / "resource_adjusted_state_latent_diffusion_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_target_gpus()
    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    monitor = start_gpu_monitor(gpu_metrics_csv)
    log_path = LOG_DIR / "level_c_resource_adjusted_state_latent_diffusion_training.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "4,7",
            "PYTHONUNBUFFERED": "1",
            "BM_STATE_LATENT_JSON": str(STATE_LATENT_JSON),
            "BM_RUN_DIR": str(run_dir),
            "BM_RESOURCE_ADJUSTED_DIFFUSION_SEED": str(SEED),
            "BM_RESOURCE_ADJUSTED_DIFFUSION_EPOCHS": str(EPOCHS),
            "BM_RESOURCE_ADJUSTED_DIFFUSION_BATCH_WINDOWS": str(BATCH_WINDOWS),
            "BM_RESOURCE_ADJUSTED_DIFFUSION_HIDDEN_DIM": str(HIDDEN_DIM),
            "BM_RESOURCE_ADJUSTED_DIFFUSION_LR": str(LEARNING_RATE),
            "BM_RESOURCE_ADJUSTED_DIFFUSION_STEPS": str(DENOISING_STEPS),
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
        failed_log = FAILED_DIR / "level_c_resource_adjusted_state_latent_diffusion_training.log"
        failed_log.write_text(output, encoding="utf-8", errors="replace")
        failed_log_copy = str(failed_log)
    gpu_summary = summarize_gpu_metrics(gpu_metrics_csv)
    status = "ok" if proc.returncode == 0 and worker_summary and all(worker_summary.get("checks", {}).values()) else "failed"
    summary = {
        "status": status,
        "experiment_type": "resource_adjusted_state_latent_diffusion_training",
        "scope": (
            "Full local resource-adjusted state-latent denoiser training over all currently generated windows; "
            "not official DAgger diffusion and not paper-level closed-loop guidance."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "settings": {
            "selected_physical_gpus": CANDIDATE_GPUS,
            "cuda_visible_devices": "4,7",
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_windows": BATCH_WINDOWS,
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "denoising_steps": DENOISING_STEPS,
        },
        "gpu_guard": guard,
        "gpu_metrics_summary": gpu_summary,
        "worker_summary": worker_summary,
        "checks": {
            "bm_diffusion_python_exists": BM_DIFFUSION_PY.is_file(),
            "state_latent_dataset_exists": STATE_LATENT_JSON.is_file(),
            "process_returned_zero": proc.returncode == 0,
            "worker_summary_recorded": bool(worker_summary),
            "uses_full_window_dataset": bool(worker_summary.get("checks", {}).get("uses_full_window_dataset")),
            "uses_two_visible_gpus": bool(worker_summary.get("checks", {}).get("uses_two_visible_gpus")),
            "data_parallel_used": bool(worker_summary.get("checks", {}).get("data_parallel_used")),
            "test_denoising_improves_over_noisy": bool(
                worker_summary.get("checks", {}).get("test_denoising_improves_over_noisy")
            ),
            "checkpoint_written_to_ignored_runs_dir": bool(
                worker_summary.get("checks", {}).get("checkpoint_written_to_ignored_runs_dir")
            ),
            "does_not_claim_official_dagger": bool(worker_summary.get("checks", {}).get("does_not_claim_official_dagger")),
            "does_not_claim_paper_level_diffusion": bool(
                worker_summary.get("checks", {}).get("does_not_claim_paper_level_diffusion")
            ),
            "gpu_metrics_recorded": gpu_summary.get("exists", False),
        },
        "outputs": {
            "json": str(OUT / "level_c_resource_adjusted_state_latent_diffusion_training.json"),
            "tsv": str(OUT / "level_c_resource_adjusted_state_latent_diffusion_training.tsv"),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "failed_log_copy": failed_log_copy,
            "worker_script": str(worker),
            "gpu_metrics_csv": str(gpu_metrics_csv),
        },
        "interpretation": {
            "official_dagger_rollout_dataset": False,
            "paper_level_diffusion": False,
            "closed_loop_guidance": False,
            "goal_complete": False,
            "boundary": (
                "This run trains a local denoising model on generated resource-adjusted windows. It advances the "
                "virtual reproduction pipeline but is not the official BeyondMimic diffusion checkpoint and does not "
                "prove Fig. 5/Fig. 6 closed-loop guidance."
            ),
        },
    }
    json_path = OUT / "level_c_resource_adjusted_state_latent_diffusion_training.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if worker_summary:
        write_tsv(OUT / "level_c_resource_adjusted_state_latent_diffusion_training.tsv", worker_summary)
    else:
        (OUT / "level_c_resource_adjusted_state_latent_diffusion_training.tsv").write_text("status\nfailed\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "returncode": proc.returncode}, sort_keys=True))


if __name__ == "__main__":
    main()
