#!/usr/bin/env python3
"""Paper-contract Transformer state-latent diffusion training entrypoint.

This script is the corrected successor to the older local MLP denoiser route.
By default it performs a dry run only: instantiate the paper-scale Transformer,
load a tiny subset of the current local state-latent window dataset, execute one
forward/backward step, and write a contract audit.  A full training run must be
requested explicitly with ``--full-train``.

The current data source is still local teacher/VAE rollouts, not the unreleased
official BeyondMimic DAgger/state-latent dataset.  Passing this script's dry run
therefore proves the code contract, not a paper-level diffusion result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
sys.path.insert(0, str(ROOT / "reproduction/scripts"))
from beyondmimic_training_hard_gate_utils import (  # noqa: E402
    pretraining_permission_block_reasons,
    state_latent_dataset_block_reasons,
    write_blocked_summary,
)

OUT = ROOT / "res/level_c/paper_contract_transformer_state_latent_diffusion_training"
RUN_ROOT = ROOT / "res/runs/level_c_paper_contract_transformer_state_latent_diffusion_training"
LOG_DIR = ROOT / "logs/level_c_paper_contract_transformer_state_latent_diffusion_training"
FAILED_DIR = ROOT / "res/failed_runs/level_c_paper_contract_transformer_state_latent_diffusion_training"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json"
)

DEFAULT_GPUS = [
    int(item.strip())
    for item in os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_GPUS", "5,6").split(",")
    if item.strip()
]


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


STATE_LATENT_JSON = Path(os.environ["BM_STATE_LATENT_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
SEED = int(os.environ["BM_SEED"])
DRY_RUN = os.environ["BM_DRY_RUN"] == "1"
EPOCHS = int(os.environ["BM_EPOCHS"])
BATCH_WINDOWS = int(os.environ["BM_BATCH_WINDOWS"])
DRY_RUN_MAX_WINDOWS = int(os.environ["BM_DRY_RUN_MAX_WINDOWS"])
EMBED_DIM = int(os.environ["BM_EMBED_DIM"])
ATTENTION_HEADS = int(os.environ["BM_ATTENTION_HEADS"])
TRANSFORMER_LAYERS = int(os.environ["BM_TRANSFORMER_LAYERS"])
DENOISING_STEPS = int(os.environ["BM_DENOISING_STEPS"])
LEARNING_RATE = float(os.environ["BM_LEARNING_RATE"])
WEIGHT_DECAY = float(os.environ["BM_WEIGHT_DECAY"])
WARMUP_STEPS = int(os.environ["BM_WARMUP_STEPS"])
EMA_POWER = float(os.environ["BM_EMA_POWER"])
EMA_MAX = float(os.environ["BM_EMA_MAX"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def alpha_bars(steps):
    # The paper reports 20 denoising steps but not the public beta schedule.
    # This local route uses the standard DDPM linear beta schedule and records it
    # explicitly so it is not confused with an unreleased official setting.
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
                    "rank_window_index": int(row["rank_window_index"]) if row.get("rank_window_index") not in (None, "") else None,
                }
            )
    return rows


def load_arrays(dataset_summary):
    obs_by_rank = {}
    latent_by_rank = {}
    for shard in dataset_summary["worker_summary"]["shards"]:
        rank = int(shard["rank"])
        with np.load(shard["latent_shard"], mmap_mode="r") as latent_data:
            if "state_windows" in latent_data.files and "latent_windows" in latent_data.files:
                obs_by_rank[rank] = latent_data["state_windows"]
                latent_by_rank[rank] = latent_data["latent_windows"]
            else:
                raise RuntimeError("Paper-contract Transformer diffusion requires window-level state_windows/latent_windows in the state-latent shard")
    return obs_by_rank, latent_by_rank


class WindowDataset(Dataset):
    def __init__(self, windows, obs_by_rank, latent_by_rank, split, max_rows=0):
        rows = [row for row in windows if row["split"] == split]
        self.rows = rows[:max_rows] if max_rows else rows
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
        if row.get("rank_window_index") is not None:
            index = row["rank_window_index"]
            obs = self.obs_by_rank[rank][index].astype(np.float32)
            latent = self.latent_by_rank[rank][index].astype(np.float32)
        else:
            obs = self.obs_by_rank[rank][start:end, env, :].astype(np.float32)
            latent = self.latent_by_rank[rank][start:end, env, :].astype(np.float32)
        return torch.from_numpy(np.concatenate([obs, latent], axis=-1))


class StateLatentDiffusionTransformer(nn.Module):
    def __init__(self, token_dim, obs_dim, latent_dim, seq_len):
        super().__init__()
        self.token_dim = token_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_proj = nn.Linear(token_dim, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, EMBED_DIM))
        self.state_step_embed = nn.Embedding(DENOISING_STEPS, EMBED_DIM)
        self.latent_step_embed = nn.Embedding(DENOISING_STEPS, EMBED_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.output_proj = nn.Linear(EMBED_DIM, token_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, noisy_tau, steps_pair):
        x = self.input_proj(noisy_tau)
        x = x + self.pos_embed[:, : noisy_tau.shape[1]]
        x = x + self.state_step_embed(steps_pair[..., 0])
        x = x + self.latent_step_embed(steps_pair[..., 1])
        return self.output_proj(self.encoder(x))


class EMA:
    def __init__(self, model):
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model, step):
        decay = min(EMA_MAX, ((1.0 + step) / (10.0 + step)) ** EMA_POWER)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        return decay


def collate(batch):
    return torch.stack(batch, dim=0)


def add_noise(clean, bars, obs_dim):
    device = clean.device
    steps_pair = torch.randint(0, DENOISING_STEPS, (*clean.shape[:2], 2), device=device)
    noise = torch.randn_like(clean)
    alpha_state = bars.to(device)[steps_pair[..., 0]].unsqueeze(-1)
    alpha_latent = bars.to(device)[steps_pair[..., 1]].unsqueeze(-1)
    clean_state, clean_latent = clean[..., :obs_dim], clean[..., obs_dim:]
    noise_state, noise_latent = noise[..., :obs_dim], noise[..., obs_dim:]
    noisy_state = torch.sqrt(alpha_state) * clean_state + torch.sqrt(1.0 - alpha_state) * noise_state
    noisy_latent = torch.sqrt(alpha_latent) * clean_latent + torch.sqrt(1.0 - alpha_latent) * noise_latent
    return torch.cat([noisy_state, noisy_latent], dim=-1), steps_pair


def lr_for_step(base_lr, step, total_steps):
    if step <= WARMUP_STEPS:
        return base_lr * step / max(WARMUP_STEPS, 1)
    denom = max(total_steps - WARMUP_STEPS, 1)
    progress = min(max((step - WARMUP_STEPS) / denom, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(model, loader, optimizer, bars, device, obs_dim, ema, epoch, total_steps, global_step):
    model.train()
    losses = []
    ema_decays = []
    for clean in loader:
        clean = clean.to(device, non_blocking=True)
        noisy, steps_pair = add_noise(clean, bars, obs_dim)
        pred = model(noisy, steps_pair)
        loss = F.mse_loss(pred, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        global_step += 1
        lr = lr_for_step(LEARNING_RATE, global_step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        ema_decays.append(float(ema.update(model, global_step)))
        losses.append(float(loss.detach().cpu()))
        if DRY_RUN:
            break
    return float(np.mean(losses)), global_step, float(np.mean(ema_decays))


def evaluate(model, loader, bars, device, obs_dim):
    model.eval()
    pred_losses = []
    noisy_losses = []
    with torch.inference_mode():
        for clean in loader:
            clean = clean.to(device, non_blocking=True)
            noisy, steps_pair = add_noise(clean, bars, obs_dim)
            pred = model(noisy, steps_pair)
            pred_losses.append(float(F.mse_loss(pred, clean).detach().cpu()))
            noisy_losses.append(float(F.mse_loss(noisy, clean).detach().cpu()))
            if DRY_RUN:
                break
    pred = float(np.mean(pred_losses))
    noisy = float(np.mean(noisy_losses))
    return {
        "pred_token_mse": pred,
        "noisy_token_mse": noisy,
        "denoising_improvement_ratio": float(1.0 - pred / max(noisy, 1e-12)),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    start = time.time()
    dataset_summary = json.loads(STATE_LATENT_JSON.read_text(encoding="utf-8"))
    ds_meta = dataset_summary["worker_summary"]["dataset"]
    windows = read_window_index(dataset_summary["worker_summary"]["outputs"]["window_index_csv"])
    obs_by_rank, latent_by_rank = load_arrays(dataset_summary)
    max_rows = DRY_RUN_MAX_WINDOWS if DRY_RUN else 0
    train_ds = WindowDataset(windows, obs_by_rank, latent_by_rank, "train", max_rows=max_rows)
    val_ds = WindowDataset(windows, obs_by_rank, latent_by_rank, "validation", max_rows=max_rows)
    test_ds = WindowDataset(windows, obs_by_rank, latent_by_rank, "test", max_rows=max_rows)
    train_loader = DataLoader(train_ds, batch_size=BATCH_WINDOWS, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_WINDOWS, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_WINDOWS, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StateLatentDiffusionTransformer(
        ds_meta["token_dim"],
        ds_meta["obs_dim"],
        ds_meta["latent_dim"],
        ds_meta["sequence_length"],
    ).to(device)
    if torch.cuda.device_count() >= 2 and not DRY_RUN:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ema = EMA(base_model)
    bars = alpha_bars(DENOISING_STEPS)
    total_steps = max(1, EPOCHS * max(len(train_loader), 1))
    epoch_rows = []
    global_step = 0
    best_val = math.inf
    checkpoint_path = RUN_DIR / "paper_contract_transformer_state_latent_diffusion.pt"
    for epoch in range(EPOCHS):
        train_loss, global_step, ema_decay = run_epoch(
            model, train_loader, optimizer, bars, device, ds_meta["obs_dim"], ema, epoch, total_steps, global_step
        )
        val_metrics = evaluate(base_model, val_loader, bars, device, ds_meta["obs_dim"])
        row = {
            "epoch": epoch + 1,
            "train_token_mse": train_loss,
            "ema_decay_mean": ema_decay,
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        }
        epoch_rows.append(row)
        print("BM_SENTINEL:epoch:" + json.dumps(row, sort_keys=True), flush=True)
        if not DRY_RUN and val_metrics["pred_token_mse"] < best_val:
            best_val = val_metrics["pred_token_mse"]
            torch.save(
                {
                    "model_state_dict": base_model.state_dict(),
                    "ema_shadow": ema.shadow,
                    "config": {
                        "token_dim": ds_meta["token_dim"],
                        "obs_dim": ds_meta["obs_dim"],
                        "latent_dim": ds_meta["latent_dim"],
                        "sequence_length": ds_meta["sequence_length"],
                        "history": 4,
                        "horizon": 16,
                        "embedding_dim": EMBED_DIM,
                        "attention_heads": ATTENTION_HEADS,
                        "transformer_layers": TRANSFORMER_LAYERS,
                        "denoising_steps": DENOISING_STEPS,
                        "batch_windows": BATCH_WINDOWS,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "warmup_steps": WARMUP_STEPS,
                        "ema_power": EMA_POWER,
                        "ema_max": EMA_MAX,
                        "noise_schedule": "local_linear_beta_1e-4_to_0.02_public_schedule_unspecified",
                    },
                },
                checkpoint_path,
            )
        if DRY_RUN:
            break
    final_val = evaluate(base_model, val_loader, bars, device, ds_meta["obs_dim"])
    final_test = evaluate(base_model, test_loader, bars, device, ds_meta["obs_dim"])
    param_count = sum(p.numel() for p in base_model.parameters())
    summary = {
        "status": "ok",
        "dry_run": DRY_RUN,
        "duration_seconds": round(time.time() - start, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "data_parallel_used": isinstance(model, nn.DataParallel),
        "source_dataset": {
            "json": str(STATE_LATENT_JSON),
            "status": dataset_summary.get("status"),
            "official_dagger_rollout_dataset": False,
            "paper_level_state_latent_dataset": False,
            "state_source": ds_meta.get("state_source"),
        },
        "dataset": {
            "sequence_length": ds_meta["sequence_length"],
            "history": 4,
            "horizon": 16,
            "obs_dim": ds_meta["obs_dim"],
            "latent_dim": ds_meta["latent_dim"],
            "token_dim": ds_meta["token_dim"],
            "full_window_count": ds_meta["window_count"],
            "used_train_windows": len(train_ds),
            "used_validation_windows": len(val_ds),
            "used_test_windows": len(test_ds),
        },
        "architecture": {
            "parameter_count": int(param_count),
            "embedding_dim": EMBED_DIM,
            "attention_heads": ATTENTION_HEADS,
            "transformer_layers": TRANSFORMER_LAYERS,
            "feedforward_dim": EMBED_DIM * 4,
            "denoising_steps": DENOISING_STEPS,
            "state_step_embedding": True,
            "latent_step_embedding": True,
            "position_embedding": True,
            "output_predicts_clean_trajectory": True,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_windows": BATCH_WINDOWS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "ema_power": EMA_POWER,
            "ema_max": EMA_MAX,
            "noise_schedule": "local_linear_beta_1e-4_to_0.02_public_schedule_unspecified",
            "epoch_rows": epoch_rows,
        },
        "evaluation": {"validation": final_val, "test": final_test},
        "outputs": {
            "checkpoint": str(checkpoint_path) if checkpoint_path.is_file() else "",
            "run_dir": str(RUN_DIR),
        },
        "checks": {
            "paper_history_horizon_4_16": ds_meta["sequence_length"] == 21,
            "paper_embedding_dim_512": EMBED_DIM == 512,
            "paper_attention_heads_8": ATTENTION_HEADS == 8,
            "paper_transformer_layers_6": TRANSFORMER_LAYERS == 6,
            "paper_denoising_steps_20": DENOISING_STEPS == 20,
            "paper_batch_size_512": BATCH_WINDOWS == 512 or DRY_RUN,
            "paper_learning_rate_1e_4": abs(LEARNING_RATE - 1e-4) < 1e-12,
            "paper_weight_decay_0_001": abs(WEIGHT_DECAY - 0.001) < 1e-12,
            "paper_warmup_steps_10000": WARMUP_STEPS == 10000,
            "paper_ema_power_0_75": abs(EMA_POWER - 0.75) < 1e-12,
            "paper_ema_max_0_9999": abs(EMA_MAX - 0.9999) < 1e-12,
            "uses_transformer_encoder": isinstance(base_model.encoder, nn.TransformerEncoder),
            "uses_individual_state_and_latent_denoising_steps": True,
            "forward_backward_ok": len(epoch_rows) > 0 and math.isfinite(epoch_rows[-1]["train_token_mse"]),
            "does_not_claim_official_diffusion_checkpoint": True,
            "does_not_claim_paper_level_diffusion": True,
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_real_robot": True,
        },
    }
    (RUN_DIR / "paper_contract_transformer_worker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("BM_SENTINEL:summary:" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-train", action="store_true", help="Launch full training instead of the default dry run.")
    parser.add_argument("--dry-run-max-windows", type=int, default=int(os.environ.get("BM_DRY_RUN_MAX_WINDOWS", "16")))
    return parser.parse_args()


def enforce_hard_gate(*, dry_run: bool) -> bool:
    if dry_run:
        return True
    json_path = OUT / "paper_contract_transformer_state_latent_diffusion_training.json"
    tsv_path = OUT / "paper_contract_transformer_state_latent_diffusion_training.tsv"
    md_path = OUT / "paper_contract_transformer_state_latent_diffusion_training.md"
    reasons = pretraining_permission_block_reasons("start_state_latent_diffusion_training")
    reasons.extend(state_latent_dataset_block_reasons(STATE_LATENT_JSON))
    if not reasons:
        return True
    summary = write_blocked_summary(
        json_path=json_path,
        tsv_path=tsv_path,
        status="blocked_paper_contract_transformer_diffusion_training_hard_gate",
        experiment_type="paper_contract_transformer_state_latent_diffusion_training",
        permission_key="start_state_latent_diffusion_training",
        blocking_reasons=reasons,
        extra={
            "dry_run": dry_run,
            "source_dataset_json": str(STATE_LATENT_JSON),
            "outputs": {"json": str(json_path), "tsv": str(tsv_path), "md": str(md_path)},
        },
    )
    md_path.write_text(
        "# Paper-Contract Transformer State-Latent Diffusion\n\n"
        f"Status: `{summary['status']}`\n\n"
        "Full training is blocked because the current state-latent data does not yet prove the paper hybrid-state "
        "contract.\n\n"
        "## Blocking Reasons\n\n"
        + "\n".join(f"- `{reason}`" for reason in reasons)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": summary["status"], "json": str(json_path), "dry_run": dry_run, "blocking_reasons": reasons},
            sort_keys=True,
        )
    )
    return False


def cuda_visible_devices() -> str:
    return ",".join(str(gpu) for gpu in DEFAULT_GPUS)


def run_command(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def summarize_gpu_snapshot() -> list[dict[str, Any]]:
    proc = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            cuda_visible_devices(),
        ],
        timeout=15,
    )
    rows = []
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(float(parts[2])),
                "memory_total_mb": int(float(parts[3])),
                "utilization_gpu_percent": int(float(parts[4])),
                "power_draw_w": float(parts[5]),
            }
        )
    return rows


def extract_worker_summary(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith("BM_SENTINEL:summary:"):
            return json.loads(line.split("BM_SENTINEL:summary:", 1)[1])
    return {}


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "status",
        "dry_run",
        "sequence_length",
        "embedding_dim",
        "attention_heads",
        "transformer_layers",
        "denoising_steps",
        "batch_windows",
        "parameter_count",
        "forward_backward_ok",
        "test_pred_token_mse",
        "test_noisy_token_mse",
        "test_denoising_improvement_ratio",
        "paper_level_diffusion",
    ]
    row = {
        "status": summary["status"],
        "dry_run": summary["dry_run"],
        "sequence_length": summary["dataset"]["sequence_length"],
        "embedding_dim": summary["architecture"]["embedding_dim"],
        "attention_heads": summary["architecture"]["attention_heads"],
        "transformer_layers": summary["architecture"]["transformer_layers"],
        "denoising_steps": summary["architecture"]["denoising_steps"],
        "batch_windows": summary["training"]["batch_windows"],
        "parameter_count": summary["architecture"]["parameter_count"],
        "forward_backward_ok": summary["checks"]["forward_backward_ok"],
        "test_pred_token_mse": summary["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": summary["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": summary["evaluation"]["test"]["denoising_improvement_ratio"],
        "paper_level_diffusion": False,
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def write_md(path: Path, summary: dict[str, Any]) -> None:
    checks = summary["checks"]
    lines = [
        "# Paper-Contract Transformer State-Latent Diffusion",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Parameter count: `{summary['architecture']['parameter_count']}`",
        f"- Sequence length: `{summary['dataset']['sequence_length']}`",
        f"- Embedding/head/layer: `{summary['architecture']['embedding_dim']}` / `{summary['architecture']['attention_heads']}` / `{summary['architecture']['transformer_layers']}`",
        f"- Denoising steps: `{summary['architecture']['denoising_steps']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in checks.items())
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a local code-contract route over local teacher/VAE rollouts. It is not an official BeyondMimic diffusion checkpoint, not a closed-loop guidance result, not Isaac-rendered evidence, and not real-robot evidence.",
            "",
            "当前不得声称完整复现 BeyondMimic；该脚本只证明 Transformer diffusion 代码合同开始对齐论文结构。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dry_run = not args.full_train
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    if not enforce_hard_gate(dry_run=dry_run):
        return 1
    run_id = (
        "paper_contract_transformer_diffusion_dry_run"
        if dry_run
        else f"paper_contract_transformer_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = run_dir / "paper_contract_transformer_diffusion_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    epochs = 1 if dry_run else int(os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_EPOCHS", "1000"))
    batch = (
        min(args.dry_run_max_windows, int(os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_BATCH_WINDOWS", "512")))
        if dry_run
        else int(os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_BATCH_WINDOWS", "512"))
    )
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices(),
            "PYTHONUNBUFFERED": "1",
            "BM_STATE_LATENT_JSON": str(STATE_LATENT_JSON),
            "BM_RUN_DIR": str(run_dir),
            "BM_SEED": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_SEED", "20260931"),
            "BM_DRY_RUN": "1" if dry_run else "0",
            "BM_EPOCHS": str(epochs),
            "BM_BATCH_WINDOWS": str(batch),
            "BM_DRY_RUN_MAX_WINDOWS": str(args.dry_run_max_windows),
            "BM_EMBED_DIM": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_EMBED_DIM", "512"),
            "BM_ATTENTION_HEADS": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_HEADS", "8"),
            "BM_TRANSFORMER_LAYERS": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_LAYERS", "6"),
            "BM_DENOISING_STEPS": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_STEPS", "20"),
            "BM_LEARNING_RATE": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_LR", "1e-4"),
            "BM_WEIGHT_DECAY": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_WEIGHT_DECAY", "0.001"),
            "BM_WARMUP_STEPS": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_WARMUP_STEPS", "10000"),
            "BM_EMA_POWER": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_EMA_POWER", "0.75"),
            "BM_EMA_MAX": os.environ.get("BM_PAPER_TRANSFORMER_DIFFUSION_EMA_MAX", "0.9999"),
        }
    )
    log_path = LOG_DIR / ("paper_contract_transformer_diffusion_dry_run.log" if dry_run else "paper_contract_transformer_diffusion_training.log")
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
    output = log_path.read_text(encoding="utf-8", errors="replace")
    worker_summary = extract_worker_summary(output)
    if proc.returncode != 0 or not worker_summary:
        failed_log = FAILED_DIR / log_path.name
        failed_log.write_text(output, encoding="utf-8", errors="replace")
    status = (
        "ok_paper_contract_transformer_diffusion_dry_run"
        if dry_run and proc.returncode == 0 and worker_summary
        else "ok_paper_contract_transformer_diffusion_training"
        if proc.returncode == 0 and worker_summary
        else "failed_paper_contract_transformer_diffusion"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "paper_contract_transformer_state_latent_diffusion_training",
        "duration_seconds": round(time.time() - start, 3),
        "returncode": proc.returncode,
        "dry_run": dry_run,
        "gpu_snapshot": summarize_gpu_snapshot(),
        "worker_summary": worker_summary,
        "checks": {
            "bm_diffusion_python_exists": BM_DIFFUSION_PY.is_file(),
            "state_latent_dataset_exists": STATE_LATENT_JSON.is_file(),
            "process_returned_zero": proc.returncode == 0,
            "worker_summary_recorded": bool(worker_summary),
            "paper_contract_architecture_checks_pass": bool(worker_summary)
            and all(
                worker_summary.get("checks", {}).get(key)
                for key in [
                    "paper_history_horizon_4_16",
                    "paper_embedding_dim_512",
                    "paper_attention_heads_8",
                    "paper_transformer_layers_6",
                    "paper_denoising_steps_20",
                    "paper_learning_rate_1e_4",
                    "paper_weight_decay_0_001",
                    "paper_warmup_steps_10000",
                    "paper_ema_power_0_75",
                    "paper_ema_max_0_9999",
                    "uses_transformer_encoder",
                    "uses_individual_state_and_latent_denoising_steps",
                    "forward_backward_ok",
                ]
            ),
            "does_not_claim_paper_level_diffusion": True,
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_real_robot": True,
        },
        "outputs": {
            "json": str(OUT / "paper_contract_transformer_state_latent_diffusion_training.json"),
            "tsv": str(OUT / "paper_contract_transformer_state_latent_diffusion_training.tsv"),
            "md": str(OUT / "paper_contract_transformer_state_latent_diffusion_training.md"),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "worker_script": str(worker),
        },
        "interpretation": {
            "paper_level_diffusion": False,
            "closed_loop_guidance": False,
            "goal_complete": False,
            "why_not_complete": (
                "The Transformer code contract is present and dry-run tested, but full training/evaluation has not "
                "been performed and the source dataset is still local teacher/VAE rollout data."
            ),
        },
    }
    if worker_summary:
        summary["architecture"] = worker_summary["architecture"]
        summary["dataset"] = worker_summary["dataset"]
        summary["training"] = worker_summary["training"]
        summary["evaluation"] = worker_summary["evaluation"]
    json_path = OUT / "paper_contract_transformer_state_latent_diffusion_training.json"
    tsv_path = OUT / "paper_contract_transformer_state_latent_diffusion_training.tsv"
    md_path = OUT / "paper_contract_transformer_state_latent_diffusion_training.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if worker_summary:
        write_tsv(tsv_path, worker_summary | {"status": status})
        write_md(md_path, worker_summary | {"status": status})
    else:
        tsv_path.write_text("status\nfailed\n", encoding="utf-8")
        md_path.write_text("# Paper-Contract Transformer State-Latent Diffusion\n\nfailed\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "dry_run": dry_run}, sort_keys=True))
    return 0 if proc.returncode == 0 and worker_summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
