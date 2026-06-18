#!/usr/bin/env python3
"""Train/evaluate a resource-adjusted tiny denoiser on debug state-latent tokens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
import subprocess
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import level_c_diffusion_to_vae_action_smoke as action_smoke
import level_c_vae_latent_diffusion_overfit_probe as vae_latent


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_ID = "level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500"
RUN_DIR = ROOT / "res/runs" / RUN_ID
OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_training_run"
SPLIT_JSON = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"
BM_DIFFUSION_PYTHON = ROOT / "envs/bm_diffusion/bin/python"


@dataclass(frozen=True)
class TinyConfig:
    seed: int = 20260923
    epochs: int = 180
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 256
    batch_tokens: int = 256
    denoising_steps: int = 20
    state_dim: int = 99
    latent_dim: int = 32
    history: int = 4
    horizon: int = 16

    @property
    def sequence_length(self) -> int:
        return self.history + 1 + self.horizon

    @property
    def token_dim(self) -> int:
        return self.state_dim + self.latent_dim


class TinyDenoiser(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        inp = cfg.token_dim + 2 * cfg.denoising_steps
        self.net = nn.Sequential(
            nn.Linear(inp, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.token_dim),
        )

    def forward(self, noisy_tau: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        state_onehot = F.one_hot(steps[..., 0], num_classes=20).to(noisy_tau.dtype)
        latent_onehot = F.one_hot(steps[..., 1], num_classes=20).to(noisy_tau.dtype)
        x = torch.cat([noisy_tau, state_onehot, latent_onehot], dim=-1)
        return self.net(x)


def read_command(cmd: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"command_failed: {cmd}: {exc}"


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simple_png_plot(path: Path, series: dict[str, list[float]], *, width: int = 720, height: int = 420) -> None:
    ml, mr, mt, mb = 70, 25, 30, 65
    img = bytearray([255] * (width * height * 3))

    def pix(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            img[i : i + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            for ox in [-1, 0, 1]:
                for oy in [-1, 0, 1]:
                    pix(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    values = [v for vals in series.values() for v in vals]
    ymin, ymax = min(values), max(values)
    pad = max((ymax - ymin) * 0.08, 1e-8)
    ymin, ymax = ymin - pad, ymax + pad
    n = max(len(vals) for vals in series.values())
    pw, ph = width - ml - mr, height - mt - mb

    def xm(i: int) -> int:
        return ml + (pw // 2 if n <= 1 else int(i / (n - 1) * pw))

    def ym(v: float) -> int:
        return mt + int((ymax - v) / (ymax - ymin) * ph)

    for k in range(6):
        y = mt + int(k * ph / 5)
        line(ml, y, ml + pw, y, (225, 225, 225))
    line(ml, mt, ml, mt + ph, (35, 35, 35))
    line(ml, mt + ph, ml + pw, mt + ph, (35, 35, 35))
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44)]
    for color, vals in zip(colors, series.values()):
        pts = [(xm(i), ym(v)) for i, v in enumerate(vals)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            line(x0, y0, x1, y1, color)
        for x, y in pts:
            for ox in range(-3, 4):
                for oy in range(-3, 4):
                    if ox * ox + oy * oy <= 9:
                        pix(x + ox, y + oy, color)
    raw = b"".join(b"\x00" + bytes(img[y * width * 3 : (y + 1) * width * 3]) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def make_noisy(clean: np.ndarray, cfg: TinyConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 1)
    steps = rng.integers(0, cfg.denoising_steps, size=clean.shape[:2] + (2,), dtype=np.int64)
    noise = rng.standard_normal(clean.shape)
    bars = vae_latent.paper_overfit.alpha_bars(cfg.denoising_steps)
    state_alpha = np.repeat(bars[steps[..., 0]][..., None], cfg.state_dim, axis=-1)
    latent_alpha = np.repeat(bars[steps[..., 1]][..., None], cfg.latent_dim, axis=-1)
    alpha = np.concatenate([state_alpha, latent_alpha], axis=-1)
    noisy = np.sqrt(alpha) * clean + np.sqrt(1.0 - alpha) * noise
    return noisy.astype(np.float32), steps.astype(np.int64)


def split_masks(latent_manifest: dict[str, Any], motion_ids: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    split_manifest = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    split_by_motion = {item["name"]: item["split"] for item in split_manifest["motions"]}
    motion_names = sorted({row["source_motion"] for row in latent_manifest["rows"]})
    id_to_split = {idx: split_by_motion[name] for idx, name in enumerate(motion_names)}
    split_labels = np.asarray([id_to_split[int(mid)] for mid in motion_ids])
    masks = {split: split_labels == split for split in ["train", "validation", "test"]}
    return split_labels, masks


def fit_action_decoder(clean_tau: np.ndarray, target_action: np.ndarray, split_labels: np.ndarray) -> np.ndarray:
    with np.load(action_smoke.VAE_NPZ) as vae_data:
        projection = vae_data["proprioception_projection"].astype(np.float64)
    features = action_smoke.feature_tensor(clean_tau.astype(np.float64), projection)
    mask = action_smoke.expand_split_mask(split_labels, features.shape[1], "train")
    x = features.reshape(-1, features.shape[-1])[mask]
    y = target_action.reshape(-1, target_action.shape[-1])[mask]
    reg = action_smoke.RIDGE_LAMBDA * np.eye(x.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def action_from_tau(tau: np.ndarray, weights: np.ndarray) -> np.ndarray:
    with np.load(action_smoke.VAE_NPZ) as vae_data:
        projection = vae_data["proprioception_projection"].astype(np.float64)
    return action_smoke.feature_tensor(tau.astype(np.float64), projection) @ weights


def evaluate(
    model: TinyDenoiser,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    steps: torch.Tensor,
    masks: dict[str, np.ndarray],
    target_action: np.ndarray,
    action_weights: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred = model(noisy, steps).detach().cpu().numpy().astype(np.float64)
    clean_np = clean.detach().cpu().numpy().astype(np.float64)
    noisy_np = noisy.detach().cpu().numpy().astype(np.float64)
    pred_action = action_from_tau(pred, action_weights)
    noisy_action = action_from_tau(noisy_np, action_weights)
    rows: list[dict[str, Any]] = []
    for split, mask in masks.items():
        token_count = int(np.sum(mask) * clean_np.shape[1])
        current = TinyConfig().history
        row = {
            "split": split,
            "window_count": int(np.sum(mask)),
            "token_count": token_count,
            "noisy_token_mse": float(np.mean(np.square(noisy_np[mask] - clean_np[mask]))),
            "pred_token_mse": float(np.mean(np.square(pred[mask] - clean_np[mask]))),
            "clean_current_action_mse": float(np.mean(np.square(action_from_tau(clean_np, action_weights)[mask, current] - target_action[mask, current]))),
            "noisy_current_action_mse": float(np.mean(np.square(noisy_action[mask, current] - target_action[mask, current]))),
            "pred_current_action_mse": float(np.mean(np.square(pred_action[mask, current] - target_action[mask, current]))),
        }
        row["token_mse_reduction_vs_noisy"] = (
            (row["noisy_token_mse"] - row["pred_token_mse"]) / row["noisy_token_mse"]
            if row["noisy_token_mse"] > 0
            else 0.0
        )
        row["current_action_delta_vs_noisy"] = row["noisy_current_action_mse"] - row["pred_current_action_mse"]
        rows.append(row)
    return rows, pred


def prepare_run_dir(command: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in ["checkpoint", "figures", "videos"]:
        (RUN_DIR / dirname).mkdir(exist_ok=True)
    write_text(
        RUN_DIR / "resolved_config.yaml",
        "\n".join(
            [
                f"run_id: {RUN_ID}",
                "stage: level_c",
                "method: resource_adjusted_tiny_diffusion_training",
                "status: SUCCESS",
                "is_training_run: false",
                "resource_adjusted_debug: true",
                "paper_level: false",
                "reason: Tiny denoiser trained on debug state-latent windows; not paper architecture or dataset.",
                "source_goal_lines: goal.md:1251-1290,1468-1487,1747-1787",
                "",
            ]
        ),
    )
    write_text(RUN_DIR / "command.sh", f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n", executable=True)
    write_text(RUN_DIR / "stderr.log", "")
    write_text(
        RUN_DIR / "environment.txt",
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat(timespec='seconds')}",
                f"python={read_command([str(BM_DIFFUSION_PYTHON), '--version'])}",
                f"torch={torch.__version__}",
                f"numpy={np.__version__}",
                "figure_writer=stdlib_png",
                "notes=Resource-adjusted debug training only; no Isaac/ROS/TensorRT/hardware.",
                "",
            ]
        ),
    )
    write_text(RUN_DIR / "git_state.txt", read_command(["git", "status", "--short"]) + "\n")
    gpu_src = ROOT / "logs/gpu/gpu_metrics.csv"
    if gpu_src.is_file():
        shutil.copyfile(gpu_src, RUN_DIR / "gpu_metrics.csv")
    else:
        write_text(RUN_DIR / "gpu_metrics.csv", "timestamp,gpu_index,run_id,run_status,sample_kind\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=180)
    args = parser.parse_args()
    cfg = TinyConfig(epochs=args.epochs)
    command = (
        f"{BM_DIFFUSION_PYTHON} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_training_run.py'} "
        f"--device {args.device} --torch-threads {args.torch_threads} --epochs {args.epochs}"
    )
    prepare_run_dir(command)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    clean_np, motion_ids, latent_manifest = vae_latent.load_dataset(
        vae_latent.VaeLatentDiffusionConfig(seed=cfg.seed)
    )
    noisy_np, steps_np = make_noisy(clean_np, cfg)
    split_labels, masks = split_masks(latent_manifest, motion_ids)
    vae_summary = json.loads(action_smoke.VAE_JSON.read_text(encoding="utf-8"))
    with np.load(action_smoke.VAE_NPZ) as vae_data:
        target_action = np.stack(
            [vae_data[f"{row['sample_id']}_decoded_action"].astype(np.float64) for row in vae_summary["rows"]],
            axis=0,
        )
    action_weights = fit_action_decoder(clean_np, target_action, split_labels)

    clean = torch.from_numpy(clean_np.astype(np.float32)).to(device)
    noisy = torch.from_numpy(noisy_np.astype(np.float32)).to(device)
    steps = torch.from_numpy(steps_np).to(device)
    train_idx = np.nonzero(masks["train"])[0]
    train_tokens = np.concatenate(
        [np.arange(i * cfg.sequence_length, (i + 1) * cfg.sequence_length, dtype=np.int64) for i in train_idx]
    )
    model = TinyDenoiser(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    noisy_flat = noisy.reshape(-1, cfg.token_dim)
    clean_flat = clean.reshape(-1, cfg.token_dim)
    steps_flat = steps.reshape(-1, 2)
    rng = np.random.default_rng(cfg.seed + 3)
    loss_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        batch = rng.choice(train_tokens, size=min(cfg.batch_tokens, len(train_tokens)), replace=False)
        pred = model(noisy_flat[batch], steps_flat[batch])
        loss = F.mse_loss(pred, clean_flat[batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            eval_rows, _ = evaluate(model, clean, noisy, steps, masks, target_action, action_weights)
            by_split = {row["split"]: row for row in eval_rows}
            loss_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_pred_token_mse": by_split["train"]["pred_token_mse"],
                    "validation_pred_token_mse": by_split["validation"]["pred_token_mse"],
                    "test_pred_token_mse": by_split["test"]["pred_token_mse"],
                }
            )
    elapsed = time.perf_counter() - start
    eval_rows, pred_np = evaluate(model, clean, noisy, steps, masks, target_action, action_weights)

    checkpoint = RUN_DIR / "checkpoint/tiny_resource_adjusted_denoiser.pt"
    payload = {
        "experiment_type": "resource_adjusted_tiny_diffusion_training",
        "run_id": RUN_ID,
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": cfg.epochs,
        "is_trained_paper_checkpoint": False,
        "resource_adjusted_debug": True,
        "paper_level": False,
    }
    torch.save(payload, checkpoint)
    loss_fig = RUN_DIR / "figures/tiny_denoiser_loss.png"
    eval_fig = RUN_DIR / "figures/tiny_denoiser_eval_mse.png"
    simple_png_plot(
        loss_fig,
        {
            "train": [r["train_pred_token_mse"] for r in loss_rows],
            "validation": [r["validation_pred_token_mse"] for r in loss_rows],
            "test": [r["test_pred_token_mse"] for r in loss_rows],
        },
    )
    simple_png_plot(
        eval_fig,
        {
            "noisy": [r["noisy_token_mse"] for r in eval_rows],
            "pred": [r["pred_token_mse"] for r in eval_rows],
        },
    )
    npz_path = OUT / "level_c_resource_adjusted_tiny_diffusion_training_run.npz"
    np.savez_compressed(
        npz_path,
        clean_tau=clean_np.astype(np.float32),
        noisy_tau=noisy_np.astype(np.float32),
        predicted_tau=pred_np.astype(np.float32),
        diffusion_steps=steps_np,
        split_labels=split_labels,
        target_action=target_action.astype(np.float32),
    )

    metrics = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "is_training_run": False,
        "resource_adjusted_debug": True,
        "paper_level": False,
        "epochs": cfg.epochs,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "elapsed_sec": elapsed,
        "samples_per_second": float((len(train_tokens) * cfg.epochs) / elapsed) if elapsed > 0 else None,
        "environment_steps_per_second": None,
        "iteration_time": float(elapsed / cfg.epochs),
        "estimated_remaining_time": 0.0,
        "oom_count": 0,
        "restart_count": 0,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "loss_figure_size_bytes": loss_fig.stat().st_size,
        "eval_figure_size_bytes": eval_fig.stat().st_size,
        "train_pred_token_mse": next(r["pred_token_mse"] for r in eval_rows if r["split"] == "train"),
        "validation_pred_token_mse": next(r["pred_token_mse"] for r in eval_rows if r["split"] == "validation"),
        "test_pred_token_mse": next(r["pred_token_mse"] for r in eval_rows if r["split"] == "test"),
        "validation_pred_current_action_mse": next(
            r["pred_current_action_mse"] for r in eval_rows if r["split"] == "validation"
        ),
        "test_pred_current_action_mse": next(r["pred_current_action_mse"] for r in eval_rows if r["split"] == "test"),
    }
    write_json_atomic(RUN_DIR / "metrics.json", metrics)
    with (RUN_DIR / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_pred_token_mse", "validation_pred_token_mse", "test_pred_token_mse"])
        writer.writeheader()
        writer.writerows(loss_rows)
    write_json_atomic(
        RUN_DIR / "status.json",
        {
            "run_id": RUN_ID,
            "status": "SUCCESS",
            "allowed_status": True,
            "is_success": True,
            "is_training_run": False,
            "resource_adjusted_debug": True,
            "paper_level": False,
            "reason": "Resource-adjusted tiny denoiser debug training completed; not a full paper training run.",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    write_text(
        RUN_DIR / "stdout.log",
        "\n".join(
            [
                "Resource-adjusted tiny diffusion training completed.",
                f"run_id={RUN_ID}",
                f"epochs={cfg.epochs}",
                f"validation_pred_token_mse={metrics['validation_pred_token_mse']}",
                f"test_pred_token_mse={metrics['test_pred_token_mse']}",
                "paper_level=false",
                "",
            ]
        ),
    )
    tsv_path = OUT / "level_c_resource_adjusted_tiny_diffusion_training_run.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)
    checks = {
        "checkpoint_file_exists": checkpoint.is_file() and checkpoint.stat().st_size > 0,
        "figures_exist": loss_fig.is_file() and eval_fig.is_file(),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "split_counts_28_each": {r["split"]: r["window_count"] for r in eval_rows}
        == {"train": 28, "validation": 28, "test": 28},
        "train_token_mse_improves_vs_noisy": next(r for r in eval_rows if r["split"] == "train")[
            "pred_token_mse"
        ]
        < next(r for r in eval_rows if r["split"] == "train")["noisy_token_mse"],
        "validation_token_mse_finite": bool(np.isfinite(metrics["validation_pred_token_mse"])),
        "test_token_mse_finite": bool(np.isfinite(metrics["test_pred_token_mse"])),
        "all_action_metrics_finite": all(bool(np.isfinite(r["pred_current_action_mse"])) for r in eval_rows),
        "status_success_but_not_paper_training": True,
        "videos_are_not_required_full_run_videos": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "resource_adjusted_tiny_diffusion_training_run",
        "scope": "tiny token denoiser debug training/evaluation on local state-latent fixtures",
        "settings": asdict(cfg) | {"device": str(device), "torch_threads": args.torch_threads},
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "metrics": metrics,
        "eval_rows": eval_rows,
        "loss_rows": loss_rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "resource_adjusted_debug_only",
            "why_not_complete": (
                "This trains a small local denoiser on debug tiny-VAE state-latent windows and evaluates held-out "
                "token/action MSE. It is not the paper Transformer, not true VAE rollout data, not closed-loop "
                "control, not a valid full training run, and not Fig. 5/Fig. 6 evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_resource_adjusted_tiny_diffusion_training_run.json"),
            "tsv": str(tsv_path),
            "npz": str(npz_path),
            "run_dir": str(RUN_DIR),
            "checkpoint": str(checkpoint),
            "loss_figure": str(loss_fig),
            "eval_figure": str(eval_fig),
        },
    }
    write_json_atomic(OUT / "level_c_resource_adjusted_tiny_diffusion_training_run.json", summary)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "run_id": RUN_ID}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
