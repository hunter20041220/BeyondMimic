#!/usr/bin/env python3
"""Evaluate the bounded debug diffusion checkpoint on fixed split batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import level_c_vae_latent_diffusion_overfit_probe as vae_latent
from level_c_bounded_debug_diffusion_training_run import RUN_DIR, RUN_ID
from level_c_paper_state_transformer_arch_probe import (
    PaperStateDiffusionTransformer,
    ProbeConfig,
    diffusion_alpha_bars,
    seed_everything,
)
from level_c_transformer_ema_smoke import clone_state, mean_state_l2_delta, state_digest


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/bounded_debug_diffusion_checkpoint_eval"
CHECKPOINT = RUN_DIR / "checkpoint/debug_bounded_diffusion_checkpoint.pt"
TRAINING_JSON = ROOT / "res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json"
SPLIT_JSON = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"


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


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "motion_names",
        "window_count",
        "token_count",
        "trained_checkpoint_mse",
        "initial_model_mse",
        "noisy_identity_mse",
        "trained_vs_initial_delta",
        "trained_vs_noisy_delta",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    tmp.replace(path)


def make_noisy_tau(
    clean_tau: torch.Tensor,
    cfg: ProbeConfig,
    state_dim: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    alpha_bars = diffusion_alpha_bars(cfg.denoising_steps, device, clean_tau.dtype)
    steps = torch.randint(0, cfg.denoising_steps, clean_tau.shape[:2] + (2,), device=device)
    noise = torch.randn_like(clean_tau)
    state_alpha = alpha_bars[steps[..., 0]].unsqueeze(-1).expand(-1, -1, state_dim)
    latent_alpha = alpha_bars[steps[..., 1]].unsqueeze(-1).expand(-1, -1, cfg.latent_dim)
    alpha = torch.cat([state_alpha, latent_alpha], dim=-1)
    return torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise, steps


def motion_split_indices(latent_manifest: dict[str, Any]) -> tuple[list[str], dict[str, list[int]]]:
    split_manifest = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    split_by_motion = {item["name"]: item["split"] for item in split_manifest["motions"]}
    motion_names = sorted({row["source_motion"] for row in latent_manifest["rows"]})
    motion_name_to_id = {name: idx for idx, name in enumerate(motion_names)}
    split_to_motion_ids: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for name in motion_names:
        split_to_motion_ids[split_by_motion[name]].append(motion_name_to_id[name])
    return motion_names, split_to_motion_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    training = json.loads(TRAINING_JSON.read_text(encoding="utf-8"))
    split_manifest = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ProbeConfig(**payload["config"])
    latent_cfg = vae_latent.VaeLatentDiffusionConfig(seed=args.seed)
    clean_np, motion_ids_np, latent_manifest = vae_latent.load_dataset(latent_cfg)
    clean_tau = torch.from_numpy(clean_np.astype(np.float32)).to(device)
    state_dim = latent_cfg.state_dim
    token_dim = int(clean_tau.shape[-1])
    noisy_tau, steps = make_noisy_tau(clean_tau, cfg, state_dim, device, args.seed)

    seed_everything(payload["seed"])
    initial_model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
    trained_model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
    trained_model.load_state_dict(payload["model_state_dict"])
    initial_model.eval()
    trained_model.eval()
    with torch.no_grad():
        trained_pred = trained_model(noisy_tau, steps)
        initial_pred = initial_model(noisy_tau, steps)

    motion_names, split_to_motion_ids = motion_split_indices(latent_manifest)
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    clean_cpu = clean_tau.detach().cpu()
    noisy_cpu = noisy_tau.detach().cpu()
    trained_cpu = trained_pred.detach().cpu()
    initial_cpu = initial_pred.detach().cpu()
    for split in ["train", "validation", "test"]:
        ids = split_to_motion_ids[split]
        mask = np.isin(motion_ids_np, np.asarray(ids, dtype=np.int64))
        indices = np.nonzero(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"empty split {split}")
        token_count = int(len(indices) * cfg.sequence_length)
        clean_split = clean_cpu[indices]
        trained_split = trained_cpu[indices]
        initial_split = initial_cpu[indices]
        noisy_split = noisy_cpu[indices]
        trained_mse = float(F.mse_loss(trained_split, clean_split).item())
        initial_mse = float(F.mse_loss(initial_split, clean_split).item())
        noisy_mse = float(F.mse_loss(noisy_split, clean_split).item())
        row = {
            "split": split,
            "motion_names": ",".join(motion_names[idx] for idx in ids),
            "window_count": int(len(indices)),
            "token_count": token_count,
            "trained_checkpoint_mse": trained_mse,
            "initial_model_mse": initial_mse,
            "noisy_identity_mse": noisy_mse,
            "trained_vs_initial_delta": initial_mse - trained_mse,
            "trained_vs_noisy_delta": noisy_mse - trained_mse,
        }
        rows.append(row)
        for key, value in row.items():
            if key not in {"split", "motion_names"}:
                metrics[f"{split}_{key}"] = value

    model_l2_from_initial = mean_state_l2_delta(clone_state(trained_model), clone_state(initial_model))
    checkpoint_sha = file_sha256(CHECKPOINT)
    finite_values = [
        value
        for row in rows
        for key, value in row.items()
        if key.endswith("_mse") or key.endswith("_delta")
    ]
    checks = {
        "training_audit_status_ok": training["status"] == "ok",
        "checkpoint_file_exists": CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 0,
        "checkpoint_sha_matches_training_audit": checkpoint_sha == training["metrics"]["checkpoint_sha256"],
        "checkpoint_marked_debug_only": payload["is_trained_paper_checkpoint"] is False
        and payload["is_bounded_debug_training_checkpoint"] is True,
        "run_id_matches_training_run": payload["run_id"] == RUN_ID,
        "token_dim_131": token_dim == 131,
        "state_dim_99": state_dim == 99,
        "latent_dim_32": cfg.latent_dim == 32,
        "window_count_84": clean_np.shape[0] == 84,
        "split_counts_28_each": {row["split"]: row["window_count"] for row in rows}
        == {"train": 28, "validation": 28, "test": 28},
        "uses_motion_level_split_manifest": split_manifest["checks"]["no_motion_crosses_splits"],
        "all_eval_losses_finite": bool(np.all(np.isfinite(finite_values))),
        "model_differs_from_initial": model_l2_from_initial > 0.0
        and state_digest(clone_state(trained_model)) != state_digest(clone_state(initial_model)),
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_paper_metrics": True,
        "does_not_claim_goal_complete": True,
    }

    json_path = OUT / "level_c_bounded_debug_diffusion_checkpoint_eval.json"
    tsv_path = OUT / "level_c_bounded_debug_diffusion_checkpoint_eval.tsv"
    write_tsv(tsv_path, rows)
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "bounded_debug_diffusion_checkpoint_eval",
        "scope": "fixed-noise per-split MSE evaluation of the bounded debug diffusion checkpoint",
        "source_artifacts": {
            "training_audit": str(TRAINING_JSON),
            "checkpoint": str(CHECKPOINT),
            "split_manifest": str(SPLIT_JSON),
            "debug_vae_latents": str(vae_latent.VAE_LATENT_JSON),
        },
        "settings": {
            "device": str(device),
            "torch_threads": args.torch_threads,
            "seed": args.seed,
            "checkpoint_seed": payload["seed"],
            "token_dim": token_dim,
            "state_dim": state_dim,
            "latent_dim": cfg.latent_dim,
            "sequence_length": cfg.sequence_length,
            "window_count": int(clean_np.shape[0]),
            "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher",
        },
        "checkpoint": {
            "sha256": checkpoint_sha,
            "size_bytes": CHECKPOINT.stat().st_size,
            "model_l2_from_initial": model_l2_from_initial,
            "trained_model_sha256": state_digest(clone_state(trained_model)),
            "initial_model_sha256": state_digest(clone_state(initial_model)),
        },
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial_debug_checkpoint_eval_only",
            "goal_complete": False,
            "why_not_complete": (
                "This evaluates the bounded debug checkpoint on fixed noisy debug state-latent windows. It is not a "
                "closed-loop diffusion controller evaluation, does not use true VAE rollout latents, does not report "
                "paper success/failure metrics, and does not validate Fig. 5/Fig. 6 behavior."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    write_json_atomic(json_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
