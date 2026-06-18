#!/usr/bin/env python3
"""Multi-seed audit for resource-adjusted tiny diffusion training."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import level_c_resource_adjusted_tiny_diffusion_training_run as tiny
import level_c_vae_latent_diffusion_overfit_probe as vae_latent


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "epochs",
        "train_pred_token_mse",
        "validation_pred_token_mse",
        "test_pred_token_mse",
        "validation_pred_current_action_mse",
        "test_pred_current_action_mse",
        "validation_token_reduction_vs_noisy",
        "test_token_reduction_vs_noisy",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    tmp.replace(path)


def train_one(seed: int, epochs: int, device: torch.device, torch_threads: int) -> tuple[dict[str, Any], np.ndarray]:
    cfg = tiny.TinyConfig(seed=seed, epochs=epochs)
    torch.set_num_threads(torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    clean_np, motion_ids, latent_manifest = vae_latent.load_dataset(
        vae_latent.VaeLatentDiffusionConfig(seed=cfg.seed)
    )
    noisy_np, steps_np = tiny.make_noisy(clean_np, cfg)
    split_labels, masks = tiny.split_masks(latent_manifest, motion_ids)
    vae_summary = json.loads(tiny.action_smoke.VAE_JSON.read_text(encoding="utf-8"))
    with np.load(tiny.action_smoke.VAE_NPZ) as vae_data:
        target_action = np.stack(
            [vae_data[f"{row['sample_id']}_decoded_action"].astype(np.float64) for row in vae_summary["rows"]],
            axis=0,
        )
    action_weights = tiny.fit_action_decoder(clean_np, target_action, split_labels)
    clean = torch.from_numpy(clean_np.astype(np.float32)).to(device)
    noisy = torch.from_numpy(noisy_np.astype(np.float32)).to(device)
    steps = torch.from_numpy(steps_np).to(device)
    train_idx = np.nonzero(masks["train"])[0]
    train_tokens = np.concatenate(
        [np.arange(i * cfg.sequence_length, (i + 1) * cfg.sequence_length, dtype=np.int64) for i in train_idx]
    )
    model = tiny.TinyDenoiser(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    noisy_flat = noisy.reshape(-1, cfg.token_dim)
    clean_flat = clean.reshape(-1, cfg.token_dim)
    steps_flat = steps.reshape(-1, 2)
    rng = np.random.default_rng(cfg.seed + 3)
    for _epoch in range(cfg.epochs):
        batch = rng.choice(train_tokens, size=min(cfg.batch_tokens, len(train_tokens)), replace=False)
        pred = model(noisy_flat[batch], steps_flat[batch])
        loss = F.mse_loss(pred, clean_flat[batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    eval_rows, pred_np = tiny.evaluate(model, clean, noisy, steps, masks, target_action, action_weights)
    by_split = {row["split"]: row for row in eval_rows}
    row = {
        "seed": seed,
        "epochs": epochs,
        "train_pred_token_mse": by_split["train"]["pred_token_mse"],
        "validation_pred_token_mse": by_split["validation"]["pred_token_mse"],
        "test_pred_token_mse": by_split["test"]["pred_token_mse"],
        "validation_pred_current_action_mse": by_split["validation"]["pred_current_action_mse"],
        "test_pred_current_action_mse": by_split["test"]["pred_current_action_mse"],
        "validation_token_reduction_vs_noisy": by_split["validation"]["token_mse_reduction_vs_noisy"],
        "test_token_reduction_vs_noisy": by_split["test"]["token_mse_reduction_vs_noisy"],
    }
    return row, pred_np.astype(np.float32)


def stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="20260931,20260932,20260933")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for seed in seeds:
        row, pred = train_one(seed, args.epochs, device, args.torch_threads)
        rows.append(row)
        predictions[f"seed_{seed}_predicted_tau"] = pred
    json_path = OUT / "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json"
    tsv_path = OUT / "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.tsv"
    npz_path = OUT / "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(npz_path, **predictions)
    stat_keys = [
        "validation_pred_token_mse",
        "test_pred_token_mse",
        "validation_pred_current_action_mse",
        "test_pred_current_action_mse",
        "validation_token_reduction_vs_noisy",
        "test_token_reduction_vs_noisy",
    ]
    statistics = {key: stats(rows, key) for key in stat_keys}
    checks = {
        "seed_count_3": len(rows) == 3,
        "all_validation_token_mse_finite": all(bool(np.isfinite(row["validation_pred_token_mse"])) for row in rows),
        "all_test_token_mse_finite": all(bool(np.isfinite(row["test_pred_token_mse"])) for row in rows),
        "all_validation_action_mse_finite": all(
            bool(np.isfinite(row["validation_pred_current_action_mse"])) for row in rows
        ),
        "all_test_action_mse_finite": all(bool(np.isfinite(row["test_pred_current_action_mse"])) for row in rows),
        "all_validation_token_improves_vs_noisy": all(row["validation_token_reduction_vs_noisy"] > 0 for row in rows),
        "all_test_token_improves_vs_noisy": all(row["test_token_reduction_vs_noisy"] > 0 for row in rows),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_paper_multiseed": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "resource_adjusted_tiny_diffusion_multiseed_audit",
        "scope": "3-seed debug statistics for tiny denoiser training on local state-latent fixtures",
        "settings": {
            "device": str(device),
            "torch_threads": args.torch_threads,
            "epochs": args.epochs,
            "seeds": seeds,
            "base_config": asdict(tiny.TinyConfig(epochs=args.epochs)),
        },
        "rows": rows,
        "statistics": statistics,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "resource_adjusted_debug_multiseed_only",
            "why_not_complete": (
                "This is a small local multi-seed debug audit. It does not train the paper Transformer, does not use "
                "true VAE rollout data, does not run closed-loop evaluation, and does not reproduce paper multi-seed "
                "metrics or figures."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    write_json_atomic(json_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "seeds": len(rows)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
