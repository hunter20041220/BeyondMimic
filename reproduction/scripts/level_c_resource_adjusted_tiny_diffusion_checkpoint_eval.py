#!/usr/bin/env python3
"""Reload/evaluate the resource-adjusted tiny diffusion checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import level_c_resource_adjusted_tiny_diffusion_training_run as tiny
import level_c_vae_latent_diffusion_overfit_probe as vae_latent


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
    / "level_c_resource_adjusted_tiny_diffusion_training_run.json"
)
OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "window_count",
        "token_count",
        "noisy_token_mse",
        "pred_token_mse",
        "source_pred_token_mse",
        "abs_pred_token_mse_delta",
        "pred_current_action_mse",
        "source_pred_current_action_mse",
        "abs_pred_current_action_mse_delta",
        "token_mse_reduction_vs_noisy",
        "current_action_delta_vs_noisy",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    tmp.replace(path)


def load_eval_inputs(seed: int, device: torch.device) -> tuple[
    tiny.TinyConfig,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    cfg = tiny.TinyConfig(seed=seed)
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
    return cfg, clean, noisy, steps, masks, target_action, action_weights


def source_row_by_split(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["split"]: row for row in source["eval_rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)

    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    checkpoint_path = Path(source["outputs"]["checkpoint"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_payload = payload["config"]
    cfg = tiny.TinyConfig(**cfg_payload)
    eval_cfg, clean, noisy, steps, masks, target_action, action_weights = load_eval_inputs(cfg.seed, device)
    model = tiny.TinyDenoiser(eval_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    eval_rows, pred_np = tiny.evaluate(model, clean, noisy, steps, masks, target_action, action_weights)

    source_rows = source_row_by_split(source)
    rows: list[dict[str, Any]] = []
    for row in eval_rows:
        src = source_rows[row["split"]]
        rows.append(
            {
                "split": row["split"],
                "window_count": row["window_count"],
                "token_count": row["token_count"],
                "noisy_token_mse": row["noisy_token_mse"],
                "pred_token_mse": row["pred_token_mse"],
                "source_pred_token_mse": src["pred_token_mse"],
                "abs_pred_token_mse_delta": abs(row["pred_token_mse"] - src["pred_token_mse"]),
                "pred_current_action_mse": row["pred_current_action_mse"],
                "source_pred_current_action_mse": src["pred_current_action_mse"],
                "abs_pred_current_action_mse_delta": abs(
                    row["pred_current_action_mse"] - src["pred_current_action_mse"]
                ),
                "token_mse_reduction_vs_noisy": row["token_mse_reduction_vs_noisy"],
                "current_action_delta_vs_noisy": row["current_action_delta_vs_noisy"],
            }
        )

    npz_path = OUT / "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.npz"
    np.savez_compressed(npz_path, reloaded_predicted_tau=pred_np.astype(np.float32))
    tsv_path = OUT / "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.tsv"
    json_path = OUT / "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json"
    write_tsv(tsv_path, rows)
    max_token_delta = max(row["abs_pred_token_mse_delta"] for row in rows)
    max_action_delta = max(row["abs_pred_current_action_mse_delta"] for row in rows)
    checkpoint_sha = sha256(checkpoint_path)
    checks = {
        "source_training_status_ok": source["status"] == "ok",
        "checkpoint_exists": checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0,
        "checkpoint_sha_matches_training_json": checkpoint_sha == source["metrics"]["checkpoint_sha256"],
        "payload_marks_not_paper_checkpoint": payload.get("paper_level") is False
        and payload.get("is_trained_paper_checkpoint") is False,
        "three_split_rows": {row["split"] for row in rows} == {"train", "validation", "test"},
        "all_split_counts_28": all(row["window_count"] == 28 for row in rows),
        "all_prediction_metrics_finite": all(
            bool(np.isfinite(row["pred_token_mse"])) and bool(np.isfinite(row["pred_current_action_mse"]))
            for row in rows
        ),
        "max_token_mse_delta_below_1e_minus_12": max_token_delta < 1e-12,
        "max_action_mse_delta_below_1e_minus_12": max_action_delta < 1e-12,
        "validation_and_test_better_than_noisy": all(
            row["pred_token_mse"] < row["noisy_token_mse"] for row in rows if row["split"] in {"validation", "test"}
        ),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_paper_checkpoint": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "resource_adjusted_tiny_diffusion_checkpoint_eval",
        "scope": "reload/evaluate the small debug tiny denoiser checkpoint and compare against source eval metrics",
        "settings": {
            "device": str(device),
            "torch_threads": args.torch_threads,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
        },
        "metrics": {
            "max_abs_pred_token_mse_delta_vs_source": max_token_delta,
            "max_abs_pred_current_action_mse_delta_vs_source": max_action_delta,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "npz": str(npz_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "debug_checkpoint_reload_only",
            "why_not_complete": (
                "This verifies that the resource-adjusted tiny denoiser checkpoint reloads and reproduces offline "
                "debug eval metrics. It is not the paper diffusion Transformer checkpoint, not closed-loop control, "
                "not TensorRT, and not Fig. 5/Fig. 6 evidence."
            ),
        },
    }
    write_json_atomic(json_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
