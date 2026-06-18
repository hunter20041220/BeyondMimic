#!/usr/bin/env python3
"""Evaluate bounded debug diffusion checkpoint tokens through the action surrogate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import level_c_diffusion_to_vae_action_smoke as action_smoke
import level_c_vae_latent_diffusion_overfit_probe as vae_latent
from level_c_bounded_debug_diffusion_checkpoint_eval import CHECKPOINT, TRAINING_JSON, make_noisy_tau
from level_c_paper_state_transformer_arch_probe import PaperStateDiffusionTransformer, ProbeConfig


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/bounded_debug_diffusion_action_eval"
SPLIT_JSON = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "motion_names",
        "window_count",
        "token_count",
        "clean_current_action_mse",
        "noisy_current_action_mse",
        "checkpoint_current_action_mse",
        "checkpoint_current_delta_vs_noisy",
        "checkpoint_current_delta_vs_clean",
        "clean_full_action_mse",
        "noisy_full_action_mse",
        "checkpoint_full_action_mse",
        "checkpoint_full_delta_vs_noisy",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    tmp.replace(path)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def split_rows(
    split_labels: np.ndarray,
    source_motion: np.ndarray,
    clean_action: np.ndarray,
    noisy_action: np.ndarray,
    checkpoint_action: np.ndarray,
    target_action: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        mask = split_labels == split
        current_idx = action_smoke.HISTORY
        clean_full = clean_action[mask].reshape(-1, clean_action.shape[-1])
        noisy_full = noisy_action[mask].reshape(-1, noisy_action.shape[-1])
        checkpoint_full = checkpoint_action[mask].reshape(-1, checkpoint_action.shape[-1])
        target_full = target_action[mask].reshape(-1, target_action.shape[-1])
        clean_current = clean_action[mask, current_idx]
        noisy_current = noisy_action[mask, current_idx]
        checkpoint_current = checkpoint_action[mask, current_idx]
        target_current = target_action[mask, current_idx]
        row = {
            "split": split,
            "motion_names": ",".join(sorted(set(source_motion[mask].tolist()))),
            "window_count": int(np.sum(mask)),
            "token_count": int(np.sum(mask) * clean_action.shape[1]),
            "clean_current_action_mse": mse(clean_current, target_current),
            "noisy_current_action_mse": mse(noisy_current, target_current),
            "checkpoint_current_action_mse": mse(checkpoint_current, target_current),
            "clean_full_action_mse": mse(clean_full, target_full),
            "noisy_full_action_mse": mse(noisy_full, target_full),
            "checkpoint_full_action_mse": mse(checkpoint_full, target_full),
        }
        row["checkpoint_current_delta_vs_noisy"] = row["noisy_current_action_mse"] - row[
            "checkpoint_current_action_mse"
        ]
        row["checkpoint_current_delta_vs_clean"] = row["clean_current_action_mse"] - row[
            "checkpoint_current_action_mse"
        ]
        row["checkpoint_full_delta_vs_noisy"] = row["noisy_full_action_mse"] - row["checkpoint_full_action_mse"]
        rows.append(row)
    return rows


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
    clean_np, _motion_ids, latent_manifest = vae_latent.load_dataset(latent_cfg)
    clean_tau_t = torch.from_numpy(clean_np.astype(np.float32)).to(device)
    noisy_tau_t, steps = make_noisy_tau(clean_tau_t, cfg, latent_cfg.state_dim, device, args.seed)
    model = PaperStateDiffusionTransformer(token_dim=int(clean_np.shape[-1]), cfg=cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        checkpoint_tau = model(noisy_tau_t, steps).detach().cpu().numpy().astype(np.float64)
    clean_tau = clean_np.astype(np.float64)
    noisy_tau = noisy_tau_t.detach().cpu().numpy().astype(np.float64)

    vae_summary = json.loads(action_smoke.VAE_JSON.read_text(encoding="utf-8"))
    with np.load(action_smoke.VAE_NPZ) as vae_data:
        projection = vae_data["proprioception_projection"].astype(np.float64)
        sample_rows = vae_summary["rows"]
        target_action = np.stack(
            [vae_data[f"{row['sample_id']}_decoded_action"].astype(np.float64) for row in sample_rows],
            axis=0,
        )
        split_labels = np.asarray([row["split"] for row in sample_rows])
        source_motion = np.asarray([row["source_motion"] for row in sample_rows])

    clean_features = action_smoke.feature_tensor(clean_tau, projection)
    noisy_features = action_smoke.feature_tensor(noisy_tau, projection)
    checkpoint_features = action_smoke.feature_tensor(checkpoint_tau, projection)
    train_mask = action_smoke.expand_split_mask(split_labels, clean_features.shape[1], "train")
    x_train = clean_features.reshape(-1, clean_features.shape[-1])[train_mask]
    y_train = target_action.reshape(-1, target_action.shape[-1])[train_mask]
    reg = action_smoke.RIDGE_LAMBDA * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)

    clean_action = clean_features @ weights
    noisy_action = noisy_features @ weights
    checkpoint_action = checkpoint_features @ weights
    rows = split_rows(split_labels, source_motion, clean_action, noisy_action, checkpoint_action, target_action)
    row_by_split = {row["split"]: row for row in rows}
    finite_values = [
        value
        for row in rows
        for key, value in row.items()
        if key.endswith("_mse") or key.endswith("_delta_vs_noisy") or key.endswith("_delta_vs_clean")
    ]
    json_path = OUT / "level_c_bounded_debug_diffusion_action_eval.json"
    tsv_path = OUT / "level_c_bounded_debug_diffusion_action_eval.tsv"
    npz_path = OUT / "level_c_bounded_debug_diffusion_action_eval.npz"
    np.savez_compressed(
        npz_path,
        clean_action=clean_action.astype(np.float32),
        noisy_action=noisy_action.astype(np.float32),
        checkpoint_action=checkpoint_action.astype(np.float32),
        target_action=target_action.astype(np.float32),
        clean_tau=clean_tau.astype(np.float32),
        noisy_tau=noisy_tau.astype(np.float32),
        checkpoint_tau=checkpoint_tau.astype(np.float32),
        split_labels=split_labels,
        source_motion=source_motion,
        decoder_surrogate_weights=weights.astype(np.float32),
    )
    write_tsv(tsv_path, rows)
    checks = {
        "training_audit_status_ok": training["status"] == "ok",
        "checkpoint_file_exists": CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 0,
        "checkpoint_marked_debug_only": payload["is_trained_paper_checkpoint"] is False
        and payload["is_bounded_debug_training_checkpoint"] is True,
        "vae_source_status_ok": vae_summary["status"] == "ok",
        "uses_motion_level_split_manifest": split_manifest["checks"]["no_motion_crosses_splits"],
        "window_count_84": clean_tau.shape[0] == 84,
        "sequence_length_21": clean_tau.shape[1] == 21,
        "token_dim_131": clean_tau.shape[2] == 131,
        "action_dim_29": target_action.shape[-1] == 29,
        "uses_train_split_only_for_decoder_surrogate": int(np.sum(split_labels == "train")) == 28,
        "all_action_metrics_finite": bool(np.all(np.isfinite(finite_values))),
        "split_counts_28_each": {row["split"]: row["window_count"] for row in rows}
        == {"train": 28, "validation": 28, "test": 28},
        "checkpoint_action_eval_recorded_even_if_poor": all(
            "checkpoint_current_action_mse" in row for row in rows
        ),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_true_vae_decoder": True,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_paper_metrics": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "bounded_debug_diffusion_action_eval",
        "scope": "offline current-action MSE evaluation for bounded debug diffusion checkpoint tokens",
        "source_artifacts": {
            "bounded_training_audit": str(TRAINING_JSON),
            "checkpoint": str(CHECKPOINT),
            "vae_debug_latents_json": str(action_smoke.VAE_JSON),
            "vae_debug_latents_npz": str(action_smoke.VAE_NPZ),
            "split_manifest": str(SPLIT_JSON),
        },
        "settings": {
            "device": str(device),
            "torch_threads": args.torch_threads,
            "seed": args.seed,
            "history": action_smoke.HISTORY,
            "current_index": action_smoke.HISTORY,
            "ridge_lambda": action_smoke.RIDGE_LAMBDA,
            "decoder_surrogate_input_dim": int(clean_features.shape[-1]),
            "decoder_surrogate_output_dim": int(target_action.shape[-1]),
            "decoder_surrogate_train_split": "train",
            "uses_token_identity_basis": False,
            "token_dim": int(clean_tau.shape[-1]),
            "state_dim": latent_cfg.state_dim,
            "latent_dim": cfg.latent_dim,
            "sequence_length": cfg.sequence_length,
        },
        "rows": rows,
        "metrics": {
            "validation_checkpoint_current_action_mse": row_by_split["validation"][
                "checkpoint_current_action_mse"
            ],
            "test_checkpoint_current_action_mse": row_by_split["test"]["checkpoint_current_action_mse"],
            "validation_checkpoint_full_action_mse": row_by_split["validation"]["checkpoint_full_action_mse"],
            "test_checkpoint_full_action_mse": row_by_split["test"]["checkpoint_full_action_mse"],
            "validation_checkpoint_current_delta_vs_noisy": row_by_split["validation"][
                "checkpoint_current_delta_vs_noisy"
            ],
            "test_checkpoint_current_delta_vs_noisy": row_by_split["test"]["checkpoint_current_delta_vs_noisy"],
            "rows": rows,
        },
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial_debug_action_eval_only",
            "goal_complete": False,
            "why_not_complete": (
                "This connects the bounded debug checkpoint to the local train-split action surrogate and records "
                "current-action MSE. It is not a true VAE decoder evaluation, not closed-loop control, not a paper "
                "success metric, and not Fig. 5/Fig. 6 evidence."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    write_json_atomic(json_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
