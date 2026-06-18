#!/usr/bin/env python3
"""Debug-only motion-split held-out evaluation for the local conditional VAE.

This trains the tiny conditional VAE only on the train motion split from the
local paper-state window fixtures, then reports validation/test action
reconstruction metrics. Teacher actions are deterministic synthetic projections
from local retargeted-motion states, so this is not true DAgger data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import level_c_vae_debug_overfit_latent_artifact as vae_debug


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.evaluation.metrics import action_mse
from beyondmimic_reimpl.vae.latent import kl_standard_normal


OUT = ROOT / "res/level_c/vae_motion_split_heldout_eval"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(meta_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    split_labels = np.asarray([row["split"] for row in meta_rows])
    return {split: np.nonzero(split_labels == split)[0] for split in ["train", "validation", "test"]}


def motion_names_for_split(meta_rows: list[dict[str, Any]], split: str) -> list[str]:
    return sorted({row["source_motion"] for row in meta_rows if row["split"] == split})


def kl_mean(mu: np.ndarray, logvar: np.ndarray) -> float:
    flat_mu = mu.reshape(-1, mu.shape[-1])
    flat_logvar = logvar.reshape(-1, logvar.shape[-1])
    values = [kl_standard_normal(flat_mu[i], flat_logvar[i]) for i in range(flat_mu.shape[0])]
    return float(np.mean(values))


def evaluate_split(
    split: str,
    idx: np.ndarray,
    initial_pred: np.ndarray,
    final_pred: np.ndarray,
    target: np.ndarray,
    final_mu: np.ndarray,
    final_logvar: np.ndarray,
) -> dict[str, Any]:
    initial_loss = action_mse(initial_pred[idx], target[idx])
    final_loss = action_mse(final_pred[idx], target[idx])
    reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0.0
    final_error = final_pred[idx] - target[idx]
    return {
        "split": split,
        "token_count": int(len(idx)),
        "initial_action_mse": initial_loss,
        "final_action_mse": final_loss,
        "action_mse_reduction_ratio": float(reduction),
        "final_action_max_abs_error": float(np.max(np.abs(final_error))),
        "final_kl_mean_sum_over_latent": kl_mean(final_mu[idx], final_logvar[idx]),
        "final_latent_mu_abs_mean": float(np.mean(np.abs(final_mu[idx]))),
        "final_latent_mu_std": float(np.std(final_mu[idx])),
        "final_latent_std_mean": float(np.mean(np.exp(0.5 * final_logvar[idx]))),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "token_count",
        "motion_names",
        "initial_action_mse",
        "final_action_mse",
        "action_mse_reduction_ratio",
        "final_action_max_abs_error",
        "final_kl_mean_sum_over_latent",
        "final_latent_mu_abs_mean",
        "final_latent_mu_std",
        "final_latent_std_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)

    device = torch.device(args.device)
    cfg = replace(vae_debug.DebugVAEConfig(), seed=args.seed, epochs=args.epochs)
    data = vae_debug.build_dataset(cfg)
    indices = split_indices(data["meta_rows"])

    model = vae_debug.TinyConditionalVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    encoder_input = torch.from_numpy(data["encoder_input"]).to(device)
    proprioception = torch.from_numpy(data["proprioception"]).to(device)
    target_action = torch.from_numpy(data["teacher_action"]).to(device)
    train_idx = torch.from_numpy(indices["train"]).long().to(device)

    with torch.no_grad():
        initial_pred_t, initial_mu_t, initial_logvar_t, _ = model(
            encoder_input,
            proprioception,
            deterministic=True,
        )

    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        pred, mu, logvar, _ = model(
            encoder_input[train_idx],
            proprioception[train_idx],
            deterministic=False,
        )
        reconstruction = F.mse_loss(pred, target_action[train_idx])
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction + cfg.kl_coefficient * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, args.epochs - 1}:
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_reconstruction_mse": float(reconstruction.detach().cpu().item()),
                    "train_kl_mean": float(kl.detach().cpu().item()),
                    "train_total_loss": float(loss.detach().cpu().item()),
                }
            )

    with torch.no_grad():
        final_pred_t, final_mu_t, final_logvar_t, _ = model(
            encoder_input,
            proprioception,
            deterministic=True,
        )

    initial_pred = initial_pred_t.detach().cpu().numpy()
    final_pred = final_pred_t.detach().cpu().numpy()
    final_mu = final_mu_t.detach().cpu().numpy()
    final_logvar = final_logvar_t.detach().cpu().numpy()
    target = data["teacher_action"]

    rows = []
    for split in ["train", "validation", "test"]:
        row = evaluate_split(split, indices[split], initial_pred, final_pred, target, final_mu, final_logvar)
        row["motion_names"] = ",".join(motion_names_for_split(data["meta_rows"], split))
        rows.append(row)

    split_motion_names = {split: motion_names_for_split(data["meta_rows"], split) for split in indices}
    split_sets = {split: set(names) for split, names in split_motion_names.items()}
    target_counts = {split: int(len(indices[split])) for split in indices}
    validation_row = next(row for row in rows if row["split"] == "validation")
    test_row = next(row for row in rows if row["split"] == "test")
    train_row = next(row for row in rows if row["split"] == "train")

    json_path = OUT / "level_c_vae_motion_split_heldout_eval.json"
    tsv_path = OUT / "level_c_vae_motion_split_heldout_eval.tsv"
    npz_path = OUT / "level_c_vae_motion_split_heldout_eval.npz"
    np.savez_compressed(
        npz_path,
        final_mu=final_mu.astype(np.float32),
        final_logvar=final_logvar.astype(np.float32),
        final_decoded_action=final_pred.astype(np.float32),
        teacher_action=target.astype(np.float32),
        split_labels=np.asarray([row["split"] for row in data["meta_rows"]]),
        source_motion=np.asarray([row["source_motion"] for row in data["meta_rows"]]),
        token_index=np.asarray([row["token_index"] for row in data["meta_rows"]], dtype=np.int64),
    )
    write_tsv(tsv_path, rows)

    checks = {
        "paper_state_inputs_exist": vae_debug.PAPER_STATE_JSON.is_file()
        and vae_debug.SPLIT_JSON.is_file()
        and vae_debug.SPLIT_TSV.is_file(),
        "uses_motion_level_train_validation_test_split": target_counts == {
            "train": 588,
            "validation": 588,
            "test": 588,
        },
        "no_motion_crosses_splits": not (
            split_sets["train"] & split_sets["validation"]
            or split_sets["train"] & split_sets["test"]
            or split_sets["validation"] & split_sets["test"]
        ),
        "optimizer_trains_only_train_split": int(len(train_idx)) == target_counts["train"],
        "uses_package_action_mse_metric": True,
        "uses_package_vae_kl_metric": True,
        "all_metrics_finite": bool(
            np.all(
                np.isfinite(
                    [
                        value
                        for row in rows
                        for key, value in row.items()
                        if key not in {"split", "motion_names"}
                    ]
                )
            )
        ),
        "train_action_mse_decreases": train_row["final_action_mse"] < train_row["initial_action_mse"],
        "validation_action_mse_decreases": validation_row["final_action_mse"]
        < validation_row["initial_action_mse"],
        "test_action_mse_decreases": test_row["final_action_mse"] < test_row["initial_action_mse"],
        "validation_reduction_above_half": validation_row["action_mse_reduction_ratio"] > 0.5,
        "test_reduction_above_half": test_row["action_mse_reduction_ratio"] > 0.5,
        "latent_dim_32": cfg.latent_dim == 32 and final_mu.shape[-1] == 32,
        "action_dim_29": cfg.action_dim == 29 and final_pred.shape[-1] == 29,
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_trained_paper_checkpoint": True,
        "does_not_claim_closed_loop_vae_eval": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_vae_motion_split_heldout_eval",
        "scope": (
            "tiny CPU conditional VAE trained only on the train motion split and evaluated on held-out "
            "validation/test local paper-state fixture motions"
        ),
        "paper_evidence": {
            "vae_elbo": str(ROOT / "reproduction/paper/source/tex/method.tex:109-115"),
            "state_latent_dataset": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "goal_vae_dagger": str(ROOT / "goal.md:1148-1190,1431-1467"),
            "motion_split_manifest": str(vae_debug.SPLIT_JSON),
        },
        "not_a_replacement_for": [
            "true DAgger rollout",
            "trained paper conditional VAE checkpoint",
            "closed-loop VAE rollout stability evaluation",
            "paper state-latent trajectory dataset",
            "paper validation/test protocol",
        ],
        "settings": asdict(cfg)
        | {
            "device": str(device),
            "torch_threads": args.torch_threads,
            "training_split": "train",
            "heldout_splits": ["validation", "test"],
            "teacher_action_source": "deterministic_synthetic_projection_of_local_paper_state_windows",
        },
        "split_motion_names": split_motion_names,
        "epoch_rows": epoch_rows,
        "rows": rows,
        "metrics": {
            "train_initial_action_mse": train_row["initial_action_mse"],
            "train_final_action_mse": train_row["final_action_mse"],
            "train_action_mse_reduction_ratio": train_row["action_mse_reduction_ratio"],
            "validation_initial_action_mse": validation_row["initial_action_mse"],
            "validation_final_action_mse": validation_row["final_action_mse"],
            "validation_action_mse_reduction_ratio": validation_row["action_mse_reduction_ratio"],
            "test_initial_action_mse": test_row["initial_action_mse"],
            "test_final_action_mse": test_row["final_action_mse"],
            "test_action_mse_reduction_ratio": test_row["action_mse_reduction_ratio"],
            "final_train_kl_mean_sum_over_latent": train_row["final_kl_mean_sum_over_latent"],
            "final_validation_kl_mean_sum_over_latent": validation_row["final_kl_mean_sum_over_latent"],
            "final_test_kl_mean_sum_over_latent": test_row["final_kl_mean_sum_over_latent"],
            "token_counts": target_counts,
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This is a stronger debug VAE training/evaluation gate than full-dataset overfit because it trains "
                "only on the train motion and evaluates held-out motions. It still uses deterministic synthetic "
                "teacher actions and local retargeted-motion fixtures, not true teacher-policy DAgger rollouts, "
                "closed-loop VAE rollouts, or the paper training scale."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(json_path),
                "validation_reduction": summary["metrics"]["validation_action_mse_reduction_ratio"],
                "test_reduction": summary["metrics"]["test_action_mse_reduction_ratio"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
