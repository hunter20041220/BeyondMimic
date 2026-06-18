#!/usr/bin/env python3
"""Offline metrics for the public-LAFAN1 paper-architecture checkpoint."""

from __future__ import annotations

import csv
import hashlib
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lafan1_paper_level_vae_diffusion as paper_train  # noqa: E402


OUT = ROOT / "res/level_c/lafan1_paper_arch_offline_metrics"
TRAINING_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
ONNX_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_onnx_latency/"
    / "level_c_lafan1_paper_arch_onnx_latency_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def split_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
    }


def smoothness(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.diff(values, axis=1)
    second = np.diff(values, n=2, axis=1)
    first_norm = np.linalg.norm(first, axis=-1).mean(axis=1)
    second_norm = np.linalg.norm(second, axis=-1).mean(axis=1)
    return first_norm, second_norm


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.output_dir, OUT)
    training_json = resolve_path(args.training_json, TRAINING_JSON)
    onnx_json = resolve_path(args.onnx_json, ONNX_JSON)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    training = load_json(training_json)
    onnx_latency = load_json(onnx_json)
    checkpoint_path = Path(training["outputs"]["checkpoint"])
    dataset_path = Path(training["outputs"]["dataset_npz"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = paper_train.TrainConfig(**payload["config"])
    vae = paper_train.ConditionalVAE(cfg)
    vae.load_state_dict(strip_module_prefix(payload["vae_state_dict"]))
    vae.eval()
    diffusion = paper_train.DiffusionTransformer(cfg)
    diffusion.load_state_dict(strip_module_prefix(payload["diffusion_state_dict"]))
    diffusion.eval()

    with np.load(dataset_path, allow_pickle=True) as data:
        states_np = data["states"].astype(np.float32)
        projected_np = data["projected_states"].astype(np.float32)
        actions_np = data["actions"].astype(np.float32)
        split_labels = data["split_labels"].astype(str)
    window_count = states_np.shape[0]
    state_tokens = torch.from_numpy(states_np.reshape(-1, cfg.state_dim))
    action_tokens = torch.from_numpy(actions_np.reshape(-1, cfg.action_dim))

    decoded_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    mu_parts: list[np.ndarray] = []
    logvar_parts: list[np.ndarray] = []
    batch = 4096
    with torch.no_grad():
        for start in range(0, state_tokens.shape[0], batch):
            end = min(start + batch, state_tokens.shape[0])
            decoded, teacher, mu, logvar, _ = vae(state_tokens[start:end], action_tokens[start:end], deterministic=True)
            decoded_parts.append(decoded.numpy())
            teacher_parts.append(teacher.numpy())
            mu_parts.append(mu.numpy())
            logvar_parts.append(logvar.numpy())
    decoded_np = np.concatenate(decoded_parts, axis=0).reshape(window_count, cfg.seq_len, cfg.action_dim)
    teacher_np = np.concatenate(teacher_parts, axis=0).reshape(window_count, cfg.seq_len, cfg.action_dim)
    mu_np = np.concatenate(mu_parts, axis=0).reshape(window_count, cfg.seq_len, cfg.latent_dim)
    logvar_np = np.concatenate(logvar_parts, axis=0).reshape(window_count, cfg.seq_len, cfg.latent_dim)
    clean_tau_np = np.concatenate([projected_np, mu_np], axis=-1).astype(np.float32)

    tau = torch.from_numpy(clean_tau_np)
    noisy, steps = paper_train.noised_tau(tau, cfg, torch.device("cpu"))
    pred_parts: list[np.ndarray] = []
    diff_batch = 128
    with torch.no_grad():
        for start in range(0, window_count, diff_batch):
            end = min(start + diff_batch, window_count)
            pred_parts.append(diffusion(noisy[start:end], steps[start:end]).numpy())
    pred_tau_np = np.concatenate(pred_parts, axis=0)
    pred_projected_np = pred_tau_np[:, :, : cfg.projected_state_dim]
    pred_latent_np = pred_tau_np[:, :, cfg.projected_state_dim :]

    pred_action_parts: list[np.ndarray] = []
    pred_latent_tokens = torch.from_numpy(pred_latent_np.reshape(-1, cfg.latent_dim).astype(np.float32))
    pred_state_tokens = torch.from_numpy(states_np.reshape(-1, cfg.state_dim).astype(np.float32))
    with torch.no_grad():
        for start in range(0, pred_latent_tokens.shape[0], batch):
            end = min(start + batch, pred_latent_tokens.shape[0])
            pred_action = vae.decoder(torch.cat([pred_latent_tokens[start:end], pred_state_tokens[start:end]], dim=-1))
            pred_action_parts.append(pred_action.numpy())
    pred_actions_np = np.concatenate(pred_action_parts, axis=0).reshape(window_count, cfg.seq_len, cfg.action_dim)

    action_first, action_second = smoothness(pred_actions_np)
    reference_action_first, reference_action_second = smoothness(actions_np)
    tau_first, tau_second = smoothness(pred_tau_np)
    clean_tau_first, clean_tau_second = smoothness(clean_tau_np)
    latent_first, latent_second = smoothness(mu_np)
    pred_latent_first, pred_latent_second = smoothness(pred_latent_np)

    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        mask = split_labels == split
        kl = -0.5 * np.mean(1.0 + logvar_np[mask] - mu_np[mask] ** 2 - np.exp(logvar_np[mask]))
        noisy_tau_mse = float(np.mean((noisy.numpy()[mask] - clean_tau_np[mask]) ** 2))
        pred_tau_mse = float(np.mean((pred_tau_np[mask] - clean_tau_np[mask]) ** 2))
        row = {
            "split": split,
            "window_count": int(mask.sum()),
            "token_count": int(mask.sum() * cfg.seq_len),
            "vae_decoded_action_mse": float(np.mean((decoded_np[mask] - actions_np[mask]) ** 2)),
            "vae_teacher_action_mse": float(np.mean((teacher_np[mask] - actions_np[mask]) ** 2)),
            "vae_teacher_student_mse": float(np.mean((teacher_np[mask] - decoded_np[mask]) ** 2)),
            "vae_kl_mean": float(kl),
            "latent_abs_mean": float(np.mean(np.abs(mu_np[mask]))),
            "latent_first_difference_mean_norm": float(np.mean(latent_first[mask])),
            "latent_second_difference_mean_norm": float(np.mean(latent_second[mask])),
            "diffusion_noisy_tau_mse": noisy_tau_mse,
            "diffusion_pred_tau_mse": pred_tau_mse,
            "diffusion_tau_mse_reduction_vs_noisy": float((noisy_tau_mse - pred_tau_mse) / noisy_tau_mse),
            "diffusion_pred_projected_state_mse": float(np.mean((pred_projected_np[mask] - projected_np[mask]) ** 2)),
            "diffusion_pred_latent_mse": float(np.mean((pred_latent_np[mask] - mu_np[mask]) ** 2)),
            "diffusion_pred_current_latent_mse": float(
                np.mean((pred_latent_np[mask, cfg.history] - mu_np[mask, cfg.history]) ** 2)
            ),
            "decoded_pred_current_action_mse": float(
                np.mean((pred_actions_np[mask, cfg.history] - actions_np[mask, cfg.history]) ** 2)
            ),
            "decoded_pred_sequence_action_mse": float(np.mean((pred_actions_np[mask] - actions_np[mask]) ** 2)),
            "pred_action_first_difference_mean_norm": float(np.mean(action_first[mask])),
            "reference_action_first_difference_mean_norm": float(np.mean(reference_action_first[mask])),
            "pred_action_second_difference_mean_norm": float(np.mean(action_second[mask])),
            "reference_action_second_difference_mean_norm": float(np.mean(reference_action_second[mask])),
            "pred_tau_first_difference_mean_norm": float(np.mean(tau_first[mask])),
            "clean_tau_first_difference_mean_norm": float(np.mean(clean_tau_first[mask])),
            "pred_tau_second_difference_mean_norm": float(np.mean(tau_second[mask])),
            "clean_tau_second_difference_mean_norm": float(np.mean(clean_tau_second[mask])),
            "pred_latent_first_difference_mean_norm": float(np.mean(pred_latent_first[mask])),
            "pred_latent_second_difference_mean_norm": float(np.mean(pred_latent_second[mask])),
        }
        rows.append(row)

    all_values = np.concatenate(
        [
            np.asarray([r["vae_decoded_action_mse"] for r in rows]),
            np.asarray([r["diffusion_pred_tau_mse"] for r in rows]),
            np.asarray([r["decoded_pred_current_action_mse"] for r in rows]),
        ]
    )
    npz_path = out / "level_c_lafan1_paper_arch_offline_metrics.npz"
    np.savez_compressed(
        npz_path,
        split_labels=split_labels,
        latent_mu=mu_np.astype(np.float32),
        clean_tau=clean_tau_np.astype(np.float32),
        predicted_tau=pred_tau_np.astype(np.float32),
        decoded_actions=decoded_np.astype(np.float32),
        predicted_actions=pred_actions_np.astype(np.float32),
    )
    checks = {
        "source_training_status_ok": training["status"] == "ok",
        "source_checkpoint_hash_matches": sha256(checkpoint_path) == training["metrics"]["checkpoint_sha256"],
        "onnx_latency_status_ok": onnx_latency["status"] == "ok",
        "paper_architecture_checkpoint": payload.get("paper_architecture") is True,
        "public_dataset_boundary_recorded": payload.get("paper_dataset") is False,
        "all_splits_present": {row["split"] for row in rows} == {"train", "validation", "test"},
        "all_metrics_finite": bool(np.all(np.isfinite(all_values)))
        and all(np.isfinite(v) for row in rows for k, v in row.items() if k not in {"split"}),
        "diffusion_improves_vs_noisy_all_splits": all(row["diffusion_tau_mse_reduction_vs_noisy"] > 0.0 for row in rows),
        "validation_and_test_action_metrics_present": all(
            next(row for row in rows if row["split"] == split)["decoded_pred_current_action_mse"] >= 0.0
            for split in ["validation", "test"]
        ),
        "smoothness_metrics_present": all(row["pred_action_second_difference_mean_norm"] >= 0.0 for row in rows),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_closed_loop_success": True,
        "does_not_claim_tensorrt_or_robot": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_offline_metrics_audit",
        "scope": (
            "Split-wise offline VAE, diffusion reconstruction, downstream decoded-action, smoothness, and latency-link "
            "metrics for the full paper-sized checkpoint trained on public retargeted LAFAN1 G1 motions."
        ),
        "settings": {
            "state_dim": cfg.state_dim,
            "projected_state_dim": cfg.projected_state_dim,
            "latent_dim": cfg.latent_dim,
            "action_dim": cfg.action_dim,
            "seq_len": cfg.seq_len,
            "window_count": int(window_count),
            "token_count": int(window_count * cfg.seq_len),
            "torch_threads": torch.get_num_threads(),
            "training_json": str(training_json),
            "onnx_latency_json": str(onnx_json),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
        },
        "metrics": {
            "validation_diffusion_pred_tau_mse": next(
                row["diffusion_pred_tau_mse"] for row in rows if row["split"] == "validation"
            ),
            "test_diffusion_pred_tau_mse": next(row["diffusion_pred_tau_mse"] for row in rows if row["split"] == "test"),
            "validation_decoded_pred_current_action_mse": next(
                row["decoded_pred_current_action_mse"] for row in rows if row["split"] == "validation"
            ),
            "test_decoded_pred_current_action_mse": next(
                row["decoded_pred_current_action_mse"] for row in rows if row["split"] == "test"
            ),
            "validation_vae_kl_mean": next(row["vae_kl_mean"] for row in rows if row["split"] == "validation"),
            "test_vae_kl_mean": next(row["vae_kl_mean"] for row in rows if row["split"] == "test"),
            "validation_action_second_difference_mean_norm": next(
                row["pred_action_second_difference_mean_norm"] for row in rows if row["split"] == "validation"
            ),
            "test_action_second_difference_mean_norm": next(
                row["pred_action_second_difference_mean_norm"] for row in rows if row["split"] == "test"
            ),
            "vae_decoder_torch_cpu_p95_ms": onnx_latency["metrics"]["vae_decoder_torch_cpu_p95_ms"],
            "diffusion_denoiser_torch_cpu_p95_ms": onnx_latency["metrics"]["diffusion_denoiser_torch_cpu_p95_ms"],
            "diffusion_denoiser_onnx_reference_cpu_p95_ms": onnx_latency["metrics"][
                "diffusion_denoiser_onnx_reference_cpu_p95_ms"
            ],
        },
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(out / "level_c_lafan1_paper_arch_offline_metrics_audit.json"),
            "tsv": str(out / "level_c_lafan1_paper_arch_offline_metrics_rows.tsv"),
            "npz": str(npz_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_offline_metrics",
            "why_not_complete": (
                "These are offline metrics from the full local public-data checkpoint. They do not measure closed-loop "
                "unconditional/guided success, falls, collisions, TensorRT deployment, Fig. 5/Fig. 6 rollouts, or real "
                "robot execution."
            ),
        },
    }
    write_json_atomic(Path(summary["outputs"]["json"]), summary)
    write_tsv(Path(summary["outputs"]["tsv"]), rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "validation_tau_mse": summary["metrics"]["validation_diffusion_pred_tau_mse"],
                "test_tau_mse": summary["metrics"]["test_diffusion_pred_tau_mse"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute split-wise offline metrics for a public-LAFAN1 paper-architecture checkpoint."
    )
    parser.add_argument(
        "--training-json",
        default=None,
        help="Training summary JSON to evaluate. Defaults to the original public-LAFAN1 paper-architecture run.",
    )
    parser.add_argument(
        "--onnx-json",
        default=None,
        help="ONNX latency audit JSON used for latency-link metrics. Defaults to the base paper-architecture audit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON/TSV/NPZ. Defaults to the original offline metrics directory.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
