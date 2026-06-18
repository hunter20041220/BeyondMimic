#!/usr/bin/env python3
"""Export the public-LAFAN1 paper-architecture VAE/diffusion checkpoint to ONNX.

The exported models come from the full paper-sized local training run, but the
dataset boundary remains important: this is public retargeted LAFAN1 data, not
the unavailable official DAgger teacher-rollout dataset and not TensorRT.
"""

from __future__ import annotations

import csv
import hashlib
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lafan1_paper_level_vae_diffusion as paper_train  # noqa: E402


OUT = ROOT / "res/level_c/lafan1_paper_arch_onnx_latency"
TRAINING_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)


class VaeDecoderExport(torch.nn.Module):
    def __init__(self, vae: paper_train.ConditionalVAE) -> None:
        super().__init__()
        self.decoder = vae.decoder

    def forward(self, latent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, state], dim=-1))


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
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def timed_loop(fn, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def export_onnx(
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    metadata: dict[str, str],
) -> onnx.ModelProto:
    torch.onnx.export(
        model,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    existing = {item.key: item for item in proto.metadata_props}
    for key, value in metadata.items():
        prop = existing.get(key) or proto.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(proto, str(path))
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    return proto


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.output_dir, OUT)
    training_json = resolve_path(args.training_json, TRAINING_JSON)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    training = load_json(training_json)
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
        states = data["states"].astype(np.float32)
        projected_states = data["projected_states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        split_labels = data["split_labels"].astype(str)
    validation_indices = np.nonzero(split_labels == "validation")[0]
    if len(validation_indices) == 0:
        raise RuntimeError("validation split is empty")
    sample_idx = int(validation_indices[0])
    current_idx = cfg.history
    state = torch.from_numpy(states[sample_idx : sample_idx + 1, current_idx].astype(np.float32))
    action = torch.from_numpy(actions[sample_idx : sample_idx + 1, current_idx].astype(np.float32))
    clean_tau_np = np.concatenate(
        [
            projected_states[sample_idx : sample_idx + 1],
            np.zeros((1, cfg.seq_len, cfg.latent_dim), dtype=np.float32),
        ],
        axis=-1,
    )
    with torch.no_grad():
        _, _, mu, _, _ = vae(state, action, deterministic=True)
        decoded_torch = VaeDecoderExport(vae)(mu, state).detach().cpu().numpy()
    clean_tau_np[:, :, cfg.projected_state_dim :] = mu.detach().cpu().numpy()[:, None, :]
    clean_tau = torch.from_numpy(clean_tau_np.astype(np.float32))
    steps = torch.zeros((1, cfg.seq_len, 2), dtype=torch.long)
    with torch.no_grad():
        diffusion_torch = diffusion(clean_tau, steps).detach().cpu().numpy()

    metadata = {
        "experiment_type": "public_lafan1_paper_arch_onnx_latency",
        "source_training_json": str(training_json),
        "source_run_id": payload["run_id"],
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "paper_architecture": str(bool(payload["paper_architecture"])).lower(),
        "paper_dataset": str(bool(payload["paper_dataset"])).lower(),
        "public_dataset": str(payload["public_dataset"]),
        "not_tensorrt": "true",
        "not_closed_loop": "true",
    }
    vae_onnx = out / "lafan1_paper_arch_vae_decoder.onnx"
    diffusion_onnx = out / "lafan1_paper_arch_diffusion_denoiser.onnx"
    vae_proto = export_onnx(
        VaeDecoderExport(vae),
        (mu, state),
        vae_onnx,
        input_names=["latent", "state"],
        output_names=["decoded_action"],
        dynamic_axes={"latent": {0: "batch"}, "state": {0: "batch"}, "decoded_action": {0: "batch"}},
        metadata=metadata | {"component": "vae_decoder"},
    )
    diffusion_proto = export_onnx(
        diffusion,
        (clean_tau, steps),
        diffusion_onnx,
        input_names=["noisy_tau", "diffusion_steps"],
        output_names=["predicted_clean_tau"],
        dynamic_axes={
            "noisy_tau": {0: "batch", 1: "sequence"},
            "diffusion_steps": {0: "batch", 1: "sequence"},
            "predicted_clean_tau": {0: "batch", 1: "sequence"},
        },
        metadata=metadata | {"component": "diffusion_denoiser"},
    )

    vae_ref = ReferenceEvaluator(vae_proto)
    diffusion_ref = ReferenceEvaluator(diffusion_proto)
    vae_onnx_pred = vae_ref.run(
        None,
        {"latent": mu.detach().cpu().numpy().astype(np.float32), "state": state.numpy().astype(np.float32)},
    )[0]
    diffusion_onnx_pred = diffusion_ref.run(
        None,
        {"noisy_tau": clean_tau.numpy().astype(np.float32), "diffusion_steps": steps.numpy().astype(np.int64)},
    )[0]
    vae_max_abs = float(np.max(np.abs(vae_onnx_pred - decoded_torch)))
    diffusion_max_abs = float(np.max(np.abs(diffusion_onnx_pred - diffusion_torch)))
    vae_mse_to_action = float(np.mean(np.square(decoded_torch - action.numpy())))
    diffusion_mse_to_tau = float(np.mean(np.square(diffusion_torch - clean_tau_np)))

    warmup = 2
    iterations = 5

    def run_vae_torch() -> None:
        with torch.no_grad():
            VaeDecoderExport(vae)(mu, state)

    def run_vae_onnx() -> None:
        vae_ref.run(None, {"latent": mu.detach().cpu().numpy().astype(np.float32), "state": state.numpy().astype(np.float32)})

    def run_diffusion_torch() -> None:
        with torch.no_grad():
            diffusion(clean_tau, steps)

    def run_diffusion_onnx() -> None:
        diffusion_ref.run(None, {"noisy_tau": clean_tau.numpy().astype(np.float32), "diffusion_steps": steps.numpy().astype(np.int64)})

    latency_rows = []
    for runtime, fn in [
        ("vae_decoder_pytorch_cpu", run_vae_torch),
        ("vae_decoder_onnx_reference_cpu", run_vae_onnx),
        ("diffusion_denoiser_pytorch_cpu", run_diffusion_torch),
        ("diffusion_denoiser_onnx_reference_cpu", run_diffusion_onnx),
    ]:
        stats = timed_loop(fn, warmup=warmup, iterations=iterations)
        latency_rows.append(
            {
                "runtime": runtime,
                "warmup_iterations": warmup,
                "timed_iterations": iterations,
                **stats,
                "paper_20ms_fraction_p95": stats["p95_ms"] / 20.0,
                "control_40ms_fraction_p95": stats["p95_ms"] / 40.0,
            }
        )

    io_npz = out / "lafan1_paper_arch_onnx_io_fixture.npz"
    np.savez_compressed(
        io_npz,
        state=state.numpy().astype(np.float32),
        action=action.numpy().astype(np.float32),
        latent=mu.detach().cpu().numpy().astype(np.float32),
        clean_tau=clean_tau_np.astype(np.float32),
        diffusion_steps=steps.numpy().astype(np.int64),
        torch_decoded_action=decoded_torch.astype(np.float32),
        onnx_decoded_action=vae_onnx_pred.astype(np.float32),
        torch_predicted_clean_tau=diffusion_torch.astype(np.float32),
        onnx_predicted_clean_tau=diffusion_onnx_pred.astype(np.float32),
    )

    rows = [
        {
            "component": "vae_decoder",
            "sample_index": sample_idx,
            "input_shape": json.dumps([list(mu.shape), list(state.shape)]),
            "output_shape": json.dumps(list(vae_onnx_pred.shape)),
            "torch_metric_mse": vae_mse_to_action,
            "max_abs_onnx_vs_torch": vae_max_abs,
            "onnx_size_bytes": vae_onnx.stat().st_size,
            "onnx_sha256": sha256(vae_onnx),
        },
        {
            "component": "diffusion_denoiser",
            "sample_index": sample_idx,
            "input_shape": json.dumps([list(clean_tau.shape), list(steps.shape)]),
            "output_shape": json.dumps(list(diffusion_onnx_pred.shape)),
            "torch_metric_mse": diffusion_mse_to_tau,
            "max_abs_onnx_vs_torch": diffusion_max_abs,
            "onnx_size_bytes": diffusion_onnx.stat().st_size,
            "onnx_sha256": sha256(diffusion_onnx),
        },
    ]
    metadata_keys = {
        "vae_decoder": sorted(item.key for item in vae_proto.metadata_props),
        "diffusion_denoiser": sorted(item.key for item in diffusion_proto.metadata_props),
    }
    checks = {
        "source_training_status_ok": training["status"] == "ok",
        "source_checkpoint_exists": checkpoint_path.is_file() and checkpoint_path.stat().st_size > 250_000_000,
        "source_checkpoint_hash_matches": sha256(checkpoint_path) == training["metrics"]["checkpoint_sha256"],
        "paper_architecture_checkpoint": payload.get("paper_architecture") is True,
        "public_lafan1_dataset_boundary_recorded": payload.get("paper_dataset") is False
        and payload.get("public_dataset") == "LAFAN1_Retargeting_Dataset/g1",
        "vae_onnx_written": vae_onnx.is_file() and vae_onnx.stat().st_size > 0,
        "diffusion_onnx_written": diffusion_onnx.is_file() and diffusion_onnx.stat().st_size > 0,
        "onnx_checker_passed": True,
        "reference_evaluator_loaded": True,
        "vae_onnx_matches_torch": vae_max_abs <= 1e-5,
        "diffusion_onnx_matches_torch": diffusion_max_abs <= 1e-4,
        "latency_rows_written": len(latency_rows) == 4
        and all(row["p95_ms"] > 0.0 and np.isfinite(row["p95_ms"]) for row in latency_rows),
        "io_fixture_written": io_npz.is_file() and io_npz.stat().st_size > 0,
        "metadata_marks_boundary": all(
            key in metadata_keys["vae_decoder"] and key in metadata_keys["diffusion_denoiser"]
            for key in ["paper_architecture", "paper_dataset", "public_dataset", "not_tensorrt", "not_closed_loop"]
        ),
        "does_not_claim_tensorrt": True,
        "does_not_claim_closed_loop_or_robot": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_onnx_latency_audit",
        "scope": (
            "ONNX export, ONNX ReferenceEvaluator parity, and host CPU latency audit for the full paper-sized "
            "VAE decoder and diffusion denoiser trained on public retargeted LAFAN1 G1 motions."
        ),
        "settings": {
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "torch_threads": torch.get_num_threads(),
            "opset": 17,
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "sample_split": "validation",
            "sample_index": sample_idx,
            "seq_len": cfg.seq_len,
            "state_dim": cfg.state_dim,
            "projected_state_dim": cfg.projected_state_dim,
            "latent_dim": cfg.latent_dim,
            "token_dim": cfg.token_dim,
            "training_json": str(training_json),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
        },
        "metrics": {
            "vae_parameter_count": int(sum(p.numel() for p in vae.parameters())),
            "diffusion_parameter_count": int(sum(p.numel() for p in diffusion.parameters())),
            "vae_onnx_size_bytes": vae_onnx.stat().st_size,
            "diffusion_onnx_size_bytes": diffusion_onnx.stat().st_size,
            "vae_onnx_sha256": sha256(vae_onnx),
            "diffusion_onnx_sha256": sha256(diffusion_onnx),
            "vae_max_abs_onnx_vs_torch": vae_max_abs,
            "diffusion_max_abs_onnx_vs_torch": diffusion_max_abs,
            "vae_decoder_current_action_mse": vae_mse_to_action,
            "diffusion_single_window_tau_mse": diffusion_mse_to_tau,
            "vae_decoder_torch_cpu_p95_ms": next(
                row["p95_ms"] for row in latency_rows if row["runtime"] == "vae_decoder_pytorch_cpu"
            ),
            "vae_decoder_onnx_reference_cpu_p95_ms": next(
                row["p95_ms"] for row in latency_rows if row["runtime"] == "vae_decoder_onnx_reference_cpu"
            ),
            "diffusion_denoiser_torch_cpu_p95_ms": next(
                row["p95_ms"] for row in latency_rows if row["runtime"] == "diffusion_denoiser_pytorch_cpu"
            ),
            "diffusion_denoiser_onnx_reference_cpu_p95_ms": next(
                row["p95_ms"] for row in latency_rows if row["runtime"] == "diffusion_denoiser_onnx_reference_cpu"
            ),
        },
        "component_rows": rows,
        "latency_rows": latency_rows,
        "checks": checks,
        "metadata_keys": metadata_keys,
        "outputs": {
            "json": str(out / "level_c_lafan1_paper_arch_onnx_latency_audit.json"),
            "component_tsv": str(out / "level_c_lafan1_paper_arch_onnx_latency_components.tsv"),
            "latency_tsv": str(out / "level_c_lafan1_paper_arch_onnx_latency_rows.tsv"),
            "npz": str(io_npz),
            "vae_decoder_onnx": str(vae_onnx),
            "diffusion_denoiser_onnx": str(diffusion_onnx),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_onnx_export",
            "why_not_complete": (
                "This exports and times the full paper-sized local VAE/diffusion checkpoint. It is still not "
                "TensorRT, not asynchronous deployment, not closed-loop simulation/robot execution, and not an "
                "official-paper checkpoint because the official teacher rollout dataset is unavailable locally."
            ),
        },
    }
    write_json_atomic(Path(summary["outputs"]["json"]), summary)
    write_tsv(Path(summary["outputs"]["component_tsv"]), rows)
    write_tsv(Path(summary["outputs"]["latency_tsv"]), latency_rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "vae_max_abs": vae_max_abs,
                "diffusion_max_abs": diffusion_max_abs,
                "diffusion_onnx_size_bytes": diffusion_onnx.stat().st_size,
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and time ONNX graphs for a public-LAFAN1 paper-architecture checkpoint."
    )
    parser.add_argument(
        "--training-json",
        default=None,
        help="Training summary JSON to export. Defaults to the original public-LAFAN1 paper-architecture run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON/TSV/NPZ/ONNX files. Defaults to the original ONNX latency directory.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
