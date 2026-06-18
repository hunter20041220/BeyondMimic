#!/usr/bin/env python3
"""Export and run the resource-adjusted tiny diffusion denoiser as ONNX."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
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

import level_c_resource_adjusted_tiny_diffusion_training_run as tiny  # noqa: E402
import level_c_vae_latent_diffusion_overfit_probe as vae_latent  # noqa: E402


OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference"
SOURCE_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
    / "level_c_resource_adjusted_tiny_diffusion_training_run.json"
)
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "split",
                "sample_index",
                "input_shape",
                "output_shape",
                "torch_mse_to_clean",
                "onnx_mse_to_clean",
                "max_abs_onnx_vs_torch",
                "matches_torch",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def split_index(motion_ids: np.ndarray, latent_manifest: dict[str, Any], split: str) -> int:
    split_labels, masks = tiny.split_masks(latent_manifest, motion_ids)
    indices = np.nonzero(masks[split])[0]
    if len(indices) == 0:
        raise RuntimeError(f"split has no windows: {split}")
    return int(indices[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = load_json(SOURCE_JSON)
    checkpoint_path = Path(source["outputs"]["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = tiny.TinyConfig(**payload["config"])
    model = tiny.TinyDenoiser(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    clean_np, motion_ids, latent_manifest = vae_latent.load_dataset(
        vae_latent.VaeLatentDiffusionConfig(seed=cfg.seed)
    )
    noisy_np, steps_np = tiny.make_noisy(clean_np, cfg)
    sample_idx = split_index(motion_ids, latent_manifest, "validation")
    noisy = torch.from_numpy(noisy_np[sample_idx : sample_idx + 1].astype(np.float32))
    steps = torch.from_numpy(steps_np[sample_idx : sample_idx + 1].astype(np.int64))
    clean = clean_np[sample_idx : sample_idx + 1].astype(np.float32)
    with torch.no_grad():
        torch_pred = model(noisy, steps).detach().cpu().numpy()

    onnx_path = OUT / "resource_adjusted_tiny_denoiser_debug.onnx"
    torch.onnx.export(
        model,
        (noisy, steps),
        onnx_path,
        input_names=["noisy_tau", "diffusion_steps"],
        output_names=["predicted_clean_tau"],
        dynamic_axes={
            "noisy_tau": {0: "batch", 1: "sequence"},
            "diffusion_steps": {0: "batch", 1: "sequence"},
            "predicted_clean_tau": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)
    metadata = {
        "experiment_type": "resource_adjusted_tiny_diffusion_onnx_export_inference",
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_run_id": source["run_id"],
        "paper_level": "false",
        "resource_adjusted_debug": "true",
        "not_trained_paper_checkpoint": "true",
    }
    existing = {item.key: item for item in proto.metadata_props}
    for key, value in metadata.items():
        prop = existing.get(key) or proto.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(proto, str(onnx_path))
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)

    evaluator = ReferenceEvaluator(proto)
    onnx_pred = evaluator.run(
        None,
        {
            "noisy_tau": noisy.numpy().astype(np.float32),
            "diffusion_steps": steps.numpy().astype(np.int64),
        },
    )[0]
    max_abs = float(np.max(np.abs(onnx_pred - torch_pred)))
    torch_mse = float(np.mean(np.square(torch_pred - clean)))
    onnx_mse = float(np.mean(np.square(onnx_pred - clean)))
    row = {
        "split": "validation",
        "sample_index": sample_idx,
        "input_shape": json.dumps(list(noisy.shape)),
        "output_shape": json.dumps(list(onnx_pred.shape)),
        "torch_mse_to_clean": torch_mse,
        "onnx_mse_to_clean": onnx_mse,
        "max_abs_onnx_vs_torch": max_abs,
        "matches_torch": max_abs <= 1e-5,
    }
    npz_path = OUT / "resource_adjusted_tiny_denoiser_onnx_debug_io.npz"
    np.savez_compressed(
        npz_path,
        noisy_tau=noisy.numpy().astype(np.float32),
        diffusion_steps=steps.numpy().astype(np.int64),
        clean_tau=clean,
        torch_predicted_clean_tau=torch_pred.astype(np.float32),
        onnx_predicted_clean_tau=onnx_pred.astype(np.float32),
    )
    metadata_keys = sorted(item.key for item in proto.metadata_props)
    checks = {
        "source_training_status_ok": source["status"] == "ok",
        "checkpoint_exists": checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0,
        "checkpoint_sha_matches_training_json": sha256(checkpoint_path) == source["metrics"]["checkpoint_sha256"],
        "payload_marks_not_paper_checkpoint": payload.get("paper_level") is False
        and payload.get("is_trained_paper_checkpoint") is False,
        "onnx_file_written": onnx_path.is_file() and onnx_path.stat().st_size > 0,
        "onnx_checker_passed": True,
        "reference_evaluator_loaded": True,
        "onnx_matches_torch": row["matches_torch"],
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "metadata_marks_debug_not_paper": all(key in metadata_keys for key in metadata),
        "does_not_claim_tensorrt": True,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "resource_adjusted_tiny_diffusion_onnx_export_inference",
        "scope": (
            "export the resource-adjusted tiny debug denoiser checkpoint to ONNX and verify deterministic ONNX "
            "reference inference against PyTorch on one validation window"
        ),
        "settings": {
            "python": str(BM_DIFFUSION_PY),
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "opset": 17,
            "validation_sample_index": sample_idx,
        },
        "metrics": {
            "onnx_size_bytes": onnx_path.stat().st_size,
            "onnx_sha256": sha256(onnx_path),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "torch_mse_to_clean": torch_mse,
            "onnx_mse_to_clean": onnx_mse,
            "max_abs_onnx_vs_torch": max_abs,
            "sequence_length": cfg.sequence_length,
            "token_dim": cfg.token_dim,
        },
        "rows": [row],
        "metadata_keys": metadata_keys,
        "checks": checks,
        "outputs": {
            "json": str(OUT / "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json"),
            "tsv": str(OUT / "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.tsv"),
            "npz": str(npz_path),
            "onnx": str(onnx_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "resource_adjusted_debug_onnx_only",
            "why_not_complete": (
                "This links the local resource-adjusted debug checkpoint to an executable ONNX graph and verifies "
                "ONNX reference inference against PyTorch. It is not the paper Transformer checkpoint, not TensorRT, "
                "not asynchronous deployment, not closed-loop guidance/control, and not Fig. 5/Fig. 6 evidence."
            ),
        },
    }
    write_json_atomic(OUT / "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json", summary)
    write_tsv(OUT / "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.tsv", [row])
    print(
        json.dumps(
            {
                "status": summary["status"],
                "onnx": str(onnx_path),
                "max_abs_onnx_vs_torch": max_abs,
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
