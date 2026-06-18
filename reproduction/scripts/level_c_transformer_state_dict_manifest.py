#!/usr/bin/env python3
"""Debug-only state-dict hash manifest for the paper-state diffusion Transformer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from level_c_paper_state_transformer_arch_probe import (
    PaperStateDiffusionTransformer,
    ProbeConfig,
    parameter_count,
    seed_everything,
)


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
ARCH_JSON = ROOT / "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json"
PARAM_JSON = ROOT / "res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json"
OUT = ROOT / "res/level_c/transformer_state_dict_manifest"


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_model_and_rows(cfg: ProbeConfig, token_dim: int) -> tuple[PaperStateDiffusionTransformer, list[dict[str, Any]], str]:
    seed_everything(cfg.seed)
    model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).cpu()
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().contiguous().numpy()
        item = {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "numel": int(arr.size),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
        rows.append(item)
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(item["shape"]).encode("utf-8"))
        digest.update(item["dtype"].encode("utf-8"))
        digest.update(arr.tobytes())
    return model, rows, digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["name", "shape", "dtype", "numel", "sha256"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "shape": json.dumps(row["shape"])})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arch = json.loads(ARCH_JSON.read_text(encoding="utf-8"))
    param_audit = json.loads(PARAM_JSON.read_text(encoding="utf-8"))
    settings = arch["settings"]
    cfg = ProbeConfig(
        seed=int(settings["seed"]),
        batch_size=int(settings["batch_size"]),
        history=int(settings["history"]),
        horizon=int(settings["horizon"]),
        latent_dim=int(settings["latent_dim"]),
        denoising_steps=int(settings["denoising_steps"]),
        embedding_dim=int(settings["embedding_dim"]),
        attention_heads=int(settings["attention_heads"]),
        transformer_layers=int(settings["transformer_layers"]),
        dim_feedforward_multiplier=int(settings["dim_feedforward_multiplier"]),
        paper_batch_size=int(settings["paper_batch_size"]),
        paper_epochs=int(settings["paper_epochs"]),
        paper_learning_rate=float(settings["paper_learning_rate"]),
        paper_weight_decay=float(settings["paper_weight_decay"]),
        paper_warmup_gradient_steps=int(settings["paper_warmup_gradient_steps"]),
        paper_ema_power=float(settings["paper_ema_power"]),
        paper_ema_max=float(settings["paper_ema_max"]),
    )
    token_dim = int(settings["token_dim"])
    model, rows, overall_hash = build_model_and_rows(cfg, token_dim)
    _model_again, rows_again, overall_hash_again = build_model_and_rows(cfg, token_dim)
    different_seed_cfg = ProbeConfig(**{**cfg.__dict__, "seed": cfg.seed + 1})
    _model_diff, _rows_diff, different_seed_hash = build_model_and_rows(different_seed_cfg, token_dim)
    parameter_total = parameter_count(model)
    state_numel = int(sum(row["numel"] for row in rows))
    checks = {
        "architecture_json_exists": ARCH_JSON.exists(),
        "parameter_count_audit_exists": PARAM_JSON.exists(),
        "uses_paper_state_token_dim_131": token_dim == 131,
        "uses_paper_transformer_hyperparameters": cfg.embedding_dim == 512
        and cfg.attention_heads == 8
        and cfg.transformer_layers == 6
        and cfg.denoising_steps == 20,
        "state_dict_hash_deterministic_for_same_seed": overall_hash == overall_hash_again
        and [row["sha256"] for row in rows] == [row["sha256"] for row in rows_again],
        "state_dict_hash_changes_for_different_seed": overall_hash != different_seed_hash,
        "parameter_count_matches_arch_probe": parameter_total == int(arch["model"]["parameter_count"]),
        "parameter_count_matches_parameter_audit": parameter_total
        == int(param_audit["metrics"]["paper_state_99d_parameter_count"]),
        "state_numel_matches_parameter_count": state_numel == parameter_total,
        "does_not_write_weight_checkpoint": True,
        "does_not_claim_training": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "transformer_state_dict_manifest",
        "scope": "debug-only state-dict tensor hash manifest for paper-state diffusion Transformer initialization",
        "settings": {
            "seed": cfg.seed,
            "token_dim": token_dim,
            "state_dim": int(settings["state_dim"]),
            "latent_dim": cfg.latent_dim,
            "sequence_length": cfg.sequence_length,
            "embedding_dim": cfg.embedding_dim,
            "attention_heads": cfg.attention_heads,
            "transformer_layers": cfg.transformer_layers,
            "denoising_steps": cfg.denoising_steps,
        },
        "metrics": {
            "state_dict_tensor_count": len(rows),
            "state_dict_numel": state_numel,
            "parameter_count": parameter_total,
            "overall_state_dict_sha256": overall_hash,
            "different_seed_state_dict_sha256": different_seed_hash,
        },
        "checks": checks,
        "rows": rows,
        "not_a_replacement_for": [
            "trained diffusion Transformer checkpoint",
            "EMA checkpoint",
            "state-latent rollout dataset",
            "TensorRT engine",
            "Fig. 5/Fig. 6 rollout evaluation",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This records reproducible initialization hashes for the local paper-state Transformer architecture. "
                "It does not save trained weights, run diffusion training, create an EMA checkpoint, or support rollout metrics."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_transformer_state_dict_manifest.json"),
            "tsv": str(OUT / "level_c_transformer_state_dict_manifest.tsv"),
        },
    }
    (OUT / "level_c_transformer_state_dict_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tsv(OUT / "level_c_transformer_state_dict_manifest.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
