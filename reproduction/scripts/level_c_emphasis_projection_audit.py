#!/usr/bin/env python3
"""Formula-level audit for the paper emphasis projection and pseudoinverse."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/emphasis_projection_audit"
PAPER_STATE_JSON = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
PAPER_STATE_DIR = ROOT / "reproduction/data/level_c_paper_state_windows"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"


@dataclass(frozen=True)
class EmphasisProjectionConfig:
    seed: int = 20260914
    root_feature_dim: int = 15
    body_feature_dim: int = 84
    emphasis_coefficient: float = 6.0
    gaussian_rows: int = 64

    @property
    def state_dim(self) -> int:
        return self.root_feature_dim + self.body_feature_dim


def load_all_paper_state_tokens() -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    state_summary = json.loads(PAPER_STATE_JSON.read_text(encoding="utf-8"))
    arrays = []
    rows = []
    for item in state_summary["rows"]:
        npz_path = PAPER_STATE_DIR / f"{item['name']}_paper_state_windows.npz"
        with np.load(npz_path) as data:
            windows = data["paper_state_windows"].astype(np.float64)
        arrays.append(windows.reshape(-1, windows.shape[-1]))
        rows.append(
            {
                "fixture_name": item["name"],
                "window_count": int(windows.shape[0]),
                "sequence_length": int(windows.shape[1]),
                "state_dim": int(windows.shape[2]),
                "sample_count": int(windows.shape[0] * windows.shape[1]),
            }
        )
    return np.concatenate(arrays, axis=0), state_summary, rows


def build_projection(cfg: EmphasisProjectionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    a = rng.normal(loc=0.0, scale=1.0, size=(cfg.gaussian_rows, cfg.root_feature_dim))
    b = cfg.emphasis_coefficient * np.eye(cfg.root_feature_dim, dtype=np.float64)
    left = np.zeros((cfg.gaussian_rows, cfg.state_dim), dtype=np.float64)
    left[:, : cfg.root_feature_dim] = a @ b
    p = np.concatenate([left, np.eye(cfg.state_dim, dtype=np.float64)], axis=0)
    return p, a, b


def text_has_patterns(path: Path, patterns: list[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["fixture_name", "window_count", "sequence_length", "state_dim", "sample_count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--gaussian-rows", type=int, default=64)
    args = parser.parse_args()

    cfg = EmphasisProjectionConfig(seed=args.seed, gaussian_rows=args.gaussian_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tokens, state_summary, rows = load_all_paper_state_tokens()
    if tokens.shape[1] != cfg.state_dim:
        raise ValueError(f"expected paper state dim {cfg.state_dim}, got {tokens.shape[1]}")

    projection, gaussian_a, emphasis_b = build_projection(cfg)
    projection_pinv = np.linalg.pinv(projection)
    projected = tokens @ projection.T
    reconstructed = projected @ projection_pinv.T
    roundtrip_error = np.abs(reconstructed - tokens)
    root_norm = np.linalg.norm(tokens[:, : cfg.root_feature_dim], axis=1)
    body_norm = np.linalg.norm(tokens[:, cfg.root_feature_dim :], axis=1)
    projected_extra_norm = np.linalg.norm(projected[:, : cfg.gaussian_rows], axis=1)
    projected_identity_norm = np.linalg.norm(projected[:, cfg.gaussian_rows :], axis=1)

    json_path = OUT / "level_c_emphasis_projection_audit.json"
    tsv_path = OUT / "level_c_emphasis_projection_audit.tsv"
    npz_path = OUT / "level_c_emphasis_projection_audit.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        projection=projection,
        projection_pinv=projection_pinv,
        gaussian_a=gaussian_a,
        emphasis_b=emphasis_b,
        projected_sample=projected[:32],
        reconstructed_sample=reconstructed[:32],
        source_token_sample=tokens[:32],
    )

    root_patterns = [
        "emphasis projection",
        "\\mathbf{P} = [\\mathbf{A}\\mathbf{B} \\ \\ \\mathbf{I}]^\\top",
        "\\mathbf{A}_{ij} \\sim \\mathcal{N}(0, 1)",
        "c = 6",
        "pseudoinverse",
    ]
    projection_rank = int(np.linalg.matrix_rank(projection))
    projection_pinv_identity = projection_pinv @ projection
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "formula_audit",
        "scope": "paper emphasis projection P=[AB I]^T and pseudoinverse round-trip over local 99-D paper-state windows",
        "paper_evidence": {
            "paper_projection_formula": str(ROOT_TEX) + ":524-532",
            "goal_projection_requirement": str(ROOT / "goal.md:1241-1249"),
            "paper_state_windows": str(PAPER_STATE_JSON),
        },
        "not_a_replacement_for": [
            "paper trainable state-latent rollout dataset",
            "trained diffusion model using projected states",
            "official random seed or projection matrix",
            "paper evaluation protocol",
        ],
        "settings": asdict(cfg),
        "input_rows": rows,
        "metrics": {
            "input_sample_count": int(tokens.shape[0]),
            "state_dim": int(tokens.shape[1]),
            "projection_shape": [int(v) for v in projection.shape],
            "projection_rank": projection_rank,
            "max_roundtrip_abs_error": float(roundtrip_error.max()),
            "mean_roundtrip_abs_error": float(roundtrip_error.mean()),
            "pinv_projection_identity_max_error": float(
                np.abs(projection_pinv_identity - np.eye(cfg.state_dim)).max()
            ),
            "gaussian_a_mean": float(gaussian_a.mean()),
            "gaussian_a_std": float(gaussian_a.std(ddof=1)),
            "mean_root_token_norm": float(root_norm.mean()),
            "mean_body_token_norm": float(body_norm.mean()),
            "mean_projected_extra_norm": float(projected_extra_norm.mean()),
            "mean_projected_identity_norm": float(projected_identity_norm.mean()),
        },
        "checks": {
            "paper_source_projection_patterns_found": text_has_patterns(ROOT_TEX, root_patterns),
            "paper_state_windows_status_ok": state_summary["status"] == "ok",
            "paper_state_windows_all_checks_pass": state_summary["checks"]["all_checks_pass"],
            "uses_paper_state_dim_99": tokens.shape[1] == 99,
            "root_feature_dim_15": cfg.root_feature_dim == 15,
            "body_feature_dim_84": cfg.body_feature_dim == 84,
            "emphasis_coefficient_c_6": cfg.emphasis_coefficient == 6.0,
            "projection_is_tall": projection.shape[0] > projection.shape[1],
            "projection_full_column_rank": projection_rank == cfg.state_dim,
            "pseudoinverse_roundtrip_below_1e_minus_10": bool(roundtrip_error.max() < 1e-10),
            "pinv_projection_identity_below_1e_minus_10": bool(
                np.abs(projection_pinv_identity - np.eye(cfg.state_dim)).max() < 1e-10
            ),
            "gaussian_a_nontrivial": bool(abs(float(gaussian_a.std(ddof=1)) - 1.0) < 0.1),
            "projected_dim_matches_formula": projection.shape[0] == cfg.gaussian_rows + cfg.state_dim,
            "debug_fixture_boundary_recorded": state_summary["experiment_type"] == "debug_only",
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This verifies the paper's emphasis projection and pseudoinverse mechanics on local paper-formula "
                "99-D state windows. It does not prove that the unpublished paper training dataset, trained "
                "diffusion checkpoint, or Fig. 5/Fig. 6 evaluations used this exact sampled projection."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
