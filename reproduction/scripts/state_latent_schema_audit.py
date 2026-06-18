#!/usr/bin/env python3
"""Audit state-latent trajectory schema using package dataset helpers."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
OUT = ROOT / "res/level_c/state_latent_schema_audit"
PAPER_STATE_JSON = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
SPLIT_JSON = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"
SPLIT_TSV = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_window_provenance.tsv"
PAPER_STATE_DIR = ROOT / "reproduction/data/level_c_paper_state_windows"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.trajectory import build_state_latent_window, split_counts, stack_state_latent_tokens


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def paper_state_npz(name: str) -> Path:
    return PAPER_STATE_DIR / f"{name}_paper_state_windows.npz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper_state = json.loads(PAPER_STATE_JSON.read_text(encoding="utf-8"))
    split_summary = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    provenance_rows = read_tsv(SPLIT_TSV)
    by_motion: dict[str, list[dict[str, str]]] = {}
    for item in provenance_rows:
        by_motion.setdefault(item["source_motion"], []).append(item)

    windows = []
    rows: list[dict[str, Any]] = []
    token_shapes: Counter[str] = Counter()
    latent_source = "synthetic_zero_placeholder_for_schema_only"
    for motion in split_summary["motions"]:
        name = motion["name"]
        npz_path = paper_state_npz(name)
        data = np.load(npz_path, allow_pickle=True)
        states = data["paper_state_windows"]
        starts = data["window_start_indices"]
        motion_rows = sorted(by_motion[name], key=lambda r: int(r["start_timestep"]))
        if len(motion_rows) != states.shape[0]:
            raise ValueError(f"motion {name} provenance rows {len(motion_rows)} != state windows {states.shape[0]}")
        for idx, prov in enumerate(motion_rows):
            state_window = states[idx]
            latents = np.zeros((state_window.shape[0], 32), dtype=np.float64)
            window = build_state_latent_window(
                sample_id=prov["sample_id"],
                source_motion=name,
                start_timestep=int(prov["start_timestep"]),
                split=prov["motion_split"],  # type: ignore[arg-type]
                accepted=prov["accept_reject"].startswith("accepted_debug"),
                states=state_window,
                latents=latents,
            )
            tokens = stack_state_latent_tokens(window.states, window.latents)
            token_shapes[str(list(tokens.shape))] += 1
            windows.append(window)
            rows.append(
                {
                    "sample_id": window.sample_id,
                    "source_motion": window.source_motion,
                    "start_timestep": window.start_timestep,
                    "split": window.split,
                    "accepted": window.accepted,
                    "state_shape": list(window.states.shape),
                    "latent_shape": list(window.latents.shape),
                    "token_shape": list(tokens.shape),
                    "paper_state_start_index": int(starts[idx]),
                    "provenance_start_timestep": int(prov["start_timestep"]),
                    "latent_source": latent_source,
                    "accept_reject": prov["accept_reject"],
                    "finite": bool(np.all(np.isfinite(tokens))),
                }
            )

    counts = split_counts(windows)
    motion_counts = Counter(row["source_motion"] for row in rows)
    accepted_count = sum(1 for row in rows if row["accepted"])
    debug_latent_count = sum(1 for row in rows if row["latent_source"] == latent_source)
    missing_evidence_rows: list[dict[str, Any]] = []
    summary = {
        "status": "ok",
        "experiment_type": "state_latent_schema_audit",
        "scope": "package-level state-latent trajectory schema over paper-state debug windows",
        "row_count": len(rows),
        "split_counts": counts,
        "motion_counts": dict(sorted(motion_counts.items())),
        "token_shape_counts": dict(sorted(token_shapes.items())),
        "accepted_count": accepted_count,
        "latent_source_counts": {latent_source: debug_latent_count},
        "settings": {
            "paper_state_dim": paper_state["settings"]["paper_state_dim"],
            "latent_dim_placeholder": 32,
            "sequence_length": paper_state["settings"]["sequence_length"],
            "history": paper_state["settings"]["history"],
            "horizon": paper_state["settings"]["horizon"],
        },
        "checks": {
            "all_evidence_paths_exist": all(path.exists() for path in [PAPER_STATE_JSON, SPLIT_JSON, SPLIT_TSV]),
            "uses_package_state_latent_api": True,
            "row_count_matches_paper_state_windows": len(rows) == paper_state["counts"]["window_count"],
            "all_motions_mapped": len(motion_counts) == paper_state["counts"]["motion_count"],
            "split_counts_match_manifest": counts == split_summary["counts"]["split_counts"],
            "all_token_shapes_21x131": token_shapes == {"[21, 131]": len(rows)},
            "all_rows_finite": all(row["finite"] for row in rows),
            "all_rows_accepted_debug_only": accepted_count == len(rows),
            "all_latents_marked_placeholder": debug_latent_count == len(rows),
            "does_not_claim_true_vae_latents": True,
            "does_not_claim_goal_complete": True,
        },
        "missing_evidence_rows": missing_evidence_rows,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "The state-latent schema can validate all 84 debug paper-state windows and split/provenance rows via "
                "the package API, but latents are explicit zero placeholders because no trained VAE rollout latents or "
                "true DAgger collection exist."
            ),
        },
        "outputs": {
            "json": str(OUT / "state_latent_schema_audit.json"),
            "tsv": str(OUT / "state_latent_schema_audit.tsv"),
        },
    }
    if not all(summary["checks"].values()):
        summary["status"] = "failed"
    (OUT / "state_latent_schema_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "state_latent_schema_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "source_motion",
            "start_timestep",
            "split",
            "accepted",
            "state_shape",
            "latent_shape",
            "token_shape",
            "latent_source",
            "accept_reject",
            "finite",
        ]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fieldnames})
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
