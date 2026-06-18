#!/usr/bin/env python3
"""Build a debug-only provenance/split manifest for the small Level C fixture dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/small_dataset_split_manifest"
DEFAULT_FIXTURE_NAMES = [
    "walk1_subject1_frames_1_180_state_fixture",
    "run2_subject1_frames_1_180_state_fixture",
    "jumps1_subject1_frames_1_180_state_fixture",
]
DEFAULT_MOTION_SPLITS = {
    "walk1_subject1_frames_1_180_state_fixture": "train",
    "run2_subject1_frames_1_180_state_fixture": "validation",
    "jumps1_subject1_frames_1_180_state_fixture": "test",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture_pair(name: str) -> tuple[Path, Path]:
    npz = ROOT / "reproduction/data/level_c_fixtures" / f"{name}.npz"
    manifest = ROOT / "res/level_c/motion_state_fixture" / f"{name}.json"
    if not npz.exists() or not manifest.exists():
        raise FileNotFoundError(f"missing fixture pair for {name}: {npz}, {manifest}")
    return npz, manifest


def source_csv_from_manifest(manifest: dict[str, Any]) -> Path:
    source = manifest.get("input_csv") or manifest.get("source", {}).get("input_csv")
    if not source:
        raise KeyError("manifest missing input_csv/source.input_csv")
    return Path(source)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "source_motion",
        "motion_id",
        "motion_split",
        "source_csv",
        "source_csv_sha256",
        "fixture_npz",
        "fixture_npz_sha256",
        "start_timestep",
        "end_timestep",
        "center_timestep",
        "state_frame",
        "latent",
        "augmentation",
        "accept_reject",
        "window_length",
        "history",
        "horizon",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_tsv(path: Path, summary: dict[str, Any]) -> None:
    flat: list[tuple[str, str]] = []

    def rec(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                rec(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, list):
            flat.append((prefix, json.dumps(value, sort_keys=True)))
        else:
            flat.append((prefix, str(value)))

    rec("", summary)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-names", type=str, default=",".join(DEFAULT_FIXTURE_NAMES))
    args = parser.parse_args()
    fixture_names = [item.strip() for item in args.fixture_names.split(",") if item.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    motion_summaries: list[dict[str, Any]] = []
    for motion_id, name in enumerate(fixture_names):
        npz_path, manifest_path = fixture_pair(name)
        data = np.load(npz_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        starts = data["window_start_indices"].astype(int)
        history = int(manifest["history"])
        horizon = int(manifest["horizon"])
        length = history + 1 + horizon
        source_csv = source_csv_from_manifest(manifest)
        split = DEFAULT_MOTION_SPLITS.get(name, "train")
        fixture_hash = sha256_file(npz_path)
        source_hash = manifest.get("input_csv_sha256") or manifest.get("source", {}).get("input_csv_sha256")
        if not source_hash:
            source_hash = sha256_file(source_csv if source_csv.is_absolute() else ROOT / source_csv)
        motion_summaries.append(
            {
                "name": name,
                "motion_id": motion_id,
                "split": split,
                "source_csv": str(source_csv),
                "fixture_npz": str(npz_path),
                "fixture_npz_sha256": fixture_hash,
                "window_count": int(len(starts)),
                "status": manifest.get("status"),
                "experiment_type": manifest.get("experiment_type"),
                "checks": manifest.get("checks", {}),
            }
        )
        for idx, center in enumerate(starts):
            rows.append(
                {
                    "sample_id": f"{name}_window_{idx:04d}",
                    "source_motion": name,
                    "motion_id": motion_id,
                    "motion_split": split,
                    "source_csv": str(source_csv),
                    "source_csv_sha256": source_hash,
                    "fixture_npz": str(npz_path),
                    "fixture_npz_sha256": fixture_hash,
                    "start_timestep": int(center - history),
                    "end_timestep": int(center + horizon),
                    "center_timestep": int(center),
                    "state_frame": "candidate_hybrid_character_yaw_centric",
                    "latent": "missing_no_teacher_or_vae_latent",
                    "augmentation": "none_original_fixture",
                    "accept_reject": "accepted_debug_fixture_no_live_stability_check",
                    "window_length": length,
                    "history": history,
                    "horizon": horizon,
                }
            )

    split_counts = {
        split: sum(row["motion_split"] == split for row in rows)
        for split in ["train", "validation", "test"]
    }
    motion_split_counts = {
        split: sum(item["split"] == split for item in motion_summaries)
        for split in ["train", "validation", "test"]
    }
    motion_to_splits: dict[str, set[str]] = {}
    for row in rows:
        motion_to_splits.setdefault(row["source_motion"], set()).add(row["motion_split"])

    rows_path = OUT / "small_dataset_window_provenance.tsv"
    summary_path = OUT / "small_dataset_split_manifest_summary.json"
    summary_tsv_path = OUT / "small_dataset_split_manifest_summary.tsv"
    write_rows(rows_path, rows)

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "small multi-motion fixture provenance and motion-level split leakage audit",
        "not_a_replacement_for": [
            "teacher rollout provenance",
            "VAE latent provenance",
            "episode rejection from live stability checks",
            "paper-exact train/validation/test split",
        ],
        "source_evidence": {
            "goal_diffusion_dataset": str(ROOT / "goal.md:1191-1228"),
            "goal_phase6_provenance": str(ROOT / "goal.md:1450-1468"),
            "small_dataset_overfit": str(
                ROOT / "res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json"
            ),
        },
        "split_policy": {
            "type": "motion_level_debug_split",
            "reason": "avoid overlapping-window leakage by assigning each debug motion to exactly one split",
            "motion_splits": DEFAULT_MOTION_SPLITS,
        },
        "motions": motion_summaries,
        "counts": {
            "motions_total": len(motion_summaries),
            "samples_total": len(rows),
            "split_counts": split_counts,
            "motion_split_counts": motion_split_counts,
        },
        "checks": {
            "all_samples_have_required_fields": bool(all(row["sample_id"] and row["source_motion"] for row in rows)),
            "uses_multiple_motions": len(motion_summaries) >= 3,
            "motion_level_splits_nonzero": bool(all(motion_split_counts[split] > 0 for split in ["train", "validation", "test"])),
            "sample_split_counts_nonzero": bool(all(split_counts[split] > 0 for split in ["train", "validation", "test"])),
            "no_motion_crosses_splits": bool(all(len(splits) == 1 for splits in motion_to_splits.values())),
            "latent_marked_missing": bool(all(row["latent"] == "missing_no_teacher_or_vae_latent" for row in rows)),
            "accept_reject_marked_debug": bool(
                all(row["accept_reject"] == "accepted_debug_fixture_no_live_stability_check" for row in rows)
            ),
            "all_fixture_manifests_ok": bool(all(item["status"] == "ok" for item in motion_summaries)),
            "debug_fixture_boundary_recorded": bool(all(item["experiment_type"] == "debug_only" for item in motion_summaries)),
        },
        "outputs": {
            "rows_tsv": str(rows_path),
            "summary_json": str(summary_path),
            "summary_tsv": str(summary_tsv_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_tsv(summary_tsv_path, summary)
    print(json.dumps({"status": "ok", "rows": str(rows_path), "summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
