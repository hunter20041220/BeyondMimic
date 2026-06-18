#!/usr/bin/env python3
"""Build a debug-only window provenance/split manifest for a Level C fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/fixture_split_manifest"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_contiguous_splits(start_indices: np.ndarray, train_frac: float, val_frac: float, guard_windows: int) -> list[str]:
    n = len(start_indices)
    train_end = max(1, int(round(n * train_frac)))
    val_end = max(train_end + 1, int(round(n * (train_frac + val_frac))))
    val_end = min(val_end, n - 1)
    splits = []
    for i in range(n):
        if train_end <= i < train_end + guard_windows:
            splits.append("excluded_guard_gap")
        elif val_end <= i < val_end + guard_windows:
            splits.append("excluded_guard_gap")
        elif i < train_end:
            splits.append("train")
        elif i < val_end:
            splits.append("validation")
        else:
            splits.append("test")
    return splits


def leakage_report(rows: list[dict[str, Any]], guard_gap: int) -> dict[str, Any]:
    by_split: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            continue
        by_split.setdefault(row["split"], []).append((row["start_timestep"], row["end_timestep"]))
    pair_reports = []
    for a in sorted(by_split):
        for b in sorted(by_split):
            if a >= b:
                continue
            min_gap = None
            overlaps = 0
            near = 0
            for a0, a1 in by_split[a]:
                for b0, b1 in by_split[b]:
                    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
                    if overlap > 0:
                        overlaps += 1
                        gap = -overlap
                    elif a1 < b0:
                        gap = b0 - a1
                    else:
                        gap = a0 - b1
                    min_gap = gap if min_gap is None else min(min_gap, gap)
                    if 0 <= gap < guard_gap:
                        near += 1
            pair_reports.append(
                {
                    "split_a": a,
                    "split_b": b,
                    "overlap_pairs": overlaps,
                    "near_guard_gap_pairs": near,
                    "min_gap_timesteps": min_gap,
                }
            )
    return {
        "guard_gap_timesteps": guard_gap,
        "pair_reports": pair_reports,
        "has_cross_split_overlap": any(item["overlap_pairs"] > 0 for item in pair_reports),
        "has_cross_split_near_gap": any(item["near_guard_gap_pairs"] > 0 for item in pair_reports),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "source_motion",
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
        "split",
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
    parser.add_argument("--fixture-npz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--guard-gap", type=int, default=1)
    parser.add_argument("--guard-windows", type=int, default=2)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fixture = np.load(args.fixture_npz)
    source_manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    starts = fixture["window_start_indices"].astype(int)
    history = int(source_manifest["history"])
    horizon = int(source_manifest["horizon"])
    length = history + 1 + horizon
    splits = assign_contiguous_splits(starts, args.train_frac, args.val_frac, args.guard_windows)
    fixture_hash = sha256_file(args.fixture_npz)
    source_csv = Path(source_manifest["input_csv"])
    source_motion = source_csv.stem

    rows: list[dict[str, Any]] = []
    for idx, (center, split) in enumerate(zip(starts, splits)):
        rows.append(
            {
                "sample_id": f"{source_motion}_window_{idx:04d}",
                "source_motion": source_motion,
                "source_csv": str(source_csv),
                "source_csv_sha256": source_manifest["input_csv_sha256"],
                "fixture_npz": str(args.fixture_npz),
                "fixture_npz_sha256": fixture_hash,
                "start_timestep": int(center - history),
                "end_timestep": int(center + horizon),
                "center_timestep": int(center),
                "state_frame": "candidate_hybrid_character_yaw_centric",
                "latent": "missing_no_teacher_or_vae_latent",
                "augmentation": "none_original_fixture",
                "accept_reject": "accepted_debug_fixture_no_live_stability_check",
                "split": split,
                "window_length": length,
                "history": history,
                "horizon": horizon,
            }
        )

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ["train", "validation", "test", "excluded_guard_gap"]
    }
    leakage = leakage_report(rows, args.guard_gap)
    rows_path = OUT / "fixture_window_provenance.tsv"
    summary_path = OUT / "fixture_split_manifest_summary.json"
    summary_tsv_path = OUT / "fixture_split_manifest_summary.tsv"
    write_rows(rows_path, rows)

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "window-level provenance and leakage checks for the motion-derived Level C fixture",
        "not_a_replacement_for": [
            "teacher rollout provenance",
            "VAE latent provenance",
            "episode rejection from live stability checks",
            "paper-exact train/validation/test split",
        ],
        "source_evidence": {
            "goal_diffusion_dataset": str(ROOT / "goal.md:1191-1228"),
            "goal_phase6_provenance": str(ROOT / "goal.md:1450-1468"),
            "fixture_manifest": str(args.manifest_json),
        },
        "inputs": {
            "fixture_npz": str(args.fixture_npz),
            "fixture_npz_sha256": fixture_hash,
            "source_csv": str(source_csv),
            "source_csv_sha256": source_manifest["input_csv_sha256"],
        },
        "split_policy": {
            "type": "contiguous_time_ordered_debug_split",
            "train_frac": args.train_frac,
            "validation_frac": args.val_frac,
            "test_frac": 1.0 - args.train_frac - args.val_frac,
            "guard_gap_timesteps": args.guard_gap,
            "guard_windows": args.guard_windows,
            "reason": "avoid random overlapping-window leakage in the single-motion fixture",
        },
        "counts": {
            "samples_total": len(rows),
            "split_counts": split_counts,
            "window_length": length,
            "history": history,
            "horizon": horizon,
        },
        "leakage": leakage,
        "checks": {
            "all_samples_have_required_fields": bool(all(row.get("sample_id") and row.get("split") for row in rows)),
            "split_counts_nonzero": bool(all(split_counts[split] > 0 for split in ["train", "validation", "test"])),
            "guard_gap_samples_present": bool(split_counts["excluded_guard_gap"] > 0),
            "no_cross_split_overlap": not leakage["has_cross_split_overlap"],
            "no_cross_split_near_gap": not leakage["has_cross_split_near_gap"],
            "latent_marked_missing": bool(all(row["latent"] == "missing_no_teacher_or_vae_latent" for row in rows)),
            "accept_reject_marked_debug": bool(
                all(row["accept_reject"] == "accepted_debug_fixture_no_live_stability_check" for row in rows)
            ),
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
