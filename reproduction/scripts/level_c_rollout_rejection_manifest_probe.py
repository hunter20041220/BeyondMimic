#!/usr/bin/env python3
"""Debug-only manifest probe for OU rollout windows, coverage, and rejection fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/rollout_rejection_manifest_probe"
FIXTURE_DIR = ROOT / "reproduction/data/level_c_fixtures"
MANIFEST_DIR = ROOT / "res/level_c/motion_state_fixture"
DEFAULT_FIXTURE_NAMES = [
    "walk1_subject1_frames_1_180_state_fixture",
    "run2_subject1_frames_1_180_state_fixture",
    "jumps1_subject1_frames_1_180_state_fixture",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ou_noise(steps: int, dims: int, theta: float, mu: float, dt: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eta = np.zeros((steps, dims), dtype=np.float64)
    for t in range(steps - 1):
        eta[t + 1] = eta[t] + theta * (mu - eta[t]) * dt + sigma * np.sqrt(dt) * rng.standard_normal(dims)
    return eta


def autocorr_lag1(x: np.ndarray) -> float:
    a = x[:-1].reshape(-1)
    b = x[1:].reshape(-1)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def source_csv_from_manifest(manifest: dict[str, Any]) -> Path:
    source = manifest.get("input_csv") or manifest.get("source", {}).get("input_csv")
    if not source:
        raise KeyError("manifest missing input_csv/source.input_csv")
    return Path(source)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id",
        "source_motion",
        "source_csv",
        "fixture_npz",
        "fixture_npz_sha256",
        "noise_seed_index",
        "ou_seed",
        "start_timestep",
        "recorded_end_timestep_exclusive",
        "stability_end_timestep_exclusive",
        "recorded_frames",
        "stability_frames",
        "recorded_seconds",
        "stability_seconds",
        "action_dim",
        "accept_reject",
        "failure_signal",
        "manifest_scope",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
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
    parser.add_argument("--fixture-names", default=",".join(DEFAULT_FIXTURE_NAMES))
    parser.add_argument("--recorded-seconds", type=float, default=2.5)
    parser.add_argument("--stability-seconds", type=float, default=5.0)
    parser.add_argument("--coverage-repeats", type=int, default=100)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--base-seed", type=int, default=20260903)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fixture_names = [item.strip() for item in args.fixture_names.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    coverage_arrays: dict[str, np.ndarray] = {}
    preview_payload: dict[str, np.ndarray] = {}

    for motion_id, name in enumerate(fixture_names):
        fixture_npz = FIXTURE_DIR / f"{name}.npz"
        manifest_json = MANIFEST_DIR / f"{name}.json"
        data = np.load(fixture_npz)
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
        fps = float(data["fps"][0])
        frame_count = int(data["joint_pos"].shape[0])
        action_dim = int(data["joint_pos"].shape[1])
        recorded_frames = int(round(args.recorded_seconds * fps))
        stability_frames = int(round(args.stability_seconds * fps))
        if stability_frames > frame_count:
            raise ValueError(f"{name}: stability window {stability_frames} exceeds frame count {frame_count}")
        valid_starts = np.arange(0, frame_count - stability_frames + 1, dtype=np.int64)
        source_csv = source_csv_from_manifest(manifest)
        fixture_hash = sha256_file(fixture_npz)

        recorded_coverage = np.zeros(frame_count, dtype=np.int64)
        stability_coverage = np.zeros(frame_count, dtype=np.int64)
        for start in valid_starts:
            recorded_coverage[start : start + recorded_frames] += args.coverage_repeats
            stability_coverage[start : start + stability_frames] += args.coverage_repeats

        coverage_arrays[f"{name}_recorded_coverage"] = recorded_coverage
        coverage_arrays[f"{name}_stability_coverage"] = stability_coverage

        first_seed = args.base_seed + motion_id * 100_000
        preview_noise = ou_noise(stability_frames, action_dim, args.theta, args.mu, args.dt, args.sigma, first_seed)
        preview_payload[f"{name}_preview_ou_noise"] = preview_noise

        central = slice(0, int(valid_starts[-1]) + recorded_frames)
        motion_rows.append(
            {
                "name": name,
                "frame_count": frame_count,
                "fps": fps,
                "valid_start_count": int(len(valid_starts)),
                "recorded_frames": recorded_frames,
                "stability_frames": stability_frames,
                "episode_rows": int(len(valid_starts) * args.coverage_repeats),
                "recorded_coverage_min_nonzero": int(recorded_coverage[recorded_coverage > 0].min()),
                "recorded_coverage_max": int(recorded_coverage.max()),
                "recorded_coverage_central_min": int(recorded_coverage[central].min()),
                "stability_coverage_min_nonzero": int(stability_coverage[stability_coverage > 0].min()),
                "stability_coverage_max": int(stability_coverage.max()),
                "preview_ou_std": float(np.std(preview_noise)),
                "preview_ou_lag1_autocorr": autocorr_lag1(preview_noise),
                "source_csv": str(source_csv),
                "fixture_npz": str(fixture_npz),
                "fixture_npz_sha256": fixture_hash,
                "manifest_status": manifest.get("status"),
                "manifest_experiment_type": manifest.get("experiment_type"),
            }
        )

        for start in valid_starts:
            for repeat_idx in range(args.coverage_repeats):
                seed = args.base_seed + motion_id * 100_000 + int(start) * args.coverage_repeats + repeat_idx
                rows.append(
                    {
                        "episode_id": f"{name}_start_{int(start):04d}_ou_{repeat_idx:03d}",
                        "source_motion": name,
                        "source_csv": str(source_csv),
                        "fixture_npz": str(fixture_npz),
                        "fixture_npz_sha256": fixture_hash,
                        "noise_seed_index": repeat_idx,
                        "ou_seed": seed,
                        "start_timestep": int(start),
                        "recorded_end_timestep_exclusive": int(start + recorded_frames),
                        "stability_end_timestep_exclusive": int(start + stability_frames),
                        "recorded_frames": recorded_frames,
                        "stability_frames": stability_frames,
                        "recorded_seconds": args.recorded_seconds,
                        "stability_seconds": args.stability_seconds,
                        "action_dim": action_dim,
                        "accept_reject": "accepted_debug_fixture_no_live_failure_signal",
                        "failure_signal": "missing_no_live_stability_rollout",
                        "manifest_scope": "debug_manifest_only_not_vae_rollout",
                    }
                )

    rows_path = OUT / "rollout_rejection_manifest_rows.tsv"
    summary_json = OUT / "level_c_rollout_rejection_manifest_probe.json"
    summary_tsv = OUT / "level_c_rollout_rejection_manifest_probe.tsv"
    npz_path = OUT / "level_c_rollout_rejection_manifest_probe.npz"
    write_rows(rows_path, rows)
    np.savez_compressed(npz_path, **coverage_arrays, **preview_payload)

    recorded_min_nonzero = min(row["recorded_coverage_min_nonzero"] for row in motion_rows)
    recorded_central_min = min(row["recorded_coverage_central_min"] for row in motion_rows)
    status = "ok"
    summary: dict[str, Any] = {
        "status": status,
        "experiment_type": "debug_only",
        "scope": "OU perturbation rollout-window, coverage, and accept/reject manifest probe",
        "paper_evidence": {
            "ou_and_rollout_protocol": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "dataset_protocol_audit": str(
                ROOT / "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json"
            ),
        },
        "not_a_replacement_for": [
            "trained VAE policy rollout",
            "true state-latent recording",
            "live 5s stability verification",
            "real episode rejection decisions",
            "paper trainable diffusion dataset",
        ],
        "settings": {
            "fixture_names": fixture_names,
            "recorded_seconds": args.recorded_seconds,
            "stability_seconds": args.stability_seconds,
            "coverage_repeats_per_start": args.coverage_repeats,
            "theta": args.theta,
            "mu": args.mu,
            "dt": args.dt,
            "sigma": args.sigma,
            "base_seed": args.base_seed,
        },
        "motion_rows": motion_rows,
        "metrics": {
            "motion_count": len(motion_rows),
            "episode_manifest_rows": len(rows),
            "valid_start_count_total": sum(row["valid_start_count"] for row in motion_rows),
            "recorded_coverage_min_nonzero": recorded_min_nonzero,
            "recorded_coverage_central_min": recorded_central_min,
            "recorded_coverage_target": args.coverage_repeats,
            "debug_accept_count": sum(row["accept_reject"].startswith("accepted_debug") for row in rows),
            "debug_reject_count": sum(row["accept_reject"].startswith("rejected") for row in rows),
            "preview_ou_lag1_autocorr_min": min(row["preview_ou_lag1_autocorr"] for row in motion_rows),
        },
        "checks": {
            "all_fixture_manifests_ok": all(row["manifest_status"] == "ok" for row in motion_rows),
            "all_fixture_manifests_debug_only": all(row["manifest_experiment_type"] == "debug_only" for row in motion_rows),
            "all_windows_match_2_5s_and_5s": all(
                row["recorded_frames"] == int(round(args.recorded_seconds * row["fps"]))
                and row["stability_frames"] == int(round(args.stability_seconds * row["fps"]))
                for row in motion_rows
            ),
            "coverage_repeats_per_valid_start_100": args.coverage_repeats == 100,
            "recorded_coverage_reaches_100_for_nonzero_frames": recorded_min_nonzero >= 100,
            "recorded_coverage_reaches_100_for_central_valid_region": recorded_central_min >= 100,
            "episode_rows_match_valid_starts_times_repeats": len(rows)
            == sum(row["valid_start_count"] for row in motion_rows) * args.coverage_repeats,
            "ou_preview_temporally_correlated": all(row["preview_ou_lag1_autocorr"] > 0.0 for row in motion_rows),
            "accept_reject_field_present_debug_only": all(
                row["accept_reject"] == "accepted_debug_fixture_no_live_failure_signal" for row in rows
            ),
            "failure_signal_marked_missing": all(
                row["failure_signal"] == "missing_no_live_stability_rollout" for row in rows
            ),
            "does_not_claim_true_vae_rollout": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "This debug manifest instantiates the paper's OU constants, 2.5s/5s windows, approximately 100x "
                "per-start synthetic coverage, and accept/reject fields over local motion fixtures. It does not run a "
                "trained VAE policy, record true latents, perform live stability verification, or produce real rejection "
                "decisions."
            ),
        },
        "outputs": {
            "rows_tsv": str(rows_path),
            "json": str(summary_json),
            "tsv": str(summary_tsv),
            "npz": str(npz_path),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_tsv(summary_tsv, summary)
    print(json.dumps({"status": status, "json": str(summary_json), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
