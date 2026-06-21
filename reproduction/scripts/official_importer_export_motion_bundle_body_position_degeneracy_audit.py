#!/usr/bin/env python3
"""Audit body-position degeneracy in the current official-importer motion bundle.

The current local official-loop bundle was generated through the official
``csv_to_npz.py`` loop body on the captured official-importer-export G1 USDA.
It has the right outer schema, but the scaled PPO endpoint trace suggested that
ankle targets sit at root-like height. This audit checks whether the saved
``body_pos_w`` array actually contains per-link positions, and compares it with
a small non-Kit URDF-FK candidate built from the same public G1 CSV source.

The FK probe is a diagnostic repair direction only. It is not official
BeyondMimic preprocessing, DAgger data, closed-loop tracking, or paper-level
rollout evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPT_DIR = ROOT / "reproduction/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_level_c_motion_state_fixture import (  # noqa: E402
    OFFICIAL_CSV_JOINT_NAMES,
    compute_fk,
    load_and_interpolate_motion,
    matrix_to_quat_xyzw,
    parse_urdf,
)


OUT = ROOT / "res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy"
BUNDLE_NPZ = ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle.npz"
CLIPS_CSV = ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle_clips.csv"
FIXTURE_JSON = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
ENDPOINT_TRACE_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/"
    "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json"
)
OFFICIAL_CSV_TO_NPZ = ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py"
DEFAULT_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "assets/unitree_description/urdf/g1/main.urdf"
)
CSV_ROOT = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"
FK_MOTION_NAME = "walk1_subject1"
FK_CSV = CSV_ROOT / f"{FK_MOTION_NAME}.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def body_spread_stats(body_pos: np.ndarray) -> dict[str, float]:
    spread = np.ptp(body_pos, axis=1)
    return {
        "x_spread_mean_m": float(np.mean(spread[:, 0])),
        "x_spread_p95_m": float(np.percentile(spread[:, 0], 95)),
        "x_spread_max_m": float(np.max(spread[:, 0])),
        "y_spread_mean_m": float(np.mean(spread[:, 1])),
        "y_spread_p95_m": float(np.percentile(spread[:, 1], 95)),
        "y_spread_max_m": float(np.max(spread[:, 1])),
        "z_spread_mean_m": float(np.mean(spread[:, 2])),
        "z_spread_p95_m": float(np.percentile(spread[:, 2], 95)),
        "z_spread_max_m": float(np.max(spread[:, 2])),
        "xyz_spread_mean_m": float(np.mean(np.linalg.norm(spread, axis=1))),
        "xyz_spread_max_m": float(np.max(np.linalg.norm(spread, axis=1))),
        "max_abs_body_minus_body0_m": float(np.max(np.abs(body_pos - body_pos[:, :1, :]))),
        "max_abs_body_minus_body0_z_m": float(np.max(np.abs(body_pos[:, :, 2] - body_pos[:, :1, 2]))),
    }


def body_z_rows(body_pos: np.ndarray, body_names: list[str], prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(body_names):
        z = body_pos[:, idx, 2]
        rows.append(
            {
                "source": prefix,
                "body_index": idx,
                "body_name": name,
                "z_mean_m": float(np.mean(z)),
                "z_min_m": float(np.min(z)),
                "z_p05_m": float(np.percentile(z, 5)),
                "z_p50_m": float(np.percentile(z, 50)),
                "z_p95_m": float(np.percentile(z, 95)),
                "z_max_m": float(np.max(z)),
            }
        )
    return rows


def build_fk_candidate(body_names: list[str]) -> dict[str, Any]:
    motion = load_and_interpolate_motion(FK_CSV, input_fps=30, output_fps=50, start_frame=1, end_frame=180)
    children_by_parent = parse_urdf(DEFAULT_URDF)
    root_pos = motion["base_pos"]
    root_quat = motion["base_quat_xyzw"]
    joint_pos = motion["joint_pos"]
    frames = int(root_pos.shape[0])
    body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float64)
    body_quat = np.zeros((frames, len(body_names), 4), dtype=np.float64)
    missing: set[str] = set()
    for t in range(frames):
        joint_values = {name: float(joint_pos[t, idx]) for idx, name in enumerate(OFFICIAL_CSV_JOINT_NAMES)}
        transforms = compute_fk(children_by_parent, root_pos[t], root_quat[t], joint_values)
        for b, body_name in enumerate(body_names):
            tf = transforms.get(body_name)
            if tf is None:
                missing.add(body_name)
                continue
            body_pos[t, b] = tf[:3, 3]
            body_quat[t, b] = matrix_to_quat_xyzw(tf[:3, :3])
    return {
        "motion_name": FK_MOTION_NAME,
        "input_csv": str(FK_CSV),
        "input_csv_sha256": sha256_file(FK_CSV),
        "urdf": str(DEFAULT_URDF),
        "urdf_sha256": sha256_file(DEFAULT_URDF),
        "frames": frames,
        "body_pos_w": body_pos,
        "body_quat_xyzw_w": body_quat,
        "missing_bodies": sorted(missing),
    }


def plot_z_means(rows: list[dict[str, Any]], path: Path) -> None:
    sources = ["official_loop_bundle", "urdf_fk_candidate_walk1"]
    body_names = [row["body_name"] for row in rows if row["source"] == sources[0]]
    x = np.arange(len(body_names))
    width = 0.42
    plt.figure(figsize=(14, 5))
    for offset, source in [(-width / 2, sources[0]), (width / 2, sources[1])]:
        values = [row["z_mean_m"] for row in rows if row["source"] == source]
        plt.bar(x + offset, values, width=width, label=source)
    plt.xticks(x, body_names, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Mean body z position (m)")
    plt.title("Current motion bundle body positions are root-like; URDF FK gives separated link heights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_spread(bundle_body_pos: np.ndarray, fk_body_pos: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(np.ptp(bundle_body_pos[:, :, 2], axis=1), label="official-loop bundle z spread")
    plt.plot(np.ptp(fk_body_pos[:, :, 2], axis=1), label="URDF-FK candidate z spread")
    plt.xlabel("Frame")
    plt.ylabel("Per-frame max-min body z spread (m)")
    plt.title("Per-link z spread diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = load_json(FIXTURE_JSON)
    endpoint = load_json(ENDPOINT_TRACE_JSON)
    body_names = fixture["body_names_urdf_order"]
    npz = np.load(BUNDLE_NPZ)
    bundle_body_pos = np.asarray(npz["body_pos_w"], dtype=np.float64)
    bundle_body_quat = np.asarray(npz["body_quat_w"], dtype=np.float64)

    fk = build_fk_candidate(body_names)
    fk_body_pos = fk["body_pos_w"]
    fk_body_quat = fk["body_quat_xyzw_w"]

    bundle_spread = body_spread_stats(bundle_body_pos)
    fk_spread = body_spread_stats(fk_body_pos)
    target_names = [
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    target_rows = []
    for name in target_names:
        idx = body_names.index(name)
        target_rows.append(
            {
                "body_name": name,
                "body_index": idx,
                "official_loop_bundle_z_mean_m": float(np.mean(bundle_body_pos[:, idx, 2])),
                "urdf_fk_candidate_z_mean_m": float(np.mean(fk_body_pos[:, idx, 2])),
                "z_mean_delta_fk_minus_bundle_m": float(np.mean(fk_body_pos[:, idx, 2]) - np.mean(bundle_body_pos[:, idx, 2])),
            }
        )

    z_rows = body_z_rows(bundle_body_pos, body_names, "official_loop_bundle")
    z_rows += body_z_rows(fk_body_pos, body_names, "urdf_fk_candidate_walk1")
    spread_rows = [
        {"source": "official_loop_bundle", **bundle_spread},
        {"source": "urdf_fk_candidate_walk1", **fk_spread},
    ]

    endpoint_body_rows = endpoint["run"]["metrics"]["body_rows"]
    ankle_endpoint_rows = [
        row
        for row in endpoint_body_rows
        if row["body_name"] in {"left_ankle_roll_link", "right_ankle_roll_link"}
    ]
    source_text = OFFICIAL_CSV_TO_NPZ.read_text(encoding="utf-8")
    source_uses_render_without_step = "sim.render()  # We don't want physic (sim.step())" in source_text

    z_csv = OUT / "body_z_summary.csv"
    spread_csv = OUT / "body_position_spread_summary.csv"
    target_csv = OUT / "target_body_height_contrast.csv"
    z_png = OUT / "body_z_mean_contrast.png"
    spread_png = OUT / "body_z_spread_timeseries.png"
    md_path = OUT / "motion_bundle_body_position_degeneracy.md"
    readme_path = OUT / "README.md"
    json_path = OUT / "motion_bundle_body_position_degeneracy_audit.json"

    write_csv(
        z_csv,
        z_rows,
        ["source", "body_index", "body_name", "z_mean_m", "z_min_m", "z_p05_m", "z_p50_m", "z_p95_m", "z_max_m"],
    )
    write_csv(spread_csv, spread_rows, ["source", *bundle_spread.keys()])
    write_csv(
        target_csv,
        target_rows,
        [
            "body_name",
            "body_index",
            "official_loop_bundle_z_mean_m",
            "urdf_fk_candidate_z_mean_m",
            "z_mean_delta_fk_minus_bundle_m",
        ],
    )
    plot_z_means(z_rows, z_png)
    plot_spread(bundle_body_pos[: fk_body_pos.shape[0]], fk_body_pos, spread_png)

    checks = {
        "bundle_npz_exists": BUNDLE_NPZ.is_file(),
        "bundle_shape_full_public_motion": list(bundle_body_pos.shape) == [11960, 40, 3],
        "bundle_body_positions_degenerate_lt_1e_minus_5m": bundle_spread["max_abs_body_minus_body0_m"] < 1e-5,
        "bundle_body_z_spread_degenerate_lt_1e_minus_5m": bundle_spread["z_spread_max_m"] < 1e-5,
        "bundle_body_quaternions_not_degenerate": float(np.max(np.ptp(bundle_body_quat, axis=1))) > 0.1,
        "fk_candidate_all_requested_bodies_present": fk["missing_bodies"] == [],
        "fk_candidate_non_degenerate_z_spread_gt_0_5m": fk_spread["z_spread_mean_m"] > 0.5,
        "fk_candidate_ankles_near_ground_lt_0_2m": all(
            row["urdf_fk_candidate_z_mean_m"] < 0.2
            for row in target_rows
            if "ankle" in row["body_name"]
        ),
        "official_loop_ankles_root_like_gt_0_7m": all(
            row["official_loop_bundle_z_mean_m"] > 0.7
            for row in target_rows
            if "ankle" in row["body_name"]
        ),
        "endpoint_trace_records_ankle_exceed_rate_gt_0_99": all(
            row["exceed_rate_mean_over_steps"] > 0.99 for row in ankle_endpoint_rows
        ),
        "source_uses_render_without_physics_step": source_uses_render_without_step,
        "does_not_claim_official_motion_fix": True,
        "does_not_claim_paper_level_eval": True,
        "does_not_claim_real_robot": True,
    }
    status = "ok_motion_bundle_body_position_degeneracy_confirmed" if all(checks.values()) else "failed"

    summary: dict[str, Any] = {
        "status": status,
        "experiment_type": "official_importer_export_motion_bundle_body_position_degeneracy_audit",
        "scope": (
            "Audit whether the current full public-motion official-loop bundle stores meaningful per-link body "
            "positions, and compare it with a non-Kit URDF-FK candidate for one public G1 motion."
        ),
        "inputs": {
            "bundle_npz": str(BUNDLE_NPZ),
            "bundle_npz_sha256": sha256_file(BUNDLE_NPZ),
            "clips_csv": str(CLIPS_CSV),
            "clips_csv_sha256": sha256_file(CLIPS_CSV),
            "motion_npz_fixture": str(FIXTURE_JSON),
            "endpoint_trace_json": str(ENDPOINT_TRACE_JSON),
            "official_csv_to_npz_script": str(OFFICIAL_CSV_TO_NPZ),
            "fk_probe": {key: value for key, value in fk.items() if key not in {"body_pos_w", "body_quat_xyzw_w"}},
        },
        "bundle": {
            "body_pos_w_shape": list(bundle_body_pos.shape),
            "body_quat_w_shape": list(bundle_body_quat.shape),
            "spread": bundle_spread,
        },
        "fk_candidate": {
            "body_pos_w_shape": list(fk_body_pos.shape),
            "body_quat_xyzw_w_shape": list(fk_body_quat.shape),
            "spread": fk_spread,
        },
        "target_body_height_contrast": target_rows,
        "endpoint_trace_context": {
            "status": endpoint["status"],
            "aggregate_exceed_rate_mean": endpoint["run"]["metrics"]["aggregate"]["exceed_rate"]["mean"],
            "ankle_rows": ankle_endpoint_rows,
        },
        "checks": checks,
        "outputs": {
            "json": str(json_path),
            "body_z_summary_csv": str(z_csv),
            "body_position_spread_summary_csv": str(spread_csv),
            "target_body_height_contrast_csv": str(target_csv),
            "body_z_mean_contrast_png": str(z_png),
            "body_z_spread_timeseries_png": str(spread_png),
            "markdown": str(md_path),
            "readme": str(readme_path),
        },
        "interpretation": {
            "claim_level": "local diagnostic / repair-direction evidence",
            "goal_complete": False,
            "finding": (
                "The current full public-motion official-loop bundle has a valid outer schema but degenerate "
                "body_pos_w: every body is effectively at the root-like position. The URDF-FK candidate from the "
                "same public CSV and G1 URDF separates pelvis, ankles, torso, and wrists into plausible heights. "
                "The current bundle should therefore not be used as trusted target-body position evidence for "
                "teacher-quality PPO, DAgger, VAE, diffusion, or paper-level closed-loop evaluation until the body "
                "position generation path is repaired and replayed."
            ),
            "not_paper_level_reasons": [
                "diagnostic audit only",
                "FK probe is non-Kit and not official csv_to_npz output",
                "no PPO training or evaluation is started",
                "no official BeyondMimic checkpoint or DAgger rollout is produced",
                "no real robot hardware is used",
            ],
            "next_recommended_step": (
                "Regenerate the public motion bundle with corrected per-link body positions, either by finding the "
                "proper IsaacLab articulation data refresh path or by creating a clearly labeled FK-repaired motion "
                "candidate, then rerun task replay before any further PPO/teacher-rollout claim."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Motion Bundle Body-Position Degeneracy Audit",
        "",
        "## Finding",
        "",
        summary["interpretation"]["finding"],
        "",
        "## Key Numbers",
        "",
        f"- Bundle shape: `{summary['bundle']['body_pos_w_shape']}`.",
        f"- Bundle max body-minus-root position spread: `{bundle_spread['max_abs_body_minus_body0_m']:.3e}` m.",
        f"- Bundle max z spread: `{bundle_spread['z_spread_max_m']:.3e}` m.",
        f"- FK candidate mean z spread: `{fk_spread['z_spread_mean_m']:.3f}` m.",
        f"- Endpoint ankle exceed-rate mean values: `{[row['exceed_rate_mean_over_steps'] for row in ankle_endpoint_rows]}`.",
        "",
        "## Claim Boundary",
        "",
        "This is local diagnostic evidence only. It is not official BeyondMimic preprocessing, not a trained teacher, not DAgger, not Fig. 5/Fig. 6, and not real-robot evidence.",
        "",
        "## Outputs",
        "",
        f"- `{z_csv}`",
        f"- `{spread_csv}`",
        f"- `{target_csv}`",
        f"- `{z_png}`",
        f"- `{spread_png}`",
        f"- `{json_path}`",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    readme_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path)}, sort_keys=True))
    if status != "ok_motion_bundle_body_position_degeneracy_confirmed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
