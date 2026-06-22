#!/usr/bin/env python3
"""Build a Stage-1 multi-source G1 motion bundle for whole_body_tracking.

This script collects only sources that can be represented as the official
whole_body_tracking generalized-coordinate CSV contract:

    root position xyz + root quaternion xyzw + 29 G1 joint coordinates

It does not modify the read-only source directories.  It writes converted CSV
work copies plus one FK-repaired, IsaacLab-runtime-body-order ``motion.npz``
under ``res/tracking``.  Sources that are present but not directly compatible
are recorded in the manifest instead of being silently padded or guessed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPT_DIR = ROOT / "reproduction/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_level_c_motion_state_fixture import (  # noqa: E402
    DEFAULT_URDF,
    OFFICIAL_CSV_JOINT_NAMES,
    angular_velocity_from_quats,
    compute_fk,
    load_and_interpolate_motion,
    matrix_to_quat_xyzw,
    parse_urdf,
)


OUT = ROOT / "res/tracking/stage1_multisource_motion_bundle"
CSV_OUT = OUT / "csv"
MOTION_ROOT = OUT / "motions"
OUT_NPZ = OUT / "stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz"
OUT_JSON = OUT / "tracking_stage1_multisource_motion_bundle.json"
OUT_CSV = OUT / "tracking_stage1_multisource_motion_bundle_rows.csv"
OUT_TSV = OUT / "tracking_stage1_multisource_motion_bundle_rows.tsv"
OUT_SKIPPED_TSV = OUT / "tracking_stage1_multisource_skipped_sources.tsv"

BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
BODY_ORDER_PROBE = (
    ROOT / "res/tracking/fk_repaired_body_order_runtime_probe/fk_repaired_body_order_runtime_probe.json"
)

LAFAN_ROOT = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"
ZENODO_TKD_CSV = ROOT / "Dataset_beyondmimic/ablation/tkd_skill.csv"
REPRO_ZENODO_TKD_CSV = ROOT / "reproduction/data/Dataset_beyondmimic/ablation/tkd_skill.csv"

HUB_PICKLES = [
    ROOT / "download/_supplemental/hub_data/drive_folder/singleleg.pkl",
    ROOT / "download/_supplemental/hub_data/drive_folder/swallow_balance.pkl",
    ROOT / "download/_supplemental/hub_data/drive_folder/bruce_lee.pkl",
    ROOT / "download/_supplemental/hub_data/drive_folder/nezha.pkl",
    ROOT / "download/_supplemental/hub_data/drive_folder/squat.pkl",
]

SKIPPED_PICKLES = [
    {
        "path": ROOT / "download/reference_code/PBHC/example/motion_data/Side_kick.pkl",
        "source_family": "KungfuBot/PBHC sidekick",
        "reason": "present as 23-DoF pkl; needs audited 23-to-29 G1 joint mapping before whole_body_tracking training",
    },
    {
        "path": ROOT
        / "download/reference_code/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/"
        "TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl",
        "source_family": "ASAP Cristiano Ronaldo celebration",
        "reason": "present as 23-DoF pkl; needs audited 23-to-29 G1 joint mapping before whole_body_tracking training",
    },
]

INPUT_FPS_DEFAULT = 30
OUTPUT_FPS = 50
ARRAY_KEYS = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


@dataclass(frozen=True)
class Candidate:
    name: str
    source_family: str
    source_path: Path
    csv_path: Path
    input_fps: int
    source_kind: str
    notes: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_name(raw: str) -> str:
    keep = []
    for ch in raw.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {"-", "_", ".", " "}:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:96] or "motion"


def numeric_csv_shape(path: Path) -> tuple[int, int] | None:
    try:
        arr = np.loadtxt(path, delimiter=",", max_rows=5)
    except Exception:
        return None
    if arr.ndim == 1:
        cols = int(arr.shape[0])
    else:
        cols = int(arr.shape[1])
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        rows = sum(1 for _ in f)
    return rows, cols


def copy_csv_candidate(src: Path, name: str, source_family: str, notes: str) -> Candidate:
    dst = CSV_OUT / f"{safe_name(name)}.csv"
    arr = np.loadtxt(src, delimiter=",", dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"{src} is not a 36-column whole_body_tracking CSV candidate, got {arr.shape}")
    np.savetxt(dst, arr, delimiter=",", fmt="%.9f")
    return Candidate(
        name=safe_name(name),
        source_family=source_family,
        source_path=src,
        csv_path=dst,
        input_fps=INPUT_FPS_DEFAULT,
        source_kind="36_column_csv",
        notes=notes,
    )


def convert_hub_pkl_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for pkl_path in HUB_PICKLES:
        if not pkl_path.is_file():
            continue
        payload = joblib.load(pkl_path)
        if not isinstance(payload, dict):
            continue
        for motion_name, motion in payload.items():
            if not isinstance(motion, dict):
                continue
            root = np.asarray(motion.get("root_trans_offset"), dtype=np.float64)
            quat = np.asarray(motion.get("root_rot"), dtype=np.float64)
            dof = np.asarray(motion.get("dof"), dtype=np.float64)
            fps = int(motion.get("fps", INPUT_FPS_DEFAULT))
            if root.ndim != 2 or quat.ndim != 2 or dof.ndim != 2:
                continue
            if root.shape[1] != 3 or quat.shape[1] != 4 or dof.shape[1] != 29:
                continue
            rows = min(root.shape[0], quat.shape[0], dof.shape[0])
            arr = np.concatenate([root[:rows], quat[:rows], dof[:rows]], axis=1)
            if arr.shape[1] != 36:
                continue
            norm = np.linalg.norm(arr[:, 3:7], axis=1, keepdims=True)
            arr[:, 3:7] = arr[:, 3:7] / np.maximum(norm, 1e-12)
            name = safe_name(f"hub_{pkl_path.stem}_{motion_name}")
            csv_path = CSV_OUT / f"{name}.csv"
            np.savetxt(csv_path, arr, delimiter=",", fmt="%.9f")
            candidates.append(
                Candidate(
                    name=name,
                    source_family="HuB supplemental 29-DoF pkl",
                    source_path=pkl_path,
                    csv_path=csv_path,
                    input_fps=fps,
                    source_kind="converted_29dof_pkl",
                    notes=f"converted from pkl motion key {motion_name}; root_rot assumed xyzw following local ASAP/PBHC viewers",
                )
            )
    return candidates


def collect_candidates() -> tuple[list[Candidate], list[dict[str, Any]]]:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    skipped: list[dict[str, Any]] = []

    for csv_path in sorted(LAFAN_ROOT.glob("*.csv")):
        shape = numeric_csv_shape(csv_path)
        if shape and shape[1] == 36:
            candidates.append(
                copy_csv_candidate(
                    csv_path,
                    f"lafan1_{csv_path.stem}",
                    "Unitree-retargeted LAFAN1",
                    "official HuggingFace LAFAN1 retargeted G1 36-column CSV",
                )
            )
        else:
            skipped.append(
                {
                    "source_path": str(csv_path),
                    "source_family": "Unitree-retargeted LAFAN1",
                    "reason": f"unexpected CSV shape {shape}",
                    "exists": csv_path.is_file(),
                }
            )

    tkd_source = ZENODO_TKD_CSV if ZENODO_TKD_CSV.is_file() else REPRO_ZENODO_TKD_CSV
    if tkd_source.is_file() and numeric_csv_shape(tkd_source) == (332, 36):
        candidates.append(
            copy_csv_candidate(
                tkd_source,
                "zenodo_tkd_skill",
                "BeyondMimic Zenodo ablation reference CSV",
                "single 36-column tkd_skill.csv; unlike rosbag/mcap files, this is a generalized-coordinate reference candidate",
            )
        )
    else:
        skipped.append(
            {
                "source_path": str(tkd_source),
                "source_family": "BeyondMimic Zenodo",
                "reason": f"tkd_skill.csv missing or unexpected shape {numeric_csv_shape(tkd_source) if tkd_source.is_file() else None}",
                "exists": tkd_source.is_file(),
            }
        )

    candidates.extend(convert_hub_pkl_candidates())

    for item in SKIPPED_PICKLES:
        path = item["path"]
        extra: dict[str, Any] = {
            "source_path": str(path),
            "source_family": item["source_family"],
            "reason": item["reason"],
            "exists": path.is_file(),
        }
        if path.is_file():
            try:
                obj = joblib.load(path)
                dof_shapes = []
                if isinstance(obj, dict):
                    for motion in obj.values():
                        if isinstance(motion, dict) and "dof" in motion:
                            dof_shapes.append(list(np.asarray(motion["dof"]).shape))
                extra["observed_dof_shapes"] = json.dumps(dof_shapes, sort_keys=True)
            except Exception as exc:  # pragma: no cover - audit only
                extra["load_error"] = repr(exc)
        skipped.append(extra)

    dataset_root = ROOT / "Dataset_beyondmimic"
    if dataset_root.is_dir():
        skipped.append(
            {
                "source_path": str(dataset_root),
                "source_family": "BeyondMimic Zenodo dataset",
                "reason": "dataset root contains rosbag/mcap, GRF, ablation and plotting artifacts; these are retained for paper-result analysis, not bulk Stage-1 PPO reference training",
                "exists": True,
            }
        )
    return candidates, skipped


def build_one_motion(
    candidate: Candidate,
    body_names: list[str],
    children_by_parent: dict[str, Any],
) -> dict[str, Any]:
    arr = np.loadtxt(candidate.csv_path, delimiter=",", dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"{candidate.csv_path}: expected 36 columns, got {arr.shape}")
    motion = load_and_interpolate_motion(
        csv_path=candidate.csv_path,
        input_fps=candidate.input_fps,
        output_fps=OUTPUT_FPS,
        start_frame=1,
        end_frame=arr.shape[0],
    )
    root_pos = motion["base_pos"]
    root_quat_xyzw = motion["base_quat_xyzw"]
    joint_pos = motion["joint_pos"].astype(np.float32)
    dt = 1.0 / OUTPUT_FPS
    frames = int(joint_pos.shape[0])
    body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    body_quat_xyzw = np.zeros((frames, len(body_names), 4), dtype=np.float32)
    missing: set[str] = set()

    for t in range(frames):
        joint_values = {name: float(joint_pos[t, idx]) for idx, name in enumerate(OFFICIAL_CSV_JOINT_NAMES)}
        transforms = compute_fk(children_by_parent, root_pos[t], root_quat_xyzw[t], joint_values)
        for b, body_name in enumerate(body_names):
            tf = transforms.get(body_name)
            if tf is None:
                missing.add(body_name)
                continue
            body_pos[t, b] = tf[:3, 3].astype(np.float32)
            body_quat_xyzw[t, b] = matrix_to_quat_xyzw(tf[:3, :3]).astype(np.float32)
    if missing:
        raise RuntimeError(f"{candidate.name}: FK missed bodies {sorted(missing)}")

    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang_vel = np.stack(
        [
            angular_velocity_from_quats(body_quat_xyzw[:, body_idx].astype(np.float64), dt)
            for body_idx in range(len(body_names))
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat_xyzw[:, :, [3, 0, 1, 2]].astype(np.float32),
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "frame_count": frames,
    }


def split_motion_npz(arrays: dict[str, np.ndarray], candidate: Candidate, cursor: int) -> Path:
    motion_dir = MOTION_ROOT / candidate.name
    motion_dir.mkdir(parents=True, exist_ok=True)
    out = motion_dir / "motion.npz"
    np.savez_compressed(out, fps=np.asarray([OUTPUT_FPS], dtype=np.int64), **arrays)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MOTION_ROOT.mkdir(parents=True, exist_ok=True)
    candidates, skipped = collect_candidates()
    body_probe = load_json(BODY_ORDER_PROBE)
    body_contract = load_json(BODY_CONTRACT)
    robot_body_names = list(body_probe.get("robot_body_names", []))
    urdf_body_names = list(body_contract.get("body_names_urdf_order", []))
    if not robot_body_names or not urdf_body_names:
        raise RuntimeError("missing body order audit inputs; run body-order audits first")
    if sorted(robot_body_names) != sorted(urdf_body_names):
        raise RuntimeError("robot and URDF body-name sets differ")

    children_by_parent = parse_urdf(DEFAULT_URDF)
    bundle_parts: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    rows: list[dict[str, Any]] = []
    cursor = 0

    for idx, candidate in enumerate(candidates):
        arrays = build_one_motion(candidate, robot_body_names, children_by_parent)
        for key in ARRAY_KEYS:
            bundle_parts[key].append(arrays[key])
        per_motion_npz = split_motion_npz({key: arrays[key] for key in ARRAY_KEYS}, candidate, cursor)
        frame_count = int(arrays["frame_count"])
        rows.append(
            {
                "index": idx,
                "motion": candidate.name,
                "source_family": candidate.source_family,
                "source_kind": candidate.source_kind,
                "source_path": str(candidate.source_path),
                "source_exists": candidate.source_path.is_file(),
                "source_sha256": sha256_file(candidate.source_path) if candidate.source_path.is_file() else "",
                "csv_path": str(candidate.csv_path),
                "csv_sha256": sha256_file(candidate.csv_path),
                "input_fps": candidate.input_fps,
                "output_fps": OUTPUT_FPS,
                "start_frame": cursor,
                "end_frame_exclusive": cursor + frame_count,
                "frame_count": frame_count,
                "duration_seconds": frame_count / OUTPUT_FPS,
                "per_motion_npz": str(per_motion_npz),
                "per_motion_npz_size": per_motion_npz.stat().st_size,
                "per_motion_npz_sha256": sha256_file(per_motion_npz),
                "joint_min": float(np.min(arrays["joint_pos"])),
                "joint_max": float(np.max(arrays["joint_pos"])),
                "root_z_min": float(np.min(arrays["body_pos_w"][:, robot_body_names.index("pelvis"), 2])),
                "root_z_max": float(np.max(arrays["body_pos_w"][:, robot_body_names.index("pelvis"), 2])),
                "notes": candidate.notes,
            }
        )
        cursor += frame_count

    bundle = {key: np.concatenate(parts, axis=0).astype(np.float32) for key, parts in bundle_parts.items()}
    bundle["fps"] = np.asarray([OUTPUT_FPS], dtype=np.int64)
    np.savez_compressed(OUT_NPZ, **bundle)

    row_fields = [
        "index",
        "motion",
        "source_family",
        "source_kind",
        "source_path",
        "source_exists",
        "source_sha256",
        "csv_path",
        "csv_sha256",
        "input_fps",
        "output_fps",
        "start_frame",
        "end_frame_exclusive",
        "frame_count",
        "duration_seconds",
        "per_motion_npz",
        "per_motion_npz_size",
        "per_motion_npz_sha256",
        "joint_min",
        "joint_max",
        "root_z_min",
        "root_z_max",
        "notes",
    ]
    skipped_fields = ["source_path", "source_family", "reason", "exists", "observed_dof_shapes", "load_error"]
    write_rows(OUT_CSV, rows, row_fields, ",")
    write_rows(OUT_TSV, rows, row_fields, "\t")
    write_rows(OUT_SKIPPED_TSV, skipped, skipped_fields, "\t")

    source_counts: dict[str, int] = {}
    source_durations: dict[str, float] = {}
    for row in rows:
        source_counts[row["source_family"]] = source_counts.get(row["source_family"], 0) + 1
        source_durations[row["source_family"]] = source_durations.get(row["source_family"], 0.0) + float(
            row["duration_seconds"]
        )

    checks = {
        "candidate_count_ge_40": len(rows) >= 40,
        "lafan1_40_csvs_included": source_counts.get("Unitree-retargeted LAFAN1", 0) == 40,
        "zenodo_tkd_skill_included": source_counts.get("BeyondMimic Zenodo ablation reference CSV", 0) == 1,
        "hub_29dof_candidates_included": source_counts.get("HuB supplemental 29-DoF pkl", 0) >= 2,
        "pb_hc_sidekick_not_silently_padded": any("KungfuBot" in row.get("source_family", "") for row in skipped),
        "asap_ronaldo_not_silently_padded": any("Ronaldo" in row.get("source_family", "") for row in skipped),
        "npz_written": OUT_NPZ.is_file() and OUT_NPZ.stat().st_size > 0,
        "joint_shape_29": list(bundle["joint_pos"].shape)[1] == 29,
        "body_shape_40": list(bundle["body_pos_w"].shape)[1] == 40,
        "all_arrays_finite": all(np.isfinite(bundle[key]).all() for key in ARRAY_KEYS),
        "fps_50": int(bundle["fps"][0]) == 50,
        "does_not_claim_2_5h_complete_paper_motion_set": True,
        "does_not_claim_official_beyondmimic_teacher_dataset": True,
        "does_not_claim_real_robot": True,
    }
    status = "ok_stage1_multisource_motion_bundle" if all(checks.values()) else "failed_stage1_multisource_motion_bundle"
    summary = {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "tracking_stage1_multisource_motion_bundle",
        "scope": (
            "Builds a local Stage-1 motion-training candidate bundle from all directly usable 36-column G1 CSVs and "
            "29-DoF G1 pkl sources currently present. It is intended for a parallel 5/6 GPU teacher-policy run."
        ),
        "claim_level": "local_public_plus_available_multisource_stage1_training_candidate",
        "checks": checks,
        "metrics": {
            "motion_count": len(rows),
            "skipped_source_count": len(skipped),
            "total_frames": int(bundle["joint_pos"].shape[0]),
            "total_duration_seconds": float(bundle["joint_pos"].shape[0] / OUTPUT_FPS),
            "total_duration_hours": float(bundle["joint_pos"].shape[0] / OUTPUT_FPS / 3600.0),
            "source_counts": source_counts,
            "source_duration_seconds": source_durations,
            "npz_size_bytes": OUT_NPZ.stat().st_size,
            "npz_sha256": sha256_file(OUT_NPZ),
            "joint_pos_shape": list(bundle["joint_pos"].shape),
            "body_pos_w_shape": list(bundle["body_pos_w"].shape),
        },
        "inputs": {
            "lafan_root": str(LAFAN_ROOT),
            "zenodo_tkd_csv": str(ZENODO_TKD_CSV),
            "reproduction_zenodo_tkd_csv": str(REPRO_ZENODO_TKD_CSV),
            "hub_pickles": [str(path) for path in HUB_PICKLES],
            "body_order_probe": str(BODY_ORDER_PROBE),
            "body_contract": str(BODY_CONTRACT),
            "urdf": str(DEFAULT_URDF),
        },
        "outputs": {
            "json": str(OUT_JSON),
            "npz": str(OUT_NPZ),
            "csv_dir": str(CSV_OUT),
            "motion_root": str(MOTION_ROOT),
            "rows_csv": str(OUT_CSV),
            "rows_tsv": str(OUT_TSV),
            "skipped_tsv": str(OUT_SKIPPED_TSV),
        },
        "rows": rows,
        "skipped_sources": skipped,
        "robot_body_names": robot_body_names,
        "interpretation": {
            "goal_complete": False,
            "uses_current_4_7_training": False,
            "suitable_for_parallel_5_6_training": status == "ok_stage1_multisource_motion_bundle",
            "why_not_paper_2_5h_complete_set": (
                "The paper describes prior-work datasets plus online animation data. The current host has LAFAN1, "
                "one Zenodo 36-column tkd reference, and several HuB 29-DoF pkl motions that can be converted. "
                "PBHC sidekick and ASAP Ronaldo files are present only as 23-DoF pkl sources and are intentionally "
                "excluded until an audited 23-to-29 G1 mapping is implemented. Online animation pack sources are not "
                "present locally as train-ready generalized-coordinate motions."
            ),
            "why_zenodo_is_not_bulk_training_data": (
                "Most Dataset_beyondmimic files are rosbag/mcap, GRF, ablation, and plotting artifacts for paper "
                "evidence analysis. Only tkd_skill.csv is currently a direct 36-column generalized-coordinate reference candidate."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "json": str(OUT_JSON),
                "npz": str(OUT_NPZ),
                "motion_count": len(rows),
                "duration_hours": summary["metrics"]["total_duration_hours"],
            },
            sort_keys=True,
        )
    )
    if status != "ok_stage1_multisource_motion_bundle":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
