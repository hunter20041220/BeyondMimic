#!/usr/bin/env python3
"""Debug-only probes for Level C OU perturbation and sagittal symmetry.

This script validates paper-specified OU noise parameters and a candidate G1
sagittal symmetry mapping on an existing motion-derived fixture. It does not
generate a paper-exact rollout dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/augmentation_probe"

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Candidate signs follow the usual sagittal mirror convention: pitch-like
# flexion/extension keeps sign, roll/yaw-like lateral axes flip sign. This is
# recorded as a candidate rule because the paper cites symmetry augmentation
# but does not spell out the G1 joint-sign table.
SYMMETRY_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1.0),
    ("left_elbow_joint", "right_elbow_joint", 1.0),
    ("left_wrist_roll_joint", "right_wrist_roll_joint", -1.0),
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1.0),
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1.0),
]

CENTER_SIGN = {
    "waist_yaw_joint": -1.0,
    "waist_roll_joint": -1.0,
    "waist_pitch_joint": 1.0,
}


def ou_noise(steps: int, dims: int, theta: float, mu: float, dt: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eta = np.zeros((steps, dims), dtype=np.float64)
    for t in range(steps - 1):
        eps = rng.standard_normal(dims)
        eta[t + 1] = eta[t] + theta * (mu - eta[t]) * dt + sigma * np.sqrt(dt) * eps
    return eta


def autocorr_lag1(x: np.ndarray) -> float:
    a = x[:-1].reshape(-1)
    b = x[1:].reshape(-1)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mirror_joint_array(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    mirrored = values.copy()
    pair_rows = []
    for left, right, sign in SYMMETRY_PAIRS:
        li = name_to_idx[left]
        ri = name_to_idx[right]
        left_src = values[:, li].copy()
        right_src = values[:, ri].copy()
        mirrored[:, li] = sign * right_src
        mirrored[:, ri] = sign * left_src
        double_left = sign * mirrored[:, ri]
        double_right = sign * mirrored[:, li]
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "sign": sign,
                "double_mirror_left_max_abs_error": float(np.max(np.abs(double_left - left_src))),
                "double_mirror_right_max_abs_error": float(np.max(np.abs(double_right - right_src))),
            }
        )
    for name, sign in CENTER_SIGN.items():
        idx = name_to_idx[name]
        mirrored[:, idx] = sign * values[:, idx]
    double_mirrored = values.copy()
    for left, right, sign in SYMMETRY_PAIRS:
        li = name_to_idx[left]
        ri = name_to_idx[right]
        double_mirrored[:, li] = sign * mirrored[:, ri]
        double_mirrored[:, ri] = sign * mirrored[:, li]
    for name, sign in CENTER_SIGN.items():
        idx = name_to_idx[name]
        double_mirrored[:, idx] = sign * mirrored[:, idx]
    info = {
        "pair_count": len(SYMMETRY_PAIRS),
        "center_joint_count": len(CENTER_SIGN),
        "pair_rows": pair_rows,
        "double_mirror_joint_max_abs_error": float(np.max(np.abs(double_mirrored - values))),
    }
    return mirrored, info


def write_tsv(path: Path, rows: dict[str, Any]) -> None:
    flat: list[tuple[str, str]] = []

    def rec(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                rec(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, list):
            flat.append((prefix, json.dumps(value, sort_keys=True)))
        else:
            flat.append((prefix, str(value)))

    rec("", rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(flat)


def save_ou_plot(path_base: Path, eta: np.ndarray, iid: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
    axes[0].plot(eta[:120, 0], label="OU joint 0", linewidth=1.5)
    axes[0].plot(iid[:120, 0], label="iid Gaussian joint 0", linewidth=1.0, alpha=0.75)
    axes[0].set_title("OU perturbation vs iid Gaussian")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("noise")
    axes[0].legend()
    axes[1].hist(eta.reshape(-1), bins=40, alpha=0.7, label="OU")
    axes[1].hist(iid.reshape(-1), bins=40, alpha=0.5, label="iid")
    axes[1].set_xlabel("noise value")
    axes[1].set_ylabel("count")
    axes[1].legend()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def save_symmetry_plot(path_base: Path, joint_pos: np.ndarray, mirrored: np.ndarray) -> None:
    name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    pairs = [
        ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
        ("left_knee_joint", "right_knee_joint", 1.0),
        ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1.0),
    ]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(8, 6), constrained_layout=True)
    for ax, (left, right, sign) in zip(axes, pairs):
        li = name_to_idx[left]
        ri = name_to_idx[right]
        ax.plot(joint_pos[:160, li], label=f"original {left}", linewidth=1.3)
        ax.plot(sign * mirrored[:160, ri], "--", label=f"double-check via mirrored {right}", linewidth=1.0)
        ax.set_ylabel("rad")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("frame")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-npz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(args.fixture_npz)
    fixture_manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    joint_pos = data["joint_pos"]
    steps, dims = joint_pos.shape

    eta = ou_noise(steps, dims, args.theta, args.mu, args.dt, args.sigma, args.seed)
    rng = np.random.default_rng(args.seed)
    iid = args.sigma * rng.standard_normal((steps, dims))
    perturbed_action_probe = joint_pos + eta
    mirrored_joint_pos, symmetry_info = mirror_joint_array(joint_pos)
    mirrored_joint_vel, velocity_symmetry_info = mirror_joint_array(data["joint_vel"])

    npz_path = OUT / "level_c_augmentation_probe.npz"
    json_path = OUT / "level_c_augmentation_probe.json"
    tsv_path = OUT / "level_c_augmentation_probe.tsv"
    np.savez_compressed(
        npz_path,
        ou_noise=eta,
        iid_gaussian_probe=iid,
        perturbed_action_probe=perturbed_action_probe,
        mirrored_joint_pos_candidate=mirrored_joint_pos,
        mirrored_joint_vel_candidate=mirrored_joint_vel,
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "OU perturbation and candidate sagittal symmetry augmentation probe",
        "not_a_replacement_for": [
            "VAE rollout with perturbation",
            "episode rejection pipeline",
            "paper-exact sagittal symmetry implementation",
            "trainable augmented diffusion dataset",
        ],
        "paper_evidence": {
            "ou_equation_and_parameters": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "rollout_rejection_window": str(ROOT / "reproduction/paper/source/root.tex:546"),
            "sagittal_symmetry_augmentation": str(ROOT / "reproduction/paper/source/root.tex:591-592"),
        },
        "fixture": {
            "npz": str(args.fixture_npz),
            "manifest": str(args.manifest_json),
            "fixture_scope": fixture_manifest.get("scope"),
        },
        "ou": {
            "theta": args.theta,
            "mu": args.mu,
            "dt": args.dt,
            "sigma": args.sigma,
            "seed": args.seed,
            "steps": int(steps),
            "dims": int(dims),
            "std": float(np.std(eta)),
            "mean_abs": float(np.mean(np.abs(eta))),
            "lag1_autocorrelation": autocorr_lag1(eta),
            "iid_lag1_autocorrelation": autocorr_lag1(iid),
        },
        "symmetry_candidate": {
            **symmetry_info,
            "joint_names": JOINT_NAMES,
            "center_signs": CENTER_SIGN,
            "velocity_double_mirror_joint_max_abs_error": velocity_symmetry_info["double_mirror_joint_max_abs_error"],
        },
        "shapes": {
            "ou_noise": list(eta.shape),
            "perturbed_action_probe": list(perturbed_action_probe.shape),
            "mirrored_joint_pos_candidate": list(mirrored_joint_pos.shape),
        },
        "checks": {
            "finite_all_arrays": bool(
                np.isfinite(eta).all()
                and np.isfinite(perturbed_action_probe).all()
                and np.isfinite(mirrored_joint_pos).all()
                and np.isfinite(mirrored_joint_vel).all()
            ),
            "ou_more_temporally_correlated_than_iid": bool(autocorr_lag1(eta) > autocorr_lag1(iid)),
            "symmetry_double_mirror_exact": bool(symmetry_info["double_mirror_joint_max_abs_error"] < 1e-12),
            "velocity_symmetry_double_mirror_exact": bool(
                velocity_symmetry_info["double_mirror_joint_max_abs_error"] < 1e-12
            ),
        },
        "outputs": {
            "npz": str(npz_path),
            "json": str(json_path),
            "tsv": str(tsv_path),
            "ou_plot_base": str(OUT / "ou_perturbation_probe"),
            "symmetry_plot_base": str(OUT / "sagittal_symmetry_probe"),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    save_ou_plot(OUT / "ou_perturbation_probe", eta, iid)
    save_symmetry_plot(OUT / "sagittal_symmetry_probe", joint_pos, mirrored_joint_pos)
    print(json.dumps({"status": "ok", "json": str(json_path), "npz": str(npz_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
