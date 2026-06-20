#!/usr/bin/env python3
"""Create Fig. 5D-style latent projection report assets.

The paper shows a latent-space visualization for walking/running transitions.
This script uses the local official-importer-export full-bundle teacher rollout
VAE posterior means and produces a PCA projection proxy. It is report evidence
for local latent structure only; it is not the paper t-SNE panel.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_latent_projection"
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json"
)
VAE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
    "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
)
FULL_BUNDLE_JSON = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
)
CLIPS_CSV = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_clips.csv"
)
FULL_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "official_csv_loop_full_public_motion_bundle.npz"
)
MAX_SAMPLES_PER_FAMILY = 1600
MAX_TRACE_POINTS_PER_MOTION = 120
SEED = 20260693


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


def read_clips(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            item["start_frame"] = int(item["start_frame"])
            item["end_frame_exclusive"] = int(item["end_frame_exclusive"])
            item["frame_count"] = int(item["frame_count"])
            item["fps"] = int(item["fps"])
            item["family"] = motion_family(item["motion"])
            rows.append(item)
    return rows


def motion_family(motion: str) -> str:
    stem = motion.split("_subject", 1)[0]
    return re.sub(r"\d+$", "", stem)


def clip_lookup(clips: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    max_frame = max(int(row["end_frame_exclusive"]) for row in clips)
    motion_by_frame = np.empty(max_frame, dtype=object)
    family_by_frame = np.empty(max_frame, dtype=object)
    for row in clips:
        motion_by_frame[row["start_frame"] : row["end_frame_exclusive"]] = row["motion"]
        family_by_frame[row["start_frame"] : row["end_frame_exclusive"]] = row["family"]
    return np.arange(max_frame), motion_by_frame.tolist(), family_by_frame.tolist()


def root_speed_by_frame() -> np.ndarray:
    with np.load(FULL_BUNDLE_NPZ) as data:
        root_xy = np.asarray(data["body_pos_w"][:, 0, :2], dtype=np.float64)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    vel = np.diff(root_xy, axis=0, prepend=root_xy[0:1]) * fps
    return np.linalg.norm(vel, axis=-1)


def load_latents(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    paths = [Path(path) for path in summary["worker_summary"]["outputs"]["latent_shards"]]
    latent_parts: list[np.ndarray] = []
    reward_parts: list[np.ndarray] = []
    done_parts: list[np.ndarray] = []
    motion_step_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            latent = np.asarray(data["latent_mu"], dtype=np.float64).reshape(-1, data["latent_mu"].shape[-1])
            rewards = np.asarray(data["rewards"], dtype=np.float64).reshape(-1)
            dones = np.asarray(data["dones"], dtype=bool).reshape(-1)
            motion_time_steps = np.asarray(data["motion_time_steps"], dtype=np.int64).reshape(-1)
            rank = int(np.asarray(data["rank"]).reshape(-1)[0])
        latent_parts.append(latent)
        reward_parts.append(rewards)
        done_parts.append(dones)
        motion_step_parts.append(motion_time_steps)
        rank_parts.append(np.full(latent.shape[0], rank, dtype=np.int16))
    return {
        "latent_mu": np.concatenate(latent_parts, axis=0),
        "rewards": np.concatenate(reward_parts, axis=0),
        "dones": np.concatenate(done_parts, axis=0),
        "motion_time_steps": np.concatenate(motion_step_parts, axis=0),
        "ranks": np.concatenate(rank_parts, axis=0),
    }


def pca_fit(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = latent.mean(axis=0)
    centered = latent - mean
    cov = centered.T @ centered / max(latent.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.maximum(eigvals.sum(), 1e-12))
    explained = eigvals / total
    return mean, eigvecs[:, :2], explained[:8]


def stratified_sample_indices(families: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices: list[np.ndarray] = []
    for family in sorted(set(families.tolist())):
        fam_idx = np.nonzero(families == family)[0]
        take = min(MAX_SAMPLES_PER_FAMILY, fam_idx.size)
        if take:
            indices.append(rng.choice(fam_idx, size=take, replace=False))
    return np.sort(np.concatenate(indices)) if indices else np.array([], dtype=np.int64)


def project(latent: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (latent - mean) @ components


def family_colors(families: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {family: cmap(i % 20) for i, family in enumerate(sorted(families))}


def plot_family_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    families = sorted({row["family"] for row in rows})
    colors = family_colors(families)
    fig, ax = plt.subplots(figsize=(9.0, 6.8), constrained_layout=True)
    for family in families:
        xs = [row["pc1"] for row in rows if row["family"] == family]
        ys = [row["pc2"] for row in rows if row["family"] == family]
        alpha = 0.85 if family in {"walk", "run"} else 0.38
        size = 9 if family in {"walk", "run"} else 5
        ax.scatter(xs, ys, s=size, alpha=alpha, color=colors[family], label=family, linewidths=0)
    ax.set_title("Official-importer local VAE latent PCA projection by motion family")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_speed_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    xs = np.array([row["pc1"] for row in rows], dtype=np.float64)
    ys = np.array([row["pc2"] for row in rows], dtype=np.float64)
    speeds = np.array([row["root_speed_mps"] for row in rows], dtype=np.float64)
    sc = ax.scatter(xs, ys, c=speeds, cmap="viridis", s=6, alpha=0.72, linewidths=0)
    fig.colorbar(sc, ax=ax, label="root speed proxy (m/s)")
    ax.set_title("Official-importer local VAE latent PCA projection by root-speed proxy")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def trace_rows_by_motion(
    latent: np.ndarray,
    motion_steps: np.ndarray,
    clips: list[dict[str, Any]],
    mean: np.ndarray,
    components: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    walk_run_clips = [row for row in clips if row["family"] in {"walk", "run"}]
    for clip in walk_run_clips:
        frame_indices = np.arange(clip["start_frame"], clip["end_frame_exclusive"], dtype=np.int64)
        frame_indices = frame_indices[frame_indices > 0]
        if frame_indices.size == 0:
            continue
        take_idx = np.linspace(0, frame_indices.size - 1, min(MAX_TRACE_POINTS_PER_MOTION, frame_indices.size))
        selected_frames = frame_indices[np.round(take_idx).astype(np.int64)]
        for frame in selected_frames:
            sample_idx = np.nonzero(motion_steps == frame)[0]
            if sample_idx.size == 0:
                continue
            pooled = latent[sample_idx].mean(axis=0, keepdims=True)
            pc = project(pooled, mean, components)[0]
            rows.append(
                {
                    "motion": clip["motion"],
                    "family": clip["family"],
                    "global_frame": int(frame),
                    "motion_frame": int(frame - clip["start_frame"]),
                    "pc1": float(pc[0]),
                    "pc2": float(pc[1]),
                    "sample_count": int(sample_idx.size),
                }
            )
    return rows


def plot_walk_run_trace(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4), constrained_layout=True)
    by_motion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_motion[row["motion"]].append(row)
    colors = {"walk": "#2563eb", "run": "#dc2626"}
    seen: set[str] = set()
    for motion, motion_rows in sorted(by_motion.items()):
        motion_rows = sorted(motion_rows, key=lambda row: row["motion_frame"])
        family = motion_rows[0]["family"]
        label = family if family not in seen else None
        ax.plot(
            [row["pc1"] for row in motion_rows],
            [row["pc2"] for row in motion_rows],
            color=colors.get(family, "#111827"),
            alpha=0.45,
            linewidth=1.2,
            label=label,
        )
        seen.add(family)
    ax.set_title("Walk/run local VAE latent PCA traces")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_summary = load_json(STATE_LATENT_JSON)
    vae_summary = load_json(VAE_JSON)
    bundle_summary = load_json(FULL_BUNDLE_JSON)
    clips = read_clips(CLIPS_CSV)
    _, motion_by_frame, family_by_frame = clip_lookup(clips)
    speeds = root_speed_by_frame()
    arrays = load_latents(state_summary)
    latent = arrays["latent_mu"]
    motion_steps = arrays["motion_time_steps"].clip(0, len(family_by_frame) - 1)
    families = np.array([family_by_frame[int(idx)] for idx in motion_steps], dtype=object)
    motions = np.array([motion_by_frame[int(idx)] for idx in motion_steps], dtype=object)
    root_speeds = speeds[motion_steps]

    mean, components, explained = pca_fit(latent)
    rng = np.random.default_rng(SEED)
    sample_idx = stratified_sample_indices(families, rng)
    projected = project(latent[sample_idx], mean, components)

    projection_rows = [
        {
            "sample_index": int(idx),
            "rank": int(arrays["ranks"][idx]),
            "motion_time_step": int(arrays["motion_time_steps"][idx]),
            "motion": str(motions[idx]),
            "family": str(families[idx]),
            "pc1": float(projected[row_idx, 0]),
            "pc2": float(projected[row_idx, 1]),
            "root_speed_mps": float(root_speeds[idx]),
            "reward": float(arrays["rewards"][idx]),
            "done": bool(arrays["dones"][idx]),
        }
        for row_idx, idx in enumerate(sample_idx.tolist())
    ]
    family_summary_rows: list[dict[str, Any]] = []
    for family in sorted(set(families.tolist())):
        mask = families == family
        family_summary_rows.append(
            {
                "family": family,
                "sample_count": int(mask.sum()),
                "motion_count": int(len({str(item) for item in motions[mask].tolist()})),
                "root_speed_mean": float(root_speeds[mask].mean()),
                "root_speed_std": float(root_speeds[mask].std()),
                "latent_norm_mean": float(np.linalg.norm(latent[mask], axis=1).mean()),
                "reward_mean": float(arrays["rewards"][mask].mean()),
                "done_fraction": float(arrays["dones"][mask].mean()),
            }
        )

    trace_rows = trace_rows_by_motion(latent, motion_steps, clips, mean, components)
    projection_csv = OUT / "latent_pca_projection_samples.csv"
    family_csv = OUT / "latent_pca_family_summary.csv"
    trace_csv = OUT / "latent_pca_walk_run_trace.csv"
    write_csv(
        projection_csv,
        projection_rows,
        ["sample_index", "rank", "motion_time_step", "motion", "family", "pc1", "pc2", "root_speed_mps", "reward", "done"],
    )
    write_csv(
        family_csv,
        family_summary_rows,
        ["family", "sample_count", "motion_count", "root_speed_mean", "root_speed_std", "latent_norm_mean", "reward_mean", "done_fraction"],
    )
    write_csv(trace_csv, trace_rows, ["motion", "family", "global_frame", "motion_frame", "pc1", "pc2", "sample_count"])

    family_png = OUT / "latent_pca_by_motion_family.png"
    speed_png = OUT / "latent_pca_by_root_speed.png"
    trace_png = OUT / "latent_pca_walk_run_trace.png"
    plot_family_scatter(projection_rows, family_png)
    plot_speed_scatter(projection_rows, speed_png)
    plot_walk_run_trace(trace_rows, trace_png)

    family_counts = Counter(families.tolist())
    metrics = {
        "total_latent_samples": int(latent.shape[0]),
        "latent_dim": int(latent.shape[1]),
        "sampled_projection_rows": int(len(projection_rows)),
        "family_count": int(len(family_counts)),
        "motion_count": int(len({str(item) for item in motions.tolist()})),
        "walk_sample_count": int(family_counts.get("walk", 0)),
        "run_sample_count": int(family_counts.get("run", 0)),
        "walk_run_trace_rows": int(len(trace_rows)),
        "walk_run_trace_motion_count": int(len({row["motion"] for row in trace_rows})),
        "pca_explained_variance_ratio_pc1": float(explained[0]),
        "pca_explained_variance_ratio_pc2": float(explained[1]),
        "pca_explained_variance_ratio_top2": float(explained[:2].sum()),
        "pca_explained_variance_ratio_top8": float(explained[:8].sum()),
        "root_speed_mean": float(root_speeds.mean()),
        "root_speed_std": float(root_speeds.std()),
        "family_counts": dict(sorted((str(k), int(v)) for k, v in family_counts.items())),
    }
    assets = {
        "json": str(OUT / "official_importer_export_full_bundle_latent_projection_assets.json"),
        "projection_samples_csv": str(projection_csv),
        "family_summary_csv": str(family_csv),
        "walk_run_trace_csv": str(trace_csv),
        "family_scatter_png": str(family_png),
        "root_speed_scatter_png": str(speed_png),
        "walk_run_trace_png": str(trace_png),
        "readme": str(OUT / "README.md"),
    }
    checks = {
        "state_latent_status_ok": state_summary["status"]
        == "ok_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset",
        "vae_status_ok": vae_summary["status"]
        == "ok_official_importer_export_full_bundle_teacher_rollout_vae_training",
        "bundle_status_ok": bundle_summary["status"] == "ok_official_csv_loop_full_bundle_motion_npz",
        "two_latent_shards_loaded": len(state_summary["worker_summary"]["outputs"]["latent_shards"]) == 2,
        "uses_full_teacher_rollout_samples": metrics["total_latent_samples"] == 306176,
        "latent_dim_32": metrics["latent_dim"] == 32,
        "motion_count_40": metrics["motion_count"] == 40,
        "has_walk_and_run_labels": metrics["walk_sample_count"] > 0 and metrics["run_sample_count"] > 0,
        "pca_variance_recorded": metrics["pca_explained_variance_ratio_top2"] > 0.0,
        "walk_run_trace_recorded": metrics["walk_run_trace_rows"] > 0,
        "all_generated_assets_exist": all(
            Path(path).is_file() and Path(path).stat().st_size > 0
            for name, path in assets.items()
            if name != "json" and path
        ),
        "does_not_claim_tsne_reproduction": True,
        "does_not_claim_paper_fig5d": True,
        "does_not_claim_official_beyondmimic_checkpoint": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    payload = {
        "status": "ok_official_importer_export_full_bundle_latent_projection_report_assets"
        if all(checks.values())
        else "failed_official_importer_export_full_bundle_latent_projection_report_assets",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "official_importer_export_full_bundle_latent_projection_report_assets",
        "scope": (
            "PCA projection proxy for local official-importer-export full-bundle VAE posterior means. "
            "This is Fig. 5D-adjacent report evidence, not the paper t-SNE panel."
        ),
        "sources": {
            "state_latent_dataset_json": str(STATE_LATENT_JSON),
            "vae_training_json": str(VAE_JSON),
            "full_bundle_json": str(FULL_BUNDLE_JSON),
            "clips_csv": str(CLIPS_CSV),
            "full_bundle_npz": str(FULL_BUNDLE_NPZ),
        },
        "metrics": metrics,
        "assets": assets,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_fig5d_latent_projection_proxy",
            "why_not_paper_level": (
                "The projection uses local official-importer-export PPO teacher rollouts, a local conditional VAE, "
                "and PCA rather than the paper's t-SNE protocol and unreleased official checkpoints. It is useful "
                "for explaining latent organization in the English report, but it is not paper Fig. 5D reproduction, "
                "not a walk-to-run closed-loop transition protocol, and not real-robot evidence."
            ),
        },
    }
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer Full-Bundle Latent Projection",
                "",
                "This folder contains a PCA projection proxy for local official-importer-export full-bundle VAE posterior means.",
                "It is intended for the English reading report and PPT as Fig. 5D-adjacent local evidence.",
                "",
                "It is not the paper t-SNE panel, not official BeyondMimic checkpoint evidence, and not real-robot evidence.",
                "",
                "## Key Metrics",
                "",
                f"- Total latent samples: `{metrics['total_latent_samples']}`",
                f"- Latent dim: `{metrics['latent_dim']}`",
                f"- Motion count: `{metrics['motion_count']}`",
                f"- Family count: `{metrics['family_count']}`",
                f"- PCA top-2 explained variance ratio: `{metrics['pca_explained_variance_ratio_top2']}`",
                f"- Walk samples: `{metrics['walk_sample_count']}`",
                f"- Run samples: `{metrics['run_sample_count']}`",
                "",
                "## Assets",
                "",
                f"- `{family_png}`",
                f"- `{speed_png}`",
                f"- `{trace_png}`",
                f"- `{projection_csv}`",
                f"- `{family_csv}`",
                f"- `{trace_csv}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(Path(assets["json"]), payload)
    print(json.dumps({"status": payload["status"], "json": assets["json"]}, sort_keys=True))
    if payload["status"] != "ok_official_importer_export_full_bundle_latent_projection_report_assets":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
