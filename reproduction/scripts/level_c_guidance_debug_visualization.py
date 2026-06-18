#!/usr/bin/env python3
"""Debug-only guidance trajectory visualization on a local motion fixture.

The script applies simple finite-difference/closed-form task guidance updates to
the integrated root XY trajectory from one local fixture window, then writes
plots and a GIF. This is a visualization of formula effects, not a trained
guided diffusion rollout or paper Fig. 5/Fig. 6 video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/guidance_debug_visualization"


def sl(feature_slices: dict[str, list[int]], name: str) -> slice:
    lo, hi = feature_slices[name]
    return slice(lo, hi)


def load_future_xy(fixture_npz: Path, manifest_json: Path, history: int) -> tuple[np.ndarray, dict[str, Any]]:
    fixture = np.load(fixture_npz)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    feature_slices = manifest["feature_slices"]
    window = fixture["candidate_hybrid_state_windows"][0]
    future = window[history:].astype(np.float64)
    root_vel = future[:, sl(feature_slices, "root_lin_vel_yaw_frame")][:, :2]
    root_xy = np.cumsum(root_vel * 0.04, axis=0)
    return root_xy, manifest


def smooth_path(path: np.ndarray) -> np.ndarray:
    out = path.copy()
    if len(path) >= 3:
        out[1:-1] = 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:]
    return out


def joystick_update(root_xy: np.ndarray, command_velocity: np.ndarray, scale: float, dt: float) -> np.ndarray:
    current_velocity = np.diff(np.vstack([np.zeros((1, 2)), root_xy]), axis=0) / dt
    velocity_error = current_velocity - command_velocity[None, :]
    updated_velocity = current_velocity - scale * velocity_error
    return np.cumsum(updated_velocity * dt, axis=0)


def waypoint_update(root_xy: np.ndarray, goal_xy: np.ndarray, scale: float) -> np.ndarray:
    distance = np.linalg.norm(root_xy - goal_xy[None, :], axis=-1, keepdims=True)
    weight_pos = 1.0 - np.exp(-2.0 * distance)
    grad = 2.0 * weight_pos * (root_xy - goal_xy[None, :])
    return root_xy - scale * grad


def obstacle_update(root_xy: np.ndarray, center: np.ndarray, radius: float, margin: float, scale: float) -> np.ndarray:
    vec = root_xy - center[None, :]
    dist = np.linalg.norm(vec, axis=-1, keepdims=True)
    direction = vec / np.maximum(dist, 1e-9)
    clearance = dist - radius
    penetration = np.maximum(margin - clearance, 0.0)
    return root_xy + scale * penetration * direction


def inpainting_update(root_xy: np.ndarray, keyframe_index: int, keyframe_xy: np.ndarray, scale: float) -> np.ndarray:
    updated = root_xy.copy()
    influence = np.exp(-0.5 * ((np.arange(len(root_xy)) - keyframe_index) / 2.0) ** 2)[:, None]
    updated += scale * influence * (keyframe_xy[None, :] - root_xy[keyframe_index][None, :])
    return updated


def path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1)))


def second_difference_mean_norm(path: np.ndarray) -> float:
    if len(path) < 3:
        return 0.0
    return float(np.mean(np.linalg.norm(path[2:] - 2.0 * path[1:-1] + path[:-2], axis=-1)))


def min_obstacle_clearance(path: np.ndarray, center: np.ndarray, radius: float) -> float:
    return float(np.min(np.linalg.norm(path - center[None, :], axis=-1) - radius))


def terminal_distance(path: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.linalg.norm(path[-1] - goal_xy))


def keyframe_error(path: np.ndarray, keyframe_index: int, keyframe_xy: np.ndarray) -> float:
    return float(np.linalg.norm(path[keyframe_index] - keyframe_xy))


def velocity_command_mse(path: np.ndarray, command_velocity: np.ndarray, dt: float) -> float:
    velocity = np.diff(np.vstack([np.zeros((1, 2)), path]), axis=0) / dt
    return float(np.mean(np.sum((velocity - command_velocity[None, :]) ** 2, axis=-1)))


def task_metrics(
    path: np.ndarray,
    *,
    goal_xy: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    keyframe_index: int,
    keyframe_xy: np.ndarray,
    command_velocity: np.ndarray,
    dt: float,
) -> dict[str, float]:
    return {
        "path_length": path_length(path),
        "velocity_command_mse": velocity_command_mse(path, command_velocity, dt),
        "terminal_goal_distance": terminal_distance(path, goal_xy),
        "min_obstacle_clearance": min_obstacle_clearance(path, obstacle_center, obstacle_radius),
        "keyframe_error": keyframe_error(path, keyframe_index, keyframe_xy),
        "second_difference_mean_norm": second_difference_mean_norm(path),
    }


def write_rows_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "metric",
        "before",
        "after",
        "delta",
        "improved",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_paths(
    paths: dict[str, np.ndarray],
    goal_xy: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    keyframe_xy: np.ndarray,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    colors = {
        "original": "0.35",
        "joystick": "tab:blue",
        "waypoint": "tab:green",
        "obstacle_avoidance": "tab:red",
        "inpainting": "tab:purple",
        "composed": "tab:orange",
    }
    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], marker="o", markersize=3, linewidth=1.8, label=name, color=colors[name])
    ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=180, color="tab:green", label="waypoint target")
    ax.scatter([keyframe_xy[0]], [keyframe_xy[1]], marker="X", s=100, color="tab:purple", label="inpaint keyframe")
    circle = plt.Circle(obstacle_center, obstacle_radius, fill=False, color="tab:red", linewidth=2.2, label="obstacle")
    ax.add_patch(circle)
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("integrated root x (m, yaw frame)")
    ax.set_ylabel("integrated root y (m, yaw frame)")
    ax.set_title("Debug Guidance Trajectory Visualization")
    ax.legend(loc="best", fontsize=8)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def world_to_pixel(points: np.ndarray, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    width, height = size
    x = (points[:, 0] - xmin) / max(xmax - xmin, 1e-9) * (width - 40) + 20
    y = height - ((points[:, 1] - ymin) / max(ymax - ymin, 1e-9) * (height - 40) + 20)
    return np.stack([x, y], axis=-1)


def write_gif(
    paths: dict[str, np.ndarray],
    goal_xy: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    keyframe_xy: np.ndarray,
    gif_path: Path,
) -> None:
    all_points = np.vstack(list(paths.values()) + [goal_xy[None, :], obstacle_center[None, :], keyframe_xy[None, :]])
    pad = 0.08
    bounds = (
        float(np.min(all_points[:, 0]) - pad),
        float(np.max(all_points[:, 0]) + pad),
        float(np.min(all_points[:, 1]) - pad),
        float(np.max(all_points[:, 1]) + pad),
    )
    size = (640, 480)
    colors = {
        "original": (80, 80, 80),
        "joystick": (31, 119, 180),
        "waypoint": (44, 160, 44),
        "obstacle_avoidance": (214, 39, 40),
        "inpainting": (148, 103, 189),
        "composed": (255, 127, 14),
    }
    frames: list[Image.Image] = []
    horizon = max(len(path) for path in paths.values())
    for frame_idx in range(horizon):
        im = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(im)
        obstacle_px = world_to_pixel(obstacle_center[None, :], bounds, size)[0]
        radius_px = obstacle_radius / max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1e-9) * (size[0] - 40)
        draw.ellipse(
            [
                obstacle_px[0] - radius_px,
                obstacle_px[1] - radius_px,
                obstacle_px[0] + radius_px,
                obstacle_px[1] + radius_px,
            ],
            outline=colors["obstacle_avoidance"],
            width=3,
        )
        for target, fill in [(goal_xy, colors["waypoint"]), (keyframe_xy, colors["inpainting"])]:
            px = world_to_pixel(target[None, :], bounds, size)[0]
            draw.ellipse([px[0] - 5, px[1] - 5, px[0] + 5, px[1] + 5], fill=fill)
        for name, path in paths.items():
            pts = world_to_pixel(path[: frame_idx + 1], bounds, size)
            if len(pts) >= 2:
                draw.line([tuple(p) for p in pts], fill=colors[name], width=3)
            px = pts[-1]
            draw.ellipse([px[0] - 4, px[1] - 4, px[0] + 4, px[1] + 4], fill=colors[name])
        draw.text((16, 12), f"debug guidance frame {frame_idx + 1}/{horizon}", fill=(0, 0, 0))
        draw.text((16, 32), "not a trained rollout or paper Fig.5/Fig.6 video", fill=(80, 80, 80))
        frames.append(im)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-npz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--history", type=int, default=4)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dt = 0.04
    original, manifest = load_future_xy(args.fixture_npz, args.manifest_json, args.history)
    goal_xy = np.asarray([0.08, 0.04], dtype=np.float64)
    obstacle_center = np.asarray([0.04, 0.0], dtype=np.float64)
    obstacle_radius = 0.2
    obstacle_margin = 0.05
    keyframe_index = min(5, len(original) - 1)
    keyframe_xy = np.asarray([0.04, -0.02], dtype=np.float64)
    command_velocity = np.asarray([0.35, 0.0], dtype=np.float64)

    joystick = smooth_path(joystick_update(original, command_velocity, scale=0.35, dt=dt))
    waypoint = smooth_path(waypoint_update(original, goal_xy, scale=0.75))
    obstacle = smooth_path(obstacle_update(original, obstacle_center, obstacle_radius, obstacle_margin, scale=0.95))
    inpainting = smooth_path(inpainting_update(original, keyframe_index, keyframe_xy, scale=0.75))
    composed = smooth_path(
        obstacle_update(
            waypoint_update(joystick, goal_xy, scale=0.35),
            obstacle_center,
            obstacle_radius,
            obstacle_margin,
            scale=0.65,
        )
    )
    paths = {
        "original": original,
        "joystick": joystick,
        "waypoint": waypoint,
        "obstacle_avoidance": obstacle,
        "inpainting": inpainting,
        "composed": composed,
    }

    base_metrics = task_metrics(
        original,
        goal_xy=goal_xy,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        keyframe_index=keyframe_index,
        keyframe_xy=keyframe_xy,
        command_velocity=command_velocity,
        dt=dt,
    )
    task_to_primary_metric = {
        "joystick": "velocity_command_mse",
        "waypoint": "terminal_goal_distance",
        "obstacle_avoidance": "min_obstacle_clearance",
        "inpainting": "keyframe_error",
        "composed": "min_obstacle_clearance",
    }
    rows: list[dict[str, Any]] = []
    per_task: dict[str, Any] = {}
    for task, path in paths.items():
        if task == "original":
            continue
        metrics = task_metrics(
            path,
            goal_xy=goal_xy,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            keyframe_index=keyframe_index,
            keyframe_xy=keyframe_xy,
            command_velocity=command_velocity,
            dt=dt,
        )
        metric_rows: dict[str, Any] = {}
        for metric, before in base_metrics.items():
            after = metrics[metric]
            lower_is_better = metric != "min_obstacle_clearance"
            improved = after < before if lower_is_better else after > before
            delta = before - after if lower_is_better else after - before
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "improved": bool(improved),
                }
            )
            metric_rows[metric] = {"before": before, "after": after, "delta": delta, "improved": bool(improved)}
        per_task[task] = {
            "primary_metric": task_to_primary_metric[task],
            "metrics": metric_rows,
            "primary_improved": metric_rows[task_to_primary_metric[task]]["improved"],
        }

    plot_base = OUT / "level_c_guidance_debug_visualization"
    gif_path = OUT / "level_c_guidance_debug_visualization.gif"
    npz_path = OUT / "level_c_guidance_debug_visualization.npz"
    json_path = OUT / "level_c_guidance_debug_visualization.json"
    tsv_path = OUT / "level_c_guidance_debug_visualization.tsv"
    plot_paths(paths, goal_xy, obstacle_center, obstacle_radius, keyframe_xy, plot_base)
    write_gif(paths, goal_xy, obstacle_center, obstacle_radius, keyframe_xy, gif_path)
    np.savez_compressed(
        npz_path,
        **{f"{name}_root_xy": path.astype(np.float64) for name, path in paths.items()},
        goal_xy=goal_xy,
        obstacle_center=obstacle_center,
        keyframe_xy=keyframe_xy,
    )
    write_rows_tsv(tsv_path, rows)

    image_outputs = [plot_base.with_suffix(ext) for ext in [".png", ".svg", ".pdf"]]
    checks = {
        "fixture_inputs_exist": args.fixture_npz.is_file() and args.manifest_json.is_file(),
        "writes_png_svg_pdf": all(path.is_file() and path.stat().st_size > 0 for path in image_outputs),
        "writes_debug_gif": gif_path.is_file() and gif_path.stat().st_size > 0,
        "writes_npz": npz_path.is_file() and npz_path.stat().st_size > 0,
        "five_debug_tasks_visualized": sorted(per_task) == sorted(
            ["joystick", "waypoint", "obstacle_avoidance", "inpainting", "composed"]
        ),
        "all_primary_metrics_improve": all(item["primary_improved"] for item in per_task.values()),
        "trajectory_arrays_finite": all(np.isfinite(path).all() for path in paths.values()),
        "does_not_claim_trained_rollout": True,
        "does_not_claim_paper_video": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_guidance_visualization",
        "scope": "formula-effect visualization for guidance tasks on one local motion-derived future trajectory",
        "paper_evidence": {
            "joystick_waypoint_obstacle_guidance": str(ROOT / "reproduction/paper/source/root.tex:548-586"),
            "inpainting_task_description": str(ROOT / "reproduction/paper/source/root.tex:237-243"),
            "goal_phase8_visuals": str(ROOT / "goal.md:1488-1507,1783-1828"),
        },
        "settings": {
            "fixture": str(args.fixture_npz),
            "manifest": str(args.manifest_json),
            "history": args.history,
            "dt": dt,
            "goal_xy": goal_xy.tolist(),
            "obstacle_center": obstacle_center.tolist(),
            "obstacle_radius": obstacle_radius,
            "obstacle_margin": obstacle_margin,
            "keyframe_index": keyframe_index,
            "keyframe_xy": keyframe_xy.tolist(),
            "command_velocity_xy": command_velocity.tolist(),
            "frame_count": int(len(original)),
        },
        "base_metrics": base_metrics,
        "per_task": per_task,
        "checks": checks,
        "not_a_replacement_for": [
            "trained guided diffusion rollout",
            "closed-loop Isaac/MuJoCo simulation",
            "paper Fig. 5/Fig. 6 reproduction",
            "success/failure task videos",
            "validation/test guidance scale protocol",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "The PNG/SVG/PDF/GIF visualize local formula guidance effects on one fixture trajectory. They are "
                "debug-only visual artifacts and must not be counted as reproduced paper videos or closed-loop "
                "guided diffusion rollouts."
            ),
        },
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "npz": str(npz_path),
            "png": str(plot_base.with_suffix(".png")),
            "svg": str(plot_base.with_suffix(".svg")),
            "pdf": str(plot_base.with_suffix(".pdf")),
            "gif": str(gif_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "gif": str(gif_path)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
