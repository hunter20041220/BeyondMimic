#!/usr/bin/env python3
"""Visualize full-checkpoint public-data guidance trajectories for representative tasks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/guidance_checkpoint_visualization"
OFFLINE_NPZ = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
    / "level_c_lafan1_paper_arch_guidance_eval.npz"
)
OFFLINE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
    / "level_c_lafan1_paper_arch_guidance_eval.json"
)
REVERSE_NPZ = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
    / "level_c_lafan1_paper_arch_reverse_guidance_full_split.npz"
)
REVERSE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
    / "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
)
DATASET_NPZ = ROOT / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_training_dataset.npz"
TASKS = ["joystick", "waypoint", "obstacle_avoidance", "inpainting", "composed_objectives"]
LABELS = {
    "clean": "clean reference",
    "unguided": "unguided prediction",
    "offline": "offline guided",
    "reverse": "reverse guided",
}
COLORS = {"clean": "#2c2c2c", "unguided": "#3977b7", "offline": "#4f8f45", "reverse": "#b85c38"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def state_from_tau(tau: np.ndarray, projection_inverse: np.ndarray) -> np.ndarray:
    return np.einsum("...p,dp->...d", tau[..., :207], projection_inverse).astype(np.float64)


def state_tensors(state99: np.ndarray) -> dict[str, np.ndarray]:
    body_pos = state99[:, 15:57].reshape(state99.shape[0], 14, 3)
    root_vel = state99[:, 9:12]
    root_xy = np.cumsum(root_vel[:, :2] * 0.04, axis=0)
    return {"root_xy": root_xy, "root_vel_xy": root_vel[:, :2], "body_pos_xy": body_pos[:, :, :2]}


def primary_metric(task: str, state99: np.ndarray) -> float:
    tensors = state_tensors(state99)
    command_velocity = np.asarray([0.35, 0.0], dtype=np.float64)
    goal_xy = np.asarray([0.08, 0.04], dtype=np.float64)
    obstacle_center = np.asarray([0.04, 0.0], dtype=np.float64)
    keyframe_xy = np.asarray([0.04, -0.02], dtype=np.float64)
    if task == "joystick":
        return float(np.mean((tensors["root_vel_xy"] - command_velocity) ** 2))
    if task == "waypoint":
        return float(np.linalg.norm(tensors["root_xy"][-1] - goal_xy))
    if task == "obstacle_avoidance":
        clearance = np.linalg.norm(tensors["body_pos_xy"] - obstacle_center, axis=-1) - 0.2 - 0.05
        return float(np.min(clearance))
    if task == "inpainting":
        idx = min(5, state99.shape[0] - 1)
        return float(np.linalg.norm(tensors["root_xy"][idx] - keyframe_xy))
    if task == "composed_objectives":
        clearance = np.linalg.norm(tensors["body_pos_xy"] - obstacle_center, axis=-1) - 0.2 - 0.05
        return float(np.min(clearance))
    raise ValueError(task)


def task_target(task: str) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    if task == "waypoint":
        return np.asarray([0.08, 0.04], dtype=np.float64), None, None
    if task in {"obstacle_avoidance", "composed_objectives"}:
        return None, np.asarray([0.04, 0.0], dtype=np.float64), 0.25
    if task == "inpainting":
        return np.asarray([0.04, -0.02], dtype=np.float64), None, None
    return None, None, None


def save_task_figure(task: str, states: dict[str, np.ndarray], out_base: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    target, obstacle, radius = task_target(task)
    for name, state in states.items():
        tensors = state_tensors(state)
        xy = tensors["root_xy"]
        axes[0].plot(xy[:, 0], xy[:, 1], marker="o", markersize=2.5, linewidth=1.5, color=COLORS[name], label=LABELS[name])
        if task == "joystick":
            values = tensors["root_vel_xy"][:, 0]
            ylabel = "root vx"
        elif task == "waypoint":
            values = np.linalg.norm(tensors["root_xy"] - np.asarray([0.08, 0.04]), axis=1)
            ylabel = "distance to waypoint"
        elif task == "inpainting":
            values = np.linalg.norm(tensors["root_xy"] - np.asarray([0.04, -0.02]), axis=1)
            ylabel = "distance to keyframe"
        else:
            clearance = np.linalg.norm(tensors["body_pos_xy"] - np.asarray([0.04, 0.0]), axis=-1) - 0.25
            values = np.min(clearance, axis=1)
            ylabel = "min obstacle clearance"
        axes[1].plot(np.arange(len(values)), values, marker="o", markersize=2.5, linewidth=1.5, color=COLORS[name], label=LABELS[name])
    if target is not None:
        axes[0].scatter([target[0]], [target[1]], marker="*", s=120, color="#111111", label="target")
    if obstacle is not None and radius is not None:
        axes[0].add_patch(plt.Circle(obstacle, radius, color="#b85c38", fill=False, linewidth=1.4, linestyle="--"))
    axes[0].set_title(f"{task} root trajectory")
    axes[0].set_xlabel("integrated x")
    axes[0].set_ylabel("integrated y")
    axes[0].axis("equal")
    axes[0].grid(color="#dddddd", linewidth=0.6)
    axes[1].set_title("task primary trace")
    axes[1].set_xlabel("token index")
    axes[1].set_ylabel(ylabel)
    axes[1].grid(color="#dddddd", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8)
    outputs = []
    for ext in ["pdf", "svg", "png"]:
        path = out_base.with_suffix(f".{ext}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 220
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def save_overview_gif(states_by_task: dict[str, dict[str, np.ndarray]], path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    task = "composed_objectives"
    states = states_by_task[task]
    all_xy = [state_tensors(state)["root_xy"] for state in states.values()]
    xy_cat = np.concatenate(all_xy, axis=0)
    pad = 0.03
    ax.set_xlim(float(xy_cat[:, 0].min() - pad), float(xy_cat[:, 0].max() + pad))
    ax.set_ylim(float(xy_cat[:, 1].min() - pad), float(xy_cat[:, 1].max() + pad))
    ax.add_patch(plt.Circle((0.04, 0.0), 0.25, color="#b85c38", fill=False, linewidth=1.4, linestyle="--"))
    lines = {
        name: ax.plot([], [], color=COLORS[name], linewidth=1.8, label=LABELS[name])[0]
        for name in ["clean", "unguided", "offline", "reverse"]
    }
    ax.set_title("Composed objective trajectory preview")
    ax.set_xlabel("integrated x")
    ax.set_ylabel("integrated y")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    def update(frame: int) -> list[Any]:
        for name, line in lines.items():
            xy = state_tensors(states[name])["root_xy"][: frame + 1]
            line.set_data(xy[:, 0], xy[:, 1])
        return list(lines.values())

    anim = FuncAnimation(fig, update, frames=states["clean"].shape[0], interval=120, blit=True)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    offline = load_json(OFFLINE_JSON)
    reverse = load_json(REVERSE_JSON)
    offline_npz = np.load(OFFLINE_NPZ)
    reverse_npz = np.load(REVERSE_NPZ, allow_pickle=True)
    dataset = np.load(DATASET_NPZ, allow_pickle=True)
    projection_inverse = dataset["projection_inverse"].astype(np.float64)

    selected_indices = offline_npz["selected_indices"]
    window_index = int(selected_indices[0])
    reverse_indices = reverse_npz["selected_indices"]
    if int(reverse_indices[0]) != window_index:
        raise RuntimeError("offline and reverse representative windows do not align")
    clean_state = state_from_tau(offline_npz["clean_tau"][0], projection_inverse)
    unguided_state = state_from_tau(offline_npz["unguided_pred_tau"][0], projection_inverse)

    rows: list[dict[str, Any]] = []
    outputs_by_task: dict[str, list[str]] = {}
    states_by_task: dict[str, dict[str, np.ndarray]] = {}
    for task in TASKS:
        offline_state = state_from_tau(offline_npz[f"guided_tau_{task}_max_scale"][0], projection_inverse)
        reverse_state = state_from_tau(reverse_npz[f"tau_{task}_batch0_scale_0.0002"][0], projection_inverse)
        states = {"clean": clean_state, "unguided": unguided_state, "offline": offline_state, "reverse": reverse_state}
        states_by_task[task] = states
        outputs_by_task[task] = save_task_figure(task, states, OUT / f"checkpoint_guidance_{task}")
        for mode, state in states.items():
            rows.append(
                {
                    "task": task,
                    "mode": mode,
                    "window_index": window_index,
                    "primary_metric": primary_metric(task, state),
                    "root_xy_final_x": float(state_tensors(state)["root_xy"][-1, 0]),
                    "root_xy_final_y": float(state_tensors(state)["root_xy"][-1, 1]),
                    "finite": bool(np.all(np.isfinite(state))),
                }
            )
    gif_path = save_overview_gif(states_by_task, OUT / "checkpoint_guidance_composed_preview.gif")

    json_path = OUT / "level_c_guidance_checkpoint_visualization.json"
    tsv_path = OUT / "level_c_guidance_checkpoint_visualization.tsv"
    write_tsv(tsv_path, rows)
    all_outputs = [path for paths in outputs_by_task.values() for path in paths] + [gif_path]
    checks = {
        "offline_status_ok": offline.get("status") == "ok",
        "reverse_status_ok": reverse.get("status") == "ok",
        "representative_window_alignment": int(reverse_indices[0]) == window_index,
        "five_tasks_visualized": sorted(outputs_by_task) == sorted(TASKS),
        "four_modes_per_task_recorded": len(rows) == len(TASKS) * 4,
        "all_rows_finite": all(row["finite"] for row in rows),
        "all_visual_files_written": all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in all_outputs),
        "gif_written": Path(gif_path).is_file() and Path(gif_path).stat().st_size > 0,
        "uses_full_split_checkpoint_sources": offline["row_count"] == 46200 and reverse["metrics"]["row_count"] == 33000,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_fig5_fig6_reproduction": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_guidance_checkpoint_visualization",
        "scope": "Representative qualitative plots from full-split public-data checkpoint guidance artifacts.",
        "inputs": {
            "offline_npz": str(OFFLINE_NPZ),
            "offline_json": str(OFFLINE_JSON),
            "reverse_npz": str(REVERSE_NPZ),
            "reverse_json": str(REVERSE_JSON),
            "dataset_npz": str(DATASET_NPZ),
        },
        "metrics": {
            "task_count": len(TASKS),
            "mode_count": 4,
            "row_count": len(rows),
            "visual_file_count": len(all_outputs),
            "representative_window_index": window_index,
            "offline_source_rows": int(offline["row_count"]),
            "reverse_source_rows": int(reverse["metrics"]["row_count"]),
        },
        "rows": rows,
        "checks": checks,
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "by_task": outputs_by_task, "gif": gif_path},
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "public_data_checkpoint_guidance_qualitative_preview",
            "why_not_complete": (
                "These are representative offline trajectory plots derived from full-split checkpoint guidance artifacts. "
                "They are not closed-loop Isaac/robot videos, paper Fig. 5/Fig. 6 reproduction, or TensorRT deployment evidence."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "visuals": len(all_outputs)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
