#!/usr/bin/env python3
"""Debug-only probe for BeyondMimic guidance cost formulas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # bm_tracking intentionally stays lean; plots are optional for this autograd probe.
    plt = None


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/guidance_formula_probe"


def relaxed_barrier(x: torch.Tensor, delta: float) -> torch.Tensor:
    delta_t = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
    safe = -torch.log(torch.clamp(x, min=1e-12))
    relaxed = -torch.log(delta_t) + 0.5 * (((x - 2.0 * delta_t) / delta_t) ** 2 - 1.0)
    return torch.where(x >= delta_t, safe, relaxed)


def extract_future(windows: np.ndarray, feature_slices: dict[str, list[int]], history: int) -> dict[str, torch.Tensor]:
    # Use the first window and the current+future range i=0..H from the paper equations.
    future = torch.tensor(windows[0, history:], dtype=torch.float64, requires_grad=True)

    def sl(name: str) -> slice:
        lo, hi = feature_slices[name]
        return slice(lo, hi)

    body_pos = future[:, sl("body_pos_yaw_frame")].reshape(future.shape[0], 14, 3)
    root_vel = future[:, sl("root_lin_vel_yaw_frame")]
    # The candidate yaw-centric fixture removes global XY; approximate future planar root position by integrating
    # root planar velocity over the 25Hz diffusion-controller interval used by the paper.
    root_xy = torch.cumsum(root_vel[:, :2] * 0.04, dim=0)
    return {
        "future": future,
        "root_xy": root_xy,
        "root_vel_xy": root_vel[:, :2],
        "body_pos_xy": body_pos[:, :, :2],
    }


def joystick_cost(root_vel_xy: torch.Tensor, command_velocity: torch.Tensor) -> torch.Tensor:
    # G_js = 1/2 sum_i ||V_xy,i - g_v||^2
    return 0.5 * torch.sum((root_vel_xy - command_velocity) ** 2)


def waypoint_cost(root_xy: torch.Tensor, root_vel_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    # G_wp = sum_i (1-exp(-2d_i))||P_xy,i-g_p||^2 + exp(-2d_i)||V_xy,i||^2
    delta = root_xy - goal_xy
    distance = torch.linalg.norm(delta, dim=-1)
    weight_pos = 1.0 - torch.exp(-2.0 * distance)
    weight_vel = torch.exp(-2.0 * distance)
    return torch.sum(weight_pos * torch.sum(delta**2, dim=-1) + weight_vel * torch.sum(root_vel_xy**2, dim=-1))


def sdf_obstacle_cost(
    body_pos_xy: torch.Tensor,
    obstacle_center_xy: torch.Tensor,
    obstacle_radius: float,
    body_radius: float,
    delta: float,
) -> torch.Tensor:
    # Circle SDF probe: SDF(P_b,i) = ||P_b,i - center|| - obstacle_radius.
    sdf = torch.linalg.norm(body_pos_xy - obstacle_center_xy, dim=-1) - obstacle_radius
    return torch.sum(relaxed_barrier(sdf - body_radius, delta))


def keyframe_candidate_cost(root_xy: torch.Tensor, keyframe_xy: torch.Tensor, target_index: int) -> torch.Tensor:
    # The paper demonstrates inpainting with sparse future keyframes, but does not provide a unique formula in S3.
    # This candidate term is kept separate from the paper-exact joystick/waypoint/SDF formulas.
    return 0.5 * torch.sum((root_xy[target_index] - keyframe_xy) ** 2)


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


def save_plot(path_base: Path, root_xy: np.ndarray, goal: np.ndarray, obstacle: np.ndarray, radius: float) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(root_xy[:, 0], root_xy[:, 1], marker="o", markersize=3, label="integrated root XY")
    ax.scatter([goal[0]], [goal[1]], marker="*", s=160, label="waypoint")
    circle = plt.Circle((obstacle[0], obstacle[1]), radius, color="tab:red", fill=False, linewidth=2, label="SDF obstacle")
    ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Guidance Formula Probe Geometry")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-npz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--obstacle-radius", type=float, default=0.2)
    parser.add_argument("--body-radius", type=float, default=0.05)
    parser.add_argument("--barrier-delta", type=float, default=0.1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fixture = np.load(args.fixture_npz)
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    tensors = extract_future(fixture["candidate_hybrid_state_windows"], manifest["feature_slices"], args.history)

    command_velocity = torch.tensor([0.35, 0.0], dtype=torch.float64)
    goal_xy = torch.tensor([0.08, 0.04], dtype=torch.float64)
    obstacle_center = torch.tensor([0.04, 0.0], dtype=torch.float64)
    keyframe_xy = torch.tensor([0.04, -0.02], dtype=torch.float64)

    costs = {
        "joystick_velocity_formula": joystick_cost(tensors["root_vel_xy"], command_velocity),
        "waypoint_formula": waypoint_cost(tensors["root_xy"], tensors["root_vel_xy"], goal_xy),
        "sdf_obstacle_barrier_formula": sdf_obstacle_cost(
            tensors["body_pos_xy"],
            obstacle_center,
            args.obstacle_radius,
            args.body_radius,
            args.barrier_delta,
        ),
        "keyframe_inpainting_candidate": keyframe_candidate_cost(
            tensors["root_xy"], keyframe_xy, target_index=min(5, tensors["root_xy"].shape[0] - 1)
        ),
    }
    composed = costs["joystick_velocity_formula"] + costs["waypoint_formula"] + costs["sdf_obstacle_barrier_formula"]
    composed_with_keyframe = composed + costs["keyframe_inpainting_candidate"]

    grad_norms: dict[str, float] = {}
    for name, cost in {**costs, "composed_paper_formula_terms": composed, "composed_with_keyframe_candidate": composed_with_keyframe}.items():
        if tensors["future"].grad is not None:
            tensors["future"].grad.zero_()
        cost.backward(retain_graph=True)
        grad_norms[name] = float(tensors["future"].grad.norm().detach().cpu())

    root_xy_np = tensors["root_xy"].detach().cpu().numpy()
    goal_np = goal_xy.detach().cpu().numpy()
    obstacle_np = obstacle_center.detach().cpu().numpy()
    plot_base = OUT / "guidance_formula_probe_geometry"
    save_plot(plot_base, root_xy_np, goal_np, obstacle_np, args.obstacle_radius)

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "autograd probe for paper guidance cost formulas on the motion-derived fixture",
        "paper_evidence": {
            "classifier_guidance_gradient": str(ROOT / "reproduction/paper/source/tex/method.tex:212-226"),
            "joystick_waypoint_sdf_formulas": str(ROOT / "reproduction/paper/source/root.tex:548-586"),
            "goal_required_keyframe_inpainting": str(ROOT / "goal.md:1290-1355"),
        },
        "not_a_replacement_for": [
            "diffusion denoising-loop guidance",
            "guidance-scale sweep",
            "validation/test scene split",
            "guided rollout metrics",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "fixture": {"npz": str(args.fixture_npz), "manifest": str(args.manifest_json)},
        "settings": {
            "history": args.history,
            "future_steps_i_0_to_H": int(tensors["root_xy"].shape[0]),
            "command_velocity_xy": command_velocity.tolist(),
            "goal_xy": goal_xy.tolist(),
            "obstacle_center_xy": obstacle_center.tolist(),
            "obstacle_radius": args.obstacle_radius,
            "body_radius": args.body_radius,
            "barrier_delta": args.barrier_delta,
            "keyframe_xy_candidate": keyframe_xy.tolist(),
        },
        "costs": {key: float(value.detach().cpu()) for key, value in costs.items()}
        | {
            "composed_paper_formula_terms": float(composed.detach().cpu()),
            "composed_with_keyframe_candidate": float(composed_with_keyframe.detach().cpu()),
        },
        "gradient_norms": grad_norms,
        "shapes": {
            "future_state": list(tensors["future"].shape),
            "root_xy": list(tensors["root_xy"].shape),
            "root_vel_xy": list(tensors["root_vel_xy"].shape),
            "body_pos_xy": list(tensors["body_pos_xy"].shape),
        },
        "checks": {
            "finite_costs": bool(all(np.isfinite(v) for v in [float(c.detach().cpu()) for c in costs.values()])),
            "finite_gradients": bool(all(np.isfinite(v) for v in grad_norms.values())),
            "all_formula_gradients_nonzero": bool(
                all(grad_norms[k] > 0.0 for k in ["joystick_velocity_formula", "waypoint_formula", "sdf_obstacle_barrier_formula"])
            ),
            "composed_gradient_nonzero": bool(grad_norms["composed_paper_formula_terms"] > 0.0),
            "keyframe_candidate_gradient_nonzero": bool(grad_norms["keyframe_inpainting_candidate"] > 0.0),
        },
        "outputs": {
            "json": str(OUT / "level_c_guidance_formula_probe.json"),
            "tsv": str(OUT / "level_c_guidance_formula_probe.tsv"),
            "geometry_plot_base": str(plot_base),
            "geometry_plot_written": bool(plt is not None),
        },
    }
    json_path = OUT / "level_c_guidance_formula_probe.json"
    tsv_path = OUT / "level_c_guidance_formula_probe.tsv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
