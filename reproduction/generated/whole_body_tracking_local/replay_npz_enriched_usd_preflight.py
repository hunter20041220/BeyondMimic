"""Bounded replay preflight for the resource-adjusted enriched G1 USD scaffold.

This is not an official BeyondMimic replay script. It replaces the official
URDF converter path with a local UsdFileCfg so the current IsaacLab/Kit runtime
can be tested against the generated enriched USD scaffold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)


parser = argparse.ArgumentParser(description="Bounded enriched-USD replay preflight.")
parser.add_argument("--motion_file", type=str, required=True, help="Local debug/fixture motion .npz path.")
parser.add_argument("--usd_path", type=str, default=str(DEFAULT_USD), help="Resource-adjusted enriched G1 USD path.")
parser.add_argument("--max_steps", type=int, default=8, help="Maximum render steps for the bounded gate.")
parser.add_argument("--metrics_file", type=str, default="", help="Optional JSON path for bounded replay metrics.")
parser.add_argument(
    "--exit_after_success",
    action="store_true",
    help="Exit the process immediately after the success sentinel to avoid known Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("BM_SENTINEL:before_app", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("BM_SENTINEL:after_app", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class EnrichedReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = G1_CYLINDER_CFG.copy()

    def __post_init__(self) -> None:
        self.robot = G1_CYLINDER_CFG.copy()
        self.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.robot.spawn = sim_utils.UsdFileCfg(
            usd_path=args_cli.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        )


def main() -> None:
    motion_path = Path(args_cli.motion_file)
    usd_path = Path(args_cli.usd_path)
    print(f"BM_SENTINEL:motion_file={motion_path}", flush=True)
    print(f"BM_SENTINEL:usd_path={usd_path}", flush=True)
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    motion = np.load(motion_path)
    joint_pos_np = motion["joint_pos"]
    joint_vel_np = motion["joint_vel"]
    body_pos_w_np = motion["body_pos_w"]
    body_quat_w_np = motion["body_quat_w"]
    body_lin_vel_w_np = motion["body_lin_vel_w"]
    body_ang_vel_w_np = motion["body_ang_vel_w"]
    print(
        "BM_SENTINEL:motion_contract="
        f"joint_pos{joint_pos_np.shape},joint_vel{joint_vel_np.shape},fps{motion['fps'].tolist()}",
        flush=True,
    )

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    print("BM_SENTINEL:sim_created", flush=True)

    scene_cfg = EnrichedReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print("BM_SENTINEL:scene_created", flush=True)
    sim.reset()
    print("BM_SENTINEL:sim_reset", flush=True)

    robot: Articulation = scene["robot"]
    print(
        "BM_SENTINEL:robot_contract="
        f"num_joints={robot.num_joints},num_bodies={robot.num_bodies},device={robot.device}",
        flush=True,
    )

    if robot.num_joints != joint_pos_np.shape[1]:
        raise RuntimeError(f"joint count mismatch: robot={robot.num_joints}, motion={joint_pos_np.shape[1]}")

    joint_pos = torch.as_tensor(joint_pos_np, dtype=torch.float32, device=sim.device)
    joint_vel = torch.as_tensor(joint_vel_np, dtype=torch.float32, device=sim.device)
    body_pos_w = torch.as_tensor(body_pos_w_np, dtype=torch.float32, device=sim.device)
    body_quat_w = torch.as_tensor(body_quat_w_np, dtype=torch.float32, device=sim.device)
    body_lin_vel_w = torch.as_tensor(body_lin_vel_w_np, dtype=torch.float32, device=sim.device)
    body_ang_vel_w = torch.as_tensor(body_ang_vel_w_np, dtype=torch.float32, device=sim.device)
    max_steps = min(int(args_cli.max_steps), int(joint_pos.shape[0]))
    joint_pos_abs_errors = []
    joint_vel_abs_errors = []
    root_pos_abs_errors = []
    root_quat_abs_errors = []
    for step in range(max_steps):
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = body_pos_w[step : step + 1, 0]
        root_states[:, 3:7] = body_quat_w[step : step + 1, 0]
        root_states[:, 7:10] = body_lin_vel_w[step : step + 1, 0]
        root_states[:, 10:] = body_ang_vel_w[step : step + 1, 0]
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos[step : step + 1], joint_vel[step : step + 1])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())
        joint_pos_abs_errors.append(float(torch.max(torch.abs(robot.data.joint_pos[0] - joint_pos[step])).item()))
        joint_vel_abs_errors.append(float(torch.max(torch.abs(robot.data.joint_vel[0] - joint_vel[step])).item()))
        root_pos_abs_errors.append(float(torch.max(torch.abs(robot.data.root_pos_w[0] - body_pos_w[step, 0])).item()))
        root_quat_abs_errors.append(float(torch.max(torch.abs(robot.data.root_quat_w[0] - body_quat_w[step, 0])).item()))
        print(f"BM_SENTINEL:step={step + 1}", flush=True)

    metrics = {
        "motion_file": str(motion_path),
        "usd_path": str(usd_path),
        "device": str(sim.device),
        "requested_steps": int(args_cli.max_steps),
        "executed_steps": int(max_steps),
        "motion_total_steps": int(joint_pos.shape[0]),
        "fps": float(motion["fps"][0]),
        "robot_num_joints": int(robot.num_joints),
        "robot_num_bodies": int(robot.num_bodies),
        "joint_pos_shape": list(joint_pos_np.shape),
        "joint_vel_shape": list(joint_vel_np.shape),
        "body_pos_w_shape": list(body_pos_w_np.shape),
        "max_joint_pos_abs_error": max(joint_pos_abs_errors) if joint_pos_abs_errors else None,
        "max_joint_vel_abs_error": max(joint_vel_abs_errors) if joint_vel_abs_errors else None,
        "max_root_pos_abs_error": max(root_pos_abs_errors) if root_pos_abs_errors else None,
        "max_root_quat_abs_error": max(root_quat_abs_errors) if root_quat_abs_errors else None,
        "root_height_min": float(body_pos_w[:max_steps, 0, 2].min().item()) if max_steps else None,
        "root_height_max": float(body_pos_w[:max_steps, 0, 2].max().item()) if max_steps else None,
        "uses_resource_adjusted_usd": True,
        "official_csv_to_npz_output": False,
        "paper_level_rollout": False,
    }
    if args_cli.metrics_file:
        metrics_path = Path(args_cli.metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"BM_SENTINEL:metrics_file={metrics_path}", flush=True)

    print("BM_SENTINEL:enriched_usd_replay_preflight_success", flush=True)
    if args_cli.exit_after_success:
        print("BM_SENTINEL:explicit_exit_after_success", flush=True)
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close(wait_for_replicator=False)
            print("BM_SENTINEL:after_close", flush=True)
        except TypeError:
            simulation_app.close()
            print("BM_SENTINEL:after_close", flush=True)
