"""Bounded replay preflight for the resource-adjusted enriched G1 USD scaffold.

This is not an official BeyondMimic replay script. It replaces the official
URDF converter path with a local UsdFileCfg so the current IsaacLab/Kit runtime
can be tested against the generated enriched USD scaffold.
"""

from __future__ import annotations

import argparse
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
    max_steps = min(int(args_cli.max_steps), int(joint_pos.shape[0]))
    for step in range(max_steps):
        robot.write_joint_state_to_sim(joint_pos[step : step + 1], joint_vel[step : step + 1])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())
        print(f"BM_SENTINEL:step={step + 1}", flush=True)

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
