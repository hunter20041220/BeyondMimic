
import json
import math
import os
from pathlib import Path

OUT = Path(os.environ["BM_BODY_ORDER_JSON"])
BODY_CONTRACT = Path(os.environ["BM_BODY_CONTRACT_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
target_gpu = os.environ.get("BM_TARGET_GPU", "4")
args.headless = True
args.enable_cameras = False
args.device = os.environ.get("BM_DEVICE", f"cuda:{target_gpu}")
args.multi_gpu = False
args.kit_args = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 "
    f"--/renderer/activeGpu={target_gpu} "
    f"--/physics/cudaDevice={target_gpu}"
)

print("BM_SENTINEL:before_app", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def f32(value):
        value = float(value)
        return value if math.isfinite(value) else None

    with BODY_CONTRACT.open("r", encoding="utf-8") as f:
        body_contract = json.load(f)
    urdf_order = list(body_contract["body_names_urdf_order"])
    tracking_body_names = list(body_contract["tracking_body_names"])
    raw = np.load(MOTION_FILE)
    raw_body_pos = raw["body_pos_w"]

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
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
    env_cfg.commands.motion.motion_file = str(MOTION_FILE)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = 0.24
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260820"))

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)
    command = env.unwrapped.command_manager.get_term("motion")
    robot = env.unwrapped.scene["robot"]

    robot_body_names = list(robot.body_names)
    cfg_body_names = list(command.cfg.body_names)
    body_indexes = [int(x) for x in command.body_indexes.detach().cpu().tolist()]
    loader_indexes = [int(x) for x in command.motion._body_indexes.detach().cpu().tolist()]
    indexed_motion_body_pos = command.motion.body_pos_w.detach().cpu().numpy()

    rows = []
    z_deltas = []
    for cfg_i, body_name in enumerate(cfg_body_names):
        robot_index = body_indexes[cfg_i]
        urdf_index = urdf_order.index(body_name)
        raw_body_at_robot_index = urdf_order[robot_index] if 0 <= robot_index < len(urdf_order) else "<out_of_range>"
        loader_z_mean = float(indexed_motion_body_pos[:, cfg_i, 2].mean())
        named_z_mean = float(raw_body_pos[:, urdf_index, 2].mean())
        robot_index_z_mean = float(raw_body_pos[:, robot_index, 2].mean())
        z_delta = abs(loader_z_mean - named_z_mean)
        z_deltas.append(z_delta)
        rows.append(
            {
                "cfg_order_index": cfg_i,
                "body_name": body_name,
                "robot_index": robot_index,
                "urdf_index": urdf_index,
                "robot_body_at_robot_index": robot_body_names[robot_index],
                "raw_body_at_robot_index": raw_body_at_robot_index,
                "raw_named_z_mean_m": named_z_mean,
                "raw_robot_index_z_mean_m": robot_index_z_mean,
                "motion_loader_z_mean_m": loader_z_mean,
                "abs_named_vs_loader_z_delta_m": z_delta,
                "index_matches": robot_index == urdf_index,
                "raw_robot_index_matches_body_name": raw_body_at_robot_index == body_name,
            }
        )

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    obs, reward, terminated, truncated, extras = env.step(action)
    endpoint_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    endpoint_indexes = [cfg_body_names.index(name) for name in endpoint_names]
    endpoint_z_error = torch.abs(
        command.body_pos_relative_w[:, endpoint_indexes, -1] - command.robot_body_pos_w[:, endpoint_indexes, -1]
    )
    endpoint_z_error_cpu = endpoint_z_error.detach().cpu().numpy()[0]
    endpoint_rows = []
    for name, value in zip(endpoint_names, endpoint_z_error_cpu):
        endpoint_rows.append({"body_name": name, "z_error_after_one_zero_action_step_m": f32(value)})

    summary = {
        "status": "ok_fk_repaired_body_order_runtime_probe",
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "experiment_type": "tracking_fk_repaired_body_order_runtime_probe",
        "scope": "Runtime body-order probe for FK-repaired motion arrays inside official Tracking-Flat-G1-v0.",
        "config": {
            "device": str(env.unwrapped.device),
            "target_gpu": int(target_gpu),
            "motion_npz": str(MOTION_FILE),
            "robot_usd": str(ROBOT_USD),
            "body_contract": str(BODY_CONTRACT),
            "action_dim": action_dim,
            "robot_num_bodies": int(robot.num_bodies),
            "robot_num_joints": int(robot.num_joints),
        },
        "checks": {
            "body_contract_exists": BODY_CONTRACT.is_file(),
            "motion_npz_exists": MOTION_FILE.is_file(),
            "official_importer_usd_exists": ROBOT_USD.is_file(),
            "robot_body_count_40": len(robot_body_names) == 40,
            "urdf_body_count_40": len(urdf_order) == 40,
            "cfg_body_count_14": len(cfg_body_names) == 14,
            "tracking_body_names_match_cfg": tracking_body_names == cfg_body_names,
            "motion_loader_indexes_equal_command_body_indexes": loader_indexes == body_indexes,
            "robot_body_order_exactly_matches_urdf_order": robot_body_names == urdf_order,
            "target_robot_indexes_equal_urdf_indexes_by_name": all(row["index_matches"] for row in rows),
            "motion_loader_matches_named_fk_targets": max(z_deltas) < 1e-6,
            "misindexed_targets_present": any(not row["raw_robot_index_matches_body_name"] for row in rows),
            "endpoint_z_error_gt_threshold_after_one_step": bool(np.max(endpoint_z_error_cpu) > 0.25),
            "does_not_start_training": True,
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_real_robot": True,
        },
        "metrics": {
            "max_abs_named_vs_loader_z_delta_m": f32(max(z_deltas)),
            "mean_abs_named_vs_loader_z_delta_m": f32(sum(z_deltas) / len(z_deltas)),
            "endpoint_z_error_after_one_step_max_m": f32(np.max(endpoint_z_error_cpu)),
            "endpoint_z_error_after_one_step_mean_m": f32(np.mean(endpoint_z_error_cpu)),
            "terminated_after_one_zero_action_step": int(terminated.detach().cpu().sum().item()),
            "truncated_after_one_zero_action_step": int(truncated.detach().cpu().sum().item()),
            "reward_after_one_zero_action_step": f32(reward.detach().cpu().mean().item()),
        },
        "robot_body_names": robot_body_names,
        "urdf_body_names": urdf_order,
        "cfg_body_names": cfg_body_names,
        "command_body_indexes": body_indexes,
        "motion_loader_indexes": loader_indexes,
        "rows": rows,
        "endpoint_rows": endpoint_rows,
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_runtime_body_order_diagnostic_not_paper_level",
            "main_finding": (
                "If robot_body_order_exactly_matches_urdf_order is false and motion_loader_matches_named_fk_targets "
                "is false, the FK-repaired motion arrays are plausibly written in the wrong full-body order for the "
                "official MotionLoader indexing contract."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BM_SENTINEL:probe_success", flush=True)
    env.close()
    simulation_app.close(wait_for_replicator=False)
    os._exit(0)
except BaseException as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    raise
