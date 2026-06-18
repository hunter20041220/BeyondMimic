#!/usr/bin/env python3
"""Convert an official G1 LAFAN CSV through the resource-adjusted enriched USD gate."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_csv_conversion"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_csv_conversion"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
INPUT_CSV = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"
OUTPUT_NPZ = OUT / "walk1_subject1_frames_1_180_resource_adjusted_motion.npz"
CONTRACT_JSON = OUT / "walk1_subject1_frames_1_180_resource_adjusted_motion_contract.json"
METRICS_JSON = OUT / "tracking_g1_resource_adjusted_csv_conversion_metrics.json"
FULL_FIXTURE_GATE = (
    ROOT
    / "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
    "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json"
)
STALL_SECONDS = 900


PROBE_CODE = r"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Resource-adjusted CSV to motion.npz conversion.")
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--input_fps", type=int, default=30)
parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"))
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_fps", type=int, default=50)
parser.add_argument("--usd_path", type=str, required=True)
parser.add_argument("--contract_file", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("BM_SENTINEL:before_app", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("BM_SENTINEL:after_app", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

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


@configclass
class EnrichedCsvConversionSceneCfg(InteractiveSceneCfg):
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


class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = int(motion.shape[0])
        self.input_columns = int(motion.shape[1])
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"BM_SENTINEL:csv_loaded=frames:{self.input_frames},columns:{self.input_columns},duration:{self.duration}",
            flush=True,
        )

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = int(times.shape[0])
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(f"BM_SENTINEL:interpolated=output_frames:{self.output_frames}", flush=True)

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return state, reset_flag


def main():
    input_path = Path(args_cli.input_file)
    output_path = Path(args_cli.output_file)
    contract_path = Path(args_cli.contract_file)
    usd_path = Path(args_cli.usd_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    print("BM_SENTINEL:sim_created", flush=True)
    scene = InteractiveScene(EnrichedCsvConversionSceneCfg(num_envs=1, env_spacing=2.0))
    print("BM_SENTINEL:scene_created", flush=True)
    sim.reset()
    print("BM_SENTINEL:sim_reset", flush=True)

    motion = MotionLoader(
        motion_file=str(input_path),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )
    robot = scene["robot"]
    joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
    print(
        "BM_SENTINEL:robot_contract="
        f"num_joints={robot.num_joints},num_bodies={robot.num_bodies},joint_indexes={len(joint_indexes)}",
        flush=True,
    )

    log = {k: [] for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]}
    for step in range(motion.output_frames):
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), reset_flag = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indexes] = motion_dof_pos
        joint_vel[:, joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
        if (step + 1) % 50 == 0 or (step + 1) == motion.output_frames:
            print(f"BM_SENTINEL:conversion_step={step + 1}/{motion.output_frames}", flush=True)
        if reset_flag and step + 1 != motion.output_frames:
            raise RuntimeError("unexpected early reset")

    arrays = {"fps": np.asarray([args_cli.output_fps], dtype=np.float32)}
    for key, value in log.items():
        arrays[key] = np.stack(value, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)

    contract = {
        "input_csv": str(input_path),
        "output_npz": str(output_path),
        "usd_path": str(usd_path),
        "input_fps": int(args_cli.input_fps),
        "output_fps": int(args_cli.output_fps),
        "frame_range": [int(x) for x in args_cli.frame_range] if args_cli.frame_range else None,
        "input_frames": int(motion.input_frames),
        "input_columns": int(motion.input_columns),
        "output_frames": int(motion.output_frames),
        "robot_num_joints": int(robot.num_joints),
        "robot_num_bodies": int(robot.num_bodies),
        "joint_index_count": int(len(joint_indexes)),
        "joint_pos_shape": list(arrays["joint_pos"].shape),
        "joint_vel_shape": list(arrays["joint_vel"].shape),
        "body_pos_w_shape": list(arrays["body_pos_w"].shape),
        "body_quat_w_shape": list(arrays["body_quat_w"].shape),
        "uses_resource_adjusted_usd": True,
        "official_csv_to_npz_output": False,
        "paper_level_rollout": False,
        "ppo_training": False,
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:motion_saved={output_path}", flush=True)
    print(f"BM_SENTINEL:contract_file={contract_path}", flush=True)
    print("BM_SENTINEL:resource_adjusted_csv_conversion_success", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close(wait_for_replicator=False)
            print("BM_SENTINEL:after_close", flush=True)
        except Exception:
            pass
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "csv_loaded": "bm_sentinel:csv_loaded=" in lowered,
        "interpolated": "bm_sentinel:interpolated=" in lowered,
        "robot_contract": "bm_sentinel:robot_contract=" in lowered,
        "step_299": "bm_sentinel:conversion_step=299/299" in lowered,
        "motion_saved": "bm_sentinel:motion_saved=" in lowered,
        "contract_file": "bm_sentinel:contract_file=" in lowered,
        "success": "bm_sentinel:resource_adjusted_csv_conversion_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "exception" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "stall_timeout": "bm_stall_timeout" in lowered,
    }


def env_vars() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_probe(script_path: Path, log_path: Path) -> dict[str, Any]:
    command = [
        str(TRACKING_PY),
        str(script_path),
        "--input_file",
        str(INPUT_CSV),
        "--input_fps",
        "30",
        "--frame_range",
        "1",
        "180",
        "--output_file",
        str(OUTPUT_NPZ),
        "--output_fps",
        "50",
        "--usd_path",
        str(ENRICHED_USD),
        "--contract_file",
        str(CONTRACT_JSON),
        "--headless",
        "--device",
        "cuda:6",
    ]
    start = time.time()
    last_size = -1
    last_change = time.time()
    stalled = False
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env_vars(),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            time.sleep(10)
            current_size = log_path.stat().st_size if log_path.is_file() else 0
            if current_size != last_size:
                last_size = current_size
                last_change = time.time()
            elif time.time() - last_change > STALL_SECONDS:
                stalled = True
                log_file.write(f"\nBM_STALL_TIMEOUT:no_log_progress_for_{STALL_SECONDS}s\n")
                log_file.flush()
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)
                break
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return {
        "command": command,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "markers": classify(text),
        "log": str(log_path),
    }


def compute_metrics(contract: dict[str, Any]) -> dict[str, Any]:
    if not OUTPUT_NPZ.is_file():
        return {}
    data = dict(np.load(OUTPUT_NPZ))
    metrics = {
        "input_csv": str(INPUT_CSV),
        "output_npz": str(OUTPUT_NPZ),
        "contract_json": str(CONTRACT_JSON),
        "usd_path": str(ENRICHED_USD),
        "fps": data["fps"].astype(float).tolist(),
        "joint_pos_shape": list(data["joint_pos"].shape),
        "joint_vel_shape": list(data["joint_vel"].shape),
        "body_pos_w_shape": list(data["body_pos_w"].shape),
        "body_quat_w_shape": list(data["body_quat_w"].shape),
        "body_lin_vel_w_shape": list(data["body_lin_vel_w"].shape),
        "body_ang_vel_w_shape": list(data["body_ang_vel_w"].shape),
        "max_joint_abs": float(np.max(np.abs(data["joint_pos"]))),
        "max_joint_vel_abs": float(np.max(np.abs(data["joint_vel"]))),
        "root_height_min": float(np.min(data["body_pos_w"][:, 0, 2])),
        "root_height_max": float(np.max(data["body_pos_w"][:, 0, 2])),
        "max_body_quat_norm_abs_error_from_1": float(
            np.max(np.abs(np.linalg.norm(data["body_quat_w"], axis=-1) - 1.0))
        ),
        "npz_size_bytes": OUTPUT_NPZ.stat().st_size,
        "contract": contract,
        "uses_resource_adjusted_usd": True,
        "official_csv_to_npz_output": False,
        "paper_level_rollout": False,
        "ppo_training": False,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT / "tracking_g1_resource_adjusted_csv_conversion_probe.py"
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_csv_conversion.log"
    for path in [OUTPUT_NPZ, CONTRACT_JSON, METRICS_JSON]:
        if path.exists():
            path.unlink()
    script_path.write_text(PROBE_CODE, encoding="utf-8")
    run = run_probe(script_path, log_path)
    contract = load_json(CONTRACT_JSON)
    metrics = compute_metrics(contract)
    full_fixture_gate = load_json(FULL_FIXTURE_GATE)
    checks = {
        "probe_script_written": script_path.is_file(),
        "input_csv_exists": INPUT_CSV.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "prior_full_fixture_gate_passed": full_fixture_gate.get("status") == "ok_resource_adjusted_multi_fixture_task_eval",
        "process_returned_zero": run["returncode"] == 0,
        "no_stall_timeout": run["stalled"] is False,
        "app_reached": run["markers"]["after_app"],
        "csv_loaded": run["markers"]["csv_loaded"],
        "interpolation_ran": run["markers"]["interpolated"],
        "robot_contract_seen": run["markers"]["robot_contract"],
        "step_299_reached": run["markers"]["step_299"],
        "success_sentinel_seen": run["markers"]["success"],
        "motion_npz_written": OUTPUT_NPZ.is_file(),
        "contract_json_written": CONTRACT_JSON.is_file(),
        "metrics_json_written": METRICS_JSON.is_file(),
        "input_frames_180": contract.get("input_frames") == 180,
        "input_columns_36": contract.get("input_columns") == 36,
        "output_frames_299": contract.get("output_frames") == 299,
        "robot_joints_29": contract.get("robot_num_joints") == 29,
        "robot_bodies_40": contract.get("robot_num_bodies") == 40,
        "joint_index_count_29": contract.get("joint_index_count") == 29,
        "joint_pos_shape_299_29": metrics.get("joint_pos_shape") == [299, 29],
        "body_pos_shape_299_40_3": metrics.get("body_pos_w_shape") == [299, 40, 3],
        "body_quaternions_unit": metrics.get("max_body_quat_norm_abs_error_from_1", 1.0) < 1e-4,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
    }
    passed = all(checks.values())
    if passed:
        status = "ok_resource_adjusted_csv_conversion"
        latest_blocker = "none_resource_adjusted_csv_conversion_passed"
    elif run["markers"]["vulkan_device_lost"]:
        status = "ok_with_resource_adjusted_csv_conversion_blocker"
        latest_blocker = "vulkan_device_lost"
    elif run["stalled"]:
        status = "ok_with_resource_adjusted_csv_conversion_blocker"
        latest_blocker = "stall_timeout"
    elif run["markers"]["traceback"]:
        status = "ok_with_resource_adjusted_csv_conversion_blocker"
        latest_blocker = "python_traceback"
    else:
        status = "ok_with_resource_adjusted_csv_conversion_blocker"
        latest_blocker = "failed_checks"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_csv_conversion",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full conversion of one official G1 LAFAN CSV segment through the official csv interpolation/state logging "
            "logic and the generated enriched G1 USD. This narrows the replay blocker but is not official "
            "csv_to_npz.py output, not official replay/evaluation, not PPO, and not paper-level evidence."
        ),
        "latest_blocker": latest_blocker,
        "run": run,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "input_csv": str(INPUT_CSV),
            "enriched_usd": str(ENRICHED_USD),
            "prior_full_fixture_gate": str(FULL_FIXTURE_GATE),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_csv_conversion_audit.json"),
            "metrics_json": str(METRICS_JSON),
            "contract_json": str(CONTRACT_JSON),
            "motion_npz": str(OUTPUT_NPZ),
            "probe_script": str(script_path),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "official_csv_to_npz_complete": False,
            "official_replay_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "The conversion uses an official CSV and the official interpolation/logging schema, but replaces the "
                "official URDF converter path with a generated resource-adjusted enriched USD. It is suitable as a "
                "diagnostic conversion artifact, not as official paper-level motion replay evidence."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_csv_conversion_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "latest_blocker": latest_blocker}, sort_keys=True))
    if status.endswith("_blocker") and not OUTPUT_NPZ.is_file():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
