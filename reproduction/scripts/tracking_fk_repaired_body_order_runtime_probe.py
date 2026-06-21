#!/usr/bin/env python3
"""Probe whether FK-repaired motion body arrays match IsaacLab robot body order.

The official ``MotionLoader`` indexes ``body_pos_w`` with IsaacLab articulation
body indexes.  The current FK-repaired bundle was generated in URDF body order,
so a mismatch between the two orders would make a numerically plausible
``body_pos_w`` file look wrong inside ``Tracking-Flat-G1-v0``.  This probe runs
one live IsaacLab environment and records the exact runtime body indexes used by
the official command term.
"""

from __future__ import annotations

import csv
import json
import os
import select
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/fk_repaired_body_order_runtime_probe"
LOG_DIR = ROOT / "logs/tracking_fk_repaired_body_order_runtime_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz/motions/"
    "dance1_subject1/motion.npz"
)
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TARGET_GPU = int(os.environ.get("BM_FK_BODY_ORDER_GPU", "4"))
STALL_SECONDS = int(os.environ.get("BM_FK_BODY_ORDER_STALL_SECONDS", "240"))


WORKER_CODE = r"""
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
"""


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def env_vars(summary_json: Path) -> dict[str, str]:
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
            "BM_BODY_ORDER_JSON": str(summary_json),
            "BM_BODY_CONTRACT_JSON": str(BODY_CONTRACT),
            "BM_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(MOTION_NPZ),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_SEED": "20260820",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_worker(worker: Path, summary_json: Path, log_path: Path) -> dict[str, Any]:
    cmd = [str(TRACKING_PY), str(worker), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    start = time.time()
    last_size = -1
    last_change = time.time()
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env_vars(summary_json),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        while proc.poll() is None:
            ready, _, _ = select.select([proc.stdout], [], [], 5)
            if ready:
                line = proc.stdout.readline()
                if line:
                    lines.append(line)
                    log_file.write(line)
                    log_file.flush()
                    if line.startswith("BM_SENTINEL") or "Traceback" in line or "Error" in line:
                        print(line.rstrip(), flush=True)
            current_size = log_path.stat().st_size if log_path.is_file() else 0
            if current_size != last_size:
                last_size = current_size
                last_change = time.time()
            elif time.time() - last_change > STALL_SECONDS:
                log_file.write(f"\nBM_STALL_TIMEOUT:no_log_progress_for_{STALL_SECONDS}s\n")
                log_file.flush()
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)
                break
        if proc.stdout:
            rest = proc.stdout.read()
            if rest:
                lines.append(rest)
                log_file.write(rest)
                print(rest, end="", flush=True)
    text = "".join(lines)
    return {
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "log": str(log_path),
        "markers": {
            "before_app": "BM_SENTINEL:before_app" in text,
            "after_app": "BM_SENTINEL:after_app" in text,
            "env_created": "BM_SENTINEL:env_created" in text,
            "env_reset": "BM_SENTINEL:env_reset" in text,
            "probe_success": "BM_SENTINEL:probe_success" in text,
            "exception": "BM_SENTINEL:exception=" in text,
            "stall_timeout": "BM_STALL_TIMEOUT" in text,
        },
    }


def write_tables(summary: dict[str, Any]) -> None:
    rows = summary.get("rows", [])
    tsv = OUT / "fk_repaired_body_order_runtime_probe_rows.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "cfg_order_index",
            "body_name",
            "robot_index",
            "urdf_index",
            "robot_body_at_robot_index",
            "raw_body_at_robot_index",
            "raw_named_z_mean_m",
            "raw_robot_index_z_mean_m",
            "motion_loader_z_mean_m",
            "abs_named_vs_loader_z_delta_m",
            "index_matches",
            "raw_robot_index_matches_body_name",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    md = OUT / "fk_repaired_body_order_runtime_probe.md"
    checks = summary.get("checks", {})
    metrics = summary.get("metrics", {})
    lines = [
        "# FK-Repaired Body Order Runtime Probe",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "This live IsaacLab probe checks whether the FK-repaired motion arrays are ordered the way the official "
        "`MotionLoader` expects when it indexes `body_pos_w` with IsaacLab robot body indexes.",
        "",
        "## Key Findings",
        "",
        f"- Robot body order exactly matches URDF order: `{checks.get('robot_body_order_exactly_matches_urdf_order')}`",
        f"- Target robot indexes equal URDF indexes by name: `{checks.get('target_robot_indexes_equal_urdf_indexes_by_name')}`",
        f"- MotionLoader matches named FK targets: `{checks.get('motion_loader_matches_named_fk_targets')}`",
        f"- Misindexed targets present: `{checks.get('misindexed_targets_present')}`",
        f"- Max named-vs-loader z delta: `{metrics.get('max_abs_named_vs_loader_z_delta_m')}` m",
        f"- Endpoint z error after one zero-action step max: `{metrics.get('endpoint_z_error_after_one_step_max_m')}` m",
        "",
        "## Target Rows",
        "",
        "| Body | Robot index | URDF index | Raw body at robot index | Loader z mean | Named z mean | Abs z delta |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{body_name}` | {robot_index} | {urdf_index} | `{raw_body_at_robot_index}` | "
            "{motion_loader_z_mean_m:.6f} | {raw_named_z_mean_m:.6f} | {abs_named_vs_loader_z_delta_m:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is a runtime diagnostic only. It does not train PPO, collect DAgger data, reproduce Fig. 5/Fig. 6, "
            "benchmark TensorRT, or validate real robot deployment.",
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    worker = OUT / "tracking_fk_repaired_body_order_runtime_probe_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    summary_json = OUT / "fk_repaired_body_order_runtime_probe.json"
    log_path = LOG_DIR / "fk_repaired_body_order_runtime_probe.log"
    run = run_worker(worker, summary_json, log_path)
    summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.is_file() else {}
    if summary:
        summary["outputs"] = {
            "json": str(summary_json),
            "rows_tsv": str(OUT / "fk_repaired_body_order_runtime_probe_rows.tsv"),
            "markdown": str(OUT / "fk_repaired_body_order_runtime_probe.md"),
            "worker": str(worker),
            "log": str(log_path),
        }
        summary["run"] = run
        write_json(summary_json, summary)
        write_tables(summary)
    else:
        summary = {
            "status": "failed_fk_repaired_body_order_runtime_probe",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_type": "tracking_fk_repaired_body_order_runtime_probe",
            "scope": "Runtime body-order probe for FK-repaired motion arrays inside official Tracking-Flat-G1-v0.",
            "checks": {"probe_success": False, "does_not_start_training": True},
            "run": run,
            "outputs": {"json": str(summary_json), "worker": str(worker), "log": str(log_path)},
            "interpretation": {"goal_complete": False, "claim_level": "failed_runtime_diagnostic"},
        }
        write_json(summary_json, summary)
    print(json.dumps({"status": summary["status"], "json": str(summary_json)}, sort_keys=True))
    if summary["status"] != "ok_fk_repaired_body_order_runtime_probe":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
