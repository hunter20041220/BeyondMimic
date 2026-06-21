#!/usr/bin/env python3
"""Evaluate the robot-order FK PPO checkpoint with reset command warmup.

This is the full-eval follow-up to
``robot_order_fk_reset_command_warmup_live_probe.py``. It keeps the same
checkpoint, official-importer USDA, and robot-order FK-repaired 40-motion
bundle as the prior robot-order eval, but injects one command-manager compute
after the RSL-RL wrapper's initial reset before the first policy observation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup"
RUN_ROOT = (
    ROOT / "res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
)
OLD_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
WARMUP_PROBE_JSON = (
    ROOT
    / "res/tracking/robot_order_fk_reset_command_warmup_live_probe/"
    "robot_order_fk_reset_command_warmup_live_probe.json"
)
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
ROBOT_ORDER_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
ROBOT_ORDER_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
ROBOT_ORDER_SPLIT_TASK_GATE = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json"
)
TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260741
DEFAULT_NUM_ENVS = 2048


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_eval_base_warmup", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_worker_code(worker_code: str) -> str:
    old = """    env = gym.make(\"Tracking-Flat-G1-v0\", cfg=env_cfg, render_mode=None)
    print(f\"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}\", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term(\"motion\")
"""
    new = """    env = gym.make(\"Tracking-Flat-G1-v0\", cfg=env_cfg, render_mode=None)
    print(f\"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}\", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term(\"motion\")

    endpoint_names = [
        \"left_ankle_roll_link\",
        \"right_ankle_roll_link\",
        \"left_wrist_yaw_link\",
        \"right_wrist_yaw_link\",
    ]

    def stats_tensor(tensor):
        if tensor.numel() == 0:
            return {\"count\": 0, \"mean\": None, \"min\": None, \"max\": None}
        flat = tensor.detach().float().reshape(-1).cpu()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {\"count\": 0, \"mean\": None, \"min\": None, \"max\": None}
        return {
            \"count\": int(finite.numel()),
            \"mean\": float(finite.mean().item()),
            \"min\": float(finite.min().item()),
            \"max\": float(finite.max().item()),
        }

    def warmup_snapshot(label):
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in endpoint_names if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        return {
            \"label\": label,
            \"endpoint_names_present\": [name for name in endpoint_names if name in body_names],
            \"endpoint_indexes\": endpoint_indexes,
            \"manual_endpoint_z_done_count\": int(endpoint_done.detach().cpu().sum().item()),
            \"manual_endpoint_z_done_rate\": float(endpoint_done.float().mean().detach().cpu().item()),
            \"endpoint_z_error_m\": stats_tensor(endpoint_z_error),
            \"body_error_m\": stats_tensor(body_error),
            \"body_pos_relative_abs_max\": float(body_target.abs().max().detach().cpu().item()),
            \"time_steps\": stats_tensor(command.time_steps.detach().float()),
        }

    reset_command_warmup_before = warmup_snapshot(\"after_wrapper_reset_before_command_warmup\")
    vec_env.unwrapped.command_manager.compute(dt=vec_env.unwrapped.step_dt)
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    reset_command_warmup_after = warmup_snapshot(\"after_wrapper_reset_command_warmup\")
    print(
        \"BM_SENTINEL:eval:reset_command_warmup=\" + json.dumps(
            {
                \"before_done_rate\": reset_command_warmup_before[\"manual_endpoint_z_done_rate\"],
                \"after_done_rate\": reset_command_warmup_after[\"manual_endpoint_z_done_rate\"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
"""
    if old not in worker_code:
        raise RuntimeError("Base worker code shape changed; cannot inject reset command warmup.")
    worker_code = worker_code.replace(old, new)
    old_metrics = """        \"critic_obs_shape\": list(extras[\"observations\"][\"critic\"].shape),
        \"robot_num_joints\": int(vec_env.unwrapped.scene[\"robot\"].num_joints),
"""
    new_metrics = """        \"critic_obs_shape\": list(extras[\"observations\"][\"critic\"].shape),
        \"reset_command_warmup\": {
            \"applied\": True,
            \"before\": reset_command_warmup_before,
            \"after\": reset_command_warmup_after,
            \"manual_endpoint_z_done_rate_delta\": (
                reset_command_warmup_after[\"manual_endpoint_z_done_rate\"]
                - reset_command_warmup_before[\"manual_endpoint_z_done_rate\"]
            ),
        },
        \"robot_num_joints\": int(vec_env.unwrapped.scene[\"robot\"].num_joints),
"""
    if old_metrics not in worker_code:
        raise RuntimeError("Base worker metrics block changed; cannot record reset command warmup.")
    return worker_code.replace(old_metrics, new_metrics)


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_robot_order_fk_repaired_full_bundle_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy adapts the robot-order FK-repaired training status enum for the shared checkpoint-eval harness. "
        "The authoritative training audit remains the robot-order FK-repaired training JSON."
    )
    shim_path = OUT / "base_compatible_robot_order_fk_repaired_full_bundle_training_run_for_eval_warmup.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shim_path


def first_timeseries_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            return {}
    converted: dict[str, Any] = {}
    for key, value in row.items():
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = value
    return converted


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    training = load_json(TRAINING_RUN_JSON)
    bundle = load_json(ROBOT_ORDER_BUNDLE_AUDIT)
    split_gate = load_json(ROBOT_ORDER_SPLIT_TASK_GATE)
    warmup_probe = load_json(WARMUP_PROBE_JSON)
    old_eval = load_json(OLD_EVAL_JSON)
    metrics = bundle.get("metrics", {})
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    final_json = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
    )
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed"
        if eval_ok
        else summary.get(
            "status",
            "failed_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup",
        )
    )
    summary["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup"
    )
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates the local robot-order FK-repaired PPO checkpoint after applying one command-manager warmup "
        "immediately after the RSL-RL wrapper's initial reset. This tests whether the previous step-0 done spike "
        "can be reduced before launching another full PPO/downstream chain."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "old_non_warmup_eval_json": str(OLD_EVAL_JSON),
            "reset_command_warmup_live_probe_json": str(WARMUP_PROBE_JSON),
            "base_compatible_training_run_json": str(
                OUT / "base_compatible_robot_order_fk_repaired_full_bundle_training_run_for_eval_warmup.json"
            ),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "robot_order_fk_repaired_motion_npz": str(ROBOT_ORDER_BUNDLE_NPZ),
            "robot_order_fk_repaired_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
            "robot_order_split_task_gate": str(ROBOT_ORDER_SPLIT_TASK_GATE),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "training_run_completed": training.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_completed",
            "old_non_warmup_eval_exists": OLD_EVAL_JSON.is_file(),
            "warmup_probe_ok": warmup_probe.get("status") == "ok_robot_order_fk_reset_command_warmup_live_probe",
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "robot_order_motion_npz_exists": ROBOT_ORDER_BUNDLE_NPZ.is_file(),
            "robot_order_bundle_audit_passed": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
            "robot_order_motion_count_40": metrics.get("motion_count") == 40,
            "robot_order_total_frames_11960": metrics.get("total_frames") == 11960,
            "robot_order_split_task_gate_passed": split_gate.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_split_task_eval",
        }
    )
    summary.setdefault("config", {})
    summary["config"]["seed"] = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_SEED", str(DEFAULT_SEED))
    )
    summary["config"]["num_envs"] = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS))
    )
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        run_metrics = summary["run"]["metrics"]
        run_metrics["uses_official_importer_export_usd"] = True
        run_metrics["uses_resource_adjusted_usd"] = False
        run_metrics["uses_fk_repaired_full_public_motion_bundle"] = False
        run_metrics["uses_robot_order_fk_repaired_full_public_motion_bundle"] = True
        run_metrics["uses_old_degenerate_full_public_motion_bundle"] = False
        run_metrics["official_csv_to_npz_unpatched_output"] = False
        run_metrics["paper_level_tracking_eval"] = False
        run_metrics["reset_command_warmup_applied"] = True
        run_metrics["motion_count"] = metrics.get("motion_count")
        run_metrics["total_motion_frames"] = metrics.get("total_frames")
    old_metrics = old_eval.get("run", {}).get("metrics", {})
    new_metrics = summary.get("run", {}).get("metrics", {})
    old_total = old_metrics.get("total_env_steps") or old_eval.get("config", {}).get("total_env_steps")
    new_total = new_metrics.get("total_env_steps") or summary.get("config", {}).get("total_env_steps")
    old_done = old_metrics.get("done_count_total")
    new_done = new_metrics.get("done_count_total")
    old_timeseries = Path(old_eval.get("outputs", {}).get("timeseries_csv", ""))
    new_timeseries = Path(summary.get("outputs", {}).get("timeseries_csv", ""))
    comparison = {
        "old_eval_json": str(OLD_EVAL_JSON),
        "old_done_count_total": old_done,
        "old_total_env_steps": old_total,
        "old_done_rate": (old_done / old_total) if old_done is not None and old_total else None,
        "old_step0": first_timeseries_row(old_timeseries),
        "warmup_eval_done_count_total": new_done,
        "warmup_eval_total_env_steps": new_total,
        "warmup_eval_done_rate": (new_done / new_total) if new_done is not None and new_total else None,
        "warmup_eval_step0": first_timeseries_row(new_timeseries),
        "warmup_snapshot": new_metrics.get("reset_command_warmup", {}),
    }
    old_step0_done = comparison["old_step0"].get("done_count")
    new_step0_done = comparison["warmup_eval_step0"].get("done_count")
    if old_step0_done is not None and new_step0_done is not None:
        comparison["step0_done_count_delta"] = new_step0_done - old_step0_done
    if comparison["old_done_rate"] is not None and comparison["warmup_eval_done_rate"] is not None:
        comparison["done_rate_delta"] = comparison["warmup_eval_done_rate"] - comparison["old_done_rate"]
    summary["comparison_to_non_warmup_eval"] = comparison
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "robot_order_fk_repaired_full_bundle_checkpoint_warmup_eval_complete": bool(eval_ok),
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "reset_command_warmup_applied": True,
        "why_not_paper_level": (
            "The evaluated checkpoint is locally trained on a robot-order FK-repaired public-motion bundle with a "
            "local official-importer USDA. The reset command warmup is a local evaluation repair, not an official "
            "BeyondMimic teacher checkpoint, not DAgger, not the paper full protocol, and not real-robot validation."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    module = load_base_module()
    compatible_training_summary = make_base_compatible_training_summary()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = ROBOT_ORDER_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.NUM_ENVS = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS))
    )
    module.SEED = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_SEED", str(DEFAULT_SEED))
    )
    module.WORKER_CODE = patch_worker_code(module.WORKER_CODE)
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
    )
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
                "num_envs": summary.get("config", {}).get("num_envs"),
                "comparison_to_non_warmup_eval": summary.get("comparison_to_non_warmup_eval", {}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
