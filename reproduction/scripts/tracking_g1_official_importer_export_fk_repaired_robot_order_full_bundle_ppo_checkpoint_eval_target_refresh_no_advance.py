#!/usr/bin/env python3
"""Evaluate robot-order FK PPO with reset-target refresh and no time advance."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.py"
)
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance"
)
OLD_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
LIVE_PROBE_JSON = (
    ROOT
    / "res/tracking/robot_order_fk_reset_target_refresh_no_advance_live_probe/"
    "robot_order_fk_reset_target_refresh_no_advance_live_probe.json"
)
DEFAULT_SEED = 20260721
DEFAULT_NUM_ENVS = 2048
SOURCE_JSON_NAME = (
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
)
FINAL_JSON_NAME = (
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_robot_order_warmup_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base warmup eval: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_timeseries_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            return {}
    out: dict[str, Any] = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def patch_worker_code(worker_code: str) -> str:
    worker_code = worker_code.replace(
        "import whole_body_tracking.tasks  # noqa: F401\n",
        "import whole_body_tracking.tasks  # noqa: F401\n"
        "    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat\n",
    )
    old = """    def warmup_snapshot(label):
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in endpoint_names if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        return {
            "label": label,
            "endpoint_names_present": [name for name in endpoint_names if name in body_names],
            "endpoint_indexes": endpoint_indexes,
            "manual_endpoint_z_done_count": int(endpoint_done.detach().cpu().sum().item()),
            "manual_endpoint_z_done_rate": float(endpoint_done.float().mean().detach().cpu().item()),
            "endpoint_z_error_m": stats_tensor(endpoint_z_error),
            "body_error_m": stats_tensor(body_error),
            "body_pos_relative_abs_max": float(body_target.abs().max().detach().cpu().item()),
            "time_steps": stats_tensor(command.time_steps.detach().float()),
        }

    reset_command_warmup_before = warmup_snapshot("after_wrapper_reset_before_command_warmup")
    vec_env.unwrapped.command_manager.compute(dt=vec_env.unwrapped.step_dt)
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    reset_command_warmup_after = warmup_snapshot("after_wrapper_reset_command_warmup")
"""
    new = """    def warmup_snapshot(label):
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in endpoint_names if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        return {
            "label": label,
            "endpoint_names_present": [name for name in endpoint_names if name in body_names],
            "endpoint_indexes": endpoint_indexes,
            "manual_endpoint_z_done_count": int(endpoint_done.detach().cpu().sum().item()),
            "manual_endpoint_z_done_rate": float(endpoint_done.float().mean().detach().cpu().item()),
            "endpoint_z_error_m": stats_tensor(endpoint_z_error),
            "body_error_m": stats_tensor(body_error),
            "body_pos_relative_abs_max": float(body_target.abs().max().detach().cpu().item()),
            "time_steps": stats_tensor(command.time_steps.detach().float()),
        }

    def refresh_motion_targets_no_advance():
        anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
        command.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, command.body_pos_w - anchor_pos_w_repeat)
        command._update_metrics()

    reset_command_warmup_before = warmup_snapshot("after_wrapper_reset_before_target_refresh")
    time_steps_before_refresh = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance()
    time_steps_after_refresh = command.time_steps.detach().clone()
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    reset_command_warmup_after = warmup_snapshot("after_wrapper_reset_target_refresh_no_advance")
    reset_command_warmup_after["time_steps_unchanged_by_refresh"] = bool(
        torch.equal(time_steps_before_refresh, time_steps_after_refresh)
    )
"""
    if old not in worker_code:
        raise RuntimeError("Warmup worker code shape changed; cannot inject no-advance target refresh.")
    worker_code = worker_code.replace(old, new)
    worker_code = worker_code.replace("reset_command_warmup=", "reset_target_refresh_no_advance=")
    return worker_code


def post_step0_done_rate(timeseries: Path, num_envs: int) -> float | None:
    if not timeseries.is_file():
        return None
    done_total = 0.0
    steps = 0
    with timeseries.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row["step"])) == 0:
                continue
            done_total += float(row["done_count"])
            steps += 1
    return done_total / float(max(steps * num_envs, 1)) if steps else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.DEFAULT_SEED = DEFAULT_SEED
    module.DEFAULT_NUM_ENVS = DEFAULT_NUM_ENVS
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_SEED"] = str(DEFAULT_SEED)
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_NUM_ENVS"] = str(DEFAULT_NUM_ENVS)
    warmup_patch_worker_code = module.patch_worker_code
    module.patch_worker_code = lambda worker_code: patch_worker_code(warmup_patch_worker_code(worker_code))
    module.main()

    source_json = OUT / SOURCE_JSON_NAME
    final_json = OUT / FINAL_JSON_NAME
    summary = load_json(source_json)
    old_eval = load_json(OLD_EVAL_JSON)
    live_probe = load_json(LIVE_PROBE_JSON)
    if not summary:
        raise RuntimeError(f"Target-refresh eval did not write expected JSON: {source_json}")
    eval_ok = summary.get("status") == (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed"
    )
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed"
        if eval_ok
        else "failed_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance"
    )
    summary["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance"
    )
    summary["scope"] = (
        "Same-seed full checkpoint evaluation using reset-target refresh without advancing MotionCommand.time_steps. "
        "This tests the mainline data-quality fix suggested by the previous warmup phase diagnostic."
    )
    summary.setdefault("config", {})
    summary["config"]["seed"] = DEFAULT_SEED
    summary["config"]["num_envs"] = DEFAULT_NUM_ENVS
    summary["config"]["same_seed_as_non_warmup_eval"] = True
    summary["config"]["reset_target_refresh_no_advance"] = True
    summary.setdefault("inputs", {})
    summary["inputs"]["reset_target_refresh_no_advance_live_probe_json"] = str(LIVE_PROBE_JSON)
    summary["input_checks"]["reset_target_refresh_live_probe_ok"] = (
        live_probe.get("status") == "ok_robot_order_fk_reset_target_refresh_no_advance_live_probe"
    )
    metrics = summary.get("run", {}).get("metrics", {})
    refresh = metrics.get("reset_command_warmup", {})
    if refresh:
        refresh["mode"] = "target_refresh_no_advance"
        refresh["time_steps_unchanged_by_refresh"] = refresh.get("after", {}).get("time_steps_unchanged_by_refresh")
    old_metrics = old_eval.get("run", {}).get("metrics", {})
    old_ts = Path(old_eval.get("outputs", {}).get("timeseries_csv", ""))
    new_ts = Path(summary.get("outputs", {}).get("timeseries_csv", ""))
    comparison = summary.get("comparison_to_non_warmup_eval", {})
    comparison["old_step0"] = first_timeseries_row(old_ts)
    comparison["target_refresh_eval_step0"] = first_timeseries_row(new_ts)
    comparison["old_post_step0_done_rate"] = post_step0_done_rate(old_ts, DEFAULT_NUM_ENVS)
    comparison["target_refresh_post_step0_done_rate"] = post_step0_done_rate(new_ts, DEFAULT_NUM_ENVS)
    if comparison["old_post_step0_done_rate"] is not None and comparison["target_refresh_post_step0_done_rate"] is not None:
        comparison["post_step0_done_rate_delta"] = (
            comparison["target_refresh_post_step0_done_rate"] - comparison["old_post_step0_done_rate"]
        )
    old_done = old_metrics.get("done_count_total")
    new_done = metrics.get("done_count_total")
    old_total = old_metrics.get("total_env_steps")
    new_total = metrics.get("total_env_steps")
    comparison.update(
        {
            "old_done_count_total": old_done,
            "target_refresh_done_count_total": new_done,
            "old_done_rate": (old_done / old_total) if old_done is not None and old_total else None,
            "target_refresh_done_rate": (new_done / new_total) if new_done is not None and new_total else None,
        }
    )
    if comparison["old_done_rate"] is not None and comparison["target_refresh_done_rate"] is not None:
        comparison["done_rate_delta"] = comparison["target_refresh_done_rate"] - comparison["old_done_rate"]
    summary["comparison_to_non_warmup_eval"] = comparison
    summary.setdefault("checks", {})
    summary["checks"] = {
        **summary.get("checks", {}),
        "same_seed_as_non_warmup_eval": True,
        "same_full_eval_scope": metrics.get("num_envs") == DEFAULT_NUM_ENVS and metrics.get("eval_steps") == 299,
        "reset_target_refresh_live_probe_ok": summary["input_checks"]["reset_target_refresh_live_probe_ok"],
        "time_steps_unchanged_by_refresh": refresh.get("time_steps_unchanged_by_refresh") is True,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_goal_complete": True,
        "does_not_claim_real_robot": True,
    }
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_tracking_eval_complete": False,
        "reset_target_refresh_no_advance_applied": True,
        "why_this_is_mainline": (
            "Unlike command_manager.compute() warmup, this refresh targets the stale reset body target without "
            "advancing the sampled motion phase."
        ),
        "why_not_paper_level": (
            "This is still a local robot-order FK-repaired public-bundle PPO checkpoint eval, not the official "
            "BeyondMimic teacher, DAgger rollout, paper Fig.5/Fig.6 protocol, TensorRT deployment, or real robot."
        ),
    }
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["source_warmup_json_before_rename"] = str(source_json)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_json.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "comparison_to_non_warmup_eval": comparison,
                "checks": summary["checks"],
            },
            sort_keys=True,
        )
    )
    if not eval_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
