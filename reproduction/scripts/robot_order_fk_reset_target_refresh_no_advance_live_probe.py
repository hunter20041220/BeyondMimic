#!/usr/bin/env python3
"""Live probe for reset-target refresh without advancing motion time.

The prior reset-command warmup probe called ``command_manager.compute()`` after
reset. That cleared the stale zero target, but it also advanced
``MotionCommand.time_steps`` by one and the full eval showed a worse post-step0
termination rate. This probe refreshes ``body_pos_relative_w`` and
``body_quat_relative_w`` directly from the current time step, then recomputes
observations without advancing the motion phase.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/robot_order_fk_reset_command_warmup_live_probe.py"
OUT = ROOT / "res/tracking/robot_order_fk_reset_target_refresh_no_advance_live_probe"
LOG_DIR = ROOT / "logs/tracking_robot_order_fk_reset_target_refresh_no_advance_live_probe"


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_reset_command_warmup_probe", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base probe: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_worker_code(worker_code: str) -> str:
    worker_code = worker_code.replace(
        "import whole_body_tracking.tasks  # noqa: F401\n",
        "import whole_body_tracking.tasks  # noqa: F401\n"
        "    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat\n",
    )
    old = """    before = snapshot("after_reset_before_command_warmup", command)
    env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)
    after_warmup = snapshot("after_manual_command_manager_compute", command)
"""
    new = """    def refresh_motion_targets_no_advance():
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

    before = snapshot("after_reset_before_target_refresh", command)
    time_steps_before_refresh = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance()
    time_steps_after_refresh = command.time_steps.detach().clone()
    obs = env.unwrapped.observation_manager.compute()
    after_warmup = snapshot("after_no_advance_target_refresh", command)
    after_warmup["time_steps_unchanged_by_refresh"] = bool(torch.equal(time_steps_before_refresh, time_steps_after_refresh))
"""
    if old not in worker_code:
        raise RuntimeError("Base warmup worker code shape changed; cannot inject no-advance refresh.")
    worker_code = worker_code.replace(old, new)
    replacements = {
        "robot_order_fk_reset_command_warmup_live_probe": "robot_order_fk_reset_target_refresh_no_advance_live_probe",
        "command_warmup_clears_reset_endpoint_z_spike": "no_advance_target_refresh_clears_reset_endpoint_z_spike",
        "command_warmup_partially_reduces_reset_endpoint_z_spike": "no_advance_target_refresh_partially_reduces_reset_endpoint_z_spike",
        "command_warmup_does_not_clear_reset_endpoint_z_spike": "no_advance_target_refresh_does_not_clear_reset_endpoint_z_spike",
        "Patch local tracking train/eval wrappers to warm command targets immediately after reset": (
            "Patch local tracking train/eval wrappers to refresh reset targets without advancing motion time"
        ),
        "command_manager_compute": "target_refresh_no_advance",
        "command warmup": "no-advance target refresh",
        "Command-Warmup": "Target-Refresh No-Advance",
    }
    for src, dst in replacements.items():
        worker_code = worker_code.replace(src, dst)
    return worker_code


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    worker = summary.get("worker_metrics", {})
    checks = summary.get("checks", {})
    rows = summary.get("snapshot_rows", [])
    lines = [
        "# Robot-Order FK Reset Target Refresh No-Advance Probe",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Worker status: `{worker.get('status')}`",
        f"- Diagnosis: `{worker.get('diagnosis')}`",
        f"- Time steps unchanged by refresh: `{summary['metrics'].get('time_steps_unchanged_by_refresh')}`",
        f"- Endpoint done-rate delta: `{summary['metrics'].get('endpoint_done_rate_delta')}`",
        f"- Endpoint z mean delta: `{summary['metrics'].get('endpoint_z_error_mean_delta')}`",
        "",
        "## Snapshot Metrics",
        "",
        "| Stage | Endpoint done rate | Endpoint z mean (m) | Body error mean (m) | Time-step mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {done} | {zmean} | {body} | {tmean} |".format(
                stage=row.get("stage"),
                done=row.get("manual_endpoint_z_done_rate"),
                zmean=row.get("endpoint_z_error_mean_m"),
                body=row.get("body_error_mean_m"),
                tmean=row.get("time_steps_mean"),
            )
        )
    lines.extend(["", "## Checks", ""])
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "This is a local live tracking data-quality gate. It is not PPO training, not DAgger, not "
            "paper-level closed-loop reproduction, and not a real-robot result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    base.OUT = OUT
    base.LOG_DIR = LOG_DIR
    base.WORKER_CODE = patch_worker_code(base.WORKER_CODE)

    worker = OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe_worker.py"
    worker_metrics_path = OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe_worker_metrics.json"
    worker.write_text(base.WORKER_CODE, encoding="utf-8")
    run = base.run_worker(worker, worker_metrics_path)
    worker_metrics = load_json(worker_metrics_path)
    snapshot_rows = base.snapshot_rows(worker_metrics)
    tsv_path = OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe.tsv"
    base.write_tsv(
        tsv_path,
        snapshot_rows,
        [
            "stage",
            "manual_endpoint_z_done_count",
            "manual_endpoint_z_done_rate",
            "endpoint_z_error_mean_m",
            "endpoint_z_error_max_m",
            "body_error_mean_m",
            "body_error_max_m",
            "anchor_error_mean_m",
            "body_pos_relative_abs_max",
            "time_steps_mean",
        ],
    )

    before = worker_metrics.get("snapshots", [{}])[0] if worker_metrics.get("snapshots") else {}
    after = worker_metrics.get("snapshots", [{}, {}])[1] if len(worker_metrics.get("snapshots", [])) > 1 else {}
    endpoint_before = before.get("endpoint_z_error_m", {})
    endpoint_after = after.get("endpoint_z_error_m", {})
    checks = worker_metrics.get("checks", {})
    status_ok = (
        run["returncode"] == 0
        and worker_metrics.get("status") == "ok_robot_order_fk_reset_target_refresh_no_advance_live_probe"
    )
    metrics = {
        "time_steps_unchanged_by_refresh": after.get("time_steps_unchanged_by_refresh"),
        "endpoint_done_rate_before": before.get("manual_endpoint_z_done_rate"),
        "endpoint_done_rate_after": after.get("manual_endpoint_z_done_rate"),
        "endpoint_done_rate_delta": (
            after.get("manual_endpoint_z_done_rate") - before.get("manual_endpoint_z_done_rate")
            if after.get("manual_endpoint_z_done_rate") is not None
            and before.get("manual_endpoint_z_done_rate") is not None
            else None
        ),
        "endpoint_z_error_mean_before": endpoint_before.get("mean"),
        "endpoint_z_error_mean_after": endpoint_after.get("mean"),
        "endpoint_z_error_mean_delta": (
            endpoint_after.get("mean") - endpoint_before.get("mean")
            if endpoint_after.get("mean") is not None and endpoint_before.get("mean") is not None
            else None
        ),
    }
    summary = {
        "status": (
            "ok_robot_order_fk_reset_target_refresh_no_advance_live_probe"
            if status_ok
            else "failed_robot_order_fk_reset_target_refresh_no_advance_live_probe"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_reset_target_refresh_no_advance_live_probe",
        "scope": (
            "Live IsaacLab reset-target refresh diagnostic that recomputes MotionCommand body targets without "
            "calling command_manager.compute() and therefore without advancing motion time_steps."
        ),
        "config": {
            "target_gpu": base.TARGET_GPU,
            "num_envs": base.NUM_ENVS,
            "robot_usd": str(base.OFFICIAL_IMPORTER_USD),
            "motion_file": str(base.ROBOT_ORDER_MOTION_NPZ),
        },
        "run": run,
        "worker_metrics": worker_metrics,
        "snapshot_rows": snapshot_rows,
        "metrics": metrics,
        "checks": {
            "worker_returned_zero": run["returncode"] == 0,
            "worker_status_ok": status_ok,
            "endpoint_names_found": checks.get("endpoint_names_found") is True,
            "pre_refresh_manual_endpoint_done_rate_high": checks.get("pre_warmup_manual_endpoint_done_rate_high")
            is True,
            "refresh_reduces_endpoint_z_error_mean": checks.get("warmup_reduces_endpoint_z_error_mean") is True,
            "refresh_reduces_manual_endpoint_done_rate": checks.get("warmup_reduces_manual_endpoint_done_rate")
            is True,
            "time_steps_unchanged_by_refresh": after.get("time_steps_unchanged_by_refresh") is True,
            "zero_action_step_after_refresh_not_all_done": checks.get("zero_action_step_after_warmup_not_all_done")
            is True,
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "claim_level": "tracking_reset_live_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "next_mainline_decision": worker_metrics.get("diagnosis", "worker_failed"),
            "why_this_matters": (
                "If this clears the stale target while preserving time_steps, the full eval can test reset-target "
                "refresh without the phase drift introduced by command_manager.compute()."
            ),
        },
        "outputs": {
            "json": str(OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe.json"),
            "tsv": str(tsv_path),
            "md": str(OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe.md"),
            "worker": str(worker),
            "worker_metrics": str(worker_metrics_path),
            "log": run["log_path"],
        },
    }
    json_path = OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe.json"
    md_path = OUT / "robot_order_fk_reset_target_refresh_no_advance_live_probe.md"
    write_json(json_path, summary)
    write_markdown(md_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(json_path),
                "metrics": metrics,
                "checks": summary["checks"],
            },
            sort_keys=True,
        )
    )
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
