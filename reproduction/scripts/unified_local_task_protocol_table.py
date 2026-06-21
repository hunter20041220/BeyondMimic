#!/usr/bin/env python3
"""Build a compact report table for local Fig.5/Fig.6-style proxy tasks."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/report_assets/unified_local_task_protocol"

TASK_METRICS = {
    "joystick": "velocity tracking error, reward delta, tracking-error delta, done/fall proxy",
    "waypoint": "final root distance, reward delta, tracking-error delta, success proxy",
    "obstacle_avoidance": "minimum clearance/collision proxy, root distance, reward delta, tracking-error delta",
    "composed": "combined joystick/waypoint/obstacle proxy score, reward delta, tracking-error delta",
    "transition": "speed-ramp profile, transition smoothness, 299-step completion/fall proxy",
    "inpainting": "keyframe/body target error, guidance cost delta, 299-step completion/fall proxy",
}


def read_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.is_file():
        return {"status": "missing", "missing_path": rel}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def task_rows_from_proxy_tables() -> list[dict[str, Any]]:
    fig_proxy_rows = {
        row["task"]: row
        for row in read_csv_rows(
            "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
            "fig5_fig6_task_protocol_proxy_aggregate.csv"
        )
        if row.get("task")
    }
    scaled_rows = {
        row["task"]: row
        for row in read_csv_rows(
            "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/"
            "importer_export_task_conditioned_guidance_multiseed_aggregate.csv"
        )
        if row.get("task")
    }
    rows: list[dict[str, Any]] = []
    for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]:
        proxy = fig_proxy_rows.get(task, {})
        scaled = scaled_rows.get(task, {})
        rows.append(
            {
                "task": task,
                "local_protocol_status": "implemented_multiseed_proxy",
                "main_metric": TASK_METRICS[task],
                "seed_count": int(as_float(proxy, "seed_group_count") or as_float(scaled, "seed_count") or 0),
                "rollout_count": int(as_float(proxy, "row_count") or 0),
                "completion_rate_or_steps": fmt(as_float(proxy, "recorded_299_step_completion_rate")),
                "guidance_signal_positive_rate": fmt(as_float(proxy, "guidance_signal_positive_rate")),
                "local_proxy_pass_rate": fmt(as_float(proxy, "local_task_protocol_proxy_pass_rate")),
                "guided_reward_mean": fmt(as_float(scaled, "guided_reward_mean")),
                "reward_improved_vs_denoised_rate": fmt(as_float(proxy, "reward_improved_vs_denoised_rate")),
                "tracking_error_not_worse_rate": fmt(as_float(proxy, "tracking_error_not_worse_vs_denoised_rate")),
                "root_final_xy_error_m_mean": fmt(as_float(proxy, "root_final_xy_error_m_mean")),
                "guided_target_body_error_mean": fmt(
                    as_float(proxy, "guided_target_body_error_mean_mean")
                    or as_float(scaled, "guided_target_body_error_mean")
                ),
                "evidence": (
                    "fig5_fig6_task_protocol_proxy aggregate plus scaled-PPO task-conditioned multiseed aggregate"
                ),
                "claim_level": "local_virtual_proxy_not_paper_level",
                "next_gap": "replace weak tracking teacher, use paper-style task thresholds, and rerun closed-loop guidance",
            }
        )
    return rows


def single_task_rows() -> list[dict[str, Any]]:
    transition = read_json(
        "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json"
    )
    inpainting = read_json(
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
    )
    rows = []
    rows.append(
        {
            "task": "transition",
            "local_protocol_status": "implemented_single_seed_proxy",
            "main_metric": TASK_METRICS["transition"],
            "seed_count": 1 if transition.get("status", "").startswith("ok_") else 0,
            "rollout_count": len(transition.get("rows", [])),
            "completion_rate_or_steps": "299" if transition.get("checks", {}).get("rollout_299_steps") else "",
            "guidance_signal_positive_rate": "",
            "local_proxy_pass_rate": "",
            "guided_reward_mean": "",
            "reward_improved_vs_denoised_rate": "",
            "tracking_error_not_worse_rate": "",
            "root_final_xy_error_m_mean": "",
            "guided_target_body_error_mean": "",
            "evidence": "transition guidance rollout asset and transition metrics",
            "claim_level": "local_virtual_single_seed_proxy_not_paper_fig5b",
            "next_gap": "upgrade to multi-seed paper-style walk-to-run transition metric and fall-rate gate",
        }
    )
    rows.append(
        {
            "task": "inpainting",
            "local_protocol_status": "implemented_single_seed_proxy",
            "main_metric": TASK_METRICS["inpainting"],
            "seed_count": 1 if inpainting.get("status", "").startswith("ok_") else 0,
            "rollout_count": len(inpainting.get("rows", [])),
            "completion_rate_or_steps": "299" if inpainting.get("checks", {}).get("rollout_299_steps") else "",
            "guidance_signal_positive_rate": "",
            "local_proxy_pass_rate": "",
            "guided_reward_mean": "",
            "reward_improved_vs_denoised_rate": "",
            "tracking_error_not_worse_rate": "",
            "root_final_xy_error_m_mean": "",
            "guided_target_body_error_mean": "",
            "evidence": "inpainting guidance rollout proxy and keyframe/body target metrics",
            "claim_level": "local_virtual_single_seed_proxy_not_paper_inpainting",
            "next_gap": "upgrade to multi-seed keyframe-error protocol with paper-facing success thresholds",
        }
    )
    return rows


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    headers = [
        "task",
        "local_protocol_status",
        "main_metric",
        "seed_count",
        "local_proxy_pass_rate",
        "reward_improved_vs_denoised_rate",
        "tracking_error_not_worse_rate",
        "claim_level",
        "next_gap",
    ]
    lines = [
        "# Unified Local Task Protocol",
        "",
        "This table consolidates the local BeyondMimic-like guidance tasks used for the reading report. It is a proxy protocol table, not a paper-level Fig. 5/Fig. 6 reproduction.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Task rows: `{summary['metrics']['task_count']}`",
        f"- Paper-level reproduced rows: `{summary['metrics']['paper_level_reproduced_count']}`",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "")).replace("|", "/") for h in headers) + "|")
    lines.extend(
        [
            "",
            "Interpretation: the current table is useful evidence for an auditable local virtual pipeline, but it should be read together with the tracking quality gate. The tracking teacher still has near-unit termination/done behavior in the latest FK-repaired PPO eval, so downstream results remain proxy evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = task_rows_from_proxy_tables() + single_task_rows()
    summary = {
        "status": "ok_unified_local_task_protocol",
        "experiment_type": "unified_local_task_protocol",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Unified local proxy task table for joystick, waypoint, obstacle, composed, transition, and inpainting guidance.",
        "rows": rows,
        "metrics": {
            "task_count": len(rows),
            "multiseed_proxy_task_count": sum(row["local_protocol_status"] == "implemented_multiseed_proxy" for row in rows),
            "single_seed_proxy_task_count": sum(row["local_protocol_status"] == "implemented_single_seed_proxy" for row in rows),
            "paper_level_reproduced_count": 0,
        },
        "checks": {
            "six_requested_tasks_present": sorted(row["task"] for row in rows)
            == ["composed", "inpainting", "joystick", "obstacle_avoidance", "transition", "waypoint"],
            "does_not_claim_fig5_fig6_paper_level": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_proxy_protocol_table_not_paper_level",
            "why_not_paper_level": (
                "The table aggregates local proxy tasks from local PPO/VAE/denoiser/guidance runs. The exact paper "
                "Fig.5/Fig.6 protocol, official checkpoints, TensorRT traces, and real-robot deployment remain absent."
            ),
        },
        "outputs": {
            "json": str(OUT / "unified_local_task_protocol.json"),
            "csv": str(OUT / "unified_local_task_protocol.csv"),
            "md": str(OUT / "unified_local_task_protocol.md"),
        },
    }
    (OUT / "unified_local_task_protocol.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "unified_local_task_protocol.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(OUT / "unified_local_task_protocol.md", summary, rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
