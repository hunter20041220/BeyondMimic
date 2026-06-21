#!/usr/bin/env python3
"""Summarize the current tracking data-quality gate for FK-repaired PPO."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/fk_repaired_data_quality_gate"


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def rel_exists(rel: str) -> bool:
    return (ROOT / rel).is_file()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    degeneracy = load_json(
        "res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/"
        "motion_bundle_body_position_degeneracy_audit.json"
    )
    fk_bundle = load_json(
        "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
        "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
    )
    fk_split = load_json(
        "res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz/"
        "tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz.json"
    )
    fk_split_eval = load_json(
        "res/tracking/g1_official_importer_export_fk_repaired_split_task_eval/"
        "tracking_g1_official_importer_export_fk_repaired_split_task_eval.json"
    )
    fk_full_eval = load_json(
        "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_task_eval/"
        "tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.json"
    )
    failed_full_eval = load_json(
        "res/failed_runs/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval/"
        "fk_repaired_full_bundle_task_eval_20260621_summary.json"
    )
    endpoint_trace = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/"
        "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json"
    )
    scaled_eval = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    )
    fk_ppo_training = load_json(
        "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run/"
        "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run.json"
    )
    fk_ppo_eval = load_json(
        "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval.json"
    )
    fk_ppo_assets = load_json(
        "res/report_assets/official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval/"
        "official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval_assets.json"
    )
    fk_eval_metrics = fk_ppo_eval.get("run", {}).get("metrics", {})
    fk_eval_total_steps = fk_eval_metrics.get("total_env_steps") or 0
    fk_eval_done_count = fk_eval_metrics.get("done_count_total") or 0
    fk_eval_done_rate = (float(fk_eval_done_count) / float(fk_eval_total_steps)) if fk_eval_total_steps else None

    checks = {
        "old_bundle_body_positions_degenerate": degeneracy.get("checks", {}).get(
            "bundle_body_positions_degenerate_lt_1e_minus_5m"
        )
        is True,
        "old_bundle_z_spread_degenerate": degeneracy.get("checks", {}).get(
            "bundle_body_z_spread_degenerate_lt_1e_minus_5m"
        )
        is True,
        "fk_repaired_bundle_non_degenerate": fk_bundle.get("checks", {}).get(
            "fk_repaired_z_spread_non_degenerate_gt_0_5m"
        )
        is True,
        "fk_repaired_bundle_has_40_motions": fk_bundle.get("bundle", {}).get("motion_count") == 40,
        "fk_repaired_split_has_40_motions": fk_split.get("metrics", {}).get("motion_count") == 40,
        "fk_repaired_split_task_eval_ok": fk_split_eval.get("status")
        == "ok_official_importer_export_fk_repaired_split_task_eval",
        "fk_repaired_full_bundle_task_eval_ok": fk_full_eval.get("status")
        == "ok_official_importer_export_fk_repaired_full_bundle_task_eval",
        "failed_full_bundle_attempt_retained": failed_full_eval.get("status") == "failed_retained",
        "endpoint_trace_records_ankle_problem": endpoint_trace.get("checks", {}).get(
            "aggregate_exceed_rate_recorded"
        )
        is True,
        "scaled_eval_exists_but_not_paper_level": scaled_eval.get("interpretation", {}).get(
            "paper_level_tracking_eval_complete"
        )
        is False,
        "fk_repaired_training_wrapper_exists": rel_exists(
            "reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run.py"
        ),
        "fk_repaired_eval_wrapper_exists": rel_exists(
            "reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval.py"
        ),
        "fk_repaired_ppo_training_completed": fk_ppo_training.get("status")
        == "ok_official_importer_export_fk_repaired_full_bundle_ppo_training_completed",
        "fk_repaired_ppo_eval_completed": fk_ppo_eval.get("status")
        == "ok_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval_completed",
        "fk_repaired_ppo_eval_uses_fk_bundle": fk_eval_metrics.get("uses_fk_repaired_full_public_motion_bundle")
        is True,
        "fk_repaired_ppo_eval_done_rate_below_0_1": fk_eval_done_rate is not None and fk_eval_done_rate < 0.1,
        "fk_repaired_ppo_report_assets_ok": fk_ppo_assets.get("checks", {}).get("eval_status_ok") is True,
    }
    downstream_ready = all(
        [
            checks["fk_repaired_ppo_training_completed"],
            checks["fk_repaired_ppo_eval_completed"],
            checks["fk_repaired_ppo_eval_uses_fk_bundle"],
            checks["fk_repaired_ppo_eval_done_rate_below_0_1"],
        ]
    )
    aggregate = fk_split_eval.get("aggregate", {})
    gate = {
        "ready_for_fk_repaired_full_ppo_attempt": all(
            [
                checks["fk_repaired_bundle_non_degenerate"],
                checks["fk_repaired_bundle_has_40_motions"],
                checks["fk_repaired_split_has_40_motions"],
                checks["fk_repaired_split_task_eval_ok"],
                checks["fk_repaired_training_wrapper_exists"],
                checks["fk_repaired_eval_wrapper_exists"],
            ]
        ),
        "fk_repaired_full_ppo_completed": checks["fk_repaired_ppo_training_completed"],
        "fk_repaired_checkpoint_eval_completed": checks["fk_repaired_ppo_eval_completed"],
        "fk_repaired_eval_done_rate": fk_eval_done_rate,
        "ready_for_teacher_rollout_downstream": downstream_ready,
        "paper_level_tracking_ready": False,
        "old_scaled_chain_trust_level": "diagnostic_only_due_to_old_body_pos_w_degeneracy_and_endpoint_z_errors",
        "next_action": (
            "Do not start teacher rollout/VAE/diffusion from the FK-repaired checkpoint yet. The full PPO/eval path "
            "now runs, but eval done/termination remains near one done per env-step. Next repair target is "
            "termination/reset/anchor alignment rather than more downstream training."
            if not downstream_ready
            else "Proceed to teacher rollout/VAE/diffusion with explicit local-virtual claim boundaries."
        ),
    }
    rows = [
        {
            "item": "old_full_bundle",
            "status": "do_not_use_for_new_teacher_training",
            "evidence": str(
                ROOT
                / "res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/"
                "motion_bundle_body_position_degeneracy_audit.json"
            ),
            "reason": "body_pos_w and z-spread are degenerate/root-like",
        },
        {
            "item": "fk_repaired_full_bundle",
            "status": "candidate_for_next_ppo",
            "evidence": str(
                ROOT
                / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
                "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
            ),
            "reason": "40 motions, 11960 frames, non-degenerate body_pos_w spread, ankles near ground",
        },
        {
            "item": "fk_repaired_split_task_eval",
            "status": "passed_task_contract_but_zero_action",
            "evidence": str(
                ROOT
                / "res/tracking/g1_official_importer_export_fk_repaired_split_task_eval/"
                "tracking_g1_official_importer_export_fk_repaired_split_task_eval.json"
            ),
            "reason": (
                f"40/40 rows ok, total_done_count={aggregate.get('total_done_count')}, "
                f"reward_mean={(aggregate.get('reward_mean') or {}).get('mean')}"
            ),
        },
        {
            "item": "old_scaled_ppo_chain",
            "status": "diagnostic_only",
            "evidence": str(
                ROOT
                / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/"
                "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json"
            ),
            "reason": "endpoint z-error/termination diagnostics are retained, but the teacher chain should be rerun",
        },
        {
            "item": "fk_repaired_full_bundle_ppo_training",
            "status": "completed_local_virtual_training",
            "evidence": str(
                ROOT
                / "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run/"
                "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run.json"
            ),
            "reason": (
                f"1000-iteration PPO completed on GPUs 4/7 with "
                f"{fk_ppo_training.get('run', {}).get('checkpoint_count')} checkpoints"
            ),
        },
        {
            "item": "fk_repaired_full_bundle_ppo_eval",
            "status": "completed_but_not_downstream_ready",
            "evidence": str(
                ROOT
                / "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval/"
                "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval.json"
            ),
            "reason": (
                f"done_count_total={fk_eval_done_count}, total_env_steps={fk_eval_total_steps}, "
                f"done_rate={fk_eval_done_rate}, reward_mean="
                f"{(fk_eval_metrics.get('reward', {}).get('mean_over_steps') or {}).get('mean')}"
            ),
        },
    ]
    summary = {
        "status": "ok_fk_repaired_data_quality_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_fk_repaired_data_quality_gate",
        "scope": "Machine-readable handoff gate from old degenerate body_pos_w tracking chain to FK-repaired PPO.",
        "checks": checks,
        "gate": gate,
        "rows": rows,
        "inputs": {
            "degeneracy_audit": str(
                ROOT
                / "res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/"
                "motion_bundle_body_position_degeneracy_audit.json"
            ),
            "fk_repaired_bundle": str(
                ROOT
                / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
                "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
            ),
            "fk_repaired_split_task_eval": str(
                ROOT
                / "res/tracking/g1_official_importer_export_fk_repaired_split_task_eval/"
                "tracking_g1_official_importer_export_fk_repaired_split_task_eval.json"
            ),
        },
        "outputs": {
            "json": str(OUT / "fk_repaired_data_quality_gate.json"),
            "tsv": str(OUT / "fk_repaired_data_quality_gate.tsv"),
            "md": str(OUT / "fk_repaired_data_quality_gate.md"),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_tracking_data_quality_gate_not_paper_level",
            "summary": (
                "The old scaled-PPO downstream chain is retained for diagnosis, and the FK-repaired full-bundle PPO "
                "path now runs end-to-end. The latest checkpoint evaluation is still not teacher-rollout ready "
                "because done/termination remains near one done per env-step."
            ),
        },
    }
    (OUT / "fk_repaired_data_quality_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUT / "fk_repaired_data_quality_gate.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "status", "evidence", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# FK-Repaired Tracking Data Quality Gate",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This gate records why the old scaled-PPO chain is diagnostic-only and why the next PPO attempt should use "
        "the FK-repaired full public-motion bundle.",
        "",
        f"- Ready for FK-repaired full PPO attempt: `{gate['ready_for_fk_repaired_full_ppo_attempt']}`",
        f"- Paper-level tracking ready: `{gate['paper_level_tracking_ready']}`",
        f"- Old scaled chain trust level: `{gate['old_scaled_chain_trust_level']}`",
        "",
        "## Rows",
        "",
        "| Item | Status | Reason |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(f"| `{row['item']}` | `{row['status']}` | {row['reason']} |")
    md_lines.extend(["", "## Next Action", "", gate["next_action"], ""])
    (OUT / "fk_repaired_data_quality_gate.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
