#!/usr/bin/env python3
"""Full-size endpoint-threshold sweep for the robot-order FK PPO eval.

The previous endpoint diagnostics isolated the official z-only ``ee_body_pos``
termination as the dominant tracking-quality gate, but simply removing endpoint
bodies is too far from the paper path.  This script keeps all four official
endpoint bodies active and sweeps only the z threshold under the same
2048-env x 299-step checkpoint-eval scope.

This is still a diagnostic candidate, not a paper-level metric.  It is intended
to decide whether the next tracking repair should be a termination calibration
candidate or a deeper motion-target/body-semantics fix before launching another
full PPO run.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
ENDPOINT_GROUP_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.py"
)
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep"
)
TARGET_REFRESH_EVAL_JSON = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json"
)
SOURCE_DIAGNOSTIC_JSON = (
    ROOT
    / "res/tracking/robot_order_fk_wrist_endpoint_source_full_diagnostic/"
    "robot_order_fk_wrist_endpoint_source_full_diagnostic.json"
)

ALL_ENDPOINTS = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def import_endpoint_group_module():
    spec = importlib.util.spec_from_file_location("bm_endpoint_threshold_sweep_base", ENDPOINT_GROUP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load endpoint-group script: {ENDPOINT_GROUP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def post_step0_rate(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if value is not None else None


def make_variant_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": f"all_endpoint_threshold_{str(threshold).replace('.', 'p')}",
            "active_ee_body_names": ALL_ENDPOINTS,
            "threshold": threshold,
            "description": (
                "All official endpoint bodies remain active; only the z-only ee_body_pos threshold is calibrated."
            ),
        }
        for threshold in THRESHOLDS
    ]


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "active_threshold",
        "done_rate",
        "post_step0_done_rate",
        "post_step0_active_ee_body_pos_rate",
        "post_step0_manual_all_endpoint_rate_0p25",
        "post_step0_manual_ankle_endpoint_rate_0p25",
        "post_step0_manual_wrist_endpoint_rate_0p25",
        "done_rate_delta_vs_target_refresh",
        "post_step0_done_rate_delta_vs_target_refresh",
        "keeps_all_official_endpoint_bodies",
        "variant_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Robot-Order FK Endpoint Threshold Sweep",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target-refresh baseline done rate: `{summary['comparison_to_baselines']['target_refresh_done_rate']}`",
        f"- Best variant: `{summary['comparison_to_baselines']['best_variant_by_done_rate']}`",
        f"- Best done rate: `{summary['comparison_to_baselines']['best_done_rate']}`",
        f"- Recommended next action: `{summary['interpretation']['recommended_next_action']}`",
        "",
        "## Rows",
        "",
        "| threshold | done rate | post-step0 done | active ee rate | manual original all-endpoint rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variant_rows"]:
        lines.append(
            "| {active_threshold} | {done_rate} | {post_step0_done_rate} | "
            "{post_step0_active_ee_body_pos_rate} | {post_step0_manual_all_endpoint_rate_0p25} |".format(**row)
        )
    lines.extend(
        [
            "",
            "This sweep keeps all official endpoint bodies active and changes only the z threshold. It remains a "
            "local diagnostic candidate, not a paper tracking metric, not DAgger/VAE/diffusion evidence, and not "
            "real-robot evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    target_refresh = load_json(TARGET_REFRESH_EVAL_JSON)
    source_diagnostic = load_json(SOURCE_DIAGNOSTIC_JSON)

    module = import_endpoint_group_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.VARIANTS = make_variant_rows()
    module.RELAXED_ENDPOINT_THRESHOLD = max(THRESHOLDS)

    compatible_training_summary = module.make_base_compatible_training_summary()
    variant_rows: list[dict[str, Any]] = []
    for variant in module.VARIANTS:
        row = module.run_variant(variant, compatible_training_summary)
        compact = {key: value for key, value in row.items() if key != "base_summary"}
        compact["keeps_all_official_endpoint_bodies"] = compact.get("active_ee_body_names") == ALL_ENDPOINTS
        variant_rows.append(compact)

    target_metrics = target_refresh.get("run", {}).get("metrics", {})
    target_done_rate = (
        target_metrics.get("done_count_total") / target_metrics.get("total_env_steps")
        if target_metrics.get("done_count_total") is not None and target_metrics.get("total_env_steps")
        else None
    )
    target_timeseries = Path(target_refresh.get("outputs", {}).get("timeseries_csv", ""))
    target_post_step0_done_rate = None
    if target_timeseries.is_file():
        total = 0.0
        steps = 0
        num_envs = int(target_refresh.get("config", {}).get("num_envs", 2048))
        with target_timeseries.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row["step"])) == 0:
                    continue
                total += float(row["done_count"])
                steps += 1
        target_post_step0_done_rate = total / float(max(steps * num_envs, 1)) if steps else None

    for row in variant_rows:
        row["done_rate_delta_vs_target_refresh"] = (
            row["done_rate"] - target_done_rate if row.get("done_rate") is not None and target_done_rate is not None else None
        )
        row["post_step0_done_rate_delta_vs_target_refresh"] = (
            row["post_step0_done_rate"] - target_post_step0_done_rate
            if row.get("post_step0_done_rate") is not None and target_post_step0_done_rate is not None
            else None
        )

    completed_rows = [row for row in variant_rows if row.get("status") == "ok" and row.get("done_rate") is not None]
    best = min(completed_rows, key=lambda row: row["done_rate"]) if completed_rows else {}
    moderate_rows = [
        row
        for row in completed_rows
        if float(row["active_threshold"]) <= 0.40
        and row.get("done_rate_delta_vs_target_refresh") is not None
        and row["done_rate_delta_vs_target_refresh"] < -0.05
    ]
    recommended = "deeper_motion_body_semantics_repair"
    if moderate_rows:
        recommended = "evaluate_threshold_candidate_before_full_ppo"
    elif best and best.get("done_rate_delta_vs_target_refresh", 0.0) < -0.05:
        recommended = "threshold_only_improves_but_requires_large_relaxation"

    rows_csv = OUT / "endpoint_threshold_sweep_rows.csv"
    write_rows_csv(rows_csv, variant_rows)
    summary = {
        "status": "ok_endpoint_threshold_sweep_completed"
        if len(completed_rows) == len(module.VARIANTS)
        else "failed_endpoint_threshold_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_endpoint_threshold_sweep",
        "scope": (
            "Same checkpoint/seed 2048-env x 299-step robot-order FK eval variants that keep all official "
            "ee_body_pos endpoint bodies active and sweep only the z threshold."
        ),
        "config": {
            "thresholds": THRESHOLDS,
            "active_ee_body_names": ALL_ENDPOINTS,
            "seed": int(module.DEFAULT_SEED),
            "num_envs": int(module.DEFAULT_NUM_ENVS),
            "eval_steps": int(module.DEFAULT_EVAL_STEPS),
            "target_gpus": list(module.TARGET_GPUS),
        },
        "inputs": {
            "target_refresh_eval_json": str(TARGET_REFRESH_EVAL_JSON),
            "source_diagnostic_json": str(SOURCE_DIAGNOSTIC_JSON),
            "endpoint_group_script": str(ENDPOINT_GROUP_SCRIPT),
        },
        "source_context": {
            "source_done_rate": source_diagnostic.get("metrics", {}).get("done_rate"),
            "source_ee_body_pos_rate": source_diagnostic.get("metrics", {}).get("ee_body_pos_rate"),
            "source_pre_wrist_done_rate_mean": source_diagnostic.get("metrics", {})
            .get("pre_wrist_done_rate", {})
            .get("mean"),
            "source_pre_ankle_done_rate_mean": source_diagnostic.get("metrics", {})
            .get("pre_ankle_done_rate", {})
            .get("mean"),
        },
        "comparison_to_baselines": {
            "target_refresh_done_rate": target_done_rate,
            "target_refresh_post_step0_done_rate": target_post_step0_done_rate,
            "best_variant_by_done_rate": best.get("variant", ""),
            "best_threshold": best.get("active_threshold"),
            "best_done_rate": best.get("done_rate"),
            "best_done_rate_delta_vs_target_refresh": best.get("done_rate_delta_vs_target_refresh"),
            "best_post_step0_done_rate": best.get("post_step0_done_rate"),
            "best_post_step0_done_rate_delta_vs_target_refresh": best.get(
                "post_step0_done_rate_delta_vs_target_refresh"
            ),
            "moderate_threshold_candidate_count": len(moderate_rows),
        },
        "variant_rows": variant_rows,
        "checks": {
            "all_variants_completed": len(completed_rows) == len(module.VARIANTS),
            "all_keep_official_endpoint_bodies": all(row["keeps_all_official_endpoint_bodies"] for row in variant_rows),
            "same_full_eval_scope_2048x299": all(
                load_json(Path(row["summary_json"])).get("config", {}).get("num_envs") == 2048
                and load_json(Path(row["summary_json"])).get("config", {}).get("eval_steps") == 299
                for row in variant_rows
            ),
            "records_original_0p25_manual_rates": all(
                row.get("post_step0_manual_all_endpoint_rate_0p25") is not None for row in variant_rows
            ),
            "does_not_remove_endpoint_bodies": True,
            "does_not_train": True,
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "claim_level": "tracking_endpoint_threshold_sweep_candidate",
            "recommended_next_action": recommended,
            "why_this_is_mainline": (
                "The sweep tests a concrete tracking-quality repair candidate while preserving the official endpoint "
                "body set. If only very large threshold relaxation helps, the next repair should target motion/body "
                "semantics rather than training a new PPO policy on a loose termination gate."
            ),
            "why_not_paper_level": (
                "Changing the termination threshold changes the evaluator, so these rows are diagnostics only and "
                "cannot be reported as official BeyondMimic tracking scores."
            ),
        },
        "outputs": {
            "json": str(OUT / "endpoint_threshold_sweep.json"),
            "rows_csv": str(rows_csv),
            "md": str(OUT / "endpoint_threshold_sweep.md"),
            "base_compatible_training_json": str(compatible_training_summary),
        },
    }
    write_json(OUT / "endpoint_threshold_sweep.json", summary)
    write_markdown(OUT / "endpoint_threshold_sweep.md", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "comparison_to_baselines": summary["comparison_to_baselines"],
                "recommended_next_action": recommended,
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok_endpoint_threshold_sweep_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
