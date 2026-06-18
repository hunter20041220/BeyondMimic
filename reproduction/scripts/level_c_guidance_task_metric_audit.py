#!/usr/bin/env python3
"""Task-level guidance metric audit for guidance objectives.

This consolidates the five before/after primary metrics from the local guidance
visualization with the formula-level task scale sweep and the full-split
public-data offline/reverse guidance summaries. It deliberately keeps the
closed-loop, success/failure video, and Fig. 5/Fig. 6 reproduction boundary
explicit.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEBUG_JSON = ROOT / "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json"
SCALE_JSON = ROOT / "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json"
FULL_SPLIT_JSON = ROOT / "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
OUT = ROOT / "res/level_c/guidance_task_metric_audit"

TASKS = ["joystick", "waypoint", "obstacle_avoidance", "inpainting", "composed"]
SCALE_TASK = {
    "joystick": "joystick",
    "waypoint": "waypoint",
    "obstacle_avoidance": "obstacle_avoidance",
    "inpainting": "inpainting",
    "composed": "composed_objectives",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task",
        "scale_sweep_task",
        "primary_metric",
        "direction",
        "before",
        "after",
        "delta",
        "improved",
        "best_scale",
        "best_cost_delta",
        "gradient_norm",
        "offline_full_split_mean_best_cost_delta",
        "offline_full_split_primary_improved_count",
        "offline_full_split_window_count",
        "offline_full_split_row_count",
        "reverse_full_split_mean_best_cost_delta",
        "reverse_full_split_primary_improved_count",
        "reverse_full_split_window_count",
        "reverse_full_split_row_count",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    debug = load_json(DEBUG_JSON)
    scale = load_json(SCALE_JSON)
    full_split = load_json(FULL_SPLIT_JSON)
    full_split_rows = {
        (row["mode"], row["task"]): row
        for row in full_split["rows"]
        if row.get("mode") in {"offline", "reverse"}
    }

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        debug_payload = debug["per_task"][task]
        primary_metric = debug_payload["primary_metric"]
        primary = debug_payload["metrics"][primary_metric]
        scale_task = SCALE_TASK[task]
        scale_summary = scale["task_summaries"][scale_task]
        offline_split = full_split_rows[("offline", scale_task)]
        reverse_split = full_split_rows[("reverse", scale_task)]
        direction = "higher_is_better" if primary_metric == "min_obstacle_clearance" else "lower_is_better"
        rows.append(
            {
                "task": task,
                "scale_sweep_task": scale_task,
                "primary_metric": primary_metric,
                "direction": direction,
                "before": float(primary["before"]),
                "after": float(primary["after"]),
                "delta": float(primary["delta"]),
                "improved": bool(primary["improved"]),
                "best_scale": float(scale_summary["best_scale"]),
                "best_cost_delta": float(scale_summary["best_cost_delta"]),
                "gradient_norm": float(scale_summary["gradient_norm"]),
                "offline_full_split_mean_best_cost_delta": float(offline_split["mean_best_cost_delta"]),
                "offline_full_split_primary_improved_count": int(offline_split["primary_improved_count"]),
                "offline_full_split_window_count": int(offline_split["window_count"]),
                "offline_full_split_row_count": int(offline_split["row_count"]),
                "reverse_full_split_mean_best_cost_delta": float(reverse_split["mean_best_cost_delta"]),
                "reverse_full_split_primary_improved_count": int(reverse_split["primary_improved_count"]),
                "reverse_full_split_window_count": int(reverse_split["window_count"]),
                "reverse_full_split_row_count": int(reverse_split["row_count"]),
                "status": "public_data_reverse_guidance_metrics",
            }
        )

    metrics = {
        "row_count": len(rows),
        "improved_task_count": sum(1 for row in rows if row["improved"]),
        "scale_sweep_task_count": len({row["scale_sweep_task"] for row in rows}),
        "mean_primary_delta": float(sum(row["delta"] for row in rows) / max(len(rows), 1)),
        "min_best_cost_delta": float(min(row["best_cost_delta"] for row in rows)),
        "min_gradient_norm": float(min(row["gradient_norm"] for row in rows)),
        "mean_offline_full_split_best_cost_delta": float(
            sum(row["offline_full_split_mean_best_cost_delta"] for row in rows) / max(len(rows), 1)
        ),
        "mean_reverse_full_split_best_cost_delta": float(
            sum(row["reverse_full_split_mean_best_cost_delta"] for row in rows) / max(len(rows), 1)
        ),
        "offline_full_split_row_count": int(sum(row["offline_full_split_row_count"] for row in rows)),
        "reverse_full_split_row_count": int(sum(row["reverse_full_split_row_count"] for row in rows)),
    }
    task_primary_metrics = {
        row["task"]: {
            "primary_metric": row["primary_metric"],
            "direction": row["direction"],
            "before": row["before"],
            "after": row["after"],
            "delta": row["delta"],
            "improved": row["improved"],
            "scale_sweep_task": row["scale_sweep_task"],
            "best_scale": row["best_scale"],
            "best_cost_delta": row["best_cost_delta"],
        }
        for row in rows
    }
    checks = {
        "debug_visualization_status_ok": debug.get("status") == "ok",
        "scale_sweep_status_ok": scale.get("status") == "ok",
        "five_tasks_recorded": sorted(row["task"] for row in rows) == sorted(TASKS),
        "all_primary_metrics_finite": all(
            finite_number(row["before"]) and finite_number(row["after"]) and finite_number(row["delta"])
            for row in rows
        ),
        "all_primary_metrics_improve": all(row["improved"] for row in rows),
        "all_tasks_have_scale_sweep_summary": all(row["scale_sweep_task"] in scale["task_summaries"] for row in rows),
        "all_scale_sweep_best_deltas_positive": all(row["best_cost_delta"] > 0.0 for row in rows),
        "all_scale_sweep_gradients_nonzero": all(row["gradient_norm"] > 0.0 for row in rows),
        "full_split_offline_links_present": all(
            row["offline_full_split_row_count"] > 0 and row["offline_full_split_mean_best_cost_delta"] is not None
            for row in rows
        ),
        "full_split_reverse_links_present": all(
            row["reverse_full_split_row_count"] > 0 and row["reverse_full_split_mean_best_cost_delta"] is not None
            for row in rows
        ),
        "does_not_claim_success_failure_videos": True,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_fig5_fig6_reproduction": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_guidance_task_metric_audit",
        "scope": "task-level guidance primary metrics plus formula-level scale-sweep and full-split public-data linkage",
        "metrics": metrics,
        "task_primary_metrics": task_primary_metrics,
        "rows": rows,
        "checks": checks,
        "inputs": {
            "guidance_debug_visualization": str(DEBUG_JSON),
            "guidance_task_scale_sweep": str(SCALE_JSON),
            "guidance_full_split_result_table": str(FULL_SPLIT_JSON),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "The audit centralizes debug task metrics, scale-sweep evidence, and full-split public-data "
                "guidance summaries. It does not run a trained denoising policy, closed-loop simulation, "
                "success/failure video evaluation, or paper Fig. 5/Fig. 6 reproduction."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_guidance_task_metric_audit.json"),
            "tsv": str(OUT / "level_c_guidance_task_metric_audit.tsv"),
        },
    }
    (OUT / "level_c_guidance_task_metric_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "level_c_guidance_task_metric_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
