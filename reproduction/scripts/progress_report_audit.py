#!/usr/bin/env python3
"""Audit PROGRESS.md against goal.md mandatory progress-report fields."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOC = ROOT / "reproduction/PROGRESS.md"
OUT = ROOT / "res/progress_report_audit"

REQUIRED_FIELDS = [
    "阶段：",
    "状态：",
    "开始时间：",
    "结束时间：",
    "使用环境：",
    "使用代码：",
    "官方/重新实现：",
    "Git commit：",
    "配置：",
    "执行命令：",
    "GPU：",
    "峰值显存：",
    "平均 GPU-Util：",
    "平均功耗：",
    "运行时间：",
    "输出文件：",
    "主要指标：",
    "与论文一致性：",
    "发现的差异：",
    "失败与风险：",
    "下一阶段：",
]

KEY_PROGRESS_MARKERS = {
    "local_inventory": "local inventory has 653 rows",
    "released_figures": "released-data summary has 13 figure rows",
    "adaptive_sampling_discrepancy": "Adaptive-sampling discrepancy evidence",
    "level_b_tracking": "Level B Tracking Notes",
    "core_math_tests": "Master audit result after adding core math unit-test evidence",
    "bm_diffusion_env": "Master audit result after adding `bm_diffusion` environment evidence",
    "gpu_resource": "Master audit result after adding GPU resource monitoring evidence",
    "run_management": "Master audit result after adding run-management schema evidence",
    "reimplementation_package": "Master audit result after adding reimplementation package evidence",
    "resolved_config": "Master audit result after adding resolved config evidence",
    "failed_run_retention": "Master audit result after adding failed-run retention evidence",
    "artifact_manifest": "Master audit result after adding artifact manifest evidence",
    "experiment_protocol": "Master audit result after adding experiment protocol evidence",
    "readme": "Master audit result after adding README evidence",
    "goal_report_path": "Master audit result after adding direct goal-report evidence",
    "final_report_requirements": "Master audit result after adding final-report requirement evidence",
    "final_deliverables": "Master audit result after adding final deliverables evidence",
}


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    field_rows = []
    for field in REQUIRED_FIELDS:
        matching_lines = [line for line in text.splitlines() if line.startswith(field)]
        field_rows.append(
            {
                "kind": "required_field",
                "name": field.rstrip("："),
                "pattern": field,
                "present": bool(matching_lines),
                "nonempty": any(line[len(field) :].strip() not in {"", "完成", "运行成功"} for line in matching_lines),
            }
        )
    marker_rows = [
        {
            "kind": "progress_marker",
            "name": name,
            "pattern": marker,
            "present": marker in text,
            "nonempty": marker in text,
        }
        for name, marker in KEY_PROGRESS_MARKERS.items()
    ]
    rows = field_rows + marker_rows
    missing = [row for row in rows if not (row["present"] and row["nonempty"])]
    master_count_mentions = text.count("Master audit result after adding")
    summary = {
        "status": "ok" if DOC.is_file() and not missing else "failed",
        "experiment_type": "doc_audit",
        "scope": "goal.md section 17 mandatory PROGRESS.md field and stage-marker coverage",
        "document": str(DOC),
        "required_field_count": len(field_rows),
        "progress_marker_count": len(marker_rows),
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing_rows": missing,
        "master_count_mentions": master_count_mentions,
        "rows": rows,
        "checks": {
            "document_exists": DOC.is_file(),
            "all_required_fields_present_and_nonempty": all(row["present"] and row["nonempty"] for row in field_rows),
            "all_key_progress_markers_present": all(row["present"] for row in marker_rows),
            "master_audit_progression_recorded": master_count_mentions >= 13,
            "records_incomplete_boundary": "goal_complete=false" in text,
            "records_missing_checkpoints_and_videos": "Checkpoints and videos are explicitly recorded as `blocked_or_missing`" in text,
            "does_not_only_say_complete_or_success": all(row["nonempty"] for row in field_rows),
            "atomic_write_used": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "PROGRESS.md records the required report fields and key progress markers, but it also records that "
                "checkpoints, videos, live training/evaluation, and deployment remain incomplete."
            ),
        },
        "outputs": {
            "json": str(OUT / "progress_report_audit.json"),
            "tsv": str(OUT / "progress_report_audit.tsv"),
        },
    }
    atomic_write_text(OUT / "progress_report_audit.json", json.dumps(summary, indent=2, sort_keys=True))
    tsv_tmp = OUT / "progress_report_audit.tsv.tmp"
    with tsv_tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["kind", "name", "pattern", "present", "nonempty"])
        writer.writeheader()
        writer.writerows(rows)
    tsv_tmp.replace(OUT / "progress_report_audit.tsv")
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
