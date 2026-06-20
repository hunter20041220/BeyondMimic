#!/usr/bin/env python3
"""Create report assets for official-importer-export full-dataset replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
AUDIT = (
    ROOT
    / "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
    "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"
)
REFERENCE_VIDEO = (
    ROOT
    / "res/visualization/official_importer_export_full_dataset_reference_replay/"
    "official_importer_export_full_dataset_reference_replay_video_asset.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_replay_full_dataset"
ASSET_JSON = OUT / "official_importer_export_replay_full_dataset_report_assets.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def motion_family(motion: str) -> str:
    head = motion.split("_", 1)[0]
    return "".join(ch for ch in head if not ch.isdigit()) or head


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_json(AUDIT)
    video = load_json(REFERENCE_VIDEO)
    rows = list(audit["rows"])
    for row in rows:
        row["motion_family"] = motion_family(row["motion"])

    family_rows: list[dict[str, Any]] = []
    for family in sorted({row["motion_family"] for row in rows}):
        group = [row for row in rows if row["motion_family"] == family]
        ok_count = sum(1 for row in group if row["status"] == "ok")
        duration_values = [float(row["duration_seconds"]) for row in group]
        family_rows.append(
            {
                "motion_family": family,
                "motion_count": len(group),
                "ok_count": ok_count,
                "failed_count": len(group) - ok_count,
                "completion_rate": ok_count / len(group) if group else 0.0,
                "duration_seconds_mean": sum(duration_values) / len(duration_values),
                "duration_seconds_min": min(duration_values),
                "duration_seconds_max": max(duration_values),
            }
        )

    rows_csv = OUT / "official_importer_export_replay_full_dataset_rows.csv"
    family_csv = OUT / "official_importer_export_replay_full_dataset_family_summary.csv"
    summary_csv = OUT / "official_importer_export_replay_full_dataset_summary.csv"
    rows_fields = [
        "motion",
        "motion_family",
        "status",
        "returncode",
        "duration_seconds",
        "official_loop_body_completed",
        "shutdown_warning",
        "official_loop_call_299",
        "fake_wandb_download",
        "uses_official_importer_export_usd",
        "motion_npz",
        "log",
        "motion_audit",
    ]
    write_csv(rows_csv, rows, rows_fields)
    write_csv(
        family_csv,
        family_rows,
        [
            "motion_family",
            "motion_count",
            "ok_count",
            "failed_count",
            "completion_rate",
            "duration_seconds_mean",
            "duration_seconds_min",
            "duration_seconds_max",
        ],
    )

    aggregate = audit["aggregate"]
    summary_rows = [
        {"metric": "status", "value": audit["status"], "claim_level": audit["interpretation"]["claim_level"]},
        {"metric": "row_count", "value": aggregate["row_count"], "claim_level": "local_virtual_replay_loop"},
        {"metric": "ok_count", "value": aggregate["ok_count"], "claim_level": "local_virtual_replay_loop"},
        {"metric": "failed_count", "value": aggregate["failed_count"], "claim_level": "local_virtual_replay_loop"},
        {
            "metric": "completion_rate",
            "value": aggregate["ok_count"] / aggregate["row_count"] if aggregate["row_count"] else 0.0,
            "claim_level": "local_virtual_replay_loop",
        },
        {
            "metric": "total_replayed_steps",
            "value": aggregate["total_replayed_steps"],
            "claim_level": "local_virtual_replay_loop",
        },
        {
            "metric": "total_duration_seconds",
            "value": aggregate["total_duration_seconds"],
            "claim_level": "local_virtual_replay_loop",
        },
        {
            "metric": "reference_video_asset",
            "value": video["assets"]["mp4"],
            "claim_level": video["claim_level"],
        },
    ]
    write_csv(summary_csv, summary_rows, ["metric", "value", "claim_level"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=False)
    families = [row["motion_family"] for row in family_rows]
    axes[0].bar(families, [row["ok_count"] for row in family_rows], color="#2563eb", label="ok")
    axes[0].bar(
        families,
        [row["failed_count"] for row in family_rows],
        bottom=[row["ok_count"] for row in family_rows],
        color="#dc2626",
        label="failed",
    )
    axes[0].set_ylabel("Motion count")
    axes[0].set_title("Official-importer-export full-dataset replay completion by motion family")
    axes[0].legend(loc="upper right")
    axes[1].bar(families, [row["duration_seconds_mean"] for row in family_rows], color="#16a34a")
    axes[1].set_ylabel("Mean replay duration (s)")
    axes[1].set_xlabel("Motion family")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    completion_png = OUT / "official_importer_export_replay_completion_by_family.png"
    fig.savefig(completion_png, dpi=180)
    plt.close(fig)

    sorted_rows = sorted(rows, key=lambda row: float(row["duration_seconds"]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, 8.0))
    colors = ["#2563eb" if row["status"] == "ok" else "#dc2626" for row in sorted_rows]
    ax.barh([row["motion"] for row in sorted_rows], [float(row["duration_seconds"]) for row in sorted_rows], color=colors)
    ax.set_xlabel("Replay duration (s)")
    ax.set_title("Official replay_npz loop duration for 40 public motions")
    ax.invert_yaxis()
    fig.tight_layout()
    duration_png = OUT / "official_importer_export_replay_duration_by_motion.png"
    fig.savefig(duration_png, dpi=180)
    plt.close(fig)

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer-Export Full-Dataset Replay Report Assets",
                "",
                "These assets summarize the full 40-motion official `replay_npz.py` loop audit using the",
                "captured official-importer-export G1 USDA and the official csv-loop NPZ inputs.",
                "",
                "## Summary",
                "",
                f"- status: `{audit['status']}`",
                f"- rows: `{aggregate['row_count']}`",
                f"- ok rows: `{aggregate['ok_count']}`",
                f"- failed rows: `{aggregate['failed_count']}`",
                f"- total replayed steps: `{aggregate['total_replayed_steps']}`",
                f"- shutdown warnings: `{aggregate['shutdown_warning_count']}`",
                "",
                "## Claim Boundary",
                "",
                "This is local virtual reference-replay evidence through the official replay loop body. It is not a",
                "trained policy evaluation, not unmodified live converter-entry success, not paper-level tracking",
                "metrics, not Fig. 5/Fig. 6 guided diffusion, and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assets = {
        "rows_csv": str(rows_csv),
        "family_summary_csv": str(family_csv),
        "summary_csv": str(summary_csv),
        "completion_by_family_png": str(completion_png),
        "duration_by_motion_png": str(duration_png),
        "readme": str(readme),
        "reference_video_asset_json": str(REFERENCE_VIDEO),
        "reference_video_mp4": video["assets"]["mp4"],
    }
    checks = {
        "source_audit_ok": audit["status"] == "ok_official_replay_npz_loop_full_dataset_with_official_importer_export",
        "all_40_rows_ok": aggregate["row_count"] == 40 and aggregate["ok_count"] == 40 and aggregate["failed_count"] == 0,
        "all_rows_reached_official_loop_299": audit["checks"]["all_rows_reached_official_loop_299"],
        "uses_official_importer_export_usd": audit["checks"]["uses_official_importer_export_usd"],
        "does_not_claim_paper_level_replay": audit["checks"]["does_not_claim_paper_level_replay"],
        "reference_video_exists": Path(video["assets"]["mp4"]).is_file() and Path(video["assets"]["mp4"]).stat().st_size > 0,
        "all_report_assets_exist": all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in assets.values() if not path.endswith(".mp4")),
    }
    summary = {
        "status": "ok_official_importer_export_replay_full_dataset_report_assets" if all(checks.values()) else "failed",
        "experiment_type": "official_importer_export_replay_full_dataset_report_assets",
        "source_audit": str(AUDIT),
        "source_status": audit["status"],
        "aggregate": aggregate,
        "family_summary": family_rows,
        "assets": assets,
        "checks": checks,
        "interpretation": {
            "claim_level": audit["interpretation"]["claim_level"],
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "These assets summarize a local official-loop reference replay over public motions. They are useful "
                "for the English report/PPT but do not constitute trained policy evaluation, paper tracking metrics, "
                "Fig.5/Fig.6 evidence, TensorRT deployment, or real-robot validation."
            ),
        },
    }
    ASSET_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(ASSET_JSON)}, sort_keys=True))
    if summary["status"] != "ok_official_importer_export_replay_full_dataset_report_assets":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
