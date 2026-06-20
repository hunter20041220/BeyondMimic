#!/usr/bin/env python3
"""Build report assets for the scaled official-importer-export PPO multiseed eval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_checkpoint_multiseed_eval"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    return float(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_json(SOURCE)
    rows = data.get("rows", [])
    seeds = [str(row["seed"]) for row in rows]

    summary_rows = [
        {
            "seed": row["seed"],
            "reward_mean": row["reward_mean"],
            "done_count_total": row["done_count_total"],
            "error_anchor_pos_mean": row["error_anchor_pos_mean"],
            "error_body_pos_mean": row["error_body_pos_mean"],
            "error_joint_pos_mean": row["error_joint_pos_mean"],
            "num_envs": row["num_envs"],
            "eval_steps": row["eval_steps"],
            "total_env_steps": row["total_env_steps"],
        }
        for row in rows
    ]
    summary_csv = OUT / "scaled_ppo_multiseed_eval_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "seed",
            "reward_mean",
            "done_count_total",
            "error_anchor_pos_mean",
            "error_body_pos_mean",
            "error_joint_pos_mean",
            "num_envs",
            "eval_steps",
            "total_env_steps",
        ],
    )

    reward_png = OUT / "scaled_ppo_multiseed_reward_done.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(seeds, [as_float(row["reward_mean"]) for row in rows], color="#2f6f73")
    axes[0].set_title("Reward Mean by Seed")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Reward mean")
    axes[1].bar(seeds, [as_float(row["done_count_total"]) for row in rows], color="#b45f3c")
    axes[1].set_title("Done Count by Seed")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Done count")
    fig.tight_layout()
    fig.savefig(reward_png, dpi=160)
    plt.close(fig)

    error_png = OUT / "scaled_ppo_multiseed_tracking_errors.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(seeds))
    width = 0.25
    ax.bar([i - width for i in x], [as_float(row["error_anchor_pos_mean"]) for row in rows], width, label="anchor pos")
    ax.bar([i for i in x], [as_float(row["error_body_pos_mean"]) for row in rows], width, label="body pos")
    ax.bar([i + width for i in x], [as_float(row["error_joint_pos_mean"]) for row in rows], width, label="joint pos")
    ax.set_xticks(list(x))
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean error")
    ax.set_title("Tracking Errors by Seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(error_png, dpi=160)
    plt.close(fig)

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer-Export Scaled PPO Multiseed Eval Assets",
                "",
                "These assets summarize three full 299-step local virtual evaluations of the iteration-999 scaled PPO checkpoint.",
                "",
                "Claim level: local virtual multiseed scaled tracking evaluation.",
                "",
                "Limitations: not an official BeyondMimic checkpoint, not paper-level tracking, not DAgger, not Fig.5/Fig.6, and not real robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    checks = {
        "source_status_ok": data["status"]
        == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_completed",
        "all_seeds_completed": data["checks"]["all_seeds_completed"],
        "all_use_official_importer_export_usd": data["checks"]["all_use_official_importer_export_usd"],
        "summary_csv_exists": summary_csv.is_file(),
        "png_assets_exist": reward_png.is_file() and error_png.is_file(),
        "readme_exists": readme.is_file(),
        "does_not_claim_paper_level_eval": True,
        "does_not_claim_real_robot": True,
    }
    asset = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_report_assets",
        "source": str(SOURCE),
        "claim_level": "local_virtual_multiseed_scaled_tracking_eval_report_asset",
        "metrics": data["metrics"],
        "aggregate": data["aggregate"],
        "checks": checks,
        "assets": {
            "summary_csv": str(summary_csv),
            "reward_done_png": str(reward_png),
            "tracking_errors_png": str(error_png),
            "readme": str(readme),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "Report assets summarize local virtual multiseed checkpoint evaluation. They do not provide official "
                "BeyondMimic checkpoints, paper-level tracking metrics, Fig.5/Fig.6 guidance, TensorRT deployment, or "
                "real-robot evidence."
            ),
        },
    }
    asset_path = OUT / "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_assets.json"
    asset_path.write_text(json.dumps(asset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": asset["status"], "json": str(asset_path)}, sort_keys=True))
    if asset["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
