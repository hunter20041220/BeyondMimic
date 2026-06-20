#!/usr/bin/env python3
"""Create report assets for the official-importer-export full-bundle VAE run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
VAE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
    "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_vae_training"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vae = load_json(VAE_JSON)
    worker = vae["worker_summary"]

    epoch_rows = worker["training"]["epoch_rows"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot([row["epoch"] for row in epoch_rows], [row["train_reconstruction_mse"] for row in epoch_rows], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train reconstruction MSE")
    ax.set_title("Official-importer full-bundle action VAE training")
    fig.tight_layout()
    curve_png = OUT / "official_importer_export_full_bundle_vae_training_curve.png"
    fig.savefig(curve_png, dpi=180)
    plt.close(fig)

    split_rows = []
    for split, metrics in worker["evaluation"].items():
        split_rows.append(
            {
                "split": split,
                "sample_count": metrics["sample_count"],
                "action_mse": metrics["action_mse"],
                "action_abs_error_mean": metrics["action_abs_error_mean"],
                "latent_mu_abs_mean": metrics["latent_mu_abs_mean"],
                "kl_mean": metrics["kl_mean"],
            }
        )
    split_csv = OUT / "official_importer_export_full_bundle_vae_split_metrics.csv"
    write_csv(
        split_csv,
        split_rows,
        ["split", "sample_count", "action_mse", "action_abs_error_mean", "latent_mu_abs_mean", "kl_mean"],
    )

    epoch_csv = OUT / "official_importer_export_full_bundle_vae_epoch_metrics.csv"
    write_csv(
        epoch_csv,
        epoch_rows,
        ["epoch", "train_reconstruction_mse", "train_kl_mean", "train_total_loss"],
    )

    metrics = {
        "sample_count": worker["dataset"]["sample_count"],
        "obs_dim": worker["dataset"]["obs_dim"],
        "action_dim": worker["dataset"]["action_dim"],
        "motion_time_step_max": worker["dataset"]["motion_time_step_max"],
        "epochs": worker["training"]["epochs"],
        "latent_dim": worker["training"]["latent_dim"],
        "test_action_mse": worker["evaluation"]["test"]["action_mse"],
        "test_action_abs_error_mean": worker["evaluation"]["test"]["action_abs_error_mean"],
        "done_count": worker["dataset"]["done_count"],
    }
    summary = {
        "status": "ok",
        "claim_level": "local_virtual_official_importer_export_full_bundle_vae_training_asset",
        "source_json": str(VAE_JSON),
        "metrics": metrics,
        "assets": {
            "training_curve_png": str(curve_png),
            "split_metrics_csv": str(split_csv),
            "epoch_metrics_csv": str(epoch_csv),
            "summary_md": str(OUT / "README.md"),
        },
        "checks": {
            "vae_status_ok": vae["status"]
            == "ok_official_importer_export_full_bundle_teacher_rollout_vae_training",
            "sample_count_306176": worker["dataset"]["sample_count"] == 306176,
            "action_dim_29": worker["dataset"]["action_dim"] == 29,
            "obs_dim_160": worker["dataset"]["obs_dim"] == 160,
            "test_action_mse_below_0_01": worker["evaluation"]["test"]["action_mse"] < 0.01,
            "curve_exists": curve_png.is_file() and curve_png.stat().st_size > 0,
            "csv_assets_exist": split_csv.is_file() and epoch_csv.is_file(),
            "does_not_claim_official_checkpoint": True,
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_real_robot": True,
        },
        "limitation": (
            "This asset summarizes a local VAE trained on local official-importer-export teacher rollout shards. "
            "It is not the official BeyondMimic VAE checkpoint, not paper-level closed-loop guidance, and not real "
            "robot evidence."
        ),
    }

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer Full-Bundle VAE Training Assets",
                "",
                "These assets summarize the local conditional action VAE trained on teacher rollout shards from the",
                "official-importer-export G1 USDA PPO checkpoint and the 40-motion public bundle.",
                "",
                "## Key Metrics",
                "",
                f"- Sample count: `{metrics['sample_count']}`",
                f"- Observation/action dims: `{metrics['obs_dim']} / {metrics['action_dim']}`",
                f"- Epochs: `{metrics['epochs']}`",
                f"- Test action MSE: `{metrics['test_action_mse']}`",
                f"- Test action abs error mean: `{metrics['test_action_abs_error_mean']}`",
                "",
                "## Claim Level",
                "",
                "local_virtual_official_importer_export_full_bundle_vae_training_asset. This is not an official",
                "BeyondMimic VAE checkpoint, not paper-level closed-loop guidance, and not real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["assets"]["summary_md"] = str(readme)
    summary["checks"]["summary_md_exists"] = readme.is_file()

    asset_json = OUT / "official_importer_export_full_bundle_vae_training_assets.json"
    asset_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "assets": summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
