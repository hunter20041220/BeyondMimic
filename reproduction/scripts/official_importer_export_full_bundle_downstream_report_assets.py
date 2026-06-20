#!/usr/bin/env python3
"""Create report assets for official-importer-export downstream VAE/diffusion runs."""

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
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json"
)
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
    "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_downstream"


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
    state_latent = load_json(STATE_LATENT_JSON)
    diffusion = load_json(DIFFUSION_JSON)

    vae_worker = vae["worker_summary"]
    state_worker = state_latent["worker_summary"]
    diffusion_worker = diffusion["worker_summary"]

    plt.style.use("seaborn-v0_8-whitegrid")

    vae_rows = vae_worker["training"]["epoch_rows"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot([row["epoch"] for row in vae_rows], [row["train_reconstruction_mse"] for row in vae_rows], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train reconstruction MSE")
    ax.set_title("Official-importer-export teacher-rollout action VAE training")
    fig.tight_layout()
    vae_curve_png = OUT / "official_importer_downstream_vae_training_curve.png"
    fig.savefig(vae_curve_png, dpi=180)
    plt.close(fig)

    diffusion_rows = diffusion_worker["training"]["epoch_rows"]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    epochs = [row["epoch"] for row in diffusion_rows]
    axes[0].plot(epochs, [row["train_token_mse"] for row in diffusion_rows], marker="o", label="train token MSE")
    axes[0].plot(
        epochs,
        [row["validation_pred_token_mse"] for row in diffusion_rows],
        marker="s",
        label="validation pred token MSE",
    )
    axes[0].set_ylabel("MSE")
    axes[0].legend(loc="upper right")
    axes[1].plot(
        epochs,
        [row["validation_denoising_improvement_ratio"] for row in diffusion_rows],
        marker="o",
        color="#16a34a",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Denoising improvement")
    axes[1].set_title("Validation denoising improvement over noisy input")
    fig.suptitle("Official-importer-export state-latent denoiser training", y=0.99)
    fig.tight_layout()
    diffusion_curve_png = OUT / "official_importer_downstream_diffusion_training_curve.png"
    fig.savefig(diffusion_curve_png, dpi=180)
    plt.close(fig)

    split_rows = []
    for split, metrics in vae_worker["evaluation"].items():
        split_rows.append(
            {
                "stage": "vae",
                "split": split,
                "sample_count": metrics["sample_count"],
                "action_mse": metrics["action_mse"],
                "action_abs_error_mean": metrics["action_abs_error_mean"],
                "kl_mean": metrics["kl_mean"],
                "pred_token_mse": "",
                "noisy_token_mse": "",
                "denoising_improvement_ratio": "",
            }
        )
    for split, metrics in diffusion_worker["evaluation"].items():
        split_rows.append(
            {
                "stage": "diffusion",
                "split": split,
                "sample_count": diffusion_worker["dataset"]["split_counts"][split],
                "action_mse": "",
                "action_abs_error_mean": "",
                "kl_mean": "",
                "pred_token_mse": metrics["pred_token_mse"],
                "noisy_token_mse": metrics["noisy_token_mse"],
                "denoising_improvement_ratio": metrics["denoising_improvement_ratio"],
            }
        )
    split_csv = OUT / "official_importer_downstream_split_metrics.csv"
    write_csv(
        split_csv,
        split_rows,
        [
            "stage",
            "split",
            "sample_count",
            "action_mse",
            "action_abs_error_mean",
            "kl_mean",
            "pred_token_mse",
            "noisy_token_mse",
            "denoising_improvement_ratio",
        ],
    )

    stage_rows = [
        {
            "stage": "teacher_rollout_vae",
            "status": vae["status"],
            "sample_count": vae_worker["dataset"]["sample_count"],
            "motion_time_step_max": vae_worker["dataset"]["motion_time_step_max"],
            "window_count": "",
            "test_action_mse": vae_worker["evaluation"]["test"]["action_mse"],
            "test_denoising_improvement_ratio": "",
        },
        {
            "stage": "state_latent_dataset",
            "status": state_latent["status"],
            "sample_count": state_worker["dataset"]["sample_count"],
            "motion_time_step_max": "",
            "window_count": state_worker["dataset"]["window_count"],
            "test_action_mse": "",
            "test_denoising_improvement_ratio": "",
        },
        {
            "stage": "state_latent_diffusion",
            "status": diffusion["status"],
            "sample_count": diffusion_worker["dataset"]["sample_count"],
            "motion_time_step_max": "",
            "window_count": diffusion_worker["dataset"]["window_count"],
            "test_action_mse": "",
            "test_denoising_improvement_ratio": diffusion_worker["evaluation"]["test"][
                "denoising_improvement_ratio"
            ],
        },
    ]
    stage_csv = OUT / "official_importer_downstream_stage_summary.csv"
    write_csv(
        stage_csv,
        stage_rows,
        [
            "stage",
            "status",
            "sample_count",
            "motion_time_step_max",
            "window_count",
            "test_action_mse",
            "test_denoising_improvement_ratio",
        ],
    )

    summary = {
        "status": "ok_official_importer_export_full_bundle_downstream_assets",
        "claim_level": "local_virtual_official_importer_export_full_bundle_downstream_report_asset",
        "source_jsons": {
            "vae": str(VAE_JSON),
            "state_latent": str(STATE_LATENT_JSON),
            "diffusion": str(DIFFUSION_JSON),
        },
        "metrics": {
            "vae_sample_count": vae_worker["dataset"]["sample_count"],
            "vae_test_action_mse": vae_worker["evaluation"]["test"]["action_mse"],
            "state_latent_window_count": state_worker["dataset"]["window_count"],
            "state_latent_weighted_posterior_reconstruction_mse": state_worker["dataset"][
                "weighted_posterior_reconstruction_mse"
            ],
            "diffusion_test_pred_token_mse": diffusion_worker["evaluation"]["test"]["pred_token_mse"],
            "diffusion_test_noisy_token_mse": diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
            "diffusion_test_denoising_improvement_ratio": diffusion_worker["evaluation"]["test"][
                "denoising_improvement_ratio"
            ],
        },
        "assets": {
            "vae_training_curve_png": str(vae_curve_png),
            "diffusion_training_curve_png": str(diffusion_curve_png),
            "split_metrics_csv": str(split_csv),
            "stage_summary_csv": str(stage_csv),
            "summary_md": str(OUT / "README.md"),
        },
        "checks": {
            "vae_status_ok": vae["status"] == "ok_official_importer_export_full_bundle_teacher_rollout_vae_training",
            "state_latent_status_ok": state_latent["status"]
            == "ok_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset",
            "diffusion_status_ok": diffusion["status"]
            == "ok_official_importer_export_full_bundle_state_latent_diffusion_training",
            "vae_curve_exists": vae_curve_png.is_file() and vae_curve_png.stat().st_size > 0,
            "diffusion_curve_exists": diffusion_curve_png.is_file() and diffusion_curve_png.stat().st_size > 0,
            "csv_assets_exist": split_csv.is_file() and stage_csv.is_file(),
            "uses_official_importer_export_usd": True,
            "does_not_claim_official_checkpoint": True,
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "limitation": (
            "These assets summarize local virtual downstream training on the official-importer-export G1 USDA chain. "
            "They are not official BeyondMimic VAE/diffusion checkpoints, not paper-level closed-loop guidance, not "
            "TensorRT deployment, and not real robot evidence."
        ),
    }

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer-Export Downstream Training Assets",
                "",
                "These assets summarize the local official-importer-export teacher-rollout VAE, state-latent",
                "dataset, and state-latent denoiser training chain.",
                "",
                "## Key Metrics",
                "",
                f"- VAE sample count: `{summary['metrics']['vae_sample_count']}`",
                f"- VAE test action MSE: `{summary['metrics']['vae_test_action_mse']}`",
                f"- State-latent window count: `{summary['metrics']['state_latent_window_count']}`",
                f"- Diffusion test pred token MSE: `{summary['metrics']['diffusion_test_pred_token_mse']}`",
                f"- Diffusion test denoising improvement ratio: "
                f"`{summary['metrics']['diffusion_test_denoising_improvement_ratio']}`",
                "",
                "## Claim Level",
                "",
                "local_virtual_official_importer_export_full_bundle_downstream_report_asset. This is not an official",
                "BeyondMimic checkpoint, not paper-level closed-loop guidance, and not real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["checks"]["summary_md_exists"] = readme.is_file()
    summary_json = OUT / "official_importer_export_full_bundle_downstream_report_assets.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(summary_json), "assets": summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
