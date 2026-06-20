#!/usr/bin/env python3
"""Run/aggregate scaled-PPO importer-export multi-seed guidance rollouts.

This wrapper reuses the validated official-importer-export multi-seed
task-conditioned guidance orchestrator, but redirects it to the scaled PPO
single-seed runner and scaled downstream evidence chain. The resulting rows are
local virtual report evidence only: they are not official BeyondMimic Fig. 5/6
success metrics, TensorRT deployment, or real-robot validation.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval as base


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
TASKS = ["joystick", "waypoint", "obstacle_avoidance", "composed"]
OUT = ROOT / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
SUMMARY_JSON = OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
BASE_SUMMARY_JSON = OUT / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
REPORT_OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed"
OK_SINGLE_STATUS = "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval"
OK_MULTI_STATUS = "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
FAILED_MULTI_STATUS = "failed_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"

DEFAULT_SEED_GROUPS = {
    "seed_group_1": {
        "joystick": 20260901,
        "waypoint": 20260902,
        "obstacle_avoidance": 20260903,
        "composed": 20260904,
    },
    "seed_group_2": {
        "joystick": 20260911,
        "waypoint": 20260912,
        "obstacle_avoidance": 20260913,
        "composed": 20260914,
    },
    "seed_group_3": {
        "joystick": 20260921,
        "waypoint": 20260922,
        "obstacle_avoidance": 20260923,
        "composed": 20260924,
    },
    "seed_group_4": {
        "joystick": 20260931,
        "waypoint": 20260932,
        "obstacle_avoidance": 20260933,
        "composed": 20260934,
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], delimiter: str = ",") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def patch_base() -> None:
    extra_seed_groups = {
        key: {task: int(seed) for task, seed in value.items()}
        for key, value in json.loads(os.environ.get("BM_SCALED_PPO_EXTRA_SEED_GROUPS_JSON", "{}")).items()
    }
    base.BASE_SCRIPT = (
        ROOT
        / "reproduction/scripts/"
        "tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py"
    )
    base.SINGLE_SUMMARY = (
        ROOT
        / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
    )
    base.OUT = OUT
    base.VIS_ROOT = ROOT / "res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rollout"
    base.LOG_ROOT = ROOT / "logs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
    base.RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
    base.FAILED_ROOT = (
        ROOT / "res/failed_runs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
    )
    base.REPORT_OUT = REPORT_OUT
    base.OK_SINGLE_STATUS = OK_SINGLE_STATUS
    base.OK_MULTI_STATUS = OK_MULTI_STATUS
    base.FAILED_MULTI_STATUS = FAILED_MULTI_STATUS
    base.BASELINE_SEED_GROUP = "seed_group_0_existing_scaled"
    base.NEW_SEED_GROUPS = {**DEFAULT_SEED_GROUPS, **extra_seed_groups}
    base.EXTRA_SEED_GROUPS = {}
    base.REUSE_EXISTING_GROUPS = os.environ.get("BM_SCALED_PPO_REUSE_EXISTING_SEED_GROUPS", "0") == "1"


def rewrite_scaled_metadata() -> None:
    source_summary_json = SUMMARY_JSON if SUMMARY_JSON.is_file() else BASE_SUMMARY_JSON
    summary = load_json(source_summary_json)
    if not summary:
        raise FileNotFoundError(source_summary_json)

    summary["status"] = OK_MULTI_STATUS if summary.get("status") == base.OK_MULTI_STATUS else summary.get("status")
    summary["experiment_type"] = "tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
    summary["scope"] = (
        "Aggregates local virtual seed groups for four scaled-PPO official-importer-export closed-loop "
        "task-conditioned latent-guidance rollouts: joystick, waypoint, obstacle avoidance, and composed objectives."
    )
    summary["inputs"] = {
        "single_seed_scaled_summary": str(base.SINGLE_SUMMARY),
        "scaled_ppo_training_run": str(
            ROOT
            / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
        ),
        "scaled_ppo_checkpoint_eval": str(
            ROOT
            / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
        ),
        "scaled_ppo_vae": str(
            ROOT
            / "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
            "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json"
        ),
        "scaled_ppo_denoiser": str(
            ROOT
            / "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
            "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
        ),
        "scaled_ppo_offline_guidance": str(
            ROOT
            / "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/"
            "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
        ),
    }
    summary["checks"]["uses_scaled_ppo_training_run"] = (
        load_json(Path(summary["inputs"]["scaled_ppo_training_run"])).get("status")
        == "ok_official_importer_export_full_bundle_scaled_ppo_training_completed"
    )
    summary["checks"]["uses_scaled_ppo_checkpoint_eval"] = (
        load_json(Path(summary["inputs"]["scaled_ppo_checkpoint_eval"])).get("status")
        == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed"
    )
    summary["checks"]["uses_scaled_ppo_vae"] = (
        load_json(Path(summary["inputs"]["scaled_ppo_vae"])).get("status")
        == "ok_official_importer_export_scaled_ppo_teacher_rollout_vae_training"
    )
    summary["checks"]["uses_scaled_ppo_denoiser"] = (
        load_json(Path(summary["inputs"]["scaled_ppo_denoiser"])).get("status")
        == "ok_official_importer_export_scaled_ppo_state_latent_diffusion_training"
    )
    summary["checks"]["uses_scaled_ppo_offline_guidance"] = (
        load_json(Path(summary["inputs"]["scaled_ppo_offline_guidance"])).get("status")
        == "ok_official_importer_export_scaled_ppo_state_latent_guidance_eval"
    )
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "local_virtual_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval",
        "why_not_complete": (
            "This is a multi-seed local virtual official-importer-export scaled-PPO closed-loop guidance evaluation "
            "using local PPO/VAE/denoiser checkpoints and proxy costs. It is useful paper-facing evidence, but not "
            "official BeyondMimic Fig. 5/Fig. 6 reproduction, TensorRT deployment, or real-robot evidence."
        ),
    }
    for row in summary.get("rows", []):
        row["claim_level"] = "local_virtual_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed"
    summary["outputs"] = {
        "json": str(SUMMARY_JSON),
        "rows_csv": str(OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rows.csv"),
        "rows_tsv": str(OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rows.tsv"),
        "aggregate_csv": str(OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_aggregate.csv"),
        "report_dir": str(REPORT_OUT),
        "compat_json": str(BASE_SUMMARY_JSON),
    }

    write_json(SUMMARY_JSON, summary)
    row_fields = [
        "seed_group",
        "task",
        "seed",
        "status",
        "selected_physical_gpu",
        "rollout_steps",
        "guided_reward_mean",
        "teacher_reward_mean",
        "vae_base_reward_mean",
        "denoised_reward_mean",
        "guided_target_body_error_mean",
        "teacher_target_body_error_mean",
        "vae_base_target_body_error_mean",
        "denoised_target_body_error_mean",
        "guided_done_count_total",
        "teacher_done_count_total",
        "guidance_cost_delta_mean",
        "guidance_grad_norm_mean",
        "guided_teacher_action_mse_mean",
        "guided_base_action_mse_mean",
        "summary_json",
        "asset_json",
        "mp4",
        "metrics_csv",
        "keyframes_png",
        "metrics_png",
        "claim_level",
    ]
    write_csv(OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rows.csv", summary["rows"], row_fields)
    write_csv(
        OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rows.tsv",
        summary["rows"],
        row_fields,
        delimiter="\t",
    )
    aggregate_fields = sorted({key for row in summary["aggregate"] for key in row})
    write_csv(
        OUT / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_aggregate.csv",
        summary["aggregate"],
        aggregate_fields,
    )


def main() -> None:
    patch_base()
    if os.environ.get("BM_METADATA_ONLY") == "1":
        rewrite_scaled_metadata()
        final = load_json(SUMMARY_JSON)
        print(json.dumps({"status": final["status"], "json": str(SUMMARY_JSON), "rows": len(final["rows"])}, sort_keys=True))
        if final["status"] != OK_MULTI_STATUS:
            raise SystemExit(1)
        return
    base.main()
    rewrite_scaled_metadata()
    final = load_json(SUMMARY_JSON)
    print(json.dumps({"status": final["status"], "json": str(SUMMARY_JSON), "rows": len(final["rows"])}, sort_keys=True))
    if final["status"] != OK_MULTI_STATUS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
