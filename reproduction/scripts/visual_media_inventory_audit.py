#!/usr/bin/env python3
"""Inventory local visual media deliverables and paper-video gaps."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/visual_media_inventory"
IMAGE_SUFFIXES = {".png", ".svg", ".pdf"}
ANIMATION_SUFFIXES = {".gif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("res/visualization/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout/"):
        return "local_task_conditioned_latent_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout/"):
        return "local_task_conditioned_latent_guidance_rollout_video"
    if rel.startswith(
        "res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rollout/"
    ):
        return "local_task_conditioned_latent_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_task_conditioned_latent_guidance_multiseed_rollout/"):
        return "local_task_conditioned_latent_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/"):
        return "local_task_conditioned_latent_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/"):
        return "local_receding_latent_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_receding_latent_guidance_rollout/"):
        return "local_receding_latent_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_action_guidance_rollout/"):
        return "local_action_guidance_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_vae_closed_loop_rollout/"):
        return "local_vae_closed_loop_rollout_video"
    if rel.startswith("res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"):
        return "local_vae_closed_loop_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_full_bundle_policy_rollout/"):
        return "local_policy_rollout_video"
    if rel.startswith("res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"):
        return "local_policy_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_policy_rollout/"):
        return "local_policy_rollout_video"
    if rel.startswith("res/visualization/official_csv_loop_reference_replay/"):
        return "local_kinematic_reference_video"
    if rel.startswith("res/released_figures/"):
        return "released_data_figure"
    if rel.startswith("res/level_c/guidance_debug_visualization/"):
        return "debug_guidance_visual"
    if rel.startswith("res/level_c/guidance_checkpoint_visualization/"):
        return "debug_checkpoint_guidance_visual"
    if "resource_adjusted_tiny_diffusion" in rel and ("/videos/" in rel or "video_preview" in rel):
        return "debug_tiny_diffusion_preview"
    if rel.startswith("res/level_c/augmentation_probe/"):
        return "debug_augmentation_visual"
    if rel.startswith("res/tracking/adaptive_sampling_probe/"):
        return "debug_tracking_visual"
    if rel.startswith("res/runs/") and "/figures/" in rel:
        return "debug_run_figure"
    return "other_visual_media"


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix[1:]
    if suffix in ANIMATION_SUFFIXES:
        return "gif"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unknown"


def row_for(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "kind": media_kind(path),
        "category": classify(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "paper_level": classify(path) == "released_data_figure",
        "debug_only": classify(path).startswith("debug_"),
    }


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["relative_path", "kind", "category", "size_bytes", "sha256", "paper_level", "debug_only"]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffixes = IMAGE_SUFFIXES | ANIMATION_SUFFIXES | VIDEO_SUFFIXES
    paths = sorted(path for path in (ROOT / "res").rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    rows = [row_for(path) for path in paths]
    kind_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for item in rows:
        kind_counts[item["kind"]] = kind_counts.get(item["kind"], 0) + 1
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1

    required_video_rows = [
        {
            "requirement": "motion_tracking_replay_or_rollout_video",
            "required_by": "goal.md section 16 videos; tracking evaluation",
            "status": "missing_or_blocked",
            "reason": "Local virtual reference/policy/VAE/action-guidance/receding-latent/task-conditioned guidance videos exist, but paper-level tracking rollout videos with official checkpoints, original assets, and paper metrics remain missing.",
        },
        {
            "requirement": "fig5_joystick_waypoint_rollout_videos",
            "required_by": "goal.md Phase 8 / paper Fig. 5",
            "status": "missing_or_blocked",
            "reason": "No trained VAE/diffusion checkpoint or closed-loop task rollout log/video exists.",
        },
        {
            "requirement": "fig6_inpainting_obstacle_rollout_videos",
            "required_by": "goal.md Phase 8 / paper Fig. 6",
            "status": "missing_or_blocked",
            "reason": "No trained diffusion rollout, obstacle scene, mocap context, or closed-loop video exists.",
        },
        {
            "requirement": "real_robot_success_failure_videos",
            "required_by": "goal.md final videos / paper qualitative results",
            "status": "out_of_scope_or_missing",
            "reason": "No Unitree G1 hardware is available and real robot commands must not be run.",
        },
    ]
    checks = {
        "visual_rows_nonempty": len(rows) > 0,
        "all_files_nonempty": all(item["size_bytes"] > 0 for item in rows),
        "all_hashes_recorded": all(len(item["sha256"]) == 64 for item in rows),
        "released_data_figures_present": category_counts.get("released_data_figure", 0) >= 30,
        "debug_guidance_visuals_present": category_counts.get("debug_guidance_visual", 0) >= 4,
        "debug_tiny_diffusion_previews_present": category_counts.get("debug_tiny_diffusion_preview", 0) >= 4,
        "gif_previews_present": kind_counts.get("gif", 0) >= 3,
        "no_paper_level_mp4_mov_mkv_reproduction_videos": all(
            item["category"]
            in {
                "local_kinematic_reference_video",
                "local_policy_rollout_video",
                "local_vae_closed_loop_rollout_video",
                "local_action_guidance_rollout_video",
                "local_receding_latent_guidance_rollout_video",
                "local_task_conditioned_latent_guidance_rollout_video",
            }
            for item in rows
            if item["kind"] == "video"
        ),
        "local_reference_video_allowed_and_labeled": all(
            item["category"]
            in {
                "local_kinematic_reference_video",
                "local_policy_rollout_video",
                "local_vae_closed_loop_rollout_video",
                "local_action_guidance_rollout_video",
                "local_receding_latent_guidance_rollout_video",
                "local_task_conditioned_latent_guidance_rollout_video",
            }
            for item in rows
            if item["kind"] == "video"
        ),
        "paper_required_video_gaps_recorded": len(required_video_rows) == 4,
        "does_not_claim_paper_fig5_fig6_or_robot_video": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "visual_media_inventory_audit",
        "scope": "Inventory of local visual media deliverables and explicit paper-required video gaps",
        "row_count": len(rows),
        "kind_counts": dict(sorted(kind_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "total_size_bytes": sum(item["size_bytes"] for item in rows),
        "required_video_rows": required_video_rows,
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(OUT / "visual_media_inventory_audit.json"),
            "tsv": str(OUT / "visual_media_inventory_audit.tsv"),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "released_figures_debug_visuals_local_reference_and_policy_videos_only",
            "why_not_complete": (
                "The inventory proves local released-data figures, debug visual previews, and any local reference "
                "videos are hashed and categorized. It also records that required closed-loop simulation, Fig. 5/"
                "Fig. 6, and real-robot videos are missing or blocked."
            ),
        },
    }
    write_json_atomic(OUT / "visual_media_inventory_audit.json", summary)
    write_tsv(OUT / "visual_media_inventory_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["row_count"],
                "kind_counts": summary["kind_counts"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
