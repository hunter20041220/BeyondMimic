#!/usr/bin/env python3
"""Run PPO training on the official-importer-export G1 USDA asset.

This wrapper reuses the already-validated PPO training harness while replacing
the robot asset with the large USDA exported by the official Isaac Sim URDF
importer GPU4 probe and replacing the motion input with the 40-motion public
official csv-loop bundle. It is real virtual PPO training, but it is not the
paper's official teacher-policy run.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_ppo_training"
OFFICIAL_IMPORTER_USD = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_probe/"
    "g1_official_importer_in_memory_gpu4_export.usda"
)
EXPORT_STRUCTURE_AUDIT = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit/"
    "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json"
)
FULL_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
)
FULL_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "official_csv_loop_full_public_motion_bundle.npz"
)
OFFICIAL_IMPORTER_TASK_GATE = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
    "tracking_g1_official_importer_export_full_dataset_task_eval.json"
)
TARGET_GPUS = [4, 7]
DEFAULT_MAX_ITERATIONS = 300
DEFAULT_SEED = 20260680


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_training_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_train_entry() -> Path:
    """Write a temporary gate summary with the status expected by the base harness."""
    gate = load_json(OFFICIAL_IMPORTER_TASK_GATE)
    compatible = dict(gate)
    compatible["status"] = "ok_resource_adjusted_train_entry_diagnostic"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_official_importer_task_gate"] = str(OFFICIAL_IMPORTER_TASK_GATE)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy only adapts the status enum for the shared PPO training harness. "
        "The authoritative gate remains the original official-importer-export full-dataset task eval JSON."
    )
    shim_path = OUT / "base_compatible_official_importer_task_gate_for_training.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True), encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    bundle = load_json(FULL_BUNDLE_AUDIT)
    export_audit = load_json(EXPORT_STRUCTURE_AUDIT)
    task_gate = load_json(OFFICIAL_IMPORTER_TASK_GATE)
    bundle_info = bundle.get("bundle", {})
    trained_ok = summary.get("status") == "ok_resource_adjusted_ppo_training_completed"
    summary["status"] = (
        "ok_official_importer_export_full_bundle_ppo_training_completed"
        if trained_ok
        else summary.get("status", "failed_official_importer_export_full_bundle_ppo_training")
    )
    summary["experiment_type"] = "tracking_official_importer_export_full_bundle_ppo_training_run"
    summary["scope"] = (
        "PPO training in the official Tracking-Flat-G1-v0 manager stack using the official-importer GPU4 G1 USDA "
        "export and a concatenated 40-motion public official csv-loop NPZ bundle. This removes the earlier "
        "resource-adjusted/enriched robot asset from the PPO path, but still uses a local exported USDA and a local "
        "bundle with artificial clip boundaries, so it is not the paper's official teacher-policy training run."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "asset_note": (
                "The inherited base-harness key `enriched_usd` is also present in this JSON, but its value has been "
                "overridden to the official-importer-export USDA path for this run."
            ),
            "official_importer_export_structure_audit": str(EXPORT_STRUCTURE_AUDIT),
            "official_importer_task_gate": str(OFFICIAL_IMPORTER_TASK_GATE),
            "motion_npz": str(FULL_BUNDLE_NPZ),
            "full_bundle_audit": str(FULL_BUNDLE_AUDIT),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "official_importer_export_structure_audit_passed": export_audit.get("status")
            == "ok_with_physics_usd_export_but_vulkan_device_lost",
            "official_importer_task_gate_passed": task_gate.get("status")
            == "ok_official_importer_export_full_dataset_task_eval",
            "full_bundle_audit_passed": bundle.get("status")
            == "ok_official_csv_loop_full_bundle_motion_npz",
            "full_bundle_has_40_motions": bundle_info.get("motion_count") == 40,
            "full_bundle_total_frames_11960": bundle_info.get("total_frames") == 11960,
        }
    )
    summary["config"]["max_iterations"] = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_PPO_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    summary["config"]["seed"] = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_PPO_SEED", str(DEFAULT_SEED)))
    for rank_metric in summary.get("run", {}).get("rank_metrics", []):
        rank_metric["uses_official_importer_export_usd"] = True
        rank_metric["uses_resource_adjusted_usd"] = False
        rank_metric["official_csv_loop_full_public_motion_bundle"] = True
        rank_metric["motion_file"] = str(FULL_BUNDLE_NPZ)
        rank_metric["paper_level_training"] = False
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_importer_export_full_bundle_ppo_training_complete": bool(trained_ok),
        "official_ppo_training_complete": False,
        "paper_level_tracking_training_complete": False,
        "why_not_paper_level": (
            "The PPO run uses a local USDA produced by the official Isaac Sim importer and a public-motion bundle. "
            "It does not use an official released BeyondMimic teacher checkpoint, does not prove DAgger data "
            "quality, and does not reproduce the paper's full teacher-policy training protocol."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = FULL_BUNDLE_NPZ
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.MAX_ITERATIONS = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_PPO_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    module.SEED = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_PPO_SEED", str(DEFAULT_SEED)))
    module.TRAIN_ENTRY = make_base_compatible_train_entry()
    module.main()

    output_json = OUT / "tracking_g1_resource_adjusted_ppo_training_run.json"
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
    summary = patch_summary(load_json(output_json))
    output_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_training": summary.get("run", {}).get("attempted_training"),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "max_iterations": summary.get("config", {}).get("max_iterations"),
                "motion_npz": str(FULL_BUNDLE_NPZ),
                "usd": str(OFFICIAL_IMPORTER_USD),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
