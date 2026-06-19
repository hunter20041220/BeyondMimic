#!/usr/bin/env python3
"""Evaluate the PPO checkpoint trained on official csv-loop motion.

The worker is inherited from the resource-adjusted checkpoint evaluation
harness. This wrapper only switches the training summary, motion NPZ, output
directories, and final audit wording so the official csv_to_npz loop chain is
kept separate from the earlier local conversion chain.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_ppo_checkpoint_eval"
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_training_run/"
    "tracking_g1_official_csv_loop_ppo_training_run.json"
)
OFFICIAL_CSV_LOOP_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
)
OFFICIAL_REPLAY_LOOP_AUDIT = (
    ROOT
    / "res/tracking/official_replay_npz_loop_with_enriched_usd/"
    "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
)
TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260630


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    official_csv_loop = load_json(OFFICIAL_CSV_LOOP_AUDIT)
    official_replay_loop = load_json(OFFICIAL_REPLAY_LOOP_AUDIT)
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    summary["status"] = (
        "ok_official_csv_loop_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_official_csv_loop_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_official_csv_loop_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Loads the PPO checkpoint trained on the official csv_to_npz.py loop motion artifact and evaluates it in "
        "the official Tracking-Flat-G1-v0 task stack. The motion artifact still depends on the enriched-USD runtime "
        "patch, so this is a local virtual checkpoint evaluation rather than unpatched official paper-level tracking."
    )
    summary.setdefault("inputs", {})
    summary["inputs"]["training_run_json"] = str(TRAINING_RUN_JSON)
    summary["inputs"]["motion_npz"] = str(OFFICIAL_LOOP_MOTION_NPZ)
    summary["inputs"]["official_csv_to_npz_loop_audit"] = str(OFFICIAL_CSV_LOOP_AUDIT)
    summary["inputs"]["official_replay_loop_audit"] = str(OFFICIAL_REPLAY_LOOP_AUDIT)
    summary.setdefault("input_checks", {})
    summary["input_checks"]["official_csv_loop_motion_npz_exists"] = OFFICIAL_LOOP_MOTION_NPZ.is_file()
    summary["input_checks"]["official_csv_loop_audit_passed"] = (
        official_csv_loop.get("status") == "ok_official_csv_to_npz_loop_with_enriched_usd_patch"
    )
    summary["input_checks"]["official_replay_loop_audit_passed"] = (
        official_replay_loop.get("status") == "ok_official_replay_loop_with_enriched_usd_patch"
    )
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        summary["run"]["metrics"]["official_csv_to_npz_loop_output"] = True
        summary["run"]["metrics"]["official_csv_to_npz_unpatched_output"] = False
        summary["run"]["metrics"]["paper_level_tracking_eval"] = False
    summary.setdefault("outputs", {})
    final_json = OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
    summary["outputs"]["json"] = str(final_json)
    summary["interpretation"] = {
        "goal_complete": False,
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "official_csv_loop_motion_checkpoint_eval_complete": bool(eval_ok),
        "why_not_paper_level": (
            "The checkpoint was trained on official-loop motion, but the converter/replay chain still uses the "
            "enriched-USD runtime patch and only runs 300 PPO iterations. It is evidence for the local virtual "
            "tracking chain, not the paper's full teacher-policy evaluation."
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
    module.CSV_MOTION_NPZ = OFFICIAL_LOOP_MOTION_NPZ
    module.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    module.main()

    output_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
    summary = patch_summary(load_json(output_json))
    output_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
