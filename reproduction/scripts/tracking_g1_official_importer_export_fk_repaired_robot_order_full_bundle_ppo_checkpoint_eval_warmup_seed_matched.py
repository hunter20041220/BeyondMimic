#!/usr/bin/env python3
"""Run the robot-order FK warmup checkpoint eval with the non-warmup seed.

The first warmup follow-up used a new seed, which made the total done-rate
regression hard to attribute. This wrapper reuses the warmup evaluator but
forces the seed to match the authoritative non-warmup robot-order eval
(`20260721`) and writes to a separate output directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.py"
)
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched"
)
SEED = 20260721
NUM_ENVS = 2048
SOURCE_JSON_NAME = (
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
)
FINAL_JSON_NAME = (
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("bm_robot_order_warmup_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load warmup evaluator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.DEFAULT_SEED = SEED
    module.DEFAULT_NUM_ENVS = NUM_ENVS
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_SEED"] = str(SEED)
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_WARMUP_EVAL_NUM_ENVS"] = str(NUM_ENVS)
    module.main()

    source_json = OUT / SOURCE_JSON_NAME
    final_json = OUT / FINAL_JSON_NAME
    summary = load_json(source_json)
    if not summary:
        raise RuntimeError(f"Warmup evaluator did not write expected JSON: {source_json}")
    eval_ok = summary.get("status") == (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed"
    )
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched_completed"
        if eval_ok
        else "failed_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched"
    )
    summary["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched"
    )
    summary["scope"] = (
        "Seed-matched full evaluation of the local robot-order FK-repaired PPO checkpoint with reset command warmup. "
        "The seed matches the non-warmup baseline so the total done-rate delta is attributable to the warmup "
        "intervention rather than a different adaptive-sampling seed."
    )
    summary.setdefault("config", {})
    summary["config"]["seed"] = SEED
    summary["config"]["num_envs"] = NUM_ENVS
    summary["config"]["same_seed_as_non_warmup_eval"] = True
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["source_warmup_json_before_rename"] = str(source_json)
    summary["seed_match"] = {
        "non_warmup_seed": SEED,
        "warmup_seed": SEED,
        "same_seed_as_non_warmup_eval": True,
        "why": (
            "The prior warmup eval used seed 20260741, while the non-warmup baseline used 20260721. This run removes "
            "that comparison ambiguity."
        ),
    }
    summary.setdefault("interpretation", {})
    summary["interpretation"]["seed_matched_to_non_warmup_eval"] = True
    summary["interpretation"]["goal_complete"] = False
    summary["interpretation"]["paper_level_tracking_eval_complete"] = False
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_json.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "seed": SEED,
                "num_envs": NUM_ENVS,
                "comparison_to_non_warmup_eval": summary.get("comparison_to_non_warmup_eval", {}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
