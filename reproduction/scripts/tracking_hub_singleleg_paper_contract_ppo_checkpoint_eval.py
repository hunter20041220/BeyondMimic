#!/usr/bin/env python3
"""Evaluate the focused Hub single-leg Stage-1 PPO teacher checkpoint."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/hub_singleleg_paper_contract_ppo_checkpoint_eval"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_training_run/"
    "tracking_hub_singleleg_paper_contract_ppo_training_run.json"
)
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
SINGLELEG_MOTION_NPZ = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/"
    "hub_singleleg_video_single_leg_stand_1/motion.npz"
)
MOTION_NPZ = Path(os.environ.get("BM_HUB_SINGLELEG_EVAL_MOTION_NPZ", str(SINGLELEG_MOTION_NPZ)))
RUN_TAG = os.environ.get("BM_HUB_SINGLELEG_EVAL_RUN_TAG", "").strip()
if RUN_TAG:
    OUT = ROOT / f"res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_{RUN_TAG}"
    LOG_DIR = ROOT / f"logs/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval_{RUN_TAG}"
    RUN_ROOT = ROOT / f"res/runs/hub_singleleg_paper_contract_ppo_checkpoint_eval_{RUN_TAG}"
FINAL_JSON = OUT / "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json"

DEFAULT_TARGET_GPUS = [5, 6]
DEFAULT_SEED = 20260912
DEFAULT_NUM_ENVS = 512
DEFAULT_EVAL_STEPS = 799


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_gpus() -> list[int]:
    raw = os.environ.get("BM_HUB_SINGLELEG_EVAL_TARGET_GPUS", "")
    if raw.strip():
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return list(DEFAULT_TARGET_GPUS)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("bm_hub_singleleg_ppo_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_worker_eval_termination(module: Any) -> None:
    old = 'env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)\n    print(f"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}", flush=True)'
    new = '''env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    ee_body_pos_threshold_patch = None
    ee_body_pos_body_names_patch = None
    ee_cfg_original = None
    if os.environ.get("BM_EE_BODY_POS_EVAL_THRESHOLD"):
        ee_cfg = env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
        ee_cfg_original = {
            "threshold": float(ee_cfg.params.get("threshold", 0.25)),
            "body_names": list(ee_cfg.params.get("body_names", [])),
        }
        ee_cfg.params = dict(ee_cfg.params)
        ee_cfg.params["threshold"] = float(os.environ["BM_EE_BODY_POS_EVAL_THRESHOLD"])
        if os.environ.get("BM_EE_BODY_POS_EVAL_BODY_NAMES"):
            ee_cfg.params["body_names"] = [
                name.strip()
                for name in os.environ["BM_EE_BODY_POS_EVAL_BODY_NAMES"].split(",")
                if name.strip()
            ]
        env.unwrapped.termination_manager.set_term_cfg("ee_body_pos", ee_cfg)
        ee_cfg_after = env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
        ee_body_pos_threshold_patch = float(ee_cfg_after.params.get("threshold"))
        ee_body_pos_body_names_patch = list(ee_cfg_after.params.get("body_names", []))
        print(
            f"BM_SENTINEL:eval:ee_body_pos_threshold_patch={ee_body_pos_threshold_patch}",
            flush=True,
        )
    print(f"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}", flush=True)'''
    if old not in module.WORKER_CODE:
        raise RuntimeError("Unable to patch eval worker termination threshold hook")
    module.WORKER_CODE = module.WORKER_CODE.replace(old, new)
    old_metrics = '"paper_level_tracking_eval": False,\n    }'
    new_metrics = '''"paper_level_tracking_eval": False,
        "ee_body_pos_threshold_patch_applied": ee_body_pos_threshold_patch is not None,
        "ee_body_pos_original_cfg": ee_cfg_original,
        "ee_body_pos_eval_threshold_after": ee_body_pos_threshold_patch,
        "ee_body_pos_eval_body_names_after": ee_body_pos_body_names_patch,
    }'''
    if old_metrics not in module.WORKER_CODE:
        raise RuntimeError("Unable to patch eval worker metrics threshold record")
    module.WORKER_CODE = module.WORKER_CODE.replace(old_metrics, new_metrics)


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = json.loads(json.dumps(training))
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    checkpoint_override = os.environ.get("BM_HUB_SINGLELEG_EVAL_CHECKPOINT", "").strip()
    if checkpoint_override:
        checkpoint_override_path = Path(checkpoint_override)
        compatible.setdefault("outputs", {})
        compatible["outputs"]["run_dir"] = str(checkpoint_override_path.parent.parent)
        compatible.setdefault("run", {})
        compatible["run"]["best_checkpoint_override"] = checkpoint_override
        compatible["run"]["checkpoint_count"] = max(1, int(compatible["run"].get("checkpoint_count", 0)))
        compatible["run"]["rank_metrics"] = [
            {
                "rank": 0,
                "checkpoints": [checkpoint_override],
                "checkpoint_count": 1,
                "checkpoint_override_for_eval": True,
            }
        ]
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_hub_singleleg_training_run_json"] = str(TRAINING_RUN_JSON)
    if checkpoint_override:
        compatible["inputs"]["checkpoint_override_for_eval"] = checkpoint_override
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This shim adapts the Hub single-leg training status enum for the shared checkpoint-eval harness. "
        "The authoritative audit remains the Hub single-leg training JSON."
    )
    shim_path = OUT / "base_compatible_hub_singleleg_training_run_for_eval.json"
    write_json(shim_path, compatible)
    return shim_path


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_mean(metrics: dict[str, Any], section: str) -> float | None:
    value = metrics.get(section, {})
    if isinstance(value, dict):
        if "mean" in value:
            return safe_float(value["mean"])
        if "mean_over_steps" in value and isinstance(value["mean_over_steps"], dict):
            return safe_float(value["mean_over_steps"].get("mean"))
    return None


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    run_metrics = summary.get("run", {}).get("metrics", {})
    motion_metrics = run_metrics.get("motion_metrics", {}) if isinstance(run_metrics, dict) else {}
    total_env_steps = safe_float(run_metrics.get("total_env_steps"), 0.0) or 0.0
    done = safe_float(run_metrics.get("done_count_total"), 0.0) or 0.0
    timeout = safe_float(run_metrics.get("timeout_count_total"), 0.0) or 0.0
    reward = metric_mean(run_metrics, "reward")
    body_error = metric_mean(motion_metrics, "error_body_pos")
    joint_error = metric_mean(motion_metrics, "error_joint_pos")
    non_timeout_done_rate = (done - timeout) / total_env_steps if total_env_steps > 0 else None

    summary["status"] = (
        "ok_hub_singleleg_paper_contract_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_hub_singleleg_paper_contract_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates the focused Hub single-leg Stage-1 teacher checkpoint in Tracking-Flat-G1-v0. "
        "This screens whether the teacher has learned the single-leg reference before downstream VAE/diffusion."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "base_compatible_training_run_json": str(OUT / "base_compatible_hub_singleleg_training_run_for_eval.json"),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "singleleg_motion_npz": str(MOTION_NPZ),
            "default_singleleg_motion_npz": str(SINGLELEG_MOTION_NPZ),
            "checkpoint_override_for_eval": os.environ.get("BM_HUB_SINGLELEG_EVAL_CHECKPOINT", ""),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "hub_singleleg_training_completed": load_json(TRAINING_RUN_JSON).get("status")
            == "ok_hub_singleleg_paper_contract_ppo_training_completed",
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "singleleg_motion_npz_exists": MOTION_NPZ.is_file(),
        }
    )
    if isinstance(run_metrics, dict):
        run_metrics.update(
            {
                "uses_official_importer_export_usd": True,
                "uses_hub_singleleg_motion": True,
                "motion_count": 1,
                "official_beyondmimic_checkpoint": False,
                "paper_level_tracking_eval": False,
            }
        )
    summary["quality_gate"] = {
        "reward_mean": reward,
        "error_body_pos_mean": body_error,
        "error_joint_pos_mean": joint_error,
        "local_non_timeout_done_rate": non_timeout_done_rate,
        "screening_thresholds": {
            "reward_mean_min": 0.10,
            "error_body_pos_mean_max_m": 0.25,
            "local_non_timeout_done_rate_max": 0.05,
        },
        "passed": bool(
            reward is not None
            and body_error is not None
            and non_timeout_done_rate is not None
            and reward >= 0.10
            and body_error <= 0.25
            and non_timeout_done_rate <= 0.05
        ),
        "interpretation": (
            "This is a conservative local screening gate. Passing it would permit downstream rollout collection; "
            "failing it means VAE/diffusion video generation should not proceed from this teacher."
        ),
    }
    summary.setdefault("config", {})
    summary["config"]["ee_body_pos_eval_threshold"] = os.environ.get("BM_EE_BODY_POS_EVAL_THRESHOLD")
    summary["config"]["ee_body_pos_eval_body_names"] = os.environ.get("BM_EE_BODY_POS_EVAL_BODY_NAMES")
    summary["config"]["run_tag"] = RUN_TAG
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(FINAL_JSON)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_beyondmimic_checkpoint": False,
        "paper_level_tracking_eval_complete": False,
        "claim_level": "local_single_motion_teacher_quality_screening",
        "why_not_paper_level": (
            "This evaluates a local single-motion teacher, not the paper's full diverse teacher set, official "
            "checkpoint, DAgger dataset, Fig.5/Fig.6 protocol, TensorRT deployment, or real robot."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    module = load_base_module()
    patch_worker_eval_termination(module)
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = MOTION_NPZ
    checkpoint_override = os.environ.get("BM_HUB_SINGLELEG_EVAL_CHECKPOINT", "").strip()
    if checkpoint_override:
        override_path = Path(checkpoint_override)
        def _select_checkpoint_override(_training_run: dict[str, Any]) -> Path:
            return override_path

        module.select_checkpoint = _select_checkpoint_override
    module.TRAINING_RUN_JSON = make_base_compatible_training_summary()
    module.CANDIDATE_GPUS = parse_gpus()
    module.NUM_ENVS = int(os.environ.get("BM_HUB_SINGLELEG_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    module.EVAL_STEPS = int(os.environ.get("BM_HUB_SINGLELEG_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    module.SEED = int(os.environ.get("BM_HUB_SINGLELEG_EVAL_SEED", str(DEFAULT_SEED)))
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    write_json(FINAL_JSON, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(FINAL_JSON),
                "checkpoint": summary.get("inputs", {}).get("checkpoint"),
                "quality_gate_passed": summary.get("quality_gate", {}).get("passed"),
            },
            sort_keys=True,
        )
    )
    if summary["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
