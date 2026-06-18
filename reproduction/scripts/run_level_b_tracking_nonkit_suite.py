#!/usr/bin/env python3
"""Unified non-Kit Level-B tracking evidence suite.

This suite intentionally avoids IsaacLab/Kit launches. It reruns the tracking
source/config/schema/fixture audits that can execute under the current host
limits and records a single pass/fail artifact for reproducibility.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/nonkit_suite"
LOG = ROOT / "logs/tracking_nonkit_suite"


STEPS = [
    (
        "official_source_contract",
        ["python3", str(ROOT / "reproduction/scripts/tracking_official_source_contract_audit.py")],
        "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json",
    ),
    (
        "g1_action_scale",
        ["python3", str(ROOT / "reproduction/scripts/tracking_g1_action_scale_audit.py")],
        "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
    ),
    (
        "reward_formula",
        ["python3", str(ROOT / "reproduction/scripts/tracking_reward_formula_audit.py")],
        "res/tracking/reward_formula_audit/tracking_reward_formula_audit.json",
    ),
    (
        "observation_action_schema",
        ["python3", str(ROOT / "reproduction/scripts/tracking_observation_action_schema_audit.py")],
        "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json",
    ),
    (
        "randomization_termination",
        ["python3", str(ROOT / "reproduction/scripts/tracking_randomization_termination_audit.py")],
        "res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json",
    ),
    (
        "motion_preprocessing_contract",
        ["python3", str(ROOT / "reproduction/scripts/motion_preprocessing_contract_audit.py")],
        "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json",
    ),
    (
        "build_tracking_motion_npz_fixture",
        ["python3", str(ROOT / "reproduction/scripts/build_tracking_motion_npz_fixture.py")],
        "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json",
    ),
    (
        "tracking_local_smoke_preflight",
        ["python3", str(ROOT / "reproduction/scripts/tracking_local_smoke_preflight.py")],
        "res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json",
    ),
    (
        "adaptive_sampling_discrepancy",
        ["python3", str(ROOT / "reproduction/scripts/adaptive_sampling_discrepancy_audit.py")],
        "res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json",
    ),
    (
        "tracking_onnx_export_contract",
        ["python3", str(ROOT / "reproduction/scripts/tracking_onnx_export_contract_audit.py")],
        "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json",
    ),
    (
        "tracking_debug_motion_policy_onnx_export",
        [str(ROOT / "envs/bm_diffusion/bin/python"), str(ROOT / "reproduction/scripts/tracking_debug_motion_policy_onnx_export.py")],
        "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json",
    ),
    (
        "tracking_debug_motion_policy_onnx_inference",
        [
            str(ROOT / "envs/bm_diffusion/bin/python"),
            str(ROOT / "reproduction/scripts/tracking_debug_motion_policy_onnx_inference_audit.py"),
        ],
        "res/tracking/debug_motion_policy_onnx_inference/tracking_debug_motion_policy_onnx_inference_audit.json",
    ),
    (
        "mujoco_ros_launch_contract",
        ["python3", str(ROOT / "reproduction/scripts/mujoco_ros_launch_contract_audit.py")],
        "res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json",
    ),
]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def run_step(name: str, command: list[str], rel_json: str) -> dict[str, Any]:
    LOG.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    log_path = LOG / f"{name}.log"
    atomic_write_text(log_path, proc.stdout)
    artifact_path = ROOT / rel_json
    artifact: dict[str, Any] = {}
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "command": " ".join(command),
        "return_code": proc.returncode,
        "passed": proc.returncode == 0 and artifact.get("status") == "ok",
        "artifact": str(artifact_path),
        "artifact_status": artifact.get("status"),
        "log": str(log_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_step(*step) for step in STEPS]
    artifacts = {
        row["name"]: json.loads(Path(row["artifact"]).read_text(encoding="utf-8"))
        for row in rows
        if Path(row["artifact"]).is_file()
    }
    checks = {
        "all_steps_pass": all(row["passed"] for row in rows),
        "step_count_13": len(rows) == 13,
        "official_source_contract_ok": artifacts["official_source_contract"]["status"] == "ok",
        "g1_action_scale_rows_29": artifacts["g1_action_scale"]["metrics"]["row_count"] == 29,
        "reward_formula_motion_terms_6": artifacts["reward_formula"]["metrics"]["motion_exp_reward_term_count"] == 6,
        "observation_policy_dim_160": artifacts["observation_action_schema"]["metrics"]["policy_dimension"] == 160,
        "observation_critic_dim_286": artifacts["observation_action_schema"]["metrics"]["critic_dimension"] == 286,
        "randomization_event_terms_4": artifacts["randomization_termination"]["metrics"]["event_term_count"] == 4,
        "motion_preprocessing_contract_ok": artifacts["motion_preprocessing_contract"]["status"] == "ok",
        "fixture_count_3": artifacts["build_tracking_motion_npz_fixture"]["metrics"]["fixture_count"] == 3,
        "local_preflight_steps_6": artifacts["tracking_local_smoke_preflight"]["step_count"] == 6,
        "adaptive_sampling_discrepancy_recorded": artifacts["adaptive_sampling_discrepancy"]["status"] == "ok"
        and artifacts["adaptive_sampling_discrepancy"]["metrics"]["l1_difference"] > 0.0,
        "onnx_contract_ok": artifacts["tracking_onnx_export_contract"]["status"] == "ok",
        "debug_onnx_export_ok": artifacts["tracking_debug_motion_policy_onnx_export"]["status"] == "ok",
        "debug_onnx_file_written": artifacts["tracking_debug_motion_policy_onnx_export"]["checks"]["onnx_file_written"],
        "debug_onnx_contract_match": artifacts["tracking_debug_motion_policy_onnx_export"]["checks"][
            "all_inputs_match_contract"
        ]
        and artifacts["tracking_debug_motion_policy_onnx_export"]["checks"]["all_outputs_match_contract"],
        "debug_onnx_inference_ok": artifacts["tracking_debug_motion_policy_onnx_inference"]["status"] == "ok",
        "debug_onnx_reference_evaluator_loaded": artifacts["tracking_debug_motion_policy_onnx_inference"]["checks"][
            "reference_evaluator_loaded"
        ],
        "debug_onnx_inference_outputs_match": artifacts["tracking_debug_motion_policy_onnx_inference"]["checks"][
            "all_outputs_match_contract_values"
        ],
        "mujoco_ros_contract_ok": artifacts["mujoco_ros_launch_contract"]["status"] == "ok",
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_rollout_or_policy_performance": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "Level-B non-Kit tracking suite: official-code/static/schema/fixture contracts only",
        "step_count": len(rows),
        "pass_count": sum(1 for row in rows if row["passed"]),
        "steps": rows,
        "metrics": {
            "official_target_body_count": artifacts["official_source_contract"]["flat_env"]["body_count"],
            "g1_action_scale_rows": artifacts["g1_action_scale"]["metrics"]["row_count"],
            "reward_motion_terms": artifacts["reward_formula"]["metrics"]["motion_exp_reward_term_count"],
            "policy_dimension": artifacts["observation_action_schema"]["metrics"]["policy_dimension"],
            "critic_dimension": artifacts["observation_action_schema"]["metrics"]["critic_dimension"],
            "randomization_event_terms": artifacts["randomization_termination"]["metrics"]["event_term_count"],
            "fixture_count": artifacts["build_tracking_motion_npz_fixture"]["metrics"]["fixture_count"],
            "local_preflight_steps": artifacts["tracking_local_smoke_preflight"]["step_count"],
            "adaptive_sampling_l1_difference": artifacts["adaptive_sampling_discrepancy"]["metrics"]["l1_difference"],
            "debug_onnx_size_bytes": artifacts["tracking_debug_motion_policy_onnx_export"]["onnx_size_bytes"],
            "debug_onnx_sha256": artifacts["tracking_debug_motion_policy_onnx_export"]["onnx_sha256"],
            "debug_onnx_inference_max_abs_error": artifacts["tracking_debug_motion_policy_onnx_inference"][
                "metrics"
            ]["max_abs_error"],
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "unified_nonkit_tracking_evidence_suite",
            "goal_complete": False,
            "remaining_gap": (
                "This suite consolidates executable non-Kit Level-B evidence. It still does not run official "
                "IsaacLab/Kit csv_to_npz/replay/PPO, trained rollout evaluation, trained policy export, ROS 2, or hardware."
            ),
        },
    }
    atomic_write_text(OUT / "level_b_tracking_nonkit_suite.json", json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_tsv(
        OUT / "level_b_tracking_nonkit_suite.tsv",
        rows,
        ["name", "command", "return_code", "passed", "artifact", "artifact_status", "log"],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(OUT / "level_b_tracking_nonkit_suite.json"),
                "steps": summary["step_count"],
                "pass": summary["pass_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
