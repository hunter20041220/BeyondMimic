#!/usr/bin/env python3
"""Audit the native normalized-action to MuJoCo PD-setpoint adapter contract.

This is a formula/ordering gate, not a rollout.  The BeyondMimic paper and the
official whole_body_tracking G1 configuration use normalized actions that are
mapped to joint position setpoints by

    theta_sp = theta_default + alpha * action

where alpha is the per-joint action scale derived from official G1 effort and
stiffness values.  The weak local videos should not be debugged by changing
that formula.  Instead, this audit verifies the formula, joint ordering,
default pose source, and MuJoCo actuator order before any native MuJoCo rollout
claims are made.
"""

from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/mujoco_native_action_adapter_contract"
JSON_OUT = OUT / "mujoco_native_action_adapter_contract.json"
TSV_OUT = OUT / "mujoco_native_action_adapter_contract.tsv"
MD_OUT = OUT / "mujoco_native_action_adapter_contract.md"

FILES = {
    "paper_method": ROOT / "reproduction/paper/source/tex/method.tex",
    "paper_supplement": ROOT / "reproduction/paper/source/root.tex",
    "official_g1_robot": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/g1.py",
    "official_g1_flat": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
    "isaaclab_joint_actions": ROOT
    / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/"
    "envs/mdp/actions/joint_actions.py",
    "isaaclab_rsl_rl_wrapper": ROOT
    / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab_rl/isaaclab_rl/"
    "rsl_rl/vecenv_wrapper.py",
    "controller_yaml": ROOT
    / "reproduction/third_party/official/motion_tracking_controller/config/g1/controllers.yaml",
    "action_scale_audit": ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
    "mujoco_joint_mapping": ROOT / "mujoco_mp4/configs/g1_joint_mapping.yaml",
    "mujoco_pd_xml": ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def parse_yaml_bracket_list(text: str, key: str, numeric: bool = False) -> list[Any]:
    match = re.search(rf"{re.escape(key)}:\s*\[(.*?)\]", text, flags=re.S)
    if not match:
        return []
    body = match.group(1)
    if numeric:
        return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", body)]
    return [token.strip().strip("\"'") for token in body.replace("\n", " ").split(",") if token.strip()]


def parse_mapping_joint_names(text: str) -> list[str]:
    names: list[str] = []
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "joint_names:":
            in_names = True
            continue
        if in_names:
            if stripped.startswith("- "):
                names.append(stripped[2:].strip())
            elif stripped and not stripped.startswith("#"):
                break
    return names


def isaaclab_default_joint_pos(joint_names: list[str]) -> np.ndarray:
    values = np.zeros(len(joint_names), dtype=np.float64)
    for idx, name in enumerate(joint_names):
        if name.endswith("_hip_pitch_joint"):
            values[idx] = -0.312
        elif name.endswith("_knee_joint"):
            values[idx] = 0.669
        elif name.endswith("_ankle_pitch_joint"):
            values[idx] = -0.363
        elif name.endswith("_elbow_joint"):
            values[idx] = 0.6
        elif name == "left_shoulder_roll_joint":
            values[idx] = 0.2
        elif name == "left_shoulder_pitch_joint":
            values[idx] = 0.2
        elif name == "right_shoulder_roll_joint":
            values[idx] = -0.2
        elif name == "right_shoulder_pitch_joint":
            values[idx] = 0.2
    return values


def parse_pd_xml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "actuator_joints": [], "ctrlrange": []}
    root = ET.parse(path).getroot()
    actuators = list(root.findall(".//actuator/position"))
    joints = [act.attrib.get("joint", "") for act in actuators]
    ctrlrange: list[list[float]] = []
    for act in actuators:
        values = [float(x) for x in act.attrib.get("ctrlrange", "").split()]
        ctrlrange.append(values if len(values) == 2 else [float("-inf"), float("inf")])
    return {"exists": True, "actuator_joints": joints, "ctrlrange": ctrlrange}


def normalized_action_to_pd_target(
    actions: np.ndarray,
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    clip_abs: float | None = 1.0,
    ctrlrange: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert normalized policy actions into PD joint-position setpoints."""
    seq = np.asarray(actions, dtype=np.float64)
    if seq.ndim == 1:
        seq = seq[None, :]
    if seq.ndim != 2 or seq.shape[1] != default_joint_pos.shape[0]:
        raise ValueError(f"Expected actions with trailing dim {default_joint_pos.shape[0]}, got {seq.shape}")
    raw = seq.copy()
    clipped = np.clip(seq, -clip_abs, clip_abs) if clip_abs is not None else seq
    raw_targets = default_joint_pos[None, :] + clipped * action_scale[None, :]
    if ctrlrange is None:
        targets = raw_targets
        ctrl_clipped_fraction = 0.0
    else:
        targets = np.clip(raw_targets, ctrlrange[:, 0], ctrlrange[:, 1])
        ctrl_clipped_fraction = float(np.mean(np.abs(targets - raw_targets) > 1e-12))
    return targets, {
        "formula": "theta_sp = theta_default + action_scale * normalized_action",
        "raw_action_abs_mean": float(np.mean(np.abs(raw))),
        "raw_action_abs_max": float(np.max(np.abs(raw))),
        "clip_abs": clip_abs,
        "action_clip_fraction": float(np.mean(np.abs(raw) > clip_abs)) if clip_abs is not None else 0.0,
        "ctrlrange_clip_fraction": ctrl_clipped_fraction,
        "raw_target_min": float(np.min(raw_targets)),
        "raw_target_max": float(np.max(raw_targets)),
        "target_min": float(np.min(targets)),
        "target_max": float(np.max(targets)),
    }


def ctrlrange_violation_rows(
    joint_names: list[str],
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    ctrlrange: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Return joints whose normalized +-1 setpoints exceed MuJoCo ctrlrange."""
    if ctrlrange is None or len(joint_names) != len(action_scale):
        return []
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(joint_names):
        target_lo = float(default_joint_pos[idx] - action_scale[idx])
        target_hi = float(default_joint_pos[idx] + action_scale[idx])
        ctrl_lo = float(ctrlrange[idx, 0])
        ctrl_hi = float(ctrlrange[idx, 1])
        lower_excess = max(0.0, ctrl_lo - target_lo)
        upper_excess = max(0.0, target_hi - ctrl_hi)
        if lower_excess > 1e-12 or upper_excess > 1e-12:
            rows.append(
                {
                    "index": idx,
                    "joint_name": name,
                    "default_joint_pos": float(default_joint_pos[idx]),
                    "action_scale": float(action_scale[idx]),
                    "raw_target_lo_for_action_minus_one": target_lo,
                    "raw_target_hi_for_action_plus_one": target_hi,
                    "mujoco_ctrlrange_lo": ctrl_lo,
                    "mujoco_ctrlrange_hi": ctrl_hi,
                    "lower_excess_rad": lower_excess,
                    "upper_excess_rad": upper_excess,
                    "max_excess_rad": max(lower_excess, upper_excess),
                }
            )
    return rows


def close(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    return bool(np.allclose(a, b, atol=atol, rtol=0.0))


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks = summary["checks"]
    notes = {
        "paper_formula_available": "Paper text contains the normalized setpoint/action-scale formula.",
        "isaaclab_affine_joint_action_semantics_available": "IsaacLab JointPositionAction applies raw_action * scale + default offset, and the RSL-RL wrapper can clip normalized actions.",
        "official_action_scale_rows_29": "The official G1 action-scale audit expands all 29 controllable joints.",
        "controller_default_pose_available": "Deployment controller standby default_position is available and nonzero.",
        "zero_default_fallback_not_used": "The adapter fixture uses an official default pose source, not all-zero fallback.",
        "mujoco_mapping_order_matches_action_rows": "MuJoCo mapping joint order matches action-scale row order.",
        "pd_actuator_order_matches_action_rows": "MuJoCo position actuator order matches action-scale row order.",
        "zero_action_returns_default_pose": "A zero normalized action maps exactly to theta_default.",
        "unit_action_delta_matches_action_scale": "A +1 action changes each joint by +action_scale.",
        "negative_unit_action_delta_matches_action_scale": "A -1 action changes each joint by -action_scale.",
        "large_action_clips_to_unit_scale": "A larger action clips to the configured normalized-action bound for the fixture.",
        "unit_targets_inside_mujoco_ctrlrange": "The fixture's unit action setpoints stay inside MuJoCo actuator ctrlrange.",
        "does_not_claim_rollout_or_success": "This audit does not run physics or claim policy/VAE/diffusion success.",
    }
    rows = []
    for key, value in checks.items():
        rows.append(
            {
                "check": key,
                "passed": bool(value),
                "notes": notes.get(key, ""),
            }
        )
    return rows


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = build_rows(summary)
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "notes"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    failed = [row["check"] for row in rows if not row["passed"]]
    lines = [
        "# MuJoCo Native Action Adapter Contract",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: formula/order/default-pose fixture only; no physics rollout and no success video claim.",
        "- Formula: `theta_sp = theta_default + action_scale * normalized_action`.",
        "- 当前不得声称完整复现 BeyondMimic；本审计只证明 action-to-PD 公式和顺序可作为后续 native rollout 的前置条件。",
        "",
        "## Checks",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['check']}`: `{row['passed']}` - {row['notes']}")
    lines.extend(["", "## Default Pose Sources", ""])
    defaults = summary["default_pose"]
    lines.append(f"- Selected source: `{defaults['selected_source']}`")
    lines.append(f"- IsaacLab vs deployment max abs delta: `{defaults['isaaclab_vs_deployment_max_abs_delta']}`")
    lines.append(f"- Delta note: {defaults['delta_note']}")
    lines.extend(["", "## Ctrlrange Violations", ""])
    violations = summary.get("ctrlrange_analysis", {}).get("violating_joints", [])
    if violations:
        for item in violations:
            lines.append(
                "- `{joint_name}`: raw setpoint range "
                "`[{raw_target_lo_for_action_minus_one:.6f}, {raw_target_hi_for_action_plus_one:.6f}]` "
                "exceeds MuJoCo ctrlrange "
                "`[{mujoco_ctrlrange_lo:.6f}, {mujoco_ctrlrange_hi:.6f}]` by max `{max_excess_rad:.6f}` rad.".format(
                    **item
                )
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Rollout-Readiness Warnings", ""])
    warnings = summary["rollout_readiness_warnings"]
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Passing this audit does not mean the teacher policy can walk, the VAE reconstructs lifted-leg poses, or the diffusion/guidance controller is stable. It only means future native MuJoCo/Isaac rollout code has a verified action-setpoint adapter to call.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paper_method = read_text(FILES["paper_method"])
    paper_supp = read_text(FILES["paper_supplement"])
    official_g1 = read_text(FILES["official_g1_robot"])
    official_flat = read_text(FILES["official_g1_flat"])
    isaaclab_joint_actions = read_text(FILES["isaaclab_joint_actions"])
    isaaclab_rsl_rl_wrapper = read_text(FILES["isaaclab_rsl_rl_wrapper"])
    controller_text = read_text(FILES["controller_yaml"])
    action_payload = read_json(FILES["action_scale_audit"])
    action_rows = action_payload.get("joint_rows", [])
    action_joint_names = [str(row["joint_name"]) for row in action_rows]
    action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
    controller_joint_names = parse_yaml_bracket_list(controller_text, "joint_names", numeric=False)
    controller_default = np.asarray(parse_yaml_bracket_list(controller_text, "default_position", numeric=True), dtype=np.float64)
    mapping_joint_names = parse_mapping_joint_names(read_text(FILES["mujoco_joint_mapping"]))
    pd_xml = parse_pd_xml(FILES["mujoco_pd_xml"])
    ctrlrange = np.asarray(pd_xml["ctrlrange"], dtype=np.float64) if len(pd_xml["ctrlrange"]) == 29 else None

    if controller_default.shape != (29,):
        selected_default = np.zeros(29, dtype=np.float64)
        selected_source = "all_zero_fallback_missing_controller_default"
    else:
        selected_default = controller_default
        selected_source = "official_motion_tracking_controller_standby_controller.default_position"
    isaac_default = isaaclab_default_joint_pos(action_joint_names) if len(action_joint_names) == 29 else np.zeros(29)
    default_delta = selected_default - isaac_default

    zero_target, zero_meta = normalized_action_to_pd_target(np.zeros(29), selected_default, action_scale, clip_abs=1.0)
    one_target, one_meta = normalized_action_to_pd_target(np.ones(29), selected_default, action_scale, clip_abs=1.0)
    neg_target, neg_meta = normalized_action_to_pd_target(-np.ones(29), selected_default, action_scale, clip_abs=1.0)
    large_target, large_meta = normalized_action_to_pd_target(np.full(29, 2.5), selected_default, action_scale, clip_abs=1.0)
    one_ctrl_target, one_ctrl_meta = normalized_action_to_pd_target(
        np.ones(29), selected_default, action_scale, clip_abs=1.0, ctrlrange=ctrlrange
    )
    neg_ctrl_target, neg_ctrl_meta = normalized_action_to_pd_target(
        -np.ones(29), selected_default, action_scale, clip_abs=1.0, ctrlrange=ctrlrange
    )
    ctrlrange_violations = ctrlrange_violation_rows(action_joint_names, selected_default, action_scale, ctrlrange)

    expected_one = selected_default[None, :] + action_scale[None, :]
    expected_neg = selected_default[None, :] - action_scale[None, :]
    checks = {
        "paper_formula_available": bool(
            "theta}^{\\text{sp}}" in paper_method and "\\boldsymbol{\\alpha}" in paper_method and "0.25\\frac" in paper_supp
        ),
        "isaaclab_affine_joint_action_semantics_available": bool(
            "self._processed_actions = self._raw_actions * self._scale + self._offset" in isaaclab_joint_actions
            and "torch.clamp(actions, -self.clip_actions, self.clip_actions)" in isaaclab_rsl_rl_wrapper
        ),
        "official_action_scale_rows_29": bool(
            len(action_rows) == 29
            and action_payload.get("checks", {}).get("action_scale_formula_matches_official")
            and "G1_ACTION_SCALE" in official_g1
            and "self.actions.joint_pos.scale = G1_ACTION_SCALE" in official_flat
        ),
        "controller_default_pose_available": bool(len(controller_joint_names) == 29 and controller_default.shape == (29,)),
        "zero_default_fallback_not_used": bool(selected_source != "all_zero_fallback_missing_controller_default"),
        "mujoco_mapping_order_matches_action_rows": bool(mapping_joint_names == action_joint_names),
        "pd_actuator_order_matches_action_rows": bool(pd_xml.get("actuator_joints") == action_joint_names),
        "zero_action_returns_default_pose": close(zero_target, selected_default[None, :]),
        "unit_action_delta_matches_action_scale": close(one_target, expected_one),
        "negative_unit_action_delta_matches_action_scale": close(neg_target, expected_neg),
        "large_action_clips_to_unit_scale": close(large_target, expected_one),
        "unit_targets_inside_mujoco_ctrlrange": bool(
            one_ctrl_meta["ctrlrange_clip_fraction"] == 0.0 and neg_ctrl_meta["ctrlrange_clip_fraction"] == 0.0
        ),
        "does_not_claim_rollout_or_success": True,
    }
    hard = [
        "paper_formula_available",
        "isaaclab_affine_joint_action_semantics_available",
        "official_action_scale_rows_29",
        "controller_default_pose_available",
        "zero_default_fallback_not_used",
        "mujoco_mapping_order_matches_action_rows",
        "pd_actuator_order_matches_action_rows",
        "zero_action_returns_default_pose",
        "unit_action_delta_matches_action_scale",
        "negative_unit_action_delta_matches_action_scale",
        "large_action_clips_to_unit_scale",
        "does_not_claim_rollout_or_success",
    ]
    warnings: list[str] = []
    if not checks["unit_targets_inside_mujoco_ctrlrange"]:
        warnings.append(
            "MuJoCo actuator ctrlrange clips unit action setpoints for "
            f"{', '.join(row['joint_name'] for row in ctrlrange_violations) or 'unknown joints'}; "
            "native rollout code must record raw setpoint versus MuJoCo-clipped setpoint and cannot claim final success until this is resolved or justified."
        )
    if float(np.max(np.abs(default_delta))) > 1e-12:
        warnings.append(
            "IsaacLab InitialStateCfg and motion_tracking_controller standby default_position differ at ankle pitch by about 0.033 rad; rollout code should prefer exported ONNX metadata when available."
        )
    formula_ready = all(checks[k] for k in hard)
    rollout_ready = formula_ready and checks["unit_targets_inside_mujoco_ctrlrange"]
    status = (
        "ok_native_action_adapter_rollout_formula_and_ctrlrange_ready"
        if rollout_ready
        else (
            "blocked_native_action_adapter_ctrlrange_rollout_gate_formula_ready"
            if formula_ready
            else "blocked_native_action_adapter_formula_contract"
        )
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "mujoco_native_action_adapter_contract",
        "scope": "Formula/order/default-pose fixture for normalized action to MuJoCo PD setpoint conversion.",
        "files": {key: str(path) for key, path in FILES.items()},
        "formula": {
            "paper": "theta_sp = theta_default + alpha * action",
            "implemented": "raw_targets = default_joint_pos + clip(action, -1, 1) * action_scale; MuJoCo ctrlrange clipping is tracked separately as simulator limit protection",
            "function": "normalized_action_to_pd_target",
            "clip_abs_for_fixture": 1.0,
            "isaaclab_evidence": {
                "joint_action_source": str(FILES["isaaclab_joint_actions"]),
                "rsl_rl_wrapper_source": str(FILES["isaaclab_rsl_rl_wrapper"]),
            },
        },
        "joint_order": {
            "action_joint_names": action_joint_names,
            "controller_joint_names": controller_joint_names,
            "mujoco_mapping_joint_names": mapping_joint_names,
            "pd_actuator_joint_names": pd_xml.get("actuator_joints", []),
        },
        "default_pose": {
            "selected_source": selected_source,
            "selected_default_joint_pos": [float(x) for x in selected_default],
            "isaaclab_initial_default_joint_pos": [float(x) for x in isaac_default],
            "isaaclab_vs_deployment_max_abs_delta": float(np.max(np.abs(default_delta))) if len(default_delta) else math.nan,
            "isaaclab_vs_deployment_nonzero_delta_joints": [
                {
                    "joint_name": name,
                    "deployment_default": float(selected_default[idx]),
                    "isaaclab_default": float(isaac_default[idx]),
                    "delta": float(default_delta[idx]),
                }
                for idx, name in enumerate(action_joint_names)
                if abs(float(default_delta[idx])) > 1e-12
            ],
            "delta_note": (
                "The official deployment standby default differs from the IsaacLab InitialStateCfg mainly at ankle pitch "
                "(-0.33 versus -0.363 rad). A real rollout should prefer the exported ONNX metadata default_joint_pos when available."
            ),
        },
        "action_scale": {
            "values": [float(x) for x in action_scale],
            "min": float(np.min(action_scale)) if len(action_scale) else math.nan,
            "max": float(np.max(action_scale)) if len(action_scale) else math.nan,
            "mean": float(np.mean(action_scale)) if len(action_scale) else math.nan,
        },
        "ctrlrange_analysis": {
            "unit_targets_inside_mujoco_ctrlrange": bool(checks["unit_targets_inside_mujoco_ctrlrange"]),
            "violating_joint_count": len(ctrlrange_violations),
            "violating_joints": ctrlrange_violations,
            "interpretation": (
                "The paper/IsaacLab normalized action formula remains correct, but the current MuJoCo actuator "
                "or joint range would clip some legal +-1 normalized setpoints before physics sees them."
            ),
        },
        "fixtures": {
            "zero_action": {"target_first_three": [float(x) for x in zero_target[0, :3]], **zero_meta},
            "unit_action": {"target_first_three": [float(x) for x in one_target[0, :3]], **one_meta},
            "negative_unit_action": {"target_first_three": [float(x) for x in neg_target[0, :3]], **neg_meta},
            "large_action_clipped": {"target_first_three": [float(x) for x in large_target[0, :3]], **large_meta},
            "unit_action_with_mujoco_ctrlrange": {
                "target_first_three": [float(x) for x in one_ctrl_target[0, :3]],
                **one_ctrl_meta,
            },
            "negative_unit_action_with_mujoco_ctrlrange": {
                "target_first_three": [float(x) for x in neg_ctrl_target[0, :3]],
                **neg_ctrl_meta,
            },
        },
        "checks": checks,
        "rollout_readiness_warnings": warnings,
        "interpretation": {
            "formula_adapter_ready": formula_ready,
            "rollout_adapter_ready": rollout_ready,
            "native_obs_adapter_ready": False,
            "physics_rollout_performed": False,
            "success_video_claim_allowed": False,
            "mujoco_ctrlrange_warning_blocks_final_rollout_claim": not checks["unit_targets_inside_mujoco_ctrlrange"],
            "goal_complete": False,
            "next_step": (
                "Use this adapter in a no-root-assist native rollout that reconstructs observation terms and feeds "
                "policy/VAE/diffusion actions through the verified formula before mj_step."
            ),
        },
    }
    write_outputs(summary)
    print(json.dumps({"status": status, "json": str(JSON_OUT), "md": str(MD_OUT), "tsv": str(TSV_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
