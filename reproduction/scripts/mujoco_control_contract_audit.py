#!/usr/bin/env python3
"""Audit MuJoCo video/control adapters against the BeyondMimic control contract.

The user-visible MuJoCo videos are useful for reporting, but the paper's control
contract is stricter: a policy/diffusion controller should emit normalized
actions, convert them to PD position setpoints by theta_sp = theta0 + alpha * a,
and let physics advance without manually fitting body poses or applying a pelvis
root-assist stabilizer.  This audit records which parts of the local MuJoCo
stack match the official parameters and which parts must remain diagnostic.
"""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/mujoco_control_contract_audit"
JSON_OUT = OUT / "mujoco_control_contract_audit.json"
TSV_OUT = OUT / "mujoco_control_contract_audit.tsv"
MD_OUT = OUT / "mujoco_control_contract_audit.md"

FILES = {
    "paper_method": ROOT / "reproduction/paper/source/tex/method.tex",
    "paper_supplement": ROOT / "reproduction/paper/source/root.tex",
    "official_tracking_env": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "official_g1_robot": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/g1.py",
    "official_g1_flat": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
    "action_scale_audit": ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
    "joint_mapping": ROOT / "mujoco_mp4/configs/g1_joint_mapping.yaml",
    "mujoco_pd_script": ROOT / "mujoco_mp4/scripts/mujoco_pd_control_video.py",
    "mujoco_trace_script": ROOT / "mujoco_mp4/scripts/mujoco_trace_mesh_video.py",
    "mujoco_base_xml": ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml",
    "mujoco_pd_xml": ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml",
    "mujoco_reference_summary": ROOT
    / "mujoco_mp4/res/control_videos/reference_control/reference_control_summary.json",
    "final_clean_walk_reference_summary": ROOT
    / "res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/reference_action_control/"
    "reference_action_control_summary.json",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def first_line(path: Path, needle: str) -> str:
    text = read_text(path)
    for idx, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return ""


def parse_pd_xml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    root = ET.parse(path).getroot()
    option = root.find("option")
    actuators = list(root.find("actuator") or [])
    floor = None
    for geom in root.findall(".//geom"):
        if geom.attrib.get("type") == "plane" or geom.attrib.get("name") == "floor":
            floor = geom.attrib
            break
    joints: dict[str, dict[str, str]] = {}
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name:
            joints[name] = joint.attrib
    return {
        "exists": True,
        "option": option.attrib if option is not None else {},
        "actuator_count": len(actuators),
        "actuator_tags": sorted({act.tag for act in actuators}),
        "actuator_names": [act.attrib.get("joint", act.attrib.get("name", "")) for act in actuators],
        "first_actuator": actuators[0].attrib if actuators else {},
        "floor": floor or {},
        "patched_joint_count": len([j for j in joints.values() if "armature" in j or "damping" in j]),
        "patched_joint_names": [name for name, attrs in joints.items() if "armature" in attrs or "damping" in attrs],
    }


def compare_action_rows_to_xml(action_rows: list[dict[str, Any]], pd_xml: dict[str, Any]) -> dict[str, Any]:
    if not pd_xml.get("exists") or not action_rows:
        return {"comparable": False}
    names = pd_xml.get("actuator_names", [])
    rows_by_joint = {row["joint_name"]: row for row in action_rows}
    actuator_names_match = names == [row["joint_name"] for row in action_rows]
    return {
        "comparable": True,
        "actuator_names_match_action_scale_order": actuator_names_match,
        "action_row_count": len(action_rows),
        "xml_actuator_count": pd_xml.get("actuator_count"),
        "all_action_rows_have_positive_scale": all(float(row.get("action_scale", 0.0)) > 0 for row in action_rows),
        "kp_kv_armature_source": str(FILES["action_scale_audit"]),
        "sample_expected": {
            name: {
                "stiffness": rows_by_joint[name]["stiffness"],
                "damping": rows_by_joint[name]["damping"],
                "armature": rows_by_joint[name]["armature"],
                "effort_limit_sim": rows_by_joint[name]["effort_limit_sim"],
                "action_scale": rows_by_joint[name]["action_scale"],
            }
            for name in names[:3]
            if name in rows_by_joint
        },
    }


def component_rows() -> list[dict[str, Any]]:
    paper_method = read_text(FILES["paper_method"])
    paper_supplement = read_text(FILES["paper_supplement"])
    official_env = read_text(FILES["official_tracking_env"])
    official_g1 = read_text(FILES["official_g1_robot"])
    official_flat = read_text(FILES["official_g1_flat"])
    pd_script = read_text(FILES["mujoco_pd_script"])
    trace_script = read_text(FILES["mujoco_trace_script"])
    action_payload = read_json(FILES["action_scale_audit"])
    action_rows = action_payload.get("joint_rows", [])
    pd_xml = parse_pd_xml(FILES["mujoco_pd_xml"])
    base_xml = parse_pd_xml(FILES["mujoco_base_xml"])
    reference_summary = read_json(FILES["mujoco_reference_summary"])
    final_walk_summary = read_json(FILES["final_clean_walk_reference_summary"])
    xml_compare = compare_action_rows_to_xml(action_rows, pd_xml)

    return [
        {
            "component": "paper_action_contract",
            "status": "available",
            "matches_paper": bool(
                "theta}^{\\text{sp}}" in paper_method
                and "\\boldsymbol{\\alpha}" in paper_method
                and "0.25\\frac" in paper_supplement
            ),
            "evidence": [
                first_line(FILES["paper_method"], "Action is designed as normalized joint position setpoints"),
                first_line(FILES["paper_supplement"], "0.25\\frac"),
            ],
            "notes": "Paper action is normalized policy output mapped to PD setpoint theta_sp = theta0 + alpha * action.",
        },
        {
            "component": "official_stage1_pd_action_scale",
            "status": action_payload.get("status", "missing"),
            "matches_paper": bool(
                action_payload.get("checks", {}).get("action_scale_formula_matches_official")
                and action_payload.get("checks", {}).get("natural_frequency_10hz_formula")
                and action_payload.get("checks", {}).get("expanded_joint_rows_29")
                and "G1_ACTION_SCALE" in official_g1
                and "self.actions.joint_pos.scale = G1_ACTION_SCALE" in official_flat
            ),
            "evidence": [
                str(FILES["official_g1_robot"]),
                str(FILES["official_g1_flat"]),
                str(FILES["action_scale_audit"]),
            ],
            "notes": "Official G1 action scale, stiffness, damping, effort limits, and armature are available and match the paper formula audit.",
            "details": {
                "constants": action_payload.get("constants", {}),
                "metrics": action_payload.get("metrics", {}),
            },
        },
        {
            "component": "official_stage1_material_randomization",
            "status": "available",
            "matches_paper": bool(
                "static_friction_range" in official_env
                and "dynamic_friction_range" in official_env
                and "restitution_range" in official_env
                and "randomize_joint_default_pos" in official_env
                and "randomize_rigid_body_com" in official_env
            ),
            "evidence": [str(FILES["official_tracking_env"])],
            "notes": "Official IsaacLab training randomizes contact coefficients, default joint positions, torso COM, and pushes as described in the paper.",
        },
        {
            "component": "mujoco_pd_xml_parameter_patch",
            "status": "available" if pd_xml.get("exists") else "missing",
            "matches_paper": bool(
                pd_xml.get("exists")
                and pd_xml.get("actuator_count") == 29
                and pd_xml.get("actuator_tags") == ["position"]
                and xml_compare.get("actuator_names_match_action_scale_order")
                and pd_xml.get("patched_joint_count") >= 29
            ),
            "evidence": [str(FILES["mujoco_pd_xml"]), str(FILES["action_scale_audit"])],
            "notes": "The local MuJoCo PD XML is numerically patched with official 29-DoF stiffness/damping/armature/effort settings, but this only covers actuator parameters.",
            "details": {"pd_xml": pd_xml, "action_xml_compare": xml_compare},
        },
        {
            "component": "mujoco_floor_material_gap",
            "status": "diagnostic_gap",
            "matches_paper": False,
            "evidence": [str(FILES["mujoco_base_xml"]), str(FILES["official_tracking_env"])],
            "notes": (
                "The official IsaacLab terrain uses static/dynamic friction 1.0 and randomized material buckets during training. "
                "The local MuJoCo base XML floor records friction around 0.6 in at least one asset path, and material randomization is not reproduced in the video controller."
            ),
            "details": {"base_floor": base_xml.get("floor"), "pd_floor": pd_xml.get("floor")},
        },
        {
            "component": "mujoco_pd_video_control_semantics",
            "status": "diagnostic_not_native_policy_control",
            "matches_paper": False,
            "evidence": [str(FILES["mujoco_pd_script"]), str(FILES["mujoco_reference_summary"])],
            "notes": (
                "The MuJoCo PD video script directly drives absolute joint targets or IK-fitted targets with 29 position actuators. "
                "It does not reconstruct the IsaacLab observation manager, run PPO/VAE/diffusion to produce normalized actions, or apply theta0 + alpha * action as the native control path."
            ),
            "details": {
                "script_uses_absolute_joint_targets": "data.ctrl[:] = target" in pd_script,
                "script_loads_reference_joint_pos": "motion[\"joint_pos\"]" in pd_script,
                "script_uses_ik_targets": "trace_to_ik_targets" in pd_script,
                "native_mujoco_ppo_obs_adapter": reference_summary.get("checks", {}).get("native_mujoco_ppo_obs_adapter"),
            },
        },
        {
            "component": "mujoco_root_assist_semantics",
            "status": "blocks_success_claim",
            "matches_paper": False,
            "evidence": [str(FILES["mujoco_pd_script"]), str(FILES["mujoco_reference_summary"])],
            "notes": (
                "The script applies an external pelvis force/torque root-assist stabilizer by default. This is useful for centered videos but prevents any claim of unassisted humanoid balance/control."
            ),
            "details": {
                "apply_root_assist_present": "def apply_root_assist" in pd_script,
                "default_root_assist_enabled": 'BM_MUJOCO_ROOT_ASSIST", "1"' in pd_script,
                "summary_root_assist_enabled": reference_summary.get("simulation", {}).get("root_assist_enabled"),
                "summary_root_assist_type": reference_summary.get("simulation", {}).get("root_assist_type"),
            },
        },
        {
            "component": "mujoco_trace_mesh_video_semantics",
            "status": "kinematic_or_ik_visualization_only",
            "matches_paper": False,
            "evidence": [str(FILES["mujoco_trace_script"])],
            "notes": (
                "Trace mesh videos fit MuJoCo qpos to captured body positions via IK or write qpos for reference replay. They are visualization assets, not closed-loop policy/diffusion controllers."
            ),
            "details": {
                "uses_ik_solver": "solve_ik_frame" in trace_script,
                "does_not_claim_policy_adapter": "not native MuJoCo PPO controller" in trace_script,
            },
        },
        {
            "component": "current_report_video_claims",
            "status": "diagnostic_claims_recorded",
            "matches_paper": False,
            "evidence": [str(FILES["final_clean_walk_reference_summary"]), str(FILES["mujoco_reference_summary"])],
            "notes": "Current summaries already mark clean-walk/MuJoCo outputs as diagnostic/local evidence, not paper-level success.",
            "details": {
                "final_clean_walk_claim": final_walk_summary.get("claim_level"),
                "mujoco_reference_claim": reference_summary.get("claim_level"),
            },
        },
    ]


def write_tsv(rows: list[dict[str, Any]]) -> None:
    fields = ["component", "status", "matches_paper", "notes", "evidence", "details"]
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "component": row["component"],
                    "status": row["status"],
                    "matches_paper": row["matches_paper"],
                    "notes": row["notes"],
                    "evidence": json.dumps(row.get("evidence", []), ensure_ascii=False),
                    "details": json.dumps(row.get("details", {}), ensure_ascii=False, sort_keys=True),
                }
            )


def write_md(summary: dict[str, Any]) -> None:
    rows = summary["component_rows"]
    failed = [row for row in rows if not row["matches_paper"]]
    lines = [
        "# MuJoCo Control Contract Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Conclusion: the MuJoCo video stack contains useful local visualization assets, but it is not yet a native BeyondMimic control reproduction path.",
        "- 当前不得声称完整复现 BeyondMimic，也不得把 root-assist/IK/absolute-target MuJoCo 视频作为 teacher/VAE/diffusion 成功结果。",
        "",
        "## Blocking Gaps",
        "",
    ]
    lines.extend(f"- `{row['component']}`: {row['notes']}" for row in failed)
    lines.extend(["", "## Component Rows", ""])
    for row in rows:
        lines.append(f"### {row['component']}")
        lines.append(f"- Status: `{row['status']}`")
        lines.append(f"- Matches paper/control contract: `{row['matches_paper']}`")
        lines.append(f"- Notes: {row['notes']}")
        lines.append("")
    lines.extend(
        [
            "## Next Gate",
            "",
            "Before producing final single-leg/walk success videos, implement or verify a native MuJoCo/IsaacLab adapter that uses the policy/VAE/diffusion output action as normalized action, applies `theta0 + alpha * action`, disables root assist, and logs observation/action contract consistency.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = component_rows()
    hard_checks = {
        "paper_action_contract_available": next(row["matches_paper"] for row in rows if row["component"] == "paper_action_contract"),
        "official_stage1_pd_action_scale_available": next(
            row["matches_paper"] for row in rows if row["component"] == "official_stage1_pd_action_scale"
        ),
        "official_stage1_material_randomization_available": next(
            row["matches_paper"] for row in rows if row["component"] == "official_stage1_material_randomization"
        ),
        "mujoco_pd_xml_parameter_patch_matches_official_numbers": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_pd_xml_parameter_patch"
        ),
        "mujoco_floor_material_matches_official_training": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_floor_material_gap"
        ),
        "mujoco_video_uses_native_policy_action_semantics": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_pd_video_control_semantics"
        ),
        "mujoco_video_has_no_root_assist": next(row["matches_paper"] for row in rows if row["component"] == "mujoco_root_assist_semantics"),
        "trace_videos_are_closed_loop_control": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_trace_mesh_video_semantics"
        ),
    }
    status = (
        "ok_mujoco_native_control_contract_ready"
        if all(hard_checks.values())
        else "blocked_mujoco_control_semantics_not_native_policy_control"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "mujoco_control_contract_audit",
        "scope": "MuJoCo control/video adapter contract versus BeyondMimic paper action/PD/material semantics.",
        "component_rows": rows,
        "checks": hard_checks,
        "conclusion": {
            "mujoco_pd_values_available": hard_checks["mujoco_pd_xml_parameter_patch_matches_official_numbers"],
            "native_policy_control_available": hard_checks["mujoco_video_uses_native_policy_action_semantics"],
            "root_assist_blocks_success_claim": not hard_checks["mujoco_video_has_no_root_assist"],
            "can_use_current_mujoco_videos_as_final_success": False,
            "should_train_from_video_adapter_now": False,
            "recommended_next_steps": [
                "Keep current MuJoCo videos as diagnostic/report assets only.",
                "Implement a native action adapter before final success videos: obs -> model -> normalized action -> theta0 + alpha * action -> MuJoCo PD -> mj_step.",
                "Disable root assist for success gates and log fall/recovery as real physics outcomes.",
                "Patch or document MuJoCo floor/material/friction differences before comparing control quality to IsaacLab or paper claims.",
            ],
        },
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(rows)
    write_md(summary)
    print(json.dumps({"status": status, "json": str(JSON_OUT), "md": str(MD_OUT), "tsv": str(TSV_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
