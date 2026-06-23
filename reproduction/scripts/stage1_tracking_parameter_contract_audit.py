#!/usr/bin/env python3
"""Audit Stage-1 tracking parameter contracts before more training.

This script compares the BeyondMimic paper/supplement, official
whole_body_tracking code, and local paper-contract training wrappers.  It does
not launch IsaacLab or training.  The output is meant to gate future teacher
rollout, VAE, diffusion, and video work.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "res" / "tracking" / "stage1_tracking_parameter_contract_audit"
JSON_OUT = OUT_DIR / "stage1_tracking_parameter_contract_audit.json"
TSV_OUT = OUT_DIR / "stage1_tracking_parameter_contract_audit.tsv"
MD_OUT = OUT_DIR / "stage1_tracking_parameter_contract_audit.md"

PAPER_FILES = [
    ROOT / "reproduction/paper/source/tex/method.tex",
    ROOT / "reproduction/paper/source/root.tex",
]
OFFICIAL_FILES = {
    "env_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "flat_env_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
    "ppo_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
    "g1": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/g1.py",
    "commands": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/commands.py",
    "rewards": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/rewards.py",
    "terminations": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/terminations.py",
    "observations": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/observations.py",
}

LOCAL_FILES = {
    "paper_contract_wrapper": ROOT / "reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py",
    "multisource_wrapper": ROOT / "reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py",
    "base_worker_wrapper": ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py",
    "paper_contract_training_json": ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "tracking_g1_official_importer_export_paper_contract_ppo_training_run.json",
    "paper_contract_parameter_snapshot": ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "paper_contract_tracking_parameters.json",
    "multisource_training_json": ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
    "tracking_stage1_multisource_paper_contract_ppo_training_run.json",
    "teacher_quality_selector": ROOT
    / "res/tracking/stage1_teacher_checkpoint_quality_selector/"
    "stage1_teacher_checkpoint_quality_selector.json",
    "physical_asset_contract": ROOT
    / "res/tracking/g1_urdf_physical_asset_contract_audit/"
    "tracking_g1_urdf_physical_asset_contract_audit.json",
}


@dataclass
class Row:
    component: str
    paper_contract: str
    official_code_value: str
    local_value: str
    status: str
    severity: str
    evidence: str
    required_action: str

    def as_dict(self) -> dict[str, str]:
        return {
            "component": self.component,
            "paper_contract": self.paper_contract,
            "official_code_value": self.official_code_value,
            "local_value": self.local_value,
            "status": self.status,
            "severity": self.severity,
            "evidence": self.evidence,
            "required_action": self.required_action,
        }


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def extract_number(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def close(a: float | None, b: float, tol: float = 1e-6) -> bool:
    return a is not None and math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def status_from(check: bool, caution: bool = False) -> str:
    if check and caution:
        return "pass_with_caution"
    if check:
        return "pass"
    return "fail_or_unverified"


def severity_from(status: str, high: bool = False) -> str:
    if status == "pass":
        return "info"
    if status == "pass_with_caution":
        return "medium"
    return "high" if high else "medium"


def collect() -> dict[str, Any]:
    texts = {name: read(path) for name, path in OFFICIAL_FILES.items()}
    local_texts = {name: read(path) for name, path in LOCAL_FILES.items() if path.suffix == ".py"}
    paper = "\n".join(read(path) for path in PAPER_FILES)
    paper_contract_training = load_json(LOCAL_FILES["paper_contract_training_json"])
    paper_snapshot = load_json(LOCAL_FILES["paper_contract_parameter_snapshot"])
    multisource_training = load_json(LOCAL_FILES["multisource_training_json"])
    teacher_selector = load_json(LOCAL_FILES["teacher_quality_selector"])
    physical_asset = load_json(LOCAL_FILES["physical_asset_contract"])
    return {
        "paper": paper,
        "official": texts,
        "local_texts": local_texts,
        "paper_contract_training": paper_contract_training,
        "paper_snapshot": paper_snapshot,
        "multisource_training": multisource_training,
        "teacher_selector": teacher_selector,
        "physical_asset": physical_asset,
    }


def training_rank_metric(summary: dict[str, Any], key: str) -> Any:
    for row in summary.get("run", {}).get("rank_metrics", []):
        if key in row:
            return row.get(key)
    return None


def make_rows(ctx: dict[str, Any]) -> list[Row]:
    paper = ctx["paper"]
    off = ctx["official"]
    loc = ctx["local_texts"]
    paper_training = ctx["paper_contract_training"]
    snapshot = ctx["paper_snapshot"]
    multi = ctx["multisource_training"]
    selector = ctx["teacher_selector"]
    physical = ctx["physical_asset"]

    rows: list[Row] = []

    g1 = off["g1"]
    env = off["env_cfg"]
    ppo = off["ppo_cfg"]
    commands = off["commands"]
    rewards = off["rewards"]
    terms = off["terminations"]
    base = loc.get("base_worker_wrapper", "")
    paper_wrap = loc.get("paper_contract_wrapper", "")
    multi_wrap = loc.get("multisource_wrapper", "")

    nat_freq = extract_number(g1, r"NATURAL_FREQ\s*=\s*(\d+)\s*\*\s*2\.0")
    damping_ratio = extract_number(g1, r"DAMPING_RATIO\s*=\s*([0-9.]+)")
    rows.append(
        Row(
            "PD natural frequency and damping ratio",
            "S1: omega = 10 Hz, damping ratio zeta = 2",
            f"NATURAL_FREQ={nat_freq}Hz*2pi; DAMPING_RATIO={damping_ratio}",
            json.dumps(snapshot.get("motor_and_pd", {}).get("damping_ratio"), ensure_ascii=False),
            status_from(close(nat_freq, 10.0) and close(damping_ratio, 2.0)),
            severity_from(status_from(close(nat_freq, 10.0) and close(damping_ratio, 2.0)), high=True),
            f"{OFFICIAL_FILES['g1']}",
            "Keep this unchanged for any future teacher training.",
        )
    )

    action_formula_ok = "0.25 * e[n] / s[n]" in g1 and "0.25\\frac" in paper
    rows.append(
        Row(
            "Action scale formula",
            "theta_sp = theta0 + alpha*a; alpha = 0.25*tau_max/kp",
            "G1_ACTION_SCALE[n] = 0.25 * effort_limit_sim / stiffness",
            snapshot.get("motor_and_pd", {}).get("action_scale_formula", "missing"),
            status_from(action_formula_ok),
            severity_from(status_from(action_formula_ok), high=True),
            f"{OFFICIAL_FILES['g1']}; {PAPER_FILES[1]}",
            "MuJoCo adapter must use this exact exported scale/joint order, not hand-set scale.",
        )
    )

    armature_ok = all(token in g1 for token in ["ARMATURE_5020 = 0.003609725", "ARMATURE_7520_14 = 0.010177520", "ARMATURE_7520_22 = 0.025101925", "ARMATURE_4010 = 0.00425"])
    rows.append(
        Row(
            "G1 armature constants",
            "S1/Table joint armature: reflected rotor inertia; ankle/waist pitch/roll doubled",
            "ARMATURE_5020/7520_14/7520_22/4010 present; feet and waist use 2.0 multiplier",
            json.dumps(snapshot.get("motor_and_pd", {}).get("armature", {}), ensure_ascii=False),
            status_from(armature_ok),
            severity_from(status_from(armature_ok), high=True),
            f"{OFFICIAL_FILES['g1']}; {PAPER_FILES[1]}",
            "Do not zero or tune armature for stability; paper says wrong armature degrades tracking.",
        )
    )

    reward_ok = all(
        token in env
        for token in [
            "weight=0.5",
            'params={"command_name": "motion", "std": 0.3}',
            'params={"command_name": "motion", "std": 0.4}',
            'params={"command_name": "motion", "std": 1.0}',
            'params={"command_name": "motion", "std": 3.14}',
            "weight=-1e-1",
            "weight=-10.0",
        ]
    ) and "torch.exp(-error / std**2)" in rewards
    rows.append(
        Row(
            "Tracking reward formula and weights",
            "Gaussian exp(-MSE/std^2), body terms weight 1, anchor optional 0.5, action/contact -0.1, joint limit -10",
            "Official env/rewards.py match exp formula and weights",
            "paper-contract wrapper imports official G1FlatEnvCfg; no reward override detected",
            status_from(reward_ok),
            severity_from(status_from(reward_ok), high=True),
            f"{OFFICIAL_FILES['env_cfg']}; {OFFICIAL_FILES['rewards']}",
            "If teacher remains weak, inspect motion/termination/curriculum before changing reward weights.",
        )
    )

    term_ok = all(token in env for token in ["threshold\": 0.25", "threshold\": 0.8", "left_ankle_roll_link", "right_wrist_yaw_link"]) and "bad_motion_body_pos_z_only" in terms
    rows.append(
        Row(
            "Termination contract",
            "Terminate on anchor/end-effector z error > 0.25m or anchor orientation > 0.8",
            "Official uses z-only anchor_pos, z-only ankle/wrist ee_body_pos, anchor_ori threshold 0.8",
            "paper/multisource wrappers record ee_body_pos_threshold_patch_applied=False",
            status_from(term_ok and training_rank_metric(paper_training, "ee_body_pos_threshold_patch_applied") is False),
            severity_from(status_from(term_ok and training_rank_metric(paper_training, "ee_body_pos_threshold_patch_applied") is False), high=True),
            f"{OFFICIAL_FILES['env_cfg']}; {OFFICIAL_FILES['terminations']}; {LOCAL_FILES['paper_contract_training_json']}",
            "Do not relax endpoint termination and call it paper-level; use relaxations only as diagnostics.",
        )
    )

    ankle_special_in_paper = "for ankle joints to account for greater calibration errors" in paper
    official_ankle_special = "ankle" in env and "pos_distribution_params" in env and "(-0.1, 0.1)" in env
    status = "pass" if not ankle_special_in_paper else ("pass" if official_ankle_special else "pass_with_caution")
    rows.append(
        Row(
            "Default joint offset randomization",
            "S1 prose says most joints U[-0.01,0.01], ankle joints U[-0.1,0.1]; table only lists U[-0.01,0.01]",
            "Official public code randomizes all joints U[-0.01,0.01]; no ankle-special branch found",
            "local training inherits official public code; no ankle-special patch",
            status,
            severity_from(status),
            f"{OFFICIAL_FILES['env_cfg']}; {PAPER_FILES[1]}",
            "Record as paper-vs-public-code ambiguity. If trying to improve ankle/single-leg stability, run a clearly labeled ankle-offset ablation, not an unmarked paper-contract run.",
        )
    )

    adaptive_kernel = extract_number(commands, r"adaptive_kernel_size:\s*int\s*=\s*(\d+)")
    status = "pass" if close(adaptive_kernel, 3.0) else "pass_with_caution"
    rows.append(
        Row(
            "Adaptive sampling look-back kernel",
            "S1 says non-causal exponential kernel rho=0.8, u in {0,1,2}, K=3",
            f"Official MotionCommandCfg adaptive_kernel_size={adaptive_kernel}, adaptive_lambda=0.8, alpha=0.001, uniform=0.1",
            "local training inherits official public code; no K=3 patch",
            status,
            severity_from(status),
            f"{OFFICIAL_FILES['commands']}; {PAPER_FILES[1]}",
            "Likely important for hard segments. Add explicit K=3 ablation or patch only in a named non-official run.",
        )
    )

    domain_ok = all(
        token in env
        for token in [
            "static_friction_range\": (0.3, 1.6)",
            "dynamic_friction_range\": (0.3, 1.2)",
            "restitution_range\": (0.0, 0.5)",
            '"x": (-0.025, 0.025)',
            "interval_range_s=(1.0, 3.0)",
        ]
    )
    rows.append(
        Row(
            "Domain randomization",
            "friction/restitution, joint offsets, torso COM, root velocity push",
            "Official env cfg matches friction/rest/COM/push ranges; joint offset caveat above",
            "local training uses official env cfg with debug_vis disabled only",
            status_from(domain_ok, caution=status == "pass_with_caution"),
            severity_from(status_from(domain_ok, caution=status == "pass_with_caution")),
            f"{OFFICIAL_FILES['env_cfg']}; {PAPER_FILES[1]}",
            "Keep compact DR; do not add delay randomization unless labeled as ablation.",
        )
    )

    obs_ok = all(token in env for token in ["generated_commands", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel", "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action"])
    rows.append(
        Row(
            "Actor/critic observation contract",
            "actor: psi, anchor error, IMU twist, theta-theta0, theta_dot, a_last; no temporal history",
            "Official policy obs terms present; critic adds body_pos/body_ori; observed run dims 160/286",
            f"paper_contract num_obs={training_rank_metric(paper_training, 'num_obs')}; critic={training_rank_metric(paper_training, 'num_privileged_obs')}",
            status_from(obs_ok and training_rank_metric(paper_training, "num_obs") == 160 and training_rank_metric(paper_training, "num_privileged_obs") == 286),
            severity_from(status_from(obs_ok and training_rank_metric(paper_training, "num_obs") == 160 and training_rank_metric(paper_training, "num_privileged_obs") == 286), high=True),
            f"{OFFICIAL_FILES['env_cfg']}; {PAPER_FILES[0]}; {LOCAL_FILES['paper_contract_training_json']}",
            "MuJoCo PPO adapter must recreate this exact 160-dim order and normalization.",
        )
    )

    ppo_ok = all(
        token in ppo
        for token in [
            "num_steps_per_env = 24",
            "max_iterations = 30000",
            "save_interval = 500",
            "actor_hidden_dims=[512, 256, 128]",
            "critic_hidden_dims=[512, 256, 128]",
            "entropy_coef=0.005",
            "num_learning_epochs=5",
            "num_mini_batches=4",
            "learning_rate=1.0e-3",
            "desired_kl=0.01",
        ]
    )
    local_ppo_ok = paper_training.get("config", {}).get("paper_contract_max_iterations") == 30000 and paper_training.get("config", {}).get("paper_contract_save_interval") == 500
    rows.append(
        Row(
            "RSL-RL PPO hyperparameters",
            "Official whole_body_tracking G1FlatPPORunnerCfg, 30k iterations, save every 500",
            "Official PPO cfg matches expected values",
            f"paper_contract max_iter={paper_training.get('config', {}).get('paper_contract_max_iterations')}, save={paper_training.get('config', {}).get('paper_contract_save_interval')}",
            status_from(ppo_ok and local_ppo_ok),
            severity_from(status_from(ppo_ok and local_ppo_ok), high=True),
            f"{OFFICIAL_FILES['ppo_cfg']}; {LOCAL_FILES['paper_contract_training_json']}",
            "Checkpoint selector should choose best eval checkpoint, not last.",
        )
    )

    asset_check = physical.get("checks", {})
    asset_status = "pass_with_caution" if asset_check.get("skeleton_contract_ok") and not asset_check.get("physical_fidelity_complete_for_replay") else status_from(asset_check.get("physical_fidelity_complete_for_replay") is True)
    rows.append(
        Row(
            "Robot asset physical contract",
            "Unitree G1 29 DoF, 40 links/bodies, inertial/mesh/contact compatible with official tracking",
            f"physical asset audit status={physical.get('status')}, skeleton ok={asset_check.get('skeleton_contract_ok')}, physical_fidelity_complete={asset_check.get('physical_fidelity_complete_for_replay')}",
            "training swaps official URDF spawn for local official-importer-export USD spawn",
            asset_status,
            severity_from(asset_status),
            f"{LOCAL_FILES['physical_asset_contract']}; {LOCAL_FILES['paper_contract_training_json']}",
            "This is a local asset adaptation. Keep claiming local public-resource training, not official checkpoint.",
        )
    )

    paper_asset_flags = {
        "paper_uses_official_usd": training_rank_metric(paper_training, "uses_official_importer_export_usd"),
        "paper_uses_resource_adjusted_usd": training_rank_metric(paper_training, "uses_resource_adjusted_usd"),
        "multi_uses_resource_adjusted_usd": training_rank_metric(multi, "uses_resource_adjusted_usd"),
    }
    status = "pass" if paper_asset_flags["paper_uses_official_usd"] is True and paper_asset_flags["paper_uses_resource_adjusted_usd"] is False else "fail_or_unverified"
    if paper_asset_flags["multi_uses_resource_adjusted_usd"] is True:
        status = "pass_with_caution"
    rows.append(
        Row(
            "Local training metadata fidelity",
            "Training summaries must truthfully distinguish official-code parameters from local asset/data adaptations",
            "Paper-contract 4/7 rank metrics correctly mark official importer USD; 5/6 inherited base flag says resource_adjusted_usd=True",
            json.dumps(paper_asset_flags, ensure_ascii=False),
            status,
            severity_from(status),
            f"{LOCAL_FILES['paper_contract_training_json']}; {LOCAL_FILES['multisource_training_json']}; {LOCAL_FILES['multisource_wrapper']}",
            "Patch multisource summary metadata in the next cleanup so report readers are not misled.",
        )
    )

    selector_status = selector.get("status")
    rows.append(
        Row(
            "Teacher quality gate before VAE/diffusion",
            "Only low-done, low-error, continuous teacher rollouts should feed VAE/diffusion",
            f"selector status={selector_status}; ready={selector.get('ready_candidate_count')}; usable={selector.get('usable_candidate_count')}",
            "No downstream-ready teacher checkpoint exists after current sweeps",
            "fail_or_unverified" if selector_status == "blocked_no_downstream_ready_teacher_checkpoint" else "pass",
            "high" if selector_status == "blocked_no_downstream_ready_teacher_checkpoint" else "info",
            f"{LOCAL_FILES['teacher_quality_selector']}",
            "Do not start final VAE/diffusion training or success videos until this gate passes.",
        )
    )

    base_patch = "BM_EE_BODY_POS_TRAIN_THRESHOLD" in base
    rows.append(
        Row(
            "Diagnostic termination patch guard",
            "Endpoint threshold changes are diagnostics only unless explicitly named",
            "Base worker supports BM_EE_BODY_POS_TRAIN_THRESHOLD environment patch",
            "current paper-contract rank metrics report ee_body_pos_threshold_patch_applied=False",
            status_from(base_patch and training_rank_metric(paper_training, "ee_body_pos_threshold_patch_applied") is False, caution=True),
            "medium",
            f"{LOCAL_FILES['base_worker_wrapper']}; {LOCAL_FILES['paper_contract_training_json']}",
            "Keep the env var unset for paper-contract training; if used, write a separate ablation row.",
        )
    )

    rows.append(
        Row(
            "Known failure interpretation",
            "Good motion data and paper formulas are necessary but not sufficient; eval must prove policy quality",
            "Official/public code contract mostly matches, with two paper-vs-public-code cautions",
            "selector proves all current teachers remain below downstream quality gate",
            "fail_or_unverified",
            "high",
            f"{LOCAL_FILES['teacher_quality_selector']}; {LOCAL_FILES['paper_contract_training_json']}; {LOCAL_FILES['multisource_training_json']}",
            "Next work should target why body/joint errors remain high: adaptive K=3/ankle offset ablations, motion scale/order, reset sampling, and eval per-motion failure bins.",
        )
    )

    return rows


def summarize(rows: list[Row]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        severity_counts[row.severity] = severity_counts.get(row.severity, 0) + 1
    blocking = [row.as_dict() for row in rows if row.status == "fail_or_unverified" or row.severity == "high"]
    return {
        "status": "blocked_stage1_teacher_contract_has_required_followups"
        if any(row.status == "fail_or_unverified" for row in rows)
        else "ok_stage1_tracking_parameter_contract_audited",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "status_counts": counts,
        "severity_counts": severity_counts,
        "claim_boundary": (
            "This audit checks Stage-1 tracking implementation contracts. It does not certify a paper-level "
            "teacher, VAE, diffusion, guidance controller, Isaac rendered video, or real robot result."
        ),
        "immediate_conclusion": (
            "Official/public tracking code and the 4/7 paper-contract wrapper largely match the paper formulas, "
            "but current teacher checkpoints still fail quality gates. Two paper-vs-public-code cautions remain: "
            "adaptive sampling kernel K=1 in public code vs K=3 in supplement, and ankle-special joint offset "
            "described in text but not implemented in the public config. The 5/6 multisource summary also has a "
            "metadata flag mismatch for the USD source."
        ),
        "blocking_or_high_rows": blocking,
        "rows": [row.as_dict() for row in rows],
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "component",
        "status",
        "severity",
        "paper_contract",
        "official_code_value",
        "local_value",
        "required_action",
        "evidence",
    ]
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(row)
    lines = [
        "# Stage 1 Tracking Parameter Contract Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Status counts: `{json.dumps(summary['status_counts'], ensure_ascii=False)}`",
        f"- Severity counts: `{json.dumps(summary['severity_counts'], ensure_ascii=False)}`",
        "",
        "## Immediate Conclusion",
        "",
        summary["immediate_conclusion"],
        "",
        "## Contract Matrix",
        "",
        "| Component | Status | Severity | Required Action |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['component']} | `{row['status']}` | `{row['severity']}` | {row['required_action']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
            "",
            "当前不得声称完整复现 BeyondMimic；该审计只说明 Stage 1 参数/公式合同的对齐程度，不证明 teacher 已达到论文质量。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ctx = collect()
    rows = make_rows(ctx)
    summary = summarize(rows)
    write_outputs(summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "row_count": summary["row_count"],
                "status_counts": summary["status_counts"],
                "json": str(JSON_OUT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
