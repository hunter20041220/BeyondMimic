#!/usr/bin/env python3
"""Trace BeyondMimic formulas/parameters to local code evidence.

This audit is deliberately conservative.  It is meant to be run before any
new long teacher/VAE/diffusion training, so it only reads source files and
small JSON summaries.  A failed gate here means downstream MuJoCo/IsaacLab
videos remain diagnostic, even if they render without falling.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/formula_parameter_trace_audit"
JSON_OUT = OUT / "beyondmimic_formula_parameter_trace_audit.json"
TSV_OUT = OUT / "beyondmimic_formula_parameter_trace_audit.tsv"
MD_OUT = OUT / "beyondmimic_formula_parameter_trace_audit.md"

PAPER_METHOD = ROOT / "reproduction/paper/source/tex/method.tex"
PAPER_SUPP = ROOT / "reproduction/paper/source/root.tex"
OFFICIAL_ROOT = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
)
TRACKING_CFG = OFFICIAL_ROOT / "tasks/tracking/tracking_env_cfg.py"
G1_CFG = OFFICIAL_ROOT / "tasks/tracking/config/g1/flat_env_cfg.py"
G1_ROBOT = OFFICIAL_ROOT / "robots/g1.py"
OBS = OFFICIAL_ROOT / "tasks/tracking/mdp/observations.py"
REWARDS = OFFICIAL_ROOT / "tasks/tracking/mdp/rewards.py"
TERMS = OFFICIAL_ROOT / "tasks/tracking/mdp/terminations.py"
COMMANDS = OFFICIAL_ROOT / "tasks/tracking/mdp/commands.py"
PPO_CFG = OFFICIAL_ROOT / "tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py"

RESOURCE_VAE = ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py"
PAPER_CONTRACT_VAE = ROOT / "reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py"
RESOURCE_DIFF = ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py"
PAPER_ARCH = ROOT / "reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py"
CLEAN_WALK_VIDEO = ROOT / "reproduction/scripts/render_clean_walk_mujoco_control_suite.py"
MUJOCO_PD_VIDEO = ROOT / "mujoco_mp4/scripts/mujoco_pd_control_video.py"
STATE_HELPER = ROOT / "reproduction/src/beyondmimic_reimpl/state.py"
SAMPLING_HELPER = ROOT / "reproduction/src/beyondmimic_reimpl/sampling.py"

PAPER_CONTRACT_VAE_JSON = (
    ROOT
    / "res/level_c/paper_contract_teacher_rollout_vae_training/"
    "level_c_paper_contract_teacher_rollout_vae_training.json"
)
MODEL_CHAIN_AUDIT_JSON = (
    ROOT
    / "res/audits/model_chain_paper_contract_audit/"
    "beyondmimic_model_chain_paper_contract_audit.json"
)
SINGLELEG_STOP_JSON = (
    ROOT
    / "res/failed_runs/hub_singleleg_low_memory_candidate_stop_audit/"
    "hub_singleleg_low_memory_candidate_stop_audit.json"
)


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_ref(path: Path, needle: str) -> str:
    if not path.is_file():
        return f"{path}:missing"
    for i, line in enumerate(text(path).splitlines(), 1):
        if needle in line:
            return f"{path}:{i}"
    return f"{path}:not_found:{needle}"


def has(path: Path, *needles: str) -> bool:
    body = text(path)
    return all(needle in body for needle in needles)


def row(
    component: str,
    paper_contract: str,
    local_status: str,
    training_gate: str,
    evidence: list[str],
    impact: str,
    required_fix: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "component": component,
        "paper_contract": paper_contract,
        "local_status": local_status,
        "training_gate": training_gate,
        "evidence": evidence,
        "impact_on_current_bad_videos": impact,
        "required_fix_before_success_claim": required_fix,
        "notes": notes,
    }


def build_rows() -> list[dict[str, Any]]:
    model_chain = load_json(MODEL_CHAIN_AUDIT_JSON)
    vae_summary = load_json(PAPER_CONTRACT_VAE_JSON)
    stop_summary = load_json(SINGLELEG_STOP_JSON)

    return [
        row(
            "Stage 1 observation and normalized PD action",
            "o=[psi,e_anchor,V_imu,theta-theta0,theta_dot,a_last]; theta_sp=theta0+alpha*a.",
            "aligned_to_official_stage1_code",
            "pass_for_formula_audit",
            [
                line_ref(PAPER_METHOD, r"\mathbf{o}=["),
                line_ref(PAPER_METHOD, r"\boldsymbol{\theta}^{\text{sp}}"),
                line_ref(TRACKING_CFG, "command = ObsTerm"),
                line_ref(TRACKING_CFG, "actions = ObsTerm(func=mdp.last_action)"),
                line_ref(G1_CFG, "self.actions.joint_pos.scale = G1_ACTION_SCALE"),
            ],
            "This part is unlikely to be the source of the generic leaning-pose failure if official obs order and exported normalizer are preserved.",
            "Keep using official observation order/export metadata. Do not hand-build a 160-D MuJoCo obs without metadata validation.",
        ),
        row(
            "Stage 1 anchor-relative tracking transform",
            "Non-anchor desired body pose is yaw-aligned, height-preserving, and translated under current anchor.",
            "aligned_to_official_stage1_code",
            "pass_for_formula_audit",
            [
                line_ref(PAPER_SUPP, r"R_{\Delta}"),
                line_ref(COMMANDS, "delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]"),
                line_ref(COMMANDS, "delta_ori_w = yaw_quat"),
                line_ref(COMMANDS, "self.body_pos_relative_w = delta_pos_w"),
            ],
            "A broken anchor transform would cause drifting/jumping references; official code implements the paper transform.",
            "For MuJoCo replay/video, use the same anchor convention or label as diagnostic only.",
        ),
        row(
            "Stage 1 reward formulation",
            "Gaussian body pos/orientation/linear velocity/angular velocity rewards with std 0.3/0.4/1.0/3.14 plus action-rate, joint-limit, contact penalties.",
            "aligned_to_official_stage1_code",
            "pass_for_formula_audit",
            [
                line_ref(PAPER_SUPP, r"0.3^{2}"),
                line_ref(TRACKING_CFG, "motion_body_pos = RewTerm"),
                line_ref(TRACKING_CFG, "motion_body_ang_vel = RewTerm"),
                line_ref(TRACKING_CFG, "action_rate_l2 = RewTerm"),
                line_ref(REWARDS, "motion_relative_body_position_error_exp"),
            ],
            "Weak teacher reward is more likely undertraining/data/asset/runtime quality than a different reward formula.",
            "Use official rewards for next teacher; compare reward components and termination counts per checkpoint before downstream VAE.",
        ),
        row(
            "Stage 1 termination",
            "Terminate when anchor or EE z-position error >0.25 m, or anchor orientation error >0.8 rad.",
            "mostly_aligned_with_one_public_code_detail",
            "pass_with_caution",
            [
                line_ref(PAPER_SUPP, r"0.25~\text{m}"),
                line_ref(TRACKING_CFG, "anchor_pos = DoneTerm"),
                line_ref(TRACKING_CFG, "ee_body_pos = DoneTerm"),
                line_ref(TERMS, "def bad_anchor_ori"),
            ],
            "High termination counts explain bad teacher rollouts and VAE collapse. The public code checks projected gravity for anchor orientation, not a full log-map norm.",
            "Gate teacher checkpoints by done rate/episode length/motion-time continuity before collecting VAE data.",
            "Public code detail should be reported instead of silently treated as exact supplementary-text implementation.",
        ),
        row(
            "Stage 1 PD, action scale, and armature",
            "kp=I*w^2, kd=2*I*zeta*w, w=10 Hz, zeta=2, alpha=0.25*tau_max/kp; dual ankles/waist use doubled armature.",
            "aligned_to_official_stage1_code",
            "pass_for_formula_audit",
            [
                line_ref(PAPER_SUPP, r"k_{\mathrm{p},j}"),
                line_ref(PAPER_SUPP, r"\boldsymbol{\alpha}"),
                line_ref(G1_ROBOT, "NATURAL_FREQ = 10 * 2.0"),
                line_ref(G1_ROBOT, "DAMPING_RATIO = 2.0"),
                line_ref(G1_ROBOT, "G1_ACTION_SCALE = {}"),
            ],
            "If MuJoCo PD uses different kp/kd/action scale/joint order, learned actions can look like leaning or tiny steps even with a good policy.",
            "Before final MuJoCo videos, export or derive exact joint order, kp, kd, armature, default pose, and action scale from official metadata/checkpoint.",
        ),
        row(
            "Stage 1 domain randomization",
            "Friction/restitution, default joint offsets, torso COM, and velocity perturbations; text mentions larger ankle offset.",
            "partial_public_code_difference",
            "pass_with_caution",
            [
                line_ref(PAPER_SUPP, r"\mu_\mathrm{static}"),
                line_ref(PAPER_SUPP, r"larger range"),
                line_ref(TRACKING_CFG, "static_friction_range"),
                line_ref(TRACKING_CFG, "pos_distribution_params"),
                line_ref(TRACKING_CFG, "base_com = EventTerm"),
                line_ref(TRACKING_CFG, "push_robot = EventTerm"),
            ],
            "This is not the main reason the current videos are generic, but it matters for final paper-contract training.",
            "Decide explicitly whether to follow public code exactly or patch ankle default offsets to the supplementary text, then document the choice.",
        ),
        row(
            "Stage 1 adaptive sampling and reset",
            "Failure-rate bins, EMA alpha=0.001, uniform floor 0.1/S, non-causal kernel rho=0.8 over u={0,1,2}, reset to reference state with perturbations.",
            "partial_public_code_difference",
            "block_for_paper_exact_claim_only",
            [
                line_ref(PAPER_SUPP, r"\bar{f}_s \leftarrow 0.999"),
                line_ref(PAPER_SUPP, r"u \in \{0,1,2\}"),
                line_ref(COMMANDS, "adaptive_alpha: float = 0.001"),
                line_ref(COMMANDS, "adaptive_kernel_size: int = 1"),
                line_ref(COMMANDS, "joint_pos += sample_uniform"),
            ],
            "Poor sampling can slow learning of hard segments such as leg lift/single-leg balance.",
            "For hard single-leg training, either patch kernel size to 3 per paper or record that public code uses kernel size 1.",
        ),
        row(
            "PPO hyperparameters",
            "Actor/critic [512,256,128], ELU, 24 steps/env, 30000 iterations, lr 1e-3, clip 0.2, entropy 0.005, gamma 0.99, GAE 0.95, desired KL 0.01.",
            "aligned_to_official_stage1_code",
            "pass_for_formula_audit",
            [
                line_ref(PAPER_SUPP, "Actor MLP hidden dimensions"),
                line_ref(PPO_CFG, "num_steps_per_env = 24"),
                line_ref(PPO_CFG, "max_iterations = 30000"),
                line_ref(PPO_CFG, "actor_hidden_dims=[512, 256, 128]"),
                line_ref(PPO_CFG, "learning_rate=1.0e-3"),
            ],
            "Earlier short/low-memory candidates are not enough to learn hard motions; bad videos should not be read as final PPO failure.",
            "Train only after formula gates pass; for full runs use checkpoint/eval gates, not last checkpoint by default.",
        ),
        row(
            "Old resource-adjusted VAE",
            "Paper VAE encoder is E(psi,e_anchor); decoder is D(z,[g,V_imu,theta,theta_dot,a_last]); modified ELBO beta=0.01 with DAgger.",
            "mismatch",
            "block_training_chain",
            [
                line_ref(PAPER_METHOD, r"\mathbf{z} = \mathcal{E}"),
                line_ref(PAPER_METHOD, r"\hat{\mathbf{a}} = \mathcal{D}"),
                line_ref(RESOURCE_VAE, "self.encoder(torch.cat([obs, action], dim=-1))"),
                line_ref(RESOURCE_VAE, "self.decoder(torch.cat([obs, z], dim=-1))"),
            ],
            "This is a direct mechanism for generic/averaged action outputs and leaning posture.",
            "Do not use old resource-adjusted VAE for success videos. Use/retrain paper-contract VAE only on high-quality continuous teacher rollouts.",
        ),
        row(
            "New paper-contract VAE script",
            "Formula-level split E(reference intent) and D(latent, proprioception), latent dim 32, hidden [2048,1024,512], lr 5e-4, beta 0.01.",
            "formula_input_contract_repaired_but_data_gate_failed",
            "block_downstream_until_teacher_quality_passes",
            [
                str(PAPER_CONTRACT_VAE),
                str(PAPER_CONTRACT_VAE_JSON),
                f"source_teacher_done_rate_low_enough_for_downstream={vae_summary.get('checks', {}).get('source_teacher_done_rate_low_enough_for_downstream')}",
            ],
            "Even a corrected VAE will reproduce a weak/reset-heavy teacher and can still lean instead of lifting the leg.",
            "Collect continuous, low-done-rate teacher rollout data before retraining this VAE for videos.",
        ),
        row(
            "Old resource-adjusted diffusion",
            "Paper diffusion models tau=[s,z,...] with hybrid yaw-centric state, individual state/latent denoising steps, Transformer denoiser.",
            "mismatch",
            "block_training_chain",
            [
                line_ref(PAPER_METHOD, r"state–latent trajectories"),
                line_ref(RESOURCE_DIFF, 'data["policy_obs"]'),
                line_ref(RESOURCE_DIFF, "class StateLatentDenoiser"),
            ],
            "MLP over policy_obs+latent cannot be claimed as the paper state-latent diffusion and will not reliably produce leg-lift semantics.",
            "Retire this path from success videos; keep it only as a debug baseline.",
        ),
        row(
            "Public LAFAN1 paper-architecture diffusion",
            "Transformer encoder, 512 dim, 8 heads, 6 layers, horizon 16, history 4, 20 denoising steps, batch 512, epochs 1000, weight decay 0.001, EMA.",
            "architecture_partial_data_mismatch",
            "block_paper_level_claim_but_useful_for_code_probes",
            [
                line_ref(PAPER_SUPP, "Embedding dimension & 512"),
                line_ref(PAPER_ARCH, "class DiffusionTransformer"),
                line_ref(PAPER_ARCH, "self.state_step_embed"),
                line_ref(PAPER_ARCH, "self.latent_step_embed"),
                line_ref(PAPER_ARCH, "CSV_ROOT"),
            ],
            "The architecture probes are useful, but the dataset and VAE encoder still differ from the paper's teacher-rollout pipeline.",
            "Rewire diffusion training to consume paper-contract VAE latents from good teacher rollouts, then use receding-horizon closed-loop inference.",
        ),
        row(
            "State projection, OU noise, symmetry augmentation",
            "Use yaw-centric state, emphasis projection c=6, OU action noise sigma=0.1/theta=0.8, and sagittal symmetry augmentation.",
            "partially_implemented_as_helpers_and_public_data_audits",
            "block_final_chain_until_integrated",
            [
                line_ref(PAPER_SUPP, r"c = 6"),
                line_ref(PAPER_SUPP, r"\theta = 0.8"),
                line_ref(PAPER_SUPP, r"\sigma = 0.1"),
                line_ref(STATE_HELPER, "def emphasis_projection"),
                line_ref(SAMPLING_HELPER, "def mirror_state_29d"),
            ],
            "Without the VAE rollout error band and symmetry in the actual chain, diffusion may overfit/static-average.",
            "Integrate these helpers into the teacher-rollout to VAE/diffusion pipeline, not only standalone audits.",
        ),
        row(
            "Classifier guidance and task costs",
            "Guidance must apply -grad_tau G(tau) inside reverse denoising for joystick, waypoint, SDF obstacle, inpainting.",
            "offline_audits_exist_but_current_video_chain_mismatch",
            "block_success_video_claim",
            [
                line_ref(PAPER_METHOD, r"nabla_{\boldsymbol{\tau}} G"),
                line_ref(PAPER_SUPP, "G_{\\mathrm{js}}"),
                line_ref(PAPER_SUPP, "G_{\\mathrm{wp}}"),
                line_ref(PAPER_SUPP, "G_{\\mathrm{sdf}}"),
                line_ref(CLEAN_WALK_VIDEO, "used_latent = torch.from_numpy((denoised + guidance_scale"),
            ],
            "The visible guided videos were driven by latent interpolation/blending, not paper classifier guidance; they cannot prove task control.",
            "Implement closed-loop receding-horizon guidance with differentiable task costs and record gradient norms/cost decrease.",
        ),
        row(
            "MuJoCo action-control video adapter",
            "Final video should execute policy/VAE/diffusion action -> theta_sp -> PD torque -> MuJoCo step, with no direct pose playback or root assist for success claims.",
            "diagnostic_only_currently",
            "block_success_video_claim",
            [
                line_ref(MUJOCO_PD_VIDEO, "def apply_root_assist"),
                line_ref(MUJOCO_PD_VIDEO, "root_assist_enabled"),
                line_ref(CLEAN_WALK_VIDEO, "model_target_weight"),
                line_ref(CLEAN_WALK_VIDEO, "reference_anchor_weight"),
            ],
            "Root assist and reference-anchor blending can make videos readable while hiding policy failure; pure learned variants still look poor.",
            "For final success folder, disable root assist/blending, center camera only visually, and require continuous motion-time and low fall metrics.",
        ),
        row(
            "Current single-leg teacher quality",
            "A teacher used for VAE/diffusion must track the target motion continuously, with low done rate and meaningful leg-lift/body errors.",
            "failed_or_unproven",
            "block_downstream_until_retrained_or_good_checkpoint_found",
            [
                str(SINGLELEG_STOP_JSON),
                f"singleleg_stop_status={stop_summary.get('status')}",
                f"model_chain_status={model_chain.get('status')}",
            ],
            "This directly explains why teacher/VAE/diffusion did not learn single-leg posture and instead leaned.",
            "Find or train a teacher that passes checkpoint eval before collecting VAE/diffusion data.",
        ),
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    gates: dict[str, int] = {}
    for item in rows:
        counts[item["local_status"]] = counts.get(item["local_status"], 0) + 1
        gates[item["training_gate"]] = gates.get(item["training_gate"], 0) + 1
    blockers = [
        item
        for item in rows
        if item["training_gate"].startswith("block") or item["local_status"] in {"mismatch", "failed_or_unproven"}
    ]
    return {
        "status": "blocked_formula_parameter_trace_has_required_fixes_before_training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": False,
        "row_count": len(rows),
        "status_counts": counts,
        "training_gate_counts": gates,
        "blocking_components": [item["component"] for item in blockers],
        "key_conclusion": (
            "官方 Stage 1 motion tracking 代码整体符合论文 tracking 公式和主要参数；但是当前本地 "
            "VAE/diffusion/guidance/MuJoCo 成功视频链并没有完成 paper-contract 对齐。新的长训练或成功视频应当等 "
            "teacher 质量 gate 和模型链 gate 修复后再启动。"
        ),
        "source_hashes": {
            "paper_method": sha256(PAPER_METHOD),
            "paper_supplement": sha256(PAPER_SUPP),
            "tracking_cfg": sha256(TRACKING_CFG),
            "g1_robot": sha256(G1_ROBOT),
            "paper_contract_vae_script": sha256(PAPER_CONTRACT_VAE),
            "paper_arch_script": sha256(PAPER_ARCH),
        },
        "rows": rows,
    }


def write_tsv(rows: list[dict[str, Any]]) -> None:
    with TSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "component",
                "paper_contract",
                "local_status",
                "training_gate",
                "impact_on_current_bad_videos",
                "required_fix_before_success_claim",
                "evidence",
                "notes",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for item in rows:
            row_item = dict(item)
            row_item["evidence"] = " | ".join(item["evidence"])
            writer.writerow(row_item)


def write_md(summary: dict[str, Any]) -> None:
    lines = [
        "# BeyondMimic 公式与参数复现审计",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 行数：`{summary['row_count']}`",
        f"- goal_complete：`{summary['goal_complete']}`",
        "",
        "## 结论",
        "",
        summary["key_conclusion"],
        "",
        "当前不得声称完整复现 BeyondMimic，也不得把现有 teacher/VAE/diffusion/MuJoCo 视频写成 paper-level 成功结果。",
        "",
        "## 阻塞项",
        "",
    ]
    for name in summary["blocking_components"]:
        lines.append(f"- {name}")
    lines.extend(["", "## 对照矩阵", ""])
    for item in summary["rows"]:
        lines.extend(
            [
                f"### {item['component']}",
                "",
                f"- 论文要求：{item['paper_contract']}",
                f"- 本地状态：`{item['local_status']}`",
                f"- 训练 gate：`{item['training_gate']}`",
                f"- 对当前坏视频的影响：{item['impact_on_current_bad_videos']}",
                f"- 成功声明前必须修复：{item['required_fix_before_success_claim']}",
                "- 证据：",
            ]
        )
        for evidence in item["evidence"]:
            lines.append(f"  - `{evidence}`")
        if item.get("notes"):
            lines.append(f"- 备注：{item['notes']}")
        lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    summary = summarize(rows)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_tsv(rows)
    write_md(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT), "markdown": str(MD_OUT), "tsv": str(TSV_OUT)}))


if __name__ == "__main__":
    main()
