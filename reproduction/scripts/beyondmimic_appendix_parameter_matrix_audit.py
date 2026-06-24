#!/usr/bin/env python3
"""Build a machine-readable appendix parameter matrix before long training.

The current failure mode is visual and behavioral: policies tend to collapse
toward a forward-leaning standing posture instead of reproducing the reference
motion.  This audit does not try to fix that by making nicer videos.  It checks
whether the paper and supplementary contracts for Stage-1 RL, VAE, diffusion,
guidance, and MuJoCo deployment have enough code-level evidence to justify the
next long run.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/appendix_parameter_matrix"
JSON_OUT = OUT / "beyondmimic_appendix_parameter_matrix_audit.json"
TSV_OUT = OUT / "beyondmimic_appendix_parameter_matrix_audit.tsv"
MD_OUT = OUT / "beyondmimic_appendix_parameter_matrix_audit.md"

PAPER_METHOD = ROOT / "reproduction/paper/source/tex/method.tex"
PAPER_SUPP = ROOT / "reproduction/paper/source/root.tex"

OFFICIAL = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
)
TRACKING_CFG = OFFICIAL / "tasks/tracking/tracking_env_cfg.py"
G1_ROBOT = OFFICIAL / "robots/g1.py"
G1_FLAT = OFFICIAL / "tasks/tracking/config/g1/flat_env_cfg.py"
PPO_CFG = OFFICIAL / "tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py"
COMMANDS = OFFICIAL / "tasks/tracking/mdp/commands.py"
OBSERVATIONS = OFFICIAL / "tasks/tracking/mdp/observations.py"
REWARDS = OFFICIAL / "tasks/tracking/mdp/rewards.py"
TERMINATIONS = OFFICIAL / "tasks/tracking/mdp/terminations.py"

PAPER_CONTRACT_VAE_SCRIPT = ROOT / "reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py"
PAPER_CONTRACT_DIFFUSION_SCRIPT = (
    ROOT / "reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py"
)
PAPER_CONTRACT_STATE_LATENT_SCRIPT = (
    ROOT / "reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py"
)
GUIDANCE_COSTS = ROOT / "reproduction/src/beyondmimic_reimpl/guidance/costs.py"

MODEL_CHAIN_JSON = ROOT / "res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json"
PRETRAINING_JSON = ROOT / "res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json"
MUJOCO_ACTION_JSON = ROOT / "res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json"
MUJOCO_OBS_JSON = ROOT / "res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json"
MUJOCO_CONTROL_JSON = ROOT / "res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json"
STATE_LATENT_SOURCE_JSON = (
    ROOT / "res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json"
)
VAE_JSON = ROOT / "res/level_c/paper_contract_teacher_rollout_vae_training/level_c_paper_contract_teacher_rollout_vae_training.json"
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/paper_contract_transformer_state_latent_diffusion_training/"
    "paper_contract_transformer_state_latent_diffusion_training.json"
)
GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/"
    "level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json"
)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def has(path: Path, *needles: str) -> bool:
    body = read_text(path)
    return all(needle in body for needle in needles)


def has_any(path: Path, *needles: str) -> bool:
    body = read_text(path)
    return any(needle in body for needle in needles)


def line_ref(path: Path, needle: str) -> str:
    if not path.is_file():
        return f"{path}:missing"
    for idx, line in enumerate(read_text(path).splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return f"{path}:not_found:{needle}"


def get_path(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def row(
    area: str,
    item: str,
    expected: str,
    observed: str,
    status: str,
    evidence: list[str],
    training_impact: str,
    required_fix: str,
    claim_boundary: str,
) -> dict[str, Any]:
    return {
        "area": area,
        "item": item,
        "expected_from_paper_or_appendix": expected,
        "observed_in_current_project": observed,
        "status": status,
        "passed": status == "pass",
        "evidence": evidence,
        "training_impact": training_impact,
        "required_fix_before_long_training_or_success_video": required_fix,
        "claim_boundary": claim_boundary,
    }


def build_rows() -> list[dict[str, Any]]:
    mujoco_action = read_json(MUJOCO_ACTION_JSON)
    mujoco_obs = read_json(MUJOCO_OBS_JSON)
    mujoco_control = read_json(MUJOCO_CONTROL_JSON)
    state_latent = read_json(STATE_LATENT_SOURCE_JSON)
    vae = read_json(VAE_JSON)
    diffusion = read_json(DIFFUSION_JSON)
    guidance = read_json(GUIDANCE_JSON)
    pretraining = read_json(PRETRAINING_JSON)

    action_scale_formula_ok = has(G1_ROBOT, "G1_ACTION_SCALE", "0.25 * e[n] / s[n]")
    native_action_formula_ready = bool(get_path(mujoco_action, "interpretation", "formula_adapter_ready", default=False))
    native_action_range_ready = bool(get_path(mujoco_action, "checks", "unit_targets_inside_mujoco_ctrlrange", default=False))
    native_obs_ready = bool(get_path(mujoco_obs, "interpretation", "native_obs_adapter_ready", default=False))
    no_root_assist = bool(get_path(mujoco_control, "checks", "mujoco_video_has_no_root_assist", default=False))
    material_ok = bool(get_path(mujoco_control, "checks", "mujoco_floor_material_matches_official", default=False))

    vae_interface_ok = (
        bool(get_path(vae, "checks", "encoder_uses_reference_intent_only", default=False))
        and bool(get_path(vae, "checks", "decoder_uses_proprioception_plus_latent", default=False))
        and bool(get_path(vae, "checks", "action_dim_29", default=False))
    )
    vae_teacher_ok = bool(get_path(vae, "checks", "source_teacher_done_rate_low_enough_for_downstream", default=False))
    diffusion_arch_ok = (
        bool(get_path(diffusion, "checks", "paper_contract_architecture_checks_pass", default=False))
        or has(
            PAPER_CONTRACT_DIFFUSION_SCRIPT,
            "nn.TransformerEncoder",
            "EMBED_DIM",
            "ATTENTION_HEADS",
            "TRANSFORMER_LAYERS",
            "DENOISING_STEPS",
        )
    )
    diffusion_full_training_ok = (
        diffusion_arch_ok
        and not bool(diffusion.get("dry_run", True))
        and bool(get_path(diffusion, "checks", "test_denoising_improves_over_noisy", default=False))
    )
    state_has_hybrid_builder = bool(get_path(state_latent, "checks", "builder_has_paper_hybrid_path", default=False))
    state_dataset_fresh = bool(get_path(state_latent, "checks", "existing_dataset_has_corrected_hybrid_state", default=False))
    teacher_raw_state = bool(get_path(state_latent, "checks", "teacher_rollout_shards_have_required_world_state", default=False))
    window_filter = bool(get_path(state_latent, "checks", "windows_filter_done_and_5s_rejection", default=False))
    guidance_offline_ok = bool(get_path(guidance, "checks", "all_best_guidance_gradients_nonzero", default=False))

    text_state = read_text(PAPER_CONTRACT_STATE_LATENT_SCRIPT).lower()
    ou_noise_code = all(term in text_state for term in ["ou", "sigma"])
    five_sec_rejection = "5" in text_state and ("reject" in text_state or "failure" in text_state)
    symmetry_aug = "symmetry" in text_state or "mirror" in text_state

    teacher_quality_failed = bool(
        get_path(pretraining, "checks", "does_not_allow_downstream_training_from_current_teacher", default=True)
    )

    rows = [
        row(
            "Stage-1 RL",
            "Policy observation vector",
            "Policy input o=[psi,e_anchor,V_imu,theta-theta0,theta_dot,a_last], no temporal stacking.",
            "Official tracking config preserves command, anchor pos/orientation, base velocities, joint state, and last action.",
            "pass"
            if has(TRACKING_CFG, "command = ObsTerm", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions")
            else "mismatch",
            [
                line_ref(PAPER_METHOD, r"\mathbf{o}=["),
                line_ref(TRACKING_CFG, "class PolicyCfg"),
                line_ref(OBSERVATIONS, "def motion_anchor_pos_b"),
            ],
            "A wrong observation frame/order can make a good checkpoint output posture-like garbage.",
            "Before MuJoCo PPO/video claims, compare each MuJoCo observation term numerically with IsaacLab observation_manager.",
            "Dimension 160 alone is not enough.",
        ),
        row(
            "Stage-1 RL",
            "Normalized PD action and no kinematic clipping",
            "theta_sp=theta0+alpha*a; setpoints intentionally not clipped by joint kinematic limits.",
            (
                f"official_action_scale_formula={action_scale_formula_ok}; "
                f"native_formula_ready={native_action_formula_ready}; "
                f"native_ctrlrange_allows_unit_targets={native_action_range_ready}."
            ),
            "blocked" if not (action_scale_formula_ok and native_action_formula_ready and native_action_range_ready) else "pass",
            [
                line_ref(PAPER_METHOD, "intentionally not clipped"),
                line_ref(G1_ROBOT, "G1_ACTION_SCALE = {}"),
                str(MUJOCO_ACTION_JSON),
            ],
            "Ctrlrange clipping can shrink or distort ankle/leg actions, producing tiny steps or leaning poses.",
            "Resolve MuJoCo ctrlrange/action-scale compatibility before trusting PPO/VAE/diffusion action videos.",
            "Approximate or clipped action videos remain diagnostic.",
        ),
        row(
            "Stage-1 RL",
            "Reward table S1",
            "Body pos/orientation/linear velocity/angular velocity stds 0.3/0.4/1.0/3.14; weights 1; optional anchor weight 0.5; action/joint/contact penalties -0.1/-10/-0.1.",
            "Official tracking config contains these reward terms, stds, and weights.",
            "pass"
            if has(TRACKING_CFG, "motion_global_anchor_pos", "weight=0.5", "motion_body_pos", "std\": 0.3", "motion_body_ori", "std\": 0.4", "motion_body_lin_vel", "std\": 1.0", "motion_body_ang_vel", "std\": 3.14", "action_rate_l2", "weight=-1e-1", "joint_limit", "weight=-10.0", "undesired_contacts", "weight=-0.1")
            else "mismatch",
            [
                line_ref(PAPER_SUPP, r"\label{tab:rewardterms}"),
                line_ref(TRACKING_CFG, "class RewardsCfg"),
                line_ref(REWARDS, "motion_relative_body_position_error_exp"),
            ],
            "Reward formula is not the first suspected bug; teacher quality failure likely comes from data/runtime/checkpoint quality or adapter mismatch.",
            "Keep reward components unchanged for paper-contract teacher training unless a row records an ablation.",
            "Reward alignment does not imply current teacher success.",
        ),
        row(
            "Stage-1 RL",
            "Termination thresholds",
            "Anchor/EE z error threshold 0.25 m and anchor orientation threshold 0.8 rad.",
            "Official public code uses z-only anchor/EE checks and projected-gravity anchor orientation threshold.",
            "partial",
            [
                line_ref(PAPER_SUPP, "0.25~\\text{m}"),
                line_ref(TRACKING_CFG, "anchor_pos = DoneTerm"),
                line_ref(TRACKING_CFG, "ee_body_pos = DoneTerm"),
                line_ref(TERMINATIONS, "motion_projected_gravity_b"),
            ],
            "High non-timeout done rates directly poison teacher rollout, VAE, and diffusion training.",
            "Gate every teacher checkpoint by done rate, episode length, body error, joint error, and continuous motion_time_steps.",
            "Public-code detail is acceptable if documented, but current reset-heavy rollouts cannot seed downstream training.",
        ),
        row(
            "Stage-1 RL",
            "PD gains, armature, and action scale",
            "omega=10 Hz, zeta=2, kp=I*omega^2, kd=2*I*zeta*omega, alpha=0.25*tau_max/kp, dual ankles/waist doubled.",
            "Official G1 code implements armature constants, stiffness, damping, and action scale; MuJoCo final video gate is still not passed.",
            "partial" if not (no_root_assist and material_ok) else "pass",
            [
                line_ref(PAPER_SUPP, r"k_{\mathrm{p},j}"),
                line_ref(PAPER_SUPP, "10\\,\\text{Hz}"),
                line_ref(G1_ROBOT, "NATURAL_FREQ"),
                line_ref(G1_ROBOT, "stiffness=2.0 * STIFFNESS_5020"),
                str(MUJOCO_CONTROL_JSON),
            ],
            "A mismatch in armature/kp/kd/action scale can convert real actions into unstable or underpowered motions.",
            "Export exact joint order, default pose, armature, kp, kd, action scale, ctrlrange and material into the MuJoCo runner manifest.",
            "Official source pass is not the same as native MuJoCo deployment pass.",
        ),
        row(
            "Stage-1 RL",
            "Domain randomization table S2",
            "Static friction U[0.3,1.6], dynamic U[0.3,1.2], restitution U[0,0.5], joint default offsets, torso COM, random root velocity pushes.",
            "Official code matches friction/restitution/COM/push and general joint offset; supplementary text also mentions larger ankle offsets not visible in the public config.",
            "partial",
            [
                line_ref(PAPER_SUPP, r"\label{tab:domain_rand}"),
                line_ref(TRACKING_CFG, "static_friction_range"),
                line_ref(TRACKING_CFG, "pos_distribution_params"),
                line_ref(TRACKING_CFG, "base_com = EventTerm"),
                line_ref(TRACKING_CFG, "push_robot = EventTerm"),
            ],
            "This affects robustness, but it does not excuse treating failed teacher/video outputs as success.",
            "Record whether the next teacher follows public code exactly or patches ankle offsets to the supplementary text.",
            "A public-code run can be described as public-code-faithful, not necessarily exact supplementary-private-config reproduction.",
        ),
        row(
            "Stage-1 RL",
            "Adaptive sampling",
            "EMA alpha 0.001, uniform floor 0.1/S, rho=0.8, non-causal look-back u={0,1,2}.",
            "Official public MotionCommandCfg has alpha/rho/floor, but default adaptive_kernel_size is 1 unless explicitly overridden.",
            "blocked" if has(COMMANDS, "adaptive_kernel_size: int = 1") else "pass",
            [
                line_ref(PAPER_SUPP, "look-back window"),
                line_ref(COMMANDS, "adaptive_alpha: float = 0.001"),
                line_ref(COMMANDS, "adaptive_kernel_size: int = 1"),
            ],
            "Hard segments such as single-leg stance can remain under-sampled if pre-failure bins are not emphasized.",
            "For paper-exact hard-motion training, set/verify kernel size 3 or explicitly label the run public-default.",
            "Do not silently call kernel_size=1 an exact appendix adaptive-sampling reproduction.",
        ),
        row(
            "Stage-1 RL",
            "PPO hyperparameters table S4",
            "Actor/critic [512,256,128], ELU, 24 steps/env, 30000 iterations, lr 1e-3, clip 0.2, entropy 0.005, gamma 0.99, GAE 0.95, KL 0.01, epochs 5, minibatches 4.",
            "Official RSL-RL config matches table S4.",
            "pass"
            if has(PPO_CFG, "num_steps_per_env = 24", "max_iterations = 30000", "actor_hidden_dims=[512, 256, 128]", "critic_hidden_dims=[512, 256, 128]", "activation=\"elu\"", "learning_rate=1.0e-3", "clip_param=0.2", "entropy_coef=0.005", "gamma=0.99", "lam=0.95", "desired_kl=0.01")
            else "mismatch",
            [
                line_ref(PAPER_SUPP, r"\label{tab:ppo_hyperparameters}"),
                line_ref(PPO_CFG, "num_steps_per_env = 24"),
                line_ref(PPO_CFG, "max_iterations = 30000"),
            ],
            "Bad videos are not explained by PPO table mismatch if this official config was used; evaluate data/teacher quality next.",
            "If batch/env count is increased for GPU utilization, record the exact deviation and preserve all other paper constants.",
            "Hyperparameter alignment does not mean a checkpoint has converged.",
        ),
        row(
            "VAE",
            "Conditional VAE input and architecture table S5",
            "Encoder E(psi,e_anchor), decoder D(z,[g,V_imu,theta,theta_dot,a_last]), latent dim 32, hidden [2048,1024,512], lr 5e-4, grad accum 15, KL 0.01.",
            f"paper_contract_vae_interface={vae_interface_ok}; source_teacher_quality_ok={vae_teacher_ok}.",
            "blocked" if not (vae_interface_ok and vae_teacher_ok) else "pass",
            [
                line_ref(PAPER_METHOD, "The encoder receives only the reference-motion components"),
                line_ref(PAPER_SUPP, r"\label{tab:vae_hyperparameters}"),
                line_ref(PAPER_CONTRACT_VAE_SCRIPT, "encoder_x = np.concatenate"),
                str(VAE_JSON),
            ],
            "The old obs+action VAE can learn an averaged standing action. The corrected VAE still fails if trained on weak teacher data.",
            "Only retrain paper-contract VAE after a continuous, low-fall teacher rollout dataset passes quality gates.",
            "VAE MSE alone is not closed-loop success.",
        ),
        row(
            "Diffusion",
            "State-latent representation and data collection",
            "Hybrid character-yaw-centric state + latent, individual state/latent denoising steps, VAE rollouts with OU action noise, 5s rejection, sagittal symmetry augmentation.",
            (
                f"hybrid_builder={state_has_hybrid_builder}; existing_dataset_fresh={state_dataset_fresh}; "
                f"teacher_raw_state={teacher_raw_state}; window_filter={window_filter}; "
                f"ou_noise_code={ou_noise_code}; five_sec_rejection={five_sec_rejection}; symmetry_aug={symmetry_aug}."
            ),
            "blocked"
            if not (state_has_hybrid_builder and state_dataset_fresh and teacher_raw_state and window_filter and ou_noise_code and five_sec_rejection and symmetry_aug)
            else "pass",
            [
                line_ref(PAPER_METHOD, "hybrid character"),
                line_ref(PAPER_SUPP, "Ornstein-Uhlenbeck"),
                line_ref(PAPER_SUPP, "sagittal-symmetric"),
                str(STATE_LATENT_SOURCE_JSON),
            ],
            "Training diffusion on old policy_obs/latent windows or reset-heavy data can produce plausible MSE but poor actions.",
            "Regenerate teacher rollout shards with raw state, then rebuild reset-safe hybrid state-latent windows with OU and symmetry manifests.",
            "Current denoising MSE is diagnostic, not a paper-level control result.",
        ),
        row(
            "Diffusion",
            "Transformer architecture table S6",
            "Horizon 16, history 4, embed 512, heads 8, layers 6, denoising steps 20, batch 512, epochs 1000, lr 1e-4, weight decay 0.001, cosine, warmup 10000, EMA 0.75/0.9999.",
            f"transformer_arch_code_present={diffusion_arch_ok}; full_training_over_accepted_dataset={diffusion_full_training_ok}.",
            "blocked" if not diffusion_full_training_ok else "pass",
            [
                line_ref(PAPER_SUPP, r"\label{tab:diffusion_hyperparameters}"),
                line_ref(PAPER_CONTRACT_DIFFUSION_SCRIPT, "nn.TransformerEncoder"),
                str(DIFFUSION_JSON),
            ],
            "A dry-run or MLP denoiser cannot explain or reproduce the paper's guided closed-loop behavior.",
            "After teacher/VAE/data gates pass, train the paper-contract Transformer on held-out splits and record full metrics.",
            "Architecture code presence is not trained-model evidence.",
        ),
        row(
            "Guidance",
            "Classifier guidance and task costs",
            "Use -grad G(tau) during denoising in a receding-horizon closed-loop controller; SDF relaxed barrier for obstacle tasks.",
            f"offline_guidance_gradients_ok={guidance_offline_ok}; native_obs_ready={native_obs_ready}; sdf_barrier_formula={has(GUIDANCE_COSTS, 'BeyondMimic relaxed SDF barrier', '-np.log(distance')}.",
            "blocked" if not (guidance_offline_ok and native_obs_ready and diffusion_full_training_ok) else "pass",
            [
                line_ref(PAPER_METHOD, "conditional gradient"),
                line_ref(GUIDANCE_COSTS, "BeyondMimic relaxed SDF barrier"),
                str(GUIDANCE_JSON),
                str(MUJOCO_OBS_JSON),
            ],
            "Offline guidance can improve a token cost while the robot still leans or falls in physics.",
            "Implement and validate receding-horizon MuJoCo/Isaac closed-loop guidance only after diffusion and native adapter gates pass.",
            "Offline guidance is not Fig.5/Fig.6 reproduction.",
        ),
        row(
            "Deployment",
            "Native MuJoCo observation/action/material gate",
            "Closed-loop video must use controller actions through PD physics, no root assist/blending, valid material/contact and verified observation semantics.",
            f"native_obs_ready={native_obs_ready}; native_action_range_ready={native_action_range_ready}; no_root_assist={no_root_assist}; material_ok={material_ok}.",
            "blocked" if not (native_obs_ready and native_action_range_ready and no_root_assist and material_ok) else "pass",
            [
                str(MUJOCO_OBS_JSON),
                str(MUJOCO_ACTION_JSON),
                str(MUJOCO_CONTROL_JSON),
            ],
            "This is the most direct explanation for why existing MuJoCo action-control videos are not trustworthy.",
            "Do not create the final success folder until this native deployment gate passes without root assist.",
            "Reference replay can be useful visualization; it is not policy/diffusion control.",
        ),
        row(
            "Training permission",
            "No long downstream training from weak teacher",
            "Teacher quality and native adapter gates must pass before VAE/diffusion/guidance long training or success videos.",
            f"teacher_quality_failed={teacher_quality_failed}.",
            "blocked" if teacher_quality_failed else "pass",
            [str(PRETRAINING_JSON), str(MODEL_CHAIN_JSON)],
            "Starting downstream training now would spend GPU time learning failed teacher/controller behavior.",
            "Allowed next work: code/parameter fixes, numeric adapter validation, and corrective Stage-1 teacher evaluation/training under recorded contracts.",
            "Current project remains incomplete.",
        ),
    ]
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "area",
        "item",
        "expected_from_paper_or_appendix",
        "observed_in_current_project",
        "status",
        "passed",
        "training_impact",
        "required_fix_before_long_training_or_success_video",
        "claim_boundary",
        "evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in rows:
            record = dict(item)
            record["evidence"] = " | ".join(str(x) for x in item["evidence"])
            writer.writerow(record)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# BeyondMimic Appendix Parameter Matrix Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Claim level: `{summary['claim_level']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        "",
        "## 结论",
        "",
        "当前不允许从弱 teacher 继续做下游 VAE/diffusion/guidance 长训练，也不允许把现有 MuJoCo 视频当成功视频。",
        "下一步应先修复/验证 native observation/action adapter、teacher quality gate、hybrid state-latent 数据生成，以及 appendix 中公开代码差异项。",
        "",
        "## Permission",
        "",
    ]
    for key, value in summary["permission"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Blocking Rows", ""])
    for item in rows:
        if item["passed"]:
            continue
        lines.extend(
            [
                f"### {item['area']} / {item['item']}",
                f"- Status: `{item['status']}`",
                f"- Expected: {item['expected_from_paper_or_appendix']}",
                f"- Observed: {item['observed_in_current_project']}",
                f"- Impact: {item['training_impact']}",
                f"- Required fix: {item['required_fix_before_long_training_or_success_video']}",
                f"- Claim boundary: {item['claim_boundary']}",
                f"- Evidence: `{'; '.join(item['evidence'])}`",
                "",
            ]
        )
    lines.extend(["## Full Matrix", ""])
    for item in rows:
        lines.extend(
            [
                f"### {item['area']} / {item['item']}",
                f"- Status: `{item['status']}`",
                f"- Expected: {item['expected_from_paper_or_appendix']}",
                f"- Observed: {item['observed_in_current_project']}",
                f"- Required fix: {item['required_fix_before_long_training_or_success_video']}",
                f"- Evidence: `{'; '.join(item['evidence'])}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    status_counts: dict[str, int] = {}
    for item in rows:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    blocking = [item for item in rows if not item["passed"]]
    checks = {
        "paper_method_readable": PAPER_METHOD.is_file(),
        "paper_supplement_readable": PAPER_SUPP.is_file(),
        "official_stage1_core_readable": all(
            path.is_file() for path in [TRACKING_CFG, G1_ROBOT, G1_FLAT, PPO_CFG, COMMANDS, OBSERVATIONS, REWARDS, TERMINATIONS]
        ),
        "blocks_downstream_training": bool(blocking),
        "blocks_success_video_claims": bool(blocking),
        "does_not_start_training": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "experiment_type": "beyondmimic_appendix_parameter_matrix_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_appendix_parameter_matrix_has_required_fixes"
        if blocking
        else "ok_appendix_parameter_matrix_all_rows_pass",
        "claim_level": "audit_only; no training; no video success claim",
        "row_count": len(rows),
        "passed_count": len(rows) - len(blocking),
        "blocked_count": len(blocking),
        "status_counts": status_counts,
        "blocking_items": [f"{item['area']} / {item['item']}" for item in blocking],
        "permission": {
            "start_new_long_stage1_training": False,
            "start_downstream_vae_training": False,
            "start_diffusion_training": False,
            "start_guided_closed_loop_video_generation": False,
            "create_final_success_video_folder": False,
            "allowed_next_work": [
                "numeric IsaacLab-vs-MuJoCo observation parity probe",
                "use the MuJoCo no-action-clipping ctrlrange patch in future no-root-assist adapter probes",
                "teacher checkpoint quality selection with done/error/continuity gates",
                "paper-contract hybrid state-latent dataset regeneration after teacher gate",
            ],
        },
        "checks": checks,
        "files": {
            "paper_method": str(PAPER_METHOD),
            "paper_supplement": str(PAPER_SUPP),
            "official_tracking_cfg": str(TRACKING_CFG),
            "official_g1_robot": str(G1_ROBOT),
            "official_ppo_cfg": str(PPO_CFG),
            "mujoco_action_audit": str(MUJOCO_ACTION_JSON),
            "mujoco_observation_audit": str(MUJOCO_OBS_JSON),
            "mujoco_control_audit": str(MUJOCO_CONTROL_JSON),
        },
        "interpretation": {
            "why_current_videos_can_fail": (
                "The appendix matrix still blocks native MuJoCo observation/action/material deployment and downstream "
                "training from weak teacher data. These are sufficient explanations for forward-leaning or generic "
                "standing videos even when reference replay is continuous."
            ),
            "goal_complete": False,
        },
        "rows": rows,
        "outputs": {"json": str(JSON_OUT), "tsv": str(TSV_OUT), "markdown": str(MD_OUT)},
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tsv(TSV_OUT, rows)
    write_markdown(MD_OUT, summary, rows)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
