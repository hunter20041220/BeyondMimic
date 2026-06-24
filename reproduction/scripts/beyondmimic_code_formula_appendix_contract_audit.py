#!/usr/bin/env python3
"""Audit code contracts against BeyondMimic formulas and appendix parameters.

This is a stricter pre-training gate than the general project audit.  It checks
whether the current teacher/RL, VAE, diffusion, guidance, PD/action-scale,
armature/material, and MuJoCo adapter code can honestly support another long
training run.  The intended answer is conservative: fix formula/code contracts
first, then train.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/code_formula_appendix_contract"
JSON_OUT = OUT / "beyondmimic_code_formula_appendix_contract_audit.json"
TSV_OUT = OUT / "beyondmimic_code_formula_appendix_contract_audit.tsv"
MD_OUT = OUT / "beyondmimic_code_formula_appendix_contract_audit.md"

FILES = {
    "paper_method": ROOT / "reproduction/paper/source/tex/method.tex",
    "paper_results": ROOT / "reproduction/paper/source/tex/results.tex",
    "paper_root": ROOT / "reproduction/paper/source/root.tex",
    "official_tracking_cfg": ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/"
    "tracking_env_cfg.py",
    "official_rewards": ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "rewards.py",
    "official_observations": ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "observations.py",
    "official_commands": ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "commands.py",
    "official_g1": ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py",
    "official_ppo": ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/"
    "config/g1/agents/rsl_rl_ppo_cfg.py",
    "motion_tracking_controller_g1": ROOT / "download/official/motion_tracking_controller/config/g1/controllers.yaml",
    "paper_contract_vae_script": ROOT / "reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py",
    "paper_contract_diffusion_script": ROOT
    / "reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py",
    "state_latent_dataset_script": ROOT
    / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py",
    "paper_contract_state_latent_dataset_script": ROOT
    / "reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py",
    "guidance_eval_script": ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py",
    "paper_contract_guidance_eval_script": ROOT
    / "reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py",
    "guidance_costs": ROOT / "reproduction/src/beyondmimic_reimpl/guidance/costs.py",
    "mujoco_action_adapter": ROOT / "reproduction/scripts/mujoco_native_action_adapter_contract.py",
    "mujoco_observation_adapter": ROOT / "reproduction/scripts/mujoco_native_observation_adapter_contract.py",
    "mujoco_control_audit": ROOT / "res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json",
    "mujoco_action_audit": ROOT
    / "res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json",
    "mujoco_observation_audit": ROOT
    / "res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json",
    "pretraining_hard_gate": ROOT
    / "res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json",
    "state_latent_source_contract": ROOT
    / "res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json",
}


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def line_ref(path: Path, needle: str) -> str:
    text = read_text(path)
    if not text:
        return f"{path}:missing"
    for idx, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return f"{path}:not_found:{needle}"


def contains(path_key: str, *needles: str) -> bool:
    text = read_text(FILES[path_key])
    return all(needle in text for needle in needles)


def contains_any(path_key: str, *needles: str) -> bool:
    text = read_text(FILES[path_key])
    return any(needle in text for needle in needles)


def get_path(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def row(
    area: str,
    contract: str,
    expected: str,
    observed: str,
    status: str,
    evidence: list[str],
    required_fix: str,
    claim_boundary: str,
) -> dict[str, Any]:
    return {
        "area": area,
        "contract": contract,
        "expected_from_paper_or_official": expected,
        "observed_in_current_project": observed,
        "status": status,
        "passed": status == "pass",
        "evidence": evidence,
        "required_fix_before_long_training": required_fix,
        "claim_boundary": claim_boundary,
    }


def build_rows() -> list[dict[str, Any]]:
    mujoco_control = read_json(FILES["mujoco_control_audit"])
    mujoco_action = read_json(FILES["mujoco_action_audit"])
    mujoco_obs = read_json(FILES["mujoco_observation_audit"])
    pretraining = read_json(FILES["pretraining_hard_gate"])
    state_latent_contract = read_json(FILES["state_latent_source_contract"])
    state_latent_checks = state_latent_contract.get("checks", {})

    stage1_obs_action_ok = (
        contains("official_tracking_cfg", "generated_commands", "last_action", "base_lin_vel", "base_ang_vel")
        and contains("official_g1", "G1_ACTION_SCALE", "0.25 * e[n] / s[n]")
    )
    reward_contract_ok = (
        contains("official_tracking_cfg", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")
        and contains("official_rewards", "torch.exp(-error.mean(-1) / std**2)")
    )
    ppo_contract_ok = contains(
        "official_ppo",
        "num_steps_per_env = 24",
        "max_iterations = 30000",
        "actor_hidden_dims=[512, 256, 128]",
        "empirical_normalization = True",
    )
    pd_contract_ok = contains("official_g1", "NATURAL_FREQ = 10 * 2.0 * 3.1415926535", "DAMPING_RATIO = 2.0")
    adaptive_kernel_code_default_is_one = contains("official_commands", "adaptive_kernel_size: int = 1")
    paper_mentions_prefailure_kernel = contains_any("paper_root", "u \\in \\{0,1,2\\}", "look-back window")

    vae_script = read_text(FILES["paper_contract_vae_script"])
    vae_input_contract_ok = all(
        needle in vae_script
        for needle in [
            "encoder_x = np.concatenate([command, anchor_pos, anchor_ori]",
            "for term in [\"base_lin_vel\", \"base_ang_vel\", \"joint_pos\", \"joint_vel\", \"actions\"]",
            "proprio_x = np.concatenate(proprio_terms",
            "LATENT_DIM = int(os.environ.get(\"BM_PAPER_CONTRACT_VAE_LATENT_DIM\", \"32\"))",
            "KL_COEF = float(os.environ.get(\"BM_PAPER_CONTRACT_VAE_KL_COEF\", \"0.01\"))",
            "LR = float(os.environ.get(\"BM_PAPER_CONTRACT_VAE_LR\", \"5e-4\"))",
        ]
    )
    vae_arch_exact = (
        "[2048, 1024, 512]" in vae_script
        or ("2048" in vae_script and "1024" in vae_script and "512" in vae_script and "hidden_dims" in vae_script)
    )
    vae_grad_accum_15 = "15" in vae_script and "accum" in vae_script.lower()

    diffusion_script = read_text(FILES["paper_contract_diffusion_script"])
    diffusion_arch_ok = all(
        needle in diffusion_script
        for needle in [
            "EMBED_DIM = int(os.environ[\"BM_EMBED_DIM\"])",
            "ATTENTION_HEADS = int(os.environ[\"BM_ATTENTION_HEADS\"])",
            "TRANSFORMER_LAYERS = int(os.environ[\"BM_TRANSFORMER_LAYERS\"])",
            "DENOISING_STEPS = int(os.environ[\"BM_DENOISING_STEPS\"])",
            "nn.TransformerEncoder",
        ]
    )
    state_dataset_text = read_text(FILES["state_latent_dataset_script"]) + read_text(
        FILES["paper_contract_state_latent_dataset_script"]
    )
    state_builder_has_hybrid_path = bool(state_latent_checks.get("builder_has_paper_hybrid_path")) and bool(
        state_latent_checks.get("paper_contract_wrapper_requires_hybrid_state")
    )
    existing_state_latent_dataset_corrected = bool(state_latent_checks.get("existing_dataset_has_corrected_hybrid_state"))
    teacher_shards_have_raw_state = bool(state_latent_checks.get("teacher_rollout_shards_have_required_world_state"))
    window_filter_ready = bool(state_latent_checks.get("windows_filter_done_and_5s_rejection"))
    state_has_ou_rollout_rejection = all(term in state_dataset_text.lower() for term in ["ou", "sigma", "reject"])
    state_has_symmetry_aug = "symmetry" in state_dataset_text.lower() or "mirror" in state_dataset_text.lower()
    guidance_text = read_text(FILES["guidance_eval_script"]) + read_text(FILES["paper_contract_guidance_eval_script"])
    guidance_is_offline = "offline" in guidance_text.lower() or "proxy" in guidance_text.lower()
    guidance_receding_mujoco = all(term in guidance_text.lower() for term in ["mujoco", "env.step", "receding"])
    sdf_barrier_ok = contains(
        "guidance_costs",
        "BeyondMimic relaxed SDF barrier",
        "-np.log(distance",
        "distance[smooth_region] - 2.0 * delta",
    )

    native_action_ready = bool(get_path(mujoco_action, "interpretation", "formula_adapter_ready", default=False))
    native_action_range_ready = bool(
        get_path(mujoco_action, "checks", "unit_targets_inside_mujoco_ctrlrange", default=False)
    )
    native_obs_ready = bool(get_path(mujoco_obs, "interpretation", "native_obs_adapter_ready", default=False))
    no_root_assist = bool(get_path(mujoco_control, "checks", "mujoco_video_has_no_root_assist", default=False))
    floor_material_ok = bool(get_path(mujoco_control, "checks", "mujoco_floor_material_matches_official", default=False))
    teacher_quality_blocks = bool(
        get_path(pretraining, "checks", "does_not_allow_downstream_training_from_current_teacher", default=True)
    )

    return [
        row(
            "Stage-1 teacher/RL",
            "Observation and PD-action contract",
            "o=[psi,e_anchor,V_imu,theta-theta0,theta_dot,a_last], theta_sp=theta0+alpha*a.",
            "Official whole_body_tracking code exposes generated command, anchor error, base velocities, joints, and last action; G1 action scale uses 0.25*tau_max/kp.",
            "pass" if stage1_obs_action_ok else "mismatch",
            [
                line_ref(FILES["paper_method"], r"\theta^{\text{sp}}"),
                line_ref(FILES["official_observations"], "generated_commands"),
                line_ref(FILES["official_g1"], "G1_ACTION_SCALE"),
            ],
            "Keep using official whole_body_tracking for Stage-1; do not train a custom 160-D obs policy unless it matches this contract.",
            "Pass here only proves the official source contract exists, not that the current teacher checkpoint is good.",
        ),
        row(
            "Stage-1 teacher/RL",
            "Reward terms and weights",
            "Gaussian body pose/orientation/velocity rewards plus anchor/regularization terms from appendix table.",
            "Official tracking config/reward code contains body_pos, body_orientation, body_lin_vel, body_ang_vel and Gaussian exp(-mean(square(error))/sigma^2) terms.",
            "pass" if reward_contract_ok else "mismatch",
            [line_ref(FILES["official_tracking_cfg"], "body_pos"), line_ref(FILES["official_rewards"], "def body_pos")],
            "Train only with the official reward config unless a row records an intentional ablation.",
            "Reward-code alignment does not make released-data or MuJoCo videos paper-level.",
        ),
        row(
            "Stage-1 teacher/RL",
            "PPO hyperparameters",
            "50 Hz policy, 24 steps/env, 30000 iterations, actor/critic [512,256,128], ELU, empirical normalization.",
            "Official RSL-RL config matches these table parameters.",
            "pass" if ppo_contract_ok else "mismatch",
            [line_ref(FILES["official_ppo"], "num_steps_per_env = 24"), line_ref(FILES["official_ppo"], "max_iterations = 30000")],
            "Use the official PPO config for any new teacher run; if GPU batch is increased, record the deviation explicitly.",
            "PPO config matching is not teacher quality evidence.",
        ),
        row(
            "PD/action scale/armature/material",
            "PD gains, action scale, armature",
            "omega=10 Hz, damping ratio 2, kp=I*omega^2, kd=2*zeta*I*omega, alpha=0.25*tau_max/kp.",
            "Official G1 code implements natural frequency, damping ratio, armature-derived stiffness/damping, and action scale.",
            "pass" if pd_contract_ok else "mismatch",
            [line_ref(FILES["official_g1"], "NATURAL_FREQ"), line_ref(FILES["official_g1"], "DAMPING_RATIO")],
            "Propagate these constants into MuJoCo XML/adapters before judging control quality.",
            "Matching constants in source do not prove the local MuJoCo XML/material/contact stack is identical.",
        ),
        row(
            "Stage-1 teacher/RL",
            "Adaptive sampling kernel",
            "Paper text states pre-failure look-back u={0,1,2} with rho^u weighting.",
            "Official MotionCommandCfg default currently exposes adaptive_kernel_size=1 unless an external config overrides it.",
            "partial" if adaptive_kernel_code_default_is_one and paper_mentions_prefailure_kernel else "pass",
            [line_ref(FILES["paper_root"], "look-back window"), line_ref(FILES["official_commands"], "adaptive_kernel_size")],
            "Before long teacher training, either set/verify adaptive_kernel_size=3 for paper-faithful runs or record the official default as a source-code discrepancy.",
            "Do not silently call a kernel_size=1 run an exact appendix reproduction of adaptive sampling.",
        ),
        row(
            "VAE",
            "Encoder/decoder input contract",
            "Encoder E(psi,e_anchor), decoder D(z, proprioception), latent dim 32, KL 0.01, lr 5e-4.",
            "Local paper-contract VAE script uses command+anchor error encoder, proprio+latent decoder, latent dim 32, KL 0.01, lr 5e-4.",
            "pass" if vae_input_contract_ok else "mismatch",
            [
                line_ref(FILES["paper_method"], "The encoder receives only the reference-motion components"),
                line_ref(FILES["paper_contract_vae_script"], "encoder_x = np.concatenate"),
            ],
            "Keep this interface; do not train the older obs+action VAE as if it were paper-faithful.",
            "Interface alignment does not mean the VAE is trained on official DAgger rollouts.",
        ),
        row(
            "VAE",
            "Network architecture and gradient accumulation",
            "Appendix table gives encoder/decoder hidden dims [2048,1024,512] and accumulated gradient steps 15.",
            "Local paper-contract VAE now exposes HIDDEN_DIMS=[2048,1024,512] and GRAD_ACCUM_STEPS=15.",
            "mismatch" if not (vae_arch_exact and vae_grad_accum_15) else "pass",
            [
                line_ref(FILES["paper_contract_vae_script"], "HIDDEN_DIMS"),
                line_ref(FILES["paper_contract_vae_script"], "optimizer.step"),
            ],
            "Keep this as a regression gate; do not reuse older obs+action VAE checkpoints for paper-facing videos.",
            "Architecture alignment does not prove official DAgger data or paper-level rollout quality.",
        ),
        row(
            "Diffusion",
            "Transformer denoiser architecture",
            "Horizon 16, history 4, sequence length 21, embed 512, heads 8, layers 6, 20 denoising steps.",
            "Local paper-contract diffusion script instantiates a Transformer encoder with configurable embed/heads/layers/steps and records dry-run/full-train mode.",
            "pass" if diffusion_arch_ok else "mismatch",
            [
                line_ref(FILES["paper_contract_diffusion_script"], "class StateLatentDiffusionTransformer"),
                line_ref(FILES["paper_contract_diffusion_script"], "DENOISING_STEPS"),
            ],
            "Use this route instead of the older MLP denoiser for paper-facing runs.",
            "Dry-run architecture pass does not prove full diffusion training or rollout quality.",
        ),
        row(
            "Diffusion",
            "State-latent trajectory representation",
            "Paper uses hybrid character-yaw-centric state plus latent, not raw 160-D policy observations.",
            (
                "Code path now supports paper_hybrid state windows and the paper-contract wrapper requires it, "
                "but the existing generated dataset is still old policy_obs/latent data and must be rebuilt."
            ),
            "pass" if state_builder_has_hybrid_path else "mismatch",
            [
                str(FILES["state_latent_source_contract"]),
                line_ref(FILES["state_latent_dataset_script"], "STATE_MODE"),
                line_ref(FILES["paper_method"], "hybrid state representation"),
            ],
            "Regenerate the dataset from teacher rollout shards containing raw root/body state; do not train from the old policy_obs dataset.",
            "A code-path pass is not a data-product pass; existing policy_obs+latent artifacts remain blocked.",
        ),
        row(
            "Diffusion",
            "Trainable state-latent dataset freshness",
            "Generated training shards must contain corrected hybrid state windows, raw rollout state, and reset-safe windows.",
            (
                f"existing_dataset_corrected={existing_state_latent_dataset_corrected}, "
                f"teacher_shards_have_raw_state={teacher_shards_have_raw_state}, "
                f"window_filter_ready={window_filter_ready}."
            ),
            "pass"
            if existing_state_latent_dataset_corrected and teacher_shards_have_raw_state and window_filter_ready
            else "blocked",
            [str(FILES["state_latent_source_contract"])],
            "Collect new teacher rollout shards with raw world-state fields, then rebuild hybrid state-latent windows with done/reset rejection.",
            "Until regenerated, downstream VAE/diffusion/guidance training from old shards is prohibited.",
        ),
        row(
            "Diffusion",
            "VAE rollout collection, OU noise, rejection, symmetry",
            "Paper collects VAE rollouts with OU action noise, rejects failures before 5 s, and applies sagittal symmetry augmentation.",
            "Current local state-latent dataset scripts do not prove OU rollout/rejection/symmetry augmentation in the paper-contract path.",
            "mismatch" if not (state_has_ou_rollout_rejection and state_has_symmetry_aug) else "pass",
            [line_ref(FILES["paper_method"], "Ornstein"), line_ref(FILES["paper_method"], "symmetry")],
            "Add OU noise collection, 5 s stability rejection, and symmetry augmentation manifests before long diffusion training.",
            "Current denoising MSE improvements are useful local probes, not paper-level data collection.",
        ),
        row(
            "Guidance",
            "Classifier guidance and task costs",
            "Guidance must optimize future states in a receding-horizon closed-loop denoising/control loop.",
            "Local guidance evaluations are still mostly offline/proxy; they do not validate MuJoCo closed-loop joystick/waypoint/inpainting/obstacle control.",
            "blocked" if guidance_is_offline and not guidance_receding_mujoco else "pass",
            [line_ref(FILES["paper_method"], "classifier guidance"), line_ref(FILES["guidance_eval_script"], "offline")],
            "Implement receding-horizon MuJoCo control where diffusion generates latent, VAE decodes action, and physics feeds back state.",
            "Offline guidance cost improvement is not a successful task rollout video.",
        ),
        row(
            "Guidance",
            "SDF relaxed barrier formula",
            "B(x,delta)=-ln(x) for x>=delta, otherwise -ln(delta)+0.5*((x-2delta)/delta)^2-0.5.",
            "Local sdf_barrier has been repaired to the paper piecewise formula and is covered by core math tests.",
            "pass" if sdf_barrier_ok else "mismatch",
            [line_ref(FILES["guidance_costs"], "BeyondMimic relaxed SDF barrier")],
            "Keep the unit test as a regression guard for obstacle guidance.",
            "This formula fix alone does not make obstacle avoidance closed-loop successful.",
        ),
        row(
            "MuJoCo deployment adapter",
            "Native action adapter",
            "Policy output action must map to theta_sp=theta0+alpha*a and MuJoCo actuator targets without semantic clipping errors.",
            "Formula adapter is recorded, but unit targets inside MuJoCo ctrlrange are not fully validated.",
            "partial" if native_action_ready and not native_action_range_ready else ("pass" if native_action_ready else "blocked"),
            [str(FILES["mujoco_action_audit"])],
            "Resolve ctrlrange/action-scale compatibility before judging PPO teacher rollout quality in MuJoCo.",
            "Approximate action adapter videos cannot be used as official deployment evidence.",
        ),
        row(
            "MuJoCo deployment adapter",
            "Native observation adapter",
            "MuJoCo state must reproduce IsaacLab/deployment observation terms without reference-frame or last-action bugs.",
            "Native observation adapter audit is still blocked; previous videos used approximate obs/root assist and showed forward-leaning stance.",
            "blocked" if not native_obs_ready else "pass",
            [str(FILES["mujoco_observation_audit"])],
            "Validate MuJoCo obs term-by-term against IsaacLab/motion_tracking_controller before long policy/video claims.",
            "A dimension-correct 160-D vector is insufficient evidence.",
        ),
        row(
            "MuJoCo deployment adapter",
            "Material/contact and no-root-assist video gate",
            "Paper-facing simulation should use valid contact/material semantics and no external root assist for success videos.",
            "Current MuJoCo contract audit still records floor material mismatch and no-root-assist/native video gate failure.",
            "blocked" if not (floor_material_ok and no_root_assist) else "pass",
            [str(FILES["mujoco_control_audit"])],
            "Fix floor/contact/material and produce no-root-assist native videos before success-folder cleanup.",
            "Root-assisted videos are diagnostic/report visuals only.",
        ),
        row(
            "Training permission",
            "Current teacher/downstream readiness",
            "Teacher quality and adapter gates must pass before downstream VAE/diffusion/guidance long training.",
            "Pretraining hard gate still blocks downstream training from the current teacher chain.",
            "blocked" if teacher_quality_blocks else "pass",
            [str(FILES["pretraining_hard_gate"])],
            "Do not start long downstream runs until the mismatches above are fixed and teacher quality is re-evaluated.",
            "The project remains goal_complete=false and cannot claim full BeyondMimic reproduction.",
        ),
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "area",
        "contract",
        "expected_from_paper_or_official",
        "observed_in_current_project",
        "status",
        "passed",
        "evidence",
        "required_fix_before_long_training",
        "claim_boundary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            record = dict(item)
            record["evidence"] = " | ".join(item["evidence"])
            writer.writerow(record)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# BeyondMimic Code/Formula/Appendix Contract Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Row count: `{summary['row_count']}`",
        f"- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        f"- Training permission: `{json.dumps(summary['permission'], sort_keys=True)}`",
        "",
        "## Required Fixes Before Long Training",
    ]
    for fix in summary["required_fixes_before_long_training"]:
        lines.append(f"- {fix}")
    lines.extend(["", "## Rows", ""])
    for item in rows:
        lines.extend(
            [
                f"### {item['area']} / {item['contract']}",
                f"- Status: `{item['status']}`",
                f"- Expected: {item['expected_from_paper_or_official']}",
                f"- Observed: {item['observed_in_current_project']}",
                f"- Required fix: {item['required_fix_before_long_training']}",
                f"- Claim boundary: {item['claim_boundary']}",
                f"- Evidence: `{'; '.join(item['evidence'])}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    status_counts: dict[str, int] = {}
    for item in rows:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    required_fixes = [
        item["required_fix_before_long_training"]
        for item in rows
        if item["status"] in {"mismatch", "blocked", "partial"}
    ]
    checks = {
        "paper_sources_readable": all(
            FILES[key].is_file() for key in ["paper_method", "paper_results", "paper_root"]
        ),
        "official_stage1_code_readable": all(
            FILES[key].is_file()
            for key in ["official_tracking_cfg", "official_rewards", "official_observations", "official_commands", "official_g1", "official_ppo"]
        ),
        "stage1_official_core_contracts_traced": all(
            item["passed"]
            for item in rows
            if item["area"] in {"Stage-1 teacher/RL", "PD/action scale/armature/material"}
            and item["contract"] != "Adaptive sampling kernel"
        ),
        "vae_input_contract_aligned": any(
            item["area"] == "VAE" and item["contract"] == "Encoder/decoder input contract" and item["passed"]
            for item in rows
        ),
        "vae_architecture_exact_to_paper": any(
            item["area"] == "VAE" and item["contract"] == "Network architecture and gradient accumulation" and item["passed"]
            for item in rows
        ),
        "state_latent_uses_hybrid_state": any(
            item["area"] == "Diffusion" and item["contract"] == "State-latent trajectory representation" and item["passed"]
            for item in rows
        ),
        "state_latent_generated_dataset_ready": any(
            item["area"] == "Diffusion" and item["contract"] == "Trainable state-latent dataset freshness" and item["passed"]
            for item in rows
        ),
        "diffusion_transformer_architecture_contract_available": any(
            item["area"] == "Diffusion" and item["contract"] == "Transformer denoiser architecture" and item["passed"]
            for item in rows
        ),
        "guidance_closed_loop_receding_horizon": any(
            item["area"] == "Guidance" and item["contract"] == "Classifier guidance and task costs" and item["passed"]
            for item in rows
        ),
        "sdf_barrier_matches_paper": any(
            item["area"] == "Guidance" and item["contract"] == "SDF relaxed barrier formula" and item["passed"]
            for item in rows
        ),
        "mujoco_native_no_root_assist_success": any(
            item["area"] == "MuJoCo deployment adapter"
            and item["contract"] == "Material/contact and no-root-assist video gate"
            and item["passed"]
            for item in rows
        ),
        "does_not_allow_long_training_yet": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "experiment_type": "code_formula_appendix_contract_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_code_formula_appendix_contract_has_required_fixes_before_training",
        "row_count": len(rows),
        "status_counts": status_counts,
        "required_fixes_before_long_training": required_fixes,
        "permission": {
            "start_new_long_stage1_teacher_training": False,
            "start_downstream_vae_training": False,
            "start_state_latent_diffusion_training": False,
            "start_guided_closed_loop_video_generation": False,
            "create_final_singleleg_success_folder": False,
            "allowed_next_work": [
                "regenerate teacher rollout shards with raw root/body state",
                "rebuild hybrid state-latent dataset with reset-safe windows",
                "validate MuJoCo native observation/action adapters without root assist",
                "run short code-level probes after fixes before long training",
            ],
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "claim": (
                "The current project has traced many official Stage-1 contracts and repaired one SDF barrier formula, "
                "and the paper-contract VAE/hybrid-state code paths are now present. However, old generated "
                "state-latent data, closed-loop guidance, and MuJoCo native adapter gates still block long training "
                "and success-video claims."
            ),
        },
        "rows": rows,
        "outputs": {"json": str(JSON_OUT), "tsv": str(TSV_OUT), "markdown": str(MD_OUT)},
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(TSV_OUT, rows)
    write_markdown(MD_OUT, summary, rows)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
