#!/usr/bin/env python3
"""Audit the current failing videos against the BeyondMimic formula chain.

This script is intentionally a hard pre-training gate.  The current local
MuJoCo videos often collapse into a generic forward-leaning posture instead of
walking or single-leg balancing.  Before spending more GPU time, we check the
actual paper contracts and the current artifacts for the whole chain:

  Stage-1 teacher/RL -> teacher rollout -> VAE -> state-latent diffusion
  -> guidance -> MuJoCo action/observation adapter -> videos.

The output is machine-readable evidence that says whether training or success
video generation is currently allowed.  A "blocked" result is not a failure of
the audit; it is the desired safety behavior when the code/data path is not yet
paper-aligned.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/formula_chain_failure_root_cause"
JSON_OUT = OUT / "beyondmimic_formula_chain_failure_root_cause_audit.json"
TSV_OUT = OUT / "beyondmimic_formula_chain_failure_root_cause_audit.tsv"
MD_OUT = OUT / "beyondmimic_formula_chain_failure_root_cause_audit.md"

FILES = {
    "paper_method": ROOT / "reproduction/paper/source/tex/method.tex",
    "paper_root": ROOT / "reproduction/paper/source/root.tex",
    "official_tracking_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "official_commands": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/commands.py",
    "official_g1": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/g1.py",
    "official_ppo": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
    "vae_script": ROOT / "reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py",
    "state_latent_script": ROOT
    / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py",
    "diffusion_script": ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py",
    "paper_transformer_script": ROOT
    / "reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py",
    "guidance_script": ROOT / "reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py",
    "stage1_multisource_sweep": ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
    "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json",
    "stage1_multisource_best": ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json",
    "hub_singleleg_eval": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/"
    "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    "vae_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_vae_training/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.json",
    "state_latent_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json",
    "diffusion_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/"
    "level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json",
    "guidance_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/"
    "level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json",
    "mujoco_action": ROOT / "res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json",
    "mujoco_obs": ROOT / "res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json",
    "mujoco_runtime_walk": ROOT
    / "res/audits/mujoco_observation_runtime_parity_walk_sample/mujoco_observation_runtime_parity_audit.json",
    "mujoco_torso_cross": ROOT
    / "res/audits/mujoco_torso_frame_offset_cross_sample/mujoco_torso_frame_offset_cross_sample_audit.json",
    "final_walk_video": ROOT
    / "res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/clean_walk_mujoco_control_suite_summary.json",
    "singleleg_video": ROOT
    / "res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/clean_walk_mujoco_control_suite_summary.json",
}


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def line_ref(path: Path, needle: str) -> str:
    text = read_text(path)
    if not text:
        return f"{path}:missing"
    for idx, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return f"{path}:not_found:{needle}"


def has(path: Path, *needles: str) -> bool:
    text = read_text(path)
    return all(needle in text for needle in needles)


def pct(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{100.0 * value:.2f}%"


def row(
    area: str,
    paper_requirement: str,
    local_evidence: str,
    status: str,
    diagnosis: str,
    required_fix: str,
    evidence_refs: list[str],
    blocks_training: bool,
    blocks_success_video: bool,
) -> dict[str, Any]:
    return {
        "area": area,
        "paper_requirement": paper_requirement,
        "local_evidence": local_evidence,
        "status": status,
        "passed": status == "pass",
        "blocks_training": blocks_training,
        "blocks_success_video": blocks_success_video,
        "diagnosis": diagnosis,
        "required_fix": required_fix,
        "evidence_refs": evidence_refs,
    }


def build_rows() -> list[dict[str, Any]]:
    stage1 = read_json(FILES["stage1_multisource_sweep"])
    best = read_json(FILES["stage1_multisource_best"])
    singleleg = read_json(FILES["hub_singleleg_eval"])
    vae = read_json(FILES["vae_json"])
    state_latent = read_json(FILES["state_latent_json"])
    diffusion = read_json(FILES["diffusion_json"])
    guidance = read_json(FILES["guidance_json"])
    mujoco_action = read_json(FILES["mujoco_action"])
    mujoco_obs = read_json(FILES["mujoco_obs"])
    mujoco_runtime = read_json(FILES["mujoco_runtime_walk"])
    torso_cross = read_json(FILES["mujoco_torso_cross"])
    final_walk_video = read_json(FILES["final_walk_video"])
    singleleg_video = read_json(FILES["singleleg_video"])

    best_ckpt = best.get("best_checkpoint") or stage1.get("best_checkpoint") or {}
    singleleg_gate = singleleg.get("quality_gate", {})
    vae_dataset = get_path(vae, "worker_summary", "dataset", default={}) or {}
    vae_eval = get_path(vae, "worker_summary", "evaluation", "test", default={}) or {}
    state_dataset = get_path(state_latent, "worker_summary", "dataset", default={}) or {}
    diffusion_dataset = get_path(diffusion, "worker_summary", "dataset", default={}) or {}
    diffusion_eval = get_path(diffusion, "worker_summary", "evaluation", "test", default={}) or {}
    guidance_checks = get_path(guidance, "worker_summary", "checks", default={}) or {}

    vae_sample_count = int(vae_dataset.get("sample_count") or 0)
    vae_done_count = int(vae_dataset.get("done_count") or 0)
    vae_done_rate = (vae_done_count / vae_sample_count) if vae_sample_count else None
    window_count = int(state_dataset.get("window_count") or 0)
    expected_window_count = int(state_dataset.get("expected_window_count") or 0)
    state_source = str(state_dataset.get("state_source", "missing"))

    rows = [
        row(
            "Stage-1 teacher/RL formula",
            (
                "论文 Stage-1 使用官方 motion tracking MDP: 160-D policy observation, "
                "normalized PD action theta_sp=theta0+alpha*a, reward table S1, domain randomization table S2, "
                "PPO 30k iterations."
            ),
            (
                "官方 whole_body_tracking 源码中 observation/order、reward、domain randomization、PPO config "
                "和 G1 action scale/armature 都能定位到。"
            ),
            "pass"
            if (
                has(FILES["official_tracking_cfg"], "class ObservationsCfg", "motion_anchor_pos_b", "motion_anchor_ori_b")
                and has(FILES["official_g1"], "G1_ACTION_SCALE", "0.25 * e[n] / s[n]")
                and has(FILES["official_ppo"], "max_iterations = 30000", "save_interval = 500")
            )
            else "mismatch",
            "公式/参数源头不是主要问题；第一阶段应该继续以官方 whole_body_tracking 为准。",
            "任何新 teacher 训练都必须沿用官方 MDP/reward/action scale，偏离项要单独记录为 ablation。",
            [
                line_ref(FILES["paper_method"], r"\mathbf{o}=["),
                line_ref(FILES["paper_method"], r"\theta}^{\text{sp}}"),
                line_ref(FILES["paper_root"], r"\label{tab:rewardterms}"),
                line_ref(FILES["paper_root"], r"\label{tab:ppo_hyperparameters}"),
                line_ref(FILES["official_tracking_cfg"], "class ObservationsCfg"),
                line_ref(FILES["official_g1"], "G1_ACTION_SCALE = {}"),
                line_ref(FILES["official_ppo"], "max_iterations = 30000"),
            ],
            blocks_training=False,
            blocks_success_video=False,
        ),
        row(
            "Stage-1 teacher quality",
            (
                "后续 VAE/diffusion 只能使用高质量 motion-tracking teacher 的闭环 rollout；"
                "teacher 应能连续跟踪 motion，而不是频繁 reset 或只维持前倾站姿。"
            ),
            (
                f"multi-source best: reward_mean={best_ckpt.get('reward_mean')}, "
                f"done_rate={pct(best_ckpt.get('local_non_timeout_done_rate'))}, "
                f"body_pos_err={best_ckpt.get('error_body_pos_mean')}, "
                f"joint_pos_err={best_ckpt.get('error_joint_pos_mean')}; "
                f"singleleg gate passed={singleleg_gate.get('passed')}, "
                f"singleleg_reward={singleleg_gate.get('reward_mean')}, "
                f"singleleg_done_rate={pct(singleleg_gate.get('local_non_timeout_done_rate'))}."
            ),
            "blocked",
            (
                "当前 teacher 质量不足是“VAE/diffusion 学成前倾站姿”的直接上游原因。"
                "如果 teacher 的动作本身没有稳定学到抬腿/走路，VAE 只能拟合弱 teacher 动作。"
            ),
            (
                "先重新筛选/训练 Stage-1 teacher，要求 non-timeout done rate < 5%, reward_mean > 0.1, "
                "body/joint error 达到本地质量门槛，并保存连续 motion_time_steps 的 rollout。"
            ),
            [
                str(FILES["stage1_multisource_best"]),
                str(FILES["hub_singleleg_eval"]),
            ],
            blocks_training=True,
            blocks_success_video=True,
        ),
        row(
            "MuJoCo observation adapter",
            (
                "IsaacLab PPO checkpoint 只能在完全一致的 observation semantics 下部署；"
                "尤其是 command、anchor pose error、base velocities、joint state、last_action 的 frame/order 必须逐项数值一致。"
            ),
            (
                f"native obs status={mujoco_obs.get('status')}; walk runtime status={mujoco_runtime.get('status')}; "
                f"cross-sample torso status={torso_cross.get('status')}."
            ),
            "blocked",
            (
                "walk 样本中 command/base/joint/action 切片已基本通过，但 anchor pos/orientation 仍不匹配；"
                "单样本 torso offset 可以拟合，但 walk+dance 跨样本 offset 不稳定。"
                "这会让 policy 以为自己一直有错误姿态，从而输出恢复/前倾动作。"
            ),
            (
                "在 walk、dance、singleleg 至少三个 non-terminated IsaacLab 样本上，"
                "把 MuJoCo 160-D obs 每个 slice 与 IsaacLab observation_manager 对齐到容差内；"
                "不要用单个固定 offset hack。"
            ),
            [
                str(FILES["mujoco_obs"]),
                str(FILES["mujoco_runtime_walk"]),
                str(FILES["mujoco_torso_cross"]),
            ],
            blocks_training=True,
            blocks_success_video=True,
        ),
        row(
            "MuJoCo action adapter / PD scale",
            (
                "动作必须按论文 theta_sp=theta0+alpha*a 转换为 PD setpoint；"
                "alpha 来自官方 G1 effort/stiffness，且 MuJoCo actuator ctrlrange 不能静默裁剪 normalized action。"
            ),
            f"native action status={mujoco_action.get('status')}; checks={mujoco_action.get('checks', {})}",
            "pass" if mujoco_action.get("status") == "ok_native_action_adapter_formula_and_no_clip_ctrlrange_patch_ready" else "blocked",
            "action adapter 当前不是主要 blocker；但任何视频仍必须同时通过 observation adapter。",
            "保留 no-clip ctrlrange XML/adapter；不要为了视觉效果改 action scale 或 PD 公式。",
            [
                line_ref(FILES["paper_method"], r"\theta}^{\text{sp}}"),
                line_ref(FILES["official_g1"], "G1_ACTION_SCALE = {}"),
                str(FILES["mujoco_action"]),
            ],
            blocks_training=False,
            blocks_success_video=mujoco_action.get("status") != "ok_native_action_adapter_formula_and_no_clip_ctrlrange_patch_ready",
        ),
        row(
            "VAE formula and data quality",
            (
                "论文 VAE encoder 只吃 reference intent E(psi,e_anchor)，decoder 吃 latent+proprioception "
                "并用 DAgger/teacher actions 训练；但训练数据必须来自高质量连续 teacher rollout。"
            ),
            (
                f"VAE status={vae.get('status')}; test_action_mse={vae_eval.get('action_mse')}; "
                f"dataset_done_count={vae_done_count}/{vae_sample_count} ({pct(vae_done_rate)}). "
                "当前 VAE loader 直接展平 shards，没有在 VAE 训练阶段过滤 done/reset 样本。"
            ),
            "blocked",
            (
                "VAE 接口公式大体已经改成 paper-contract，但源 teacher rollout 弱且包含大量 done/reset；"
                "低 action MSE 只说明它能拟合弱 teacher action，不能说明学到了单脚站立或走路姿态。"
            ),
            (
                "先修 teacher 和 MuJoCo obs adapter；重新采集连续、低 done-rate 的 teacher rollout；"
                "VAE 数据加载必须过滤 done、timeout、motion_time_steps 跳变，并按 motion/episode 切分。"
            ),
            [
                line_ref(FILES["paper_method"], r"\mathbf{z} = \mathcal{E}"),
                line_ref(FILES["paper_method"], r"\hat{\mathbf{a}} = \mathcal{D}"),
                str(FILES["vae_json"]),
                line_ref(FILES["vae_script"], "encoder_x = np.concatenate([command, anchor_pos, anchor_ori]"),
                line_ref(FILES["vae_script"], "dones = data[\"dones\"].reshape(-1)"),
            ],
            blocks_training=True,
            blocks_success_video=True,
        ),
        row(
            "State-latent diffusion dataset",
            (
                "论文 diffusion 数据是 state-latent trajectory: hybrid character/yaw-centric state + VAE latent；"
                "不应把 tracking policy 的 160-D observation 当作 state，也不应包含 reference command。"
            ),
            (
                f"state_source={state_source!r}; token_dim={state_dataset.get('token_dim')}; "
                f"window_count={window_count}; expected_window_count={expected_window_count}; "
                f"split_counts={state_dataset.get('split_counts')}."
            ),
            "blocked",
            (
                "当前官方 importer/export paper-contract state-latent artifact 仍显示 state_source 是 policy_obs。"
                "这与论文第 S2/S3 的 hybrid state 设计不一致，而且 policy_obs 含 reference command/anchor error，"
                "会把 tracking cue 泄漏进 diffusion，而不是学习 task-agnostic human-like behavior distribution。"
            ),
            (
                "重新生成 state-latent dataset：使用 raw world root/body state 构造 paper hybrid state；"
                "启用 OU noise rollout、5 秒稳定性验证、done/timeout/不连续窗口 rejection；"
                "输出 state_source 必须是 paper_hybrid 或 paper_projected，不得是 policy_obs。"
            ),
            [
                line_ref(FILES["paper_method"], "hybrid character"),
                line_ref(FILES["paper_root"], "Diffusion Dataset Collection"),
                line_ref(FILES["paper_root"], "Ornstein-Uhlenbeck"),
                str(FILES["state_latent_json"]),
                line_ref(FILES["state_latent_script"], "STATE_MODE = os.environ.get"),
                line_ref(FILES["state_latent_script"], "valid_contiguous_window_mask"),
            ],
            blocks_training=True,
            blocks_success_video=True,
        ),
        row(
            "Diffusion architecture/training",
            (
                "论文 diffusion 使用 Transformer Encoder，horizon=16, history=4, embed=512, heads=8, layers=6, "
                "20 denoising steps, batch=512, 1000 epochs, cosine/warmup/EMA。"
            ),
            (
                f"current denoiser status={diffusion.get('status')}; token_dim={diffusion_dataset.get('token_dim')}; "
                f"test_pred_token_mse={diffusion_eval.get('pred_token_mse')}; "
                f"denoising_improvement={diffusion_eval.get('denoising_improvement_ratio')}. "
                "当前 official_importer_export result 包装的是 resource_adjusted MLP StateLatentDenoiser。"
            ),
            "blocked",
            (
                "denoising MSE 改善是有价值的 debug 结果，但它是在错误/弱数据和 MLP denoiser 上得到的，"
                "不能证明论文 Transformer state-latent diffusion 复现成功。"
            ),
            (
                "在通过 teacher+state-latent dataset gate 后，改用 paper Transformer 架构和训练 schedule；"
                "输出必须记录 512/8/6/20、batch/epoch/scheduler/EMA，并在验证集上评估。"
            ),
            [
                line_ref(FILES["paper_method"], r"\mathcal{L}_\text{Diffusion}"),
                line_ref(FILES["paper_root"], r"\label{tab:diffusion_hyperparameters}"),
                str(FILES["diffusion_json"]),
                line_ref(FILES["diffusion_script"], "class StateLatentDenoiser"),
                line_ref(FILES["paper_transformer_script"], "nn.TransformerEncoder"),
            ],
            blocks_training=True,
            blocks_success_video=True,
        ),
        row(
            "Guidance and closed-loop control",
            (
                "论文 guidance 是 receding-horizon classifier guidance：每帧对 state-latent trajectory cost 求梯度，"
                "取当前 latent，经 VAE decoder 输出 action，再由物理仿真反馈下一帧。"
            ),
            (
                f"guidance status={guidance.get('status')}; checks={guidance_checks}; "
                f"scope={guidance.get('scope')}."
            ),
            "blocked",
            (
                "当前 guidance 是 offline proxy over denoiser outputs，不是 MuJoCo/IsaacLab 闭环；"
                "没有证明 joystick/waypoint/inpainting/obstacle cost 能驱动物理机器人。"
            ),
            (
                "等 teacher、VAE、paper hybrid diffusion 和 MuJoCo obs/action adapter 通过后，"
                "再实现 receding-horizon closed-loop guidance，保存每步 action、state、cost、fall/done。"
            ),
            [
                line_ref(FILES["paper_method"], "classifier guidance"),
                line_ref(FILES["paper_root"], "Task Costs"),
                str(FILES["guidance_json"]),
            ],
            blocks_training=True,
            blocks_success_video=True,
        ),
        row(
            "Existing MuJoCo videos",
            (
                "成功视频必须是 controller 输出 action -> PD torque/setpoint -> MuJoCo physics step -> next state feedback，"
                "且不能靠 root assist/reference blend 伪装。"
            ),
            (
                f"walk claim={final_walk_video.get('claim_level')}; "
                f"singleleg claim={singleleg_video.get('claim_level')}; "
                f"walk checks={final_walk_video.get('checks')}; singleleg checks={singleleg_video.get('checks')}."
            ),
            "blocked",
            (
                "现有视频可作为 diagnostic/presentation asset，但不是 paper-level 或可信闭环模型链证据；"
                "当前不能把它们作为最终成功文件夹。"
            ),
            (
                "保留失败记录和 summary，不再从这些目录挑成功；只有 adapter/teacher/model-chain 全部通过后，"
                "才生成新的 single-leg/walk 成功目录，并清理旧失败视频到 failed_runs 或 archive。"
            ),
            [
                str(FILES["final_walk_video"]),
                str(FILES["singleleg_video"]),
            ],
            blocks_training=False,
            blocks_success_video=True,
        ),
    ]
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocking_rows = [r for r in rows if r["blocks_training"] or r["blocks_success_video"]]
    checks = {
        "paper_stage1_formula_sources_found": rows[0]["passed"],
        "teacher_quality_passed": rows[1]["passed"],
        "mujoco_observation_adapter_ready": rows[2]["passed"],
        "mujoco_action_adapter_ready": rows[3]["passed"],
        "vae_formula_and_data_ready": rows[4]["passed"],
        "paper_state_latent_dataset_ready": rows[5]["passed"],
        "paper_transformer_diffusion_ready": rows[6]["passed"],
        "closed_loop_guidance_ready": rows[7]["passed"],
        "existing_success_video_claim_allowed": rows[8]["passed"],
    }
    training_allowed = (
        checks["paper_stage1_formula_sources_found"]
        and checks["mujoco_action_adapter_ready"]
        and checks["teacher_quality_passed"]
        and checks["mujoco_observation_adapter_ready"]
        and checks["vae_formula_and_data_ready"]
        and checks["paper_state_latent_dataset_ready"]
        and checks["paper_transformer_diffusion_ready"]
        and checks["closed_loop_guidance_ready"]
    )
    stage1_corrective_training_allowed = checks["paper_stage1_formula_sources_found"] and checks["mujoco_action_adapter_ready"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "beyondmimic_formula_chain_failure_root_cause_audit",
        "status": "blocked_formula_chain_has_required_fixes_before_training_or_success_video"
        if not training_allowed
        else "ok_formula_chain_ready_for_training_and_success_video_generation",
        "claim_level": "audit_only; no new training; no success video claim",
        "goal_complete": False,
        "checks": checks,
        "decision": {
            "long_downstream_vae_diffusion_guidance_training_allowed": training_allowed,
            "success_singleleg_or_walk_video_generation_allowed": training_allowed,
            "stage1_corrective_teacher_training_allowed": stage1_corrective_training_allowed,
            "why_stage1_corrective_training_is_different": (
                "Stage-1 corrective teacher retraining can be allowed because the official Stage-1 formula source "
                "is available; downstream VAE/diffusion/guidance training remains blocked until teacher quality "
                "and native MuJoCo observation parity pass."
            ),
        },
        "root_cause_ranking": [
            {
                "rank": 1,
                "cause": "MuJoCo native observation adapter 的 anchor pose 项还没有与 IsaacLab 数值等价。",
                "effect": "policy 会看到假的位置/姿态误差，容易输出恢复姿态或前倾站姿，而不是抬腿/迈步。",
            },
            {
                "rank": 2,
                "cause": "当前 teacher checkpoint non-timeout done rate 高、reward 低。",
                "effect": "VAE/diffusion 学到的是弱 teacher 行为，而不是完整走路或单脚站立姿态。",
            },
            {
                "rank": 3,
                "cause": "VAE 训练数据来自弱 teacher rollout，且包含 reset/done 样本。",
                "effect": "低 reconstruction MSE 只证明拟合了弱 action，不证明学会了 reference motion。",
            },
            {
                "rank": 4,
                "cause": "当前 state-latent artifact 使用 160-D policy_obs，而不是论文 hybrid state。",
                "effect": "diffusion 训练在错误 state 表示上，还可能把 reference tracking cue 泄漏进模型。",
            },
            {
                "rank": 5,
                "cause": "当前 denoiser/guidance 证据不是论文 Transformer + closed-loop guidance 链。",
                "effect": "offline MSE/cost 改善不能证明 MuJoCo action-control 成功。",
            },
        ],
        "blocking_count": len(blocking_rows),
        "blocking_areas": [r["area"] for r in blocking_rows],
        "rows": rows,
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "area",
        "status",
        "passed",
        "blocks_training",
        "blocks_success_video",
        "paper_requirement",
        "local_evidence",
        "diagnosis",
        "required_fix",
        "evidence_refs",
    ]
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in summary["rows"]:
            row_copy = dict(item)
            row_copy["evidence_refs"] = " | ".join(item["evidence_refs"])
            writer.writerow({field: row_copy.get(field, "") for field in fields})
    lines = [
        "# BeyondMimic 公式链路失败根因审计",
        "",
        f"- status: `{summary['status']}`",
        f"- claim level: `{summary['claim_level']}`",
        f"- downstream training allowed: `{summary['decision']['long_downstream_vae_diffusion_guidance_training_allowed']}`",
        f"- success video generation allowed: `{summary['decision']['success_singleleg_or_walk_video_generation_allowed']}`",
        f"- stage1 corrective teacher training allowed: `{summary['decision']['stage1_corrective_teacher_training_allowed']}`",
        "",
        "## 当前根因排序",
        "",
    ]
    for item in summary["root_cause_ranking"]:
        lines.append(f"{item['rank']}. **{item['cause']}**")
        lines.append(f"   - 影响：{item['effect']}")
    lines.extend(["", "## 逐项门禁", ""])
    for item in summary["rows"]:
        lines.extend(
            [
                f"### {item['area']} - `{item['status']}`",
                "",
                f"- 论文/官方要求：{item['paper_requirement']}",
                f"- 本地证据：{item['local_evidence']}",
                f"- 诊断：{item['diagnosis']}",
                f"- 必须修复：{item['required_fix']}",
                f"- blocks training: `{item['blocks_training']}`",
                f"- blocks success video: `{item['blocks_success_video']}`",
                f"- evidence: {'; '.join(item['evidence_refs'])}",
                "",
            ]
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    summary = build_summary(rows)
    write_outputs(summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(JSON_OUT),
                "blocking_count": summary["blocking_count"],
                "training_allowed": summary["decision"]["long_downstream_vae_diffusion_guidance_training_allowed"],
            },
            sort_keys=True,
        )
    )
    if summary["status"].startswith("blocked"):
        return


if __name__ == "__main__":
    main()
