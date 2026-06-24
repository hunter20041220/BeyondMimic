#!/usr/bin/env python3
"""Pre-training hard gate for the BeyondMimic model/control chain.

This audit answers one narrow question before any new long teacher/VAE/
diffusion/video run:

    Are the paper formulas, official parameters, teacher quality gates,
    VAE/diffusion/guidance contracts, and MuJoCo control adapters sufficiently
    aligned to start training or to claim success videos?

The answer is intentionally conservative.  It allows future Stage-1 teacher
retraining only as the next corrective step on the official whole_body_tracking
route, while blocking downstream VAE/diffusion/guidance training and final
MuJoCo success videos until the teacher quality and native adapter gates pass.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/pretraining_hard_gate"
JSON_OUT = OUT / "beyondmimic_pretraining_hard_gate_audit.json"
TSV_OUT = OUT / "beyondmimic_pretraining_hard_gate_audit.tsv"
MD_OUT = OUT / "beyondmimic_pretraining_hard_gate_audit.md"

FILES = {
    "formula_parameter_trace": ROOT
    / "res/audits/formula_parameter_trace_audit/beyondmimic_formula_parameter_trace_audit.json",
    "appendix_parameter_matrix": ROOT
    / "res/audits/appendix_parameter_matrix/beyondmimic_appendix_parameter_matrix_audit.json",
    "model_chain_contract": ROOT
    / "res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json",
    "stage1_tracking_parameter_contract": ROOT
    / "res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.json",
    "mujoco_control_contract": ROOT / "res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json",
    "mujoco_native_action_adapter": ROOT
    / "res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json",
    "mujoco_native_observation_adapter": ROOT
    / "res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json",
    "paper_contract_vae": ROOT
    / "res/level_c/paper_contract_teacher_rollout_vae_training/"
    "level_c_paper_contract_teacher_rollout_vae_training.json",
    "paper_transformer_diffusion": ROOT
    / "res/level_c/paper_contract_transformer_state_latent_diffusion_training/"
    "paper_contract_transformer_state_latent_diffusion_training.json",
    "paper_contract_guidance": ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/"
    "level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json",
    "stage1_multisource_best_teacher": ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json",
    "stage1_multisource_checkpoint_sweep": ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
    "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json",
    "hub_singleleg_teacher_eval": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/"
    "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    "clean_walk_weight_sweep": ROOT
    / "res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.json",
    "final_clean_walk_suite": ROOT
    / "res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/"
    "clean_walk_mujoco_control_suite_summary.json",
    "official_tracking_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "official_g1_robot": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/robots/g1.py",
    "official_ppo_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def line_ref(path: Path, needle: str) -> str:
    text = read_text(path)
    if not text:
        return f"{path}:missing"
    for idx, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return f"{path}:not_found:{needle}"


def bool_path(data: dict[str, Any], *keys: str) -> bool:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return bool(cur)


def get_path(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def gate_row(
    gate: str,
    status: str,
    passed: bool,
    decision: str,
    evidence: list[str],
    reason: str,
    required_fix: str,
    claim_boundary: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": status,
        "passed": bool(passed),
        "decision": decision,
        "evidence": evidence,
        "reason": reason,
        "required_fix": required_fix,
        "claim_boundary": claim_boundary,
    }


def build_rows() -> list[dict[str, Any]]:
    formula = read_json(FILES["formula_parameter_trace"])
    appendix_matrix = read_json(FILES["appendix_parameter_matrix"])
    model_chain = read_json(FILES["model_chain_contract"])
    stage1 = read_json(FILES["stage1_tracking_parameter_contract"])
    mujoco_control = read_json(FILES["mujoco_control_contract"])
    action_adapter = read_json(FILES["mujoco_native_action_adapter"])
    obs_adapter = read_json(FILES["mujoco_native_observation_adapter"])
    vae = read_json(FILES["paper_contract_vae"])
    transformer = read_json(FILES["paper_transformer_diffusion"])
    guidance = read_json(FILES["paper_contract_guidance"])
    multisource = read_json(FILES["stage1_multisource_checkpoint_sweep"])
    singleleg = read_json(FILES["hub_singleleg_teacher_eval"])
    clean_sweep = read_json(FILES["clean_walk_weight_sweep"])
    final_walk = read_json(FILES["final_clean_walk_suite"])

    stage1_formula_ok = (
        formula.get("status") == "blocked_formula_parameter_trace_has_required_fixes_before_training"
        and stage1.get("status")
        in {"ok_stage1_tracking_parameter_contract_audited", "blocked_stage1_teacher_contract_has_required_followups"}
    )
    stage1_quality_pass = bool(get_path(singleleg, "quality_gate", "passed", default=False))
    multisource_done_rate = get_path(
        multisource, "selected_checkpoint", "local_non_timeout_done_rate", default=None
    )
    multisource_reward = get_path(multisource, "selected_checkpoint", "reward_mean", default=None)
    if multisource_done_rate is None:
        multisource_done_rate = get_path(multisource, "best_checkpoint", "local_non_timeout_done_rate", default=None)
    if multisource_reward is None:
        multisource_reward = get_path(multisource, "best_checkpoint", "reward_mean", default=None)

    paper_contract_vae_interface_ok = (
        bool_path(vae, "checks", "encoder_uses_reference_intent_only")
        and bool_path(vae, "checks", "decoder_uses_proprioception_plus_latent")
        and bool_path(vae, "checks", "action_dim_29")
    )
    vae_source_teacher_ok = bool_path(vae, "checks", "source_teacher_done_rate_low_enough_for_downstream")
    transformer_code_contract_ok = (
        bool_path(transformer, "checks", "paper_contract_architecture_checks_pass")
        and bool_path(transformer, "worker_summary", "checks", "uses_transformer_encoder")
        and bool_path(transformer, "worker_summary", "checks", "uses_individual_state_and_latent_denoising_steps")
    )
    transformer_full_training_ok = (
        transformer_code_contract_ok
        and not bool(transformer.get("dry_run", True))
        and bool_path(transformer, "checks", "test_denoising_improves_over_noisy")
    )
    guidance_offline_ok = (
        bool_path(guidance, "checks", "all_best_guidance_gradients_nonzero")
        and bool_path(guidance, "checks", "all_best_costs_improve")
    )
    native_action_formula_ok = bool_path(action_adapter, "interpretation", "formula_adapter_ready")
    native_obs_ok = bool_path(obs_adapter, "interpretation", "native_obs_adapter_ready")
    no_root_assist_native_video_ok = (
        bool_path(mujoco_control, "checks", "mujoco_video_has_no_root_assist")
        and bool_path(mujoco_control, "checks", "mujoco_video_uses_native_policy_action_semantics")
    )
    pure_model_walk_ok = bool(clean_sweep.get("pure_model_weight_1p0_ok", False))
    final_walk_claim = str(final_walk.get("claim_level", "missing"))
    final_walk_has_root_assist_boundary = "root assist" in final_walk_claim.lower()
    appendix_matrix_pass = appendix_matrix.get("status") == "ok_appendix_parameter_matrix_all_rows_pass"
    appendix_blocking_items = appendix_matrix.get("blocking_items", [])

    return [
        gate_row(
            "appendix_parameter_matrix_contract",
            "blocked_appendix_matrix_has_required_fixes",
            appendix_matrix_pass,
            "block_long_training_until_appendix_matrix_passes",
            [
                str(FILES["appendix_parameter_matrix"]),
                f"appendix_status={appendix_matrix.get('status')}",
                f"appendix_blocking_count={len(appendix_blocking_items)}",
                f"appendix_blocking_items={appendix_blocking_items[:6]}",
            ],
            (
                "The new appendix matrix keeps paper/appendix parameters, official public-code differences, "
                "and native MuJoCo deployment gates in one machine-readable pre-training checklist."
            ),
            "Resolve or explicitly waive every appendix-matrix blocker before starting long downstream training or creating final success videos.",
            "This is an audit-only gate; it does not claim a trained policy, VAE, diffusion model, or video succeeded.",
        ),
        gate_row(
            "paper_and_official_stage1_formula_contract",
            "pass_for_formula_audit_but_not_teacher_quality",
            stage1_formula_ok,
            "allow_stage1_teacher_retraining_only_as_corrective_work",
            [
                str(FILES["formula_parameter_trace"]),
                str(FILES["stage1_tracking_parameter_contract"]),
                line_ref(FILES["official_tracking_cfg"], "class ObservationsCfg"),
                line_ref(FILES["official_g1_robot"], "G1_ACTION_SCALE = {}"),
                line_ref(FILES["official_ppo_cfg"], "max_iterations = 30000"),
            ],
            (
                "Official whole_body_tracking remains the right Stage-1 route and the main observation/action/"
                "reward/PD/PPO contracts are traced. This does not mean the current checkpoint is good."
            ),
            "Use the official Stage-1 path for the next teacher run; do not change rewards or action semantics just to get better videos.",
            "This gate permits only corrective teacher training/evaluation, not downstream VAE/diffusion success claims.",
        ),
        gate_row(
            "teacher_quality_for_downstream_rollout_dataset",
            "blocked_teacher_quality_failed",
            stage1_quality_pass,
            "block_downstream_dataset_vae_diffusion_video",
            [
                str(FILES["hub_singleleg_teacher_eval"]),
                str(FILES["stage1_multisource_checkpoint_sweep"]),
                f"singleleg_quality_gate={get_path(singleleg, 'quality_gate', 'passed', default=None)}",
                f"singleleg_done_rate={get_path(singleleg, 'quality_gate', 'local_non_timeout_done_rate', default=None)}",
                f"singleleg_reward_mean={get_path(singleleg, 'quality_gate', 'reward_mean', default=None)}",
                f"multisource_done_rate={multisource_done_rate}",
                f"multisource_reward_mean={multisource_reward}",
            ],
            "The current teachers are weak/reset-heavy; this directly explains front-leaning or generic posture outputs.",
            "Select/train a teacher with low non-timeout done rate, continuous motion-time, meaningful reward, and target pose/body error gates before collecting VAE data.",
            "Current teacher/VAE/diffusion videos remain diagnostic only.",
        ),
        gate_row(
            "conditional_vae_paper_contract",
            "blocked_until_teacher_quality_passes",
            paper_contract_vae_interface_ok and vae_source_teacher_ok,
            "block_vae_long_training_from_current_teacher",
            [
                str(FILES["paper_contract_vae"]),
                f"paper_contract_vae_interface_ok={paper_contract_vae_interface_ok}",
                f"source_teacher_done_rate_low_enough={vae_source_teacher_ok}",
            ],
            (
                "The corrected VAE interface exists, but using a failed/reset-heavy teacher would train the VAE "
                "to imitate poor control. The old obs+action encoder route is explicitly disallowed for success videos."
            ),
            "Retrain the paper-contract VAE only after the accepted teacher rollout dataset exists.",
            "VAE MSE alone is not enough for paper-level or video success claims.",
        ),
        gate_row(
            "state_latent_diffusion_paper_contract",
            "blocked_transformer_full_training_not_done",
            transformer_full_training_ok,
            "block_diffusion_long_training_from_current_dataset",
            [
                str(FILES["paper_transformer_diffusion"]),
                f"transformer_code_contract_ok={transformer_code_contract_ok}",
                f"dry_run={transformer.get('dry_run')}",
            ],
            (
                "A paper-style Transformer code contract is present, but the full trained Transformer over an "
                "accepted teacher/VAE dataset is not available. MLP/resource-adjusted denoisers remain debug baselines."
            ),
            "Use the paper-contract Transformer only after teacher and VAE gates pass; then run full train/eval with held-out splits.",
            "Current denoising MSE improvements are local diagnostic evidence, not proof of successful control.",
        ),
        gate_row(
            "classifier_guidance_closed_loop_contract",
            "blocked_offline_only",
            guidance_offline_ok and transformer_full_training_ok and native_obs_ok,
            "block_guided_success_videos",
            [
                str(FILES["paper_contract_guidance"]),
                f"offline_guidance_ok={guidance_offline_ok}",
                f"native_obs_adapter_ready={native_obs_ok}",
            ],
            "Guidance has offline gradient/cost evidence, but no validated receding-horizon native closed-loop control path.",
            "After a trained paper-contract diffusion model exists, run receding-horizon closed-loop guidance with task metrics.",
            "Current guided videos are not Fig.5/Fig.6-level guidance evidence.",
        ),
        gate_row(
            "mujoco_native_action_adapter",
            "partial_formula_ready_but_rollout_not_ready",
            native_action_formula_ok and bool_path(action_adapter, "checks", "unit_targets_inside_mujoco_ctrlrange"),
            "allow_formula_fixture_only",
            [
                str(FILES["mujoco_native_action_adapter"]),
                f"formula_adapter_ready={native_action_formula_ok}",
                f"unit_targets_inside_ctrlrange={bool_path(action_adapter, 'checks', 'unit_targets_inside_mujoco_ctrlrange')}",
            ],
            "The theta0 + alpha * action fixture is available, but ctrlrange clipping is still recorded for some joints.",
            "Log raw and clipped setpoints; patch or justify MuJoCo ctrlrange before no-root-assist success videos.",
            "Formula fixture success is not a physics rollout success.",
        ),
        gate_row(
            "mujoco_native_observation_adapter",
            "blocked_not_validated_against_isaaclab_or_deployment",
            native_obs_ok,
            "block_native_ppo_vae_diffusion_rollout",
            [
                str(FILES["mujoco_native_observation_adapter"]),
                f"native_adapter_validated_against_isaaclab={bool_path(obs_adapter, 'checks', 'native_adapter_validated_against_isaaclab_observation_manager')}",
                f"native_adapter_validated_against_deployment={bool_path(obs_adapter, 'checks', 'native_adapter_validated_against_deployment_controller')}",
            ],
            "A dimension-correct 160-D vector can still be semantically wrong, which can produce leaning or collapsed postures.",
            "Numerically validate the MuJoCo observation builder against IsaacLab observation_manager and motion_tracking_controller frame semantics.",
            "No native MuJoCo PPO/VAE/diffusion rollout claim is allowed until this passes.",
        ),
        gate_row(
            "mujoco_physics_video_success_gate",
            "blocked_root_assist_blending_and_material_gap",
            no_root_assist_native_video_ok and pure_model_walk_ok and not final_walk_has_root_assist_boundary,
            "block_final_success_folder",
            [
                str(FILES["mujoco_control_contract"]),
                str(FILES["clean_walk_weight_sweep"]),
                str(FILES["final_clean_walk_suite"]),
                f"pure_model_weight_1p0_ok={pure_model_walk_ok}",
                f"no_root_assist_native_video_ok={no_root_assist_native_video_ok}",
                f"final_walk_has_root_assist_boundary={final_walk_has_root_assist_boundary}",
            ],
            (
                "Readable videos currently rely on diagnostic aids or weak/local chains. Root assist, blending, "
                "absolute targets, and material mismatch prevent final success claims."
            ),
            "Generate the final folder only from no-root-assist native action control with continuous motion-time and low fall metrics.",
            "Existing clean-walk/single-leg videos should be treated as failed/diagnostic unless later superseded.",
        ),
    ]


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_rows = [row for row in rows if not row["passed"]]
    partial_rows = [row for row in rows if str(row["status"]).startswith("partial")]
    passed_rows = [row for row in rows if row["passed"]]
    stage1_formula_pass = next(row for row in rows if row["gate"] == "paper_and_official_stage1_formula_contract")[
        "passed"
    ]
    teacher_pass = next(row for row in rows if row["gate"] == "teacher_quality_for_downstream_rollout_dataset")[
        "passed"
    ]
    downstream_pass = all(
        row["passed"]
        for row in rows
        if row["gate"]
        in {
            "conditional_vae_paper_contract",
            "state_latent_diffusion_paper_contract",
            "classifier_guidance_closed_loop_contract",
            "mujoco_native_observation_adapter",
            "mujoco_physics_video_success_gate",
        }
    )
    checks = {
        "paper_sources_and_official_stage1_code_traced": stage1_formula_pass,
        "teacher_quality_gate_passed": teacher_pass,
        "downstream_vae_training_allowed": teacher_pass
        and next(row for row in rows if row["gate"] == "conditional_vae_paper_contract")["passed"],
        "diffusion_training_allowed": downstream_pass,
        "guided_success_video_allowed": downstream_pass,
        "final_singleleg_success_folder_allowed": downstream_pass,
        "does_not_allow_downstream_training_from_current_teacher": not teacher_pass,
        "does_not_allow_current_mujoco_videos_as_success": True,
        "does_not_claim_paper_level_or_real_robot": True,
        "keeps_goal_incomplete": True,
    }
    return {
        "status": "blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "beyondmimic_pretraining_hard_gate_audit",
        "scope": "Machine-readable gate before new long teacher/VAE/diffusion training or final success video generation.",
        "claim_level": "audit_and_gate_only; no new training; no success video claim",
        "files": {key: str(path) for key, path in FILES.items()},
        "row_count": len(rows),
        "passed_count": len(passed_rows),
        "blocked_count": len(blocked_rows),
        "partial_count": len(partial_rows),
        "rows": rows,
        "blocking_gates": [row["gate"] for row in blocked_rows],
        "permission": {
            "start_stage1_teacher_retraining": "conditional_only_as_corrective_work_on_official_whole_body_tracking_route",
            "start_downstream_vae_training": False,
            "start_state_latent_diffusion_training": False,
            "start_guided_closed_loop_video_generation": False,
            "create_final_singleleg_success_folder": False,
        },
        "checks": checks,
        "interpretation": {
            "why_current_videos_are_bad": (
                "Current evidence points to weak/reset-heavy teacher rollouts plus non-native MuJoCo control/observation "
                "adapters and legacy VAE/diffusion routes. These can collapse learned behavior into a forward-leaning "
                "standing pose even when reference replay itself is continuous."
            ),
            "next_allowed_work": [
                "Continue code/parameter audits and native observation validation.",
                "Run only corrective Stage-1 teacher training/evaluation after preserving exact official contracts.",
                "Do not start downstream VAE/diffusion/guidance long training from the current weak teacher.",
            ],
            "goal_complete": False,
        },
        "outputs": {"json": str(JSON_OUT), "tsv": str(TSV_OUT), "markdown": str(MD_OUT)},
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "gate",
            "status",
            "passed",
            "decision",
            "reason",
            "required_fix",
            "claim_boundary",
            "evidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in summary["rows"]:
            item = dict(row)
            item["evidence"] = " | ".join(str(x) for x in item["evidence"])
            writer.writerow({key: item.get(key, "") for key in fieldnames})

    lines = [
        "# BeyondMimic 训练前硬门控审计",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 行数：`{summary['row_count']}`",
        f"- 阻塞项：`{summary['blocked_count']}`",
        f"- claim level：`{summary['claim_level']}`",
        "",
        "## 结论",
        "",
        "当前只能把 Stage-1 teacher 重新训练/评估作为纠错方向；不能从当前 weak teacher 继续训练 VAE/diffusion，也不能把现有 MuJoCo 视频作为成功视频。",
        "",
        "当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。",
        "",
        "## Permission",
        "",
    ]
    for key, value in summary["permission"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gate Rows", ""])
    for row in summary["rows"]:
        mark = "PASS" if row["passed"] else "BLOCK"
        lines.extend(
            [
                f"### {row['gate']}",
                "",
                f"- 结果：`{mark}` / `{row['status']}`",
                f"- 决策：`{row['decision']}`",
                f"- 原因：{row['reason']}",
                f"- 修复要求：{row['required_fix']}",
                f"- 声明边界：{row['claim_boundary']}",
                "- 证据：",
            ]
        )
        for evidence in row["evidence"]:
            lines.append(f"  - `{evidence}`")
        lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    summary = build_summary(rows)
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT), "tsv": str(TSV_OUT), "markdown": str(MD_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
