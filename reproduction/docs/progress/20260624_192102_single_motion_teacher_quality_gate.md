# Progress Update

## Goal

本轮目标是把 `jumps1_subject1` 和 Short Sequence `Single Leg Balance` 的当前 teacher 质量做成可复跑、可审计的硬门控。重点不是继续生成前倾/失败视频，而是判断现有 teacher 是否足以进入后续 VAE、diffusion、guidance 训练。

## Files Read

- `report/audits/model_chain_hard_gate_review_20260624.md`
- `reproduction/paper/source/tex/method.tex`
- `reproduction/docs/paper_parameter_map.md`
- `res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.json`
- `res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json`
- `res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/motions/jumps1_subject1/jumps1_subject1_task_eval_metrics.json`
- `res/tracking/g1_official_csv_loop_full_dataset_task_eval/motions/jumps1_subject1/jumps1_subject1_task_eval_metrics.json`
- `res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/lafan1_jumps1_subject1_mujoco_summary.json`
- `res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/source_singleleg_reference_replay/source_singleleg_reference_replay_summary.json`

## Files Modified

- `reproduction/scripts/single_motion_teacher_quality_gate_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/progress/20260624_192102_single_motion_teacher_quality_gate.md`
- `res/audits/single_motion_teacher_quality_gate/single_motion_teacher_quality_gate_audit.json`
- `res/audits/single_motion_teacher_quality_gate/single_motion_teacher_quality_gate_audit.tsv`
- `res/audits/single_motion_teacher_quality_gate/single_motion_teacher_quality_gate_audit.md`

## Commands Run

- `python3 -m py_compile reproduction/scripts/single_motion_teacher_quality_gate_audit.py`
- `python3 reproduction/scripts/single_motion_teacher_quality_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

新增脚本会统一读取现有 teacher/eval/reference 证据，并用保守阈值判定：

- `reward_mean >= 0.10`
- `error_body_pos_mean <= 0.25 m`
- `error_joint_pos_mean <= 1.00 rad`
- `local_non_timeout_done_rate <= 0.05`

该门控会把 reference replay / reference action-control 与 learned teacher eval 分开。Reference 能看不等于 teacher 学会动作，也不能解锁 VAE/diffusion/guidance 成功视频。

实际结果：

- `paper_contract_public_bundle_best_teacher`：reward `0.0210356`，body error `0.253434`，joint error `1.72818`，done rate `0.154369`，不通过。
- `stage1_multisource_best_teacher`：reward `0.0241314`，body error `1.0095`，joint error `1.67395`，done rate `0.194137`，不通过。
- `hub_singleleg_video_single_leg_stand_1`：reward `0.0411416`，body error `0.166332`，joint error `0.876853`，done rate `0.27933`，不通过。
- `lafan1_jumps1_subject1`：现有 task eval reward `0.0184674` 或 `0.0279945`，done rate 仍高于阈值，不通过。

## Verification

已通过标准验证，日志保存到：

- `logs/verification/20260624_single_motion_teacher_quality_gate/full_verification.log`

标准验证输出包括：

- artifact manifest: `ok`, `1940` artifacts
- paper-vs-reproduction comparison: `ok`
- final reproduction report: `ok`
- completion matrix status audit: `ok`
- verification command syntax audit: `ok`
- verification command script manifest: `ok`
- verification command coverage audit: `ok`
- reproduction master audit: `ok`

## Failed / Blocked Items

当前预期状态仍是 blocked：

- `jumps1_subject1` 没有通过单动作 teacher 质量门控。
- `Single Leg Balance` 没有通过单动作 teacher 质量门控。
- 当前 teacher 不应继续作为 VAE/diffusion/guidance 成功视频的数据来源。

## Effect on English Reading Report

这份审计能支撑报告中一个关键诚实结论：本项目已经实现/检查了论文公式和本地模型链，但当前视频失败的主要原因不是渲染问题，而是 Stage-1 teacher 质量和 MuJoCo/IsaacLab obs/action 契约仍未满足 paper-level 下游训练条件。

## Next Step

先运行并提交本轮 hard gate。下一步应做 corrective single-motion Stage-1 teacher training/evaluation，而不是继续训练下游 VAE/diffusion。

## Git Commit

待提交。
