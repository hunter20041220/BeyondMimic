# Progress Update

## Goal

本轮目标是回应当前视频“只会前倾站姿、不会完整走路/单脚站立”的问题：不继续盲目训练，不生成新视频，而是重新对照 BeyondMimic 论文公式、附录参数、官方 `whole_body_tracking` 代码和本地复现链路，建立一个明确的失败根因审计。

## Files Read

- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json`
- `res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json`
- `res/level_c/official_importer_export_paper_contract_teacher_rollout_vae_training/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.json`
- `res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json`
- `res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json`
- `res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json`
- `res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `res/audits/mujoco_observation_runtime_parity_walk_sample/mujoco_observation_runtime_parity_audit.json`
- `res/audits/mujoco_torso_frame_offset_cross_sample/mujoco_torso_frame_offset_cross_sample_audit.json`

## Files Modified

- Added `reproduction/scripts/beyondmimic_formula_chain_failure_root_cause_audit.py`.
- Updated `reproduction/scripts/artifact_manifest.py`.
- Added this progress report.

## Commands Run

```bash
python3 reproduction/scripts/beyondmimic_formula_chain_failure_root_cause_audit.py
sed -n '1,220p' res/audits/formula_chain_failure_root_cause/beyondmimic_formula_chain_failure_root_cause_audit.md
```

## Results

新增审计输出：

- `res/audits/formula_chain_failure_root_cause/beyondmimic_formula_chain_failure_root_cause_audit.json`
- `res/audits/formula_chain_failure_root_cause/beyondmimic_formula_chain_failure_root_cause_audit.tsv`
- `res/audits/formula_chain_failure_root_cause/beyondmimic_formula_chain_failure_root_cause_audit.md`

审计状态：

```text
blocked_formula_chain_has_required_fixes_before_training_or_success_video
```

关键结论：

1. Stage-1 官方公式/参数源头基本可定位：官方 `whole_body_tracking` 的 observation、reward、PPO、G1 action scale、PD/armature 参数与论文主线一致，可以作为后续 Stage-1 corrective teacher training 的基础。
2. 当前 teacher 质量不够：multi-source best teacher 的 `reward_mean=0.0241`，`local_non_timeout_done_rate=19.41%`；single-leg teacher gate 未通过，`local_non_timeout_done_rate=27.93%`。
3. MuJoCo observation adapter 未通过：walk sample 中 command/base/joint/action 切片基本通过，但 anchor position/orientation 仍不匹配；固定 torso offset 不跨样本稳定。
4. VAE 公式接口已经接近论文：encoder 使用 `command + anchor error`，decoder 使用 `proprioception + latent`。但训练数据来自弱 teacher，且 VAE 训练阶段直接展平 shard，源数据中 `done_count=47200/306176`，不能说明模型学到了完整 reference motion。
5. 当前 state-latent dataset 仍显示 `state_source="policy_obs in local paper-contract best-teacher rollout shards"`，不是论文要求的 hybrid character/yaw-centric state。
6. 当前 denoiser/guidance 仍是本地 debug/proxy 证据：有 denoising MSE improvement，但不是 paper Transformer + closed-loop guidance 控制链。
7. 现有 MuJoCo 视频只能作为 diagnostic/presentation assets，不能作为最终成功文件夹。

## Verification

本轮新增审计脚本已经成功运行，输出 `training_allowed=false` 和 `success_video_generation_allowed=false`。这符合当前证据，不允许盲目启动下游 VAE/diffusion/guidance 长训练或生成“成功视频”。

后续还需运行标准项目验证链：

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

- MuJoCo native observation adapter 未通过，尤其是 anchor pose terms。
- 当前 teacher 质量不足，不应继续用它生成 VAE/diffusion 训练集。
- VAE 训练集包含大量 done/reset 样本，需要重新过滤并基于高质量连续 rollout 重建。
- state-latent dataset 不能继续使用 `policy_obs` 当作论文 state。
- diffusion 训练需要回到 paper Transformer 架构和正确 hybrid state 数据。
- guidance 仍不是 MuJoCo closed-loop receding-horizon control。

## Effect on English Reading Report

这份审计可直接支撑报告中的失败分析章节：当前项目不是简单“动作选得不好”，而是模型链路仍存在可审计的非 paper-contract 环节。报告中可以诚实写明：

- 官方 Stage-1 公式和参数已经定位；
- 本地 teacher 和 adapter 尚未通过；
- VAE/diffusion 的数值指标只能解释为 debug/proxy 指标；
- 现有 MuJoCo 视频不能作为 paper-level closed-loop evidence。

## Next Step

1. 优先修 MuJoCo observation adapter：在 walk、dance、single-leg 三个 non-terminated IsaacLab 样本上逐 slice 通过 parity。
2. 重新筛选或训练 Stage-1 teacher：先达成 low done-rate/high reward/连续 motion_time_steps。
3. 重新采集 teacher rollout，并过滤 done、timeout、motion_time_steps 跳变。
4. 重建 paper hybrid state-latent dataset，禁止继续把 160-D policy obs 当 diffusion state。
5. 然后才允许训练 paper Transformer diffusion 和 closed-loop guidance。

## Git Commit

Pending.
