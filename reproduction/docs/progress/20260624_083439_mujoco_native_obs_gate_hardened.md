# Progress Update

## Goal

继续执行训练前硬门控：先把 BeyondMimic 论文、附录参数、官方 `whole_body_tracking` 代码和本地 MuJoCo 控制链的公式/语义对齐，再允许继续 teacher/VAE/diffusion 长训或生成成功视频。本轮重点是加硬 `160D` native MuJoCo observation adapter 审计，避免“维度正确但语义错误”的 obs 继续驱动 PPO actor 产生前倾、站不稳、不会抬腿的失败视频。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_native_observation_adapter_contract.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
```

## Results

- 加硬了 native MuJoCo 160D observation adapter 审计。现在每个 policy obs slice 都记录：
  - official semantics；
  - MuJoCo 需要的 source；
  - required validation；
  - failure mode；
  - 是否已和 IsaacLab observation manager / motion_tracking_controller 数值验证。
- 明确记录官方 actor policy obs 顺序：
  - `command` 58D；
  - `motion_anchor_pos_b` 3D；
  - `motion_anchor_ori_b` 6D；
  - `base_lin_vel` 3D；
  - `base_ang_vel` 3D；
  - `joint_pos_rel` 29D；
  - `joint_vel_rel` 29D；
  - `last_action` 29D；
  - 合计 `160D`。
- 纠正 normalizer shape 判定：当前 best checkpoint 的 obs normalizer 为 `[1, 160]`，尾维是 `160`，因此 normalizer shape 可用；它不是当前 blocker。
- 当前真正 blockers 更清晰：
  - native adapter 未和 IsaacLab observation manager 数值对齐；
  - native adapter 未和 deployment `motion_tracking_controller` frame semantics 数值对齐；
  - 8 个 obs term 均未通过 runtime validation matrix；
  - native action adapter 仍因 MuJoCo ankle roll ctrlrange 截断而不能进入 rollout-ready；
  - 当前没有 no-root-assist native PPO/VAE/diffusion rollout 成功证据。

## Verification

当前已通过的局部验证：

- `mujoco_native_observation_adapter_contract.py` 编译通过；
- `mujoco_native_observation_adapter_contract.py` 运行通过，状态保持 `blocked_native_mujoco_observation_adapter_not_validated`；
- `beyondmimic_model_chain_paper_contract_audit.py` 运行通过，状态保持 `blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready`；
- `beyondmimic_code_formula_appendix_contract_audit.py` 运行通过，状态保持 `blocked_code_formula_appendix_contract_has_required_fixes_before_training`；
- `beyondmimic_pretraining_hard_gate_audit.py` 运行通过，状态保持 `blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`。

## Failed / Blocked Items

- 目前不能继续把 teacher/VAE/diffusion 视频解释为成功控制复现。
- 当前可看的 `reference_action_control` 更接近 reference/Pose 或 PD target 诊断，不等价于 learned policy closed-loop。
- `teacher_policy_action_control`、`vae_reconstructed_action_control`、`diffusion_denoised_latent_action_control`、`guided_latent_action_control`、`guided_vs_unguided_action_control` 仍需要 native obs/action adapter 通过后才能重做。
- 现在的失败不能简单归因于“训练迭代数不够”：维度正确但语义错误的 160D obs 足以导致前倾站姿、不会抬腿、不会正常走路。

## Effect on English Reading Report

这轮给报告提供了更诚实的失败归因：

1. 论文和官方 Stage-1 obs/reward/PPO/PD/action-scale 合同已经被追踪；
2. 本地 MuJoCo 控制视频失败的关键风险是 adapter semantics，而不是单纯训练时间；
3. 当前项目应把现有视频归类为 diagnostic visualization，不应写成 paper-level closed-loop result；
4. 后续报告可以展示这套审计作为“复现工程没有过度声称”的证据。

## Next Step

下一步应实现一个小型 numeric parity probe：

1. 在 IsaacLab 中固定同一个 motion、reset state、time step、last action；
2. 导出 observation manager 的 160D policy obs；
3. 在 MuJoCo 中用同一状态构造 native 160D obs；
4. 对八个 slice 分别计算误差；
5. 只有误差进入可解释阈值后，才允许继续 native PPO rollout 视频。

## Git Commit

待标准 8 项验证和 git staging 后提交。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
