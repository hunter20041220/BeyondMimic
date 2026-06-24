# Progress Update

## Goal

本轮目标是继续执行“先审论文公式、代码链路和参数门禁，再开始训练”的要求，重点修正 state-latent 数据集的 paper-contract 判定逻辑。当前 MuJoCo 单脚站立、走路和 teacher/VAE/diffusion 视频效果差，不能靠换动作掩盖；必须先确保训练数据链没有把错误的 Stage-1 `policy_obs` 当成论文 diffusion 所需的 hybrid state-latent trajectory。

## Files Read

- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `res/audits/formula_chain_failure_root_cause/beyondmimic_formula_chain_failure_root_cause_audit.json`

## Files Modified

- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_142002_state_latent_paper_contract_gate.md`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.tsv`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py
python3 reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py
python3 reproduction/scripts/beyondmimic_formula_chain_failure_root_cause_audit.py
python3 reproduction/scripts/beyondmimic_appendix_parameter_matrix_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
```

## Results

- 修复前，`official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json` 顶层 `status` 是 `ok_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset`，但内部 `worker_summary.dataset.state_source` 是 `policy_obs in local paper-contract best-teacher rollout shards`，`obs_dim=160`，`token_dim=192`。这与论文 state-latent diffusion 的 hybrid state token 要求不一致。
- 已修复 wrapper：只有当状态源是 raw hybrid/projected state、维度符合 99/163 状态合同、窗口过滤连续性通过、teacher/VAE 来源通过时，才允许写 `ok_official...`。
- 已修复 source-contract audit：现在显式检查“dataset status 是否与 state_source 一致”。旧 artifact 的 `ok` 状态和 `policy_obs` 状态源会被判为矛盾。
- 当前专项审计状态为 `blocked_state_latent_dataset_source_uses_policy_obs_and_missing_rollout_state`，这是正确阻断，不是训练成功。

## Verification

- `beyondmimic_state_latent_dataset_source_contract_audit.py`: 通过运行，输出 blocked，`blocked_count=5`。
- `beyondmimic_formula_chain_failure_root_cause_audit.py`: blocked，`training_allowed=false`。
- `beyondmimic_appendix_parameter_matrix_audit.py`: blocked。
- `beyondmimic_pretraining_hard_gate_audit.py`: blocked。
- `beyondmimic_code_formula_appendix_contract_audit.py`: blocked。
- `beyondmimic_model_chain_paper_contract_audit.py`: blocked。
- `artifact_manifest.py`: ok。
- `final_reproduction_report.py`: ok。
- `reproduction_master_audit.py`: ok。
- verification command syntax/script manifest/coverage: ok。

## Failed / Blocked Items

- 现有 trainable state-latent 数据仍不可用于正式 diffusion/VAE 后续训练，因为它来自 160-D Stage-1 `policy_obs`，不是论文 hybrid state。
- teacher rollout shards 仍缺失足够可信的 raw world-state 字段和 accepted continuous windows。
- OU perturbation collection、5s rejection filtering、symmetry augmentation 元数据仍未构成 paper-contract trainable dataset。
- MuJoCo observation adapter 仍未完成和 IsaacLab anchor state 的数值等价验证。
- 当前不得生成或声称成功的 teacher/VAE/diffusion 单脚站立闭环视频。

## Effect on English Reading Report

这轮给报告提供了一个关键诚实结论：当前视频差不是单纯“选错动作”，而是模型链训练数据和论文定义的 state-latent trajectory contract 不一致。报告中应明确区分 reference replay、local MuJoCo closed-loop attempt、paper-contract blocked gate，以及未完成的 official/paper-level diffusion policy reproduction。

## Next Step

下一步应先修 teacher rollout collector 和 dataset builder：

1. 确保 rollout shard 记录 raw root pose、root twist、body world positions、body velocities、done/reset、motion phase。
2. 用 `build_paper_hybrid_state_window` 生成 99-D yaw-centric hybrid state 或 163-D projected state。
3. 严格过滤 done/reset discontinuity，并实现/记录 5s acceptance rejection、OU perturbation、symmetry augmentation。
4. 重新生成 state-latent dataset 后再允许 VAE/diffusion/guidance 训练。

## Git Commit

待提交。
