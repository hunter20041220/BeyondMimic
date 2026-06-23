# Progress Update

## Goal

在继续任何 teacher/VAE/diffusion/guidance 长训练前，审计当前 state-latent trainable dataset 是否真的来自 BeyondMimic 论文定义的 hybrid state-latent trajectory，而不是误用 Stage-1 PPO policy observation。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_072430_hybrid_state_schema_contract.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/res/audits/hybrid_state_schema_contract/beyondmimic_hybrid_state_schema_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Commands Run

- `python3 reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `python3 - <<'PY' ... inspect state-latent audit summary ... PY`
- `git status --short`

## Results

新增审计输出：

- `/mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.md`

审计状态：`blocked_state_latent_dataset_source_uses_policy_obs_and_missing_rollout_state`。

关键数据：

- 当前 paper-contract dataset `state_source`: `policy_obs in local paper-contract best-teacher rollout shards`
- 当前 obs/state dim: `160`
- 当前 token dim: `192`
- 论文 hybrid state dim: `99`
- 本地 corrected projected state dim: `163`
- 加 32-D latent 后 token 应为 `131` 或 `195`
- teacher rollout source shards `done_count`: `47200`
- source shards 缺少构造论文 hybrid state 所需的 root/body world-state fields。

## Verification

本轮新增 audit 已成功生成，并接入 artifact manifest、final reproduction report 和 master audit 的检查项。完整验证命令将在本轮收尾前统一运行。

## Failed / Blocked Items

当前仍 blocked：

- 不能使用现有 `policy_obs` state-latent dataset 继续长 VAE/diffusion/guidance 训练；
- teacher rollout collection 尚未保存 `root_pos_w/root_quat_w/root_lin_vel_w/root_ang_vel_w/body_pos_w/body_lin_vel_w`；
- state-latent window builder 尚未实现 done/reset discontinuity filter 和 5s rejection；
- OU perturbation metadata 和 sagittal symmetry augmentation 尚未作为 trainable dataset protocol 固化；
- diffusion training scripts 仍读取 `source_shard["policy_obs"]`。

## Effect on English Reading Report

这轮为报告提供了一个关键诚实结论：当前视频姿态差不应简单归因于“训练不够久”，而是下游数据源契约本身不符合论文 state-latent trajectory 定义。报告应把现有 VAE/diffusion/guidance 结果标为 local diagnostic / blocked reproduction，而不是 paper-level closed-loop result。

## Next Step

修 teacher rollout collector：在 IsaacLab/whole_body_tracking rollout 时保存 raw root/body world state 和 velocities，然后重建连续 accepted 99-D hybrid-state windows，再重新训练 VAE/diffusion。

## Git Commit

Pending.
