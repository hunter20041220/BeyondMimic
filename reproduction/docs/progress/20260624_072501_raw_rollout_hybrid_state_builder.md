# Progress Update

## Goal

本轮目标是先修正训练前最关键的数据公式契约：把 raw teacher rollout 的 root/body world-state 转成论文 S3 定义的 99-D hybrid yaw-centric state。该工作服务于后续重新采集 teacher rollout、重建 state-latent dataset、再训练 VAE/diffusion/guidance；本轮不启动长训练，也不声称视频效果已修好。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/state.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_paper_state_windows.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_reimpl_package_api.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/state.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_reimpl_package_api.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_072501_raw_rollout_hybrid_state_builder.md`

## Commands Run

```bash
PYTHONPATH=reproduction/src python3 reproduction/tests/test_core_math.py
PYTHONPATH=reproduction/src python3 reproduction/tests/test_reimpl_package_api.py
python3 -m py_compile reproduction/src/beyondmimic_reimpl/state.py reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py reproduction/tests/test_core_math.py reproduction/tests/test_reimpl_package_api.py
python3 reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py
```

## Results

- 新增 `build_paper_hybrid_state_window()`，输入连续 raw rollout 的 root/body world-frame state，输出论文 99-D hybrid state。
- 新增 `quat_to_matrix_array()`、`matrix_to_rot6d_array()`、`yaw_from_matrix_array()`，显式处理 `xyzw` 与 `wxyz` 四元数格式。
- 99-D state layout 仍为 root `15` + target-body local position/velocity `84`。
- Core math tests 从 `23` 行扩展到 `26` 行，全部通过。
- Package API tests 仍为 `8` 行，全部通过。
- State-latent source audit 现在显示 `row_count=10`、`pass_count=4`、`blocked_count=6`。

## Verification

- `core_math_unit_tests`: `ok`, rows `26`, failed `0`
- `reimpl_package_api_tests`: `ok`, rows `8`, failed `0`
- `state_latent_dataset_source_contract_audit`: `blocked_state_latent_dataset_source_uses_policy_obs_and_missing_rollout_state`

## Failed / Blocked Items

- 旧 paper-contract state-latent dataset 仍使用 `policy_obs` 160-D，token dim 192-D，不是论文 99-D hybrid state 或 163-D projected state。
- 旧 teacher rollout shards 仍缺少 raw root/body world-state 字段，无法回溯构造正确 trainable state-latent shards。
- diffusion training scripts 仍有读取 `source_shard["policy_obs"]` 的路径。
- OU perturbation、5s rejection、done/reset discontinuity 过滤、symmetry augmentation 还没有完整进入可训练数据协议。

## Effect on English Reading Report

这轮可以作为报告里的“复现审计纠错”证据：我们没有把前倾站姿视频包装成成功结果，而是确认了 VAE/diffusion 训练数据源与论文 state-latent 定义不一致，并补上了后续重建数据集必须使用的公式级 helper。

## Next Step

下一步应修改 state-latent dataset builder，使它从新采集的 raw teacher rollout shards 调用 `build_paper_hybrid_state_window()`，并拒绝所有跨 done/reset 的窗口；然后重新采集连续 accepted teacher rollout，重建 99-D/163-D state-latent dataset，再启动 VAE/diffusion 训练。

## Git Commit

待本轮验证全部完成后提交。
