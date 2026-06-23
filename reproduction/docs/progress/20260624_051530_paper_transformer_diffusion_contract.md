# Progress Update

## Goal

先停止把弱视频当作成功结果，继续按论文公式和补充材料审计模型链。本轮聚焦 Level-C state-latent diffusion：把旧 MLP denoiser 路径与论文的 Transformer diffusion 合同区分开，并新增一个 paper-contract Transformer dry-run gate。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_051530_paper_transformer_diffusion_contract.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py --dry-run-max-windows 4
python3 -m py_compile reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py reproduction/scripts/artifact_manifest.py reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
```

## Results

新增 paper-contract Transformer state-latent diffusion dry-run gate。该 gate 创建 6-layer Transformer、512 embedding、8 attention heads、20 denoising steps、separate state/latent denoising-step embeddings，并在当前 local paper-contract state-latent dataset 的 4 个窗口上完成一次 forward/backward。

输出：

- `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_transformer_state_latent_diffusion_training/paper_contract_transformer_state_latent_diffusion_training.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_transformer_state_latent_diffusion_training/paper_contract_transformer_state_latent_diffusion_training.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_transformer_state_latent_diffusion_training/paper_contract_transformer_state_latent_diffusion_training.md`

关键结果：

- status: `ok_paper_contract_transformer_diffusion_dry_run`
- parameter_count: `19142848`
- sequence_length: `21`
- obs_dim: `160`
- latent_dim: `32`
- token_dim: `192`
- dry-run train/val/test windows: `4/4/4`

## Verification

通过：

- Transformer diffusion script py_compile
- Transformer diffusion dry-run process returned zero
- paper_contract_architecture_checks_pass
- model-chain audit regeneration

仍不通过/仍 blocked：

- Transformer diffusion 只是 dry-run，不是 full training。
- test denoising MSE 在一次 dry-run 后没有意义，不能作为模型质量结论。
- state source 仍是 local `policy_obs`，不是论文完整 hybrid character-yaw state。
- teacher quality gate 未通过。
- guidance 仍是 offline cost-gradient audit，不是 receding-horizon closed-loop MuJoCo/Isaac control。

## Failed / Blocked Items

- 当前不得声称 teacher/VAE/diffusion 已学会 walk 或 single-leg。
- 当前不得把 clean-walk 或 single-leg 视频作为成功复现。
- 当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。

## Effect on English Reading Report

报告中可以新增一条诚实证据：本项目已把旧 MLP diffusion route 与论文 Transformer diffusion route 区分开，并建立了 paper-contract Transformer code gate。但必须写明它只是 architecture/gradient viability gate，不是 full diffusion training result，也不是 Fig.5/Fig.6 结果。

## Next Step

在启动任何长训练前，继续优先修 teacher quality gate 和 state construction gap；只有 teacher rollout 连续、稳定且姿态质量通过后，才值得 full train paper-contract VAE/diffusion。

## Git Commit

Pending.
