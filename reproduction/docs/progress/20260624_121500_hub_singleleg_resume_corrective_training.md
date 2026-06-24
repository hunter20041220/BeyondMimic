# Progress Update

## Goal

排查当前 teacher/VAE/diffusion/guidance 视频退化成前倾站姿的问题，并在不继续盲训 downstream 的前提下，启动一个可审计的 Hub Single Leg Balance Stage-1 teacher 纠偏训练线。

## Files Read

- `goal.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/paper/source/root.tex`
- `reproduction/paper/source/tex/method.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `envs/bm_tracking/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
- `res/audits/single_motion_teacher_quality_gate/single_motion_teacher_quality_gate_audit.json`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.json`

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py`
- `reproduction/docs/progress/20260624_121500_hub_singleleg_resume_corrective_training.md`

## Commands Run

```bash
envs/bm_tracking/bin/python -m py_compile \
  reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py \
  reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py

BM_HUB_SINGLELEG_RUN_TAG=resume_smoke_20260624_115934 \
BM_HUB_SINGLELEG_TARGET_GPUS=5,6 \
BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=512 \
BM_HUB_SINGLELEG_MAX_ITERATIONS=1 \
BM_HUB_SINGLELEG_SAVE_INTERVAL=1 \
BM_HUB_SINGLELEG_RESUME_CHECKPOINT=/mnt/infini-data/test/BeyondMimic/res/runs/hub_singleleg_paper_contract_ppo_training/resource_adjusted_ppo_20260623_194051_seed20260911/rank_0/model_250.pt \
BM_HUB_SINGLELEG_RESUME_LOAD_OPTIMIZER=1 \
BM_EE_BODY_POS_TRAIN_THRESHOLD=0.5 \
envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py

BM_HUB_SINGLELEG_RUN_TAG=corrective_resume_relaxed4096_foreground_20260624_121000 \
BM_HUB_SINGLELEG_TARGET_GPUS=5,6 \
BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=4096 \
BM_HUB_SINGLELEG_MAX_ITERATIONS=5000 \
BM_HUB_SINGLELEG_SAVE_INTERVAL=250 \
BM_HUB_SINGLELEG_RESUME_CHECKPOINT=/mnt/infini-data/test/BeyondMimic/res/runs/hub_singleleg_paper_contract_ppo_training/resource_adjusted_ppo_20260623_194051_seed20260911/rank_0/model_250.pt \
BM_HUB_SINGLELEG_RESUME_LOAD_OPTIMIZER=1 \
BM_EE_BODY_POS_TRAIN_THRESHOLD=0.5 \
envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py
```

## Results

- 新增 `BM_RESUME_CHECKPOINT` 支持：PPO worker 会调用 RSL-RL `runner.load(..., load_optimizer=True)`，并在 `training_metrics.json` 中记录 resume checkpoint、loaded iteration、optimizer 是否加载。
- 新增 `BM_HUB_SINGLELEG_EVAL_CHECKPOINT` / `BM_HUB_SINGLELEG_EVAL_RUN_TAG` 支持：Hub single-leg eval wrapper 可以评估指定 checkpoint，避免继续误读旧 `model_99.pt`。
- resume smoke 成功：日志中出现 `resume_loaded=...model_250.pt:iter=250`，训练从 `Learning iteration 250/251` 继续，mean reward 约 `0.36`。
- 正在运行的 corrective 训练：
  - tag：`corrective_resume_relaxed4096_foreground_20260624_121000`
  - GPUs：physical 5 and 6
  - envs：4096 per rank, 8192 total
  - resume checkpoint：`res/runs/hub_singleleg_paper_contract_ppo_training/resource_adjusted_ppo_20260623_194051_seed20260911/rank_0/model_250.pt`
  - endpoint threshold：`BM_EE_BODY_POS_TRAIN_THRESHOLD=0.5`
  - 当前状态：训练已超过 `Learning iteration 646/5250`，并已写出 `model_500.pt`
  - 当前观察：mean reward 约 `0.38-0.41`，body position error 约 `0.240`，joint position error 约 `3.02`，说明还没有学到可展示的抬腿关节姿态。

## Verification

- `py_compile` 通过。
- resume smoke 通过，并写出：
  - `res/tracking/hub_singleleg_paper_contract_ppo_training_run_resume_smoke_20260624_115934/tracking_hub_singleleg_paper_contract_ppo_training_run.json`
  - `logs/tracking_hub_singleleg_paper_contract_ppo_training_run_resume_smoke_20260624_115934/tracking_g1_resource_adjusted_ppo_training_run.log`
- 长训仍在运行，尚未完成最终验证；必须等 checkpoint 后用 strict/default endpoint gate 评估。
- `tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py` 语法检查通过；由于 5/6 卡正在训练，暂未抢占 GPU 做 strict eval。

## Failed / Blocked Items

- 这条训练使用 relaxed endpoint threshold `0.5`，只能作为 local corrective/curriculum line，不能称为 paper-level teacher。
- 当前 Stage-1 strict teacher gate 仍未通过。
- MuJoCo native IsaacLab 160-D observation adapter 仍未完成。
- 当前 VAE/diffusion/guidance 不能继续使用旧 weak teacher 或旧 policy-obs-derived state-latent dataset 宣称成功。

## Effect on English Reading Report

这轮提供了一个清晰的失败诊断：当前坏视频不是单纯渲染问题，而是 Stage-1 teacher quality、MuJoCo native adapter、state-latent dataset contract 三层同时未过。报告可以把本轮作为“why the first closed-loop videos failed and how the corrective line was started”的证据。

## Next Step

1. 等待 `corrective_resume_relaxed4096_foreground_20260624_121000` 产生新 checkpoint。
2. 用 `BM_HUB_SINGLELEG_EVAL_CHECKPOINT=<model_500_or_later.pt>` 做 strict/default endpoint teacher evaluation，不直接生成成功视频。
3. 如果 strict gate 仍失败，检查 joint-error 高的原因：reference joint order/default pose/action scale/termination curriculum，而不是继续训练 VAE/diffusion。
4. 只有 teacher 通过 gate 后，才重新采集 continuous accepted rollout，重建 99-D hybrid state-latent dataset，再训练 VAE/diffusion/guidance。

## Git Commit

Pending after verification.
