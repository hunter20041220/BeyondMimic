# Progress Update

## Goal

按新的主线重新启动 LAFAN1 `walk1_subject1` 单动作 Stage-1 teacher policy 训练，为后续 `teacher rollout dataset -> DAgger VAE -> state-latent diffusion -> test-time guidance -> MuJoCo control videos` 提供更可信的 teacher。当前不再把旧的全 motion / single-leg / jumps 坏视频当作成功结果。

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py`
- `res/tracking/stage1_multisource_motion_bundle_robot_joint_order/motions/lafan1_walk1_subject1/motion.npz`

## Files Modified

- `reproduction/scripts/monitor_walk1_teacher_training.sh`
- `reproduction/scripts/eval_walk1_teacher_checkpoint.sh`
- `reproduction/docs/progress/20260626_105705_walk1_only_teacher_training_started.md`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
ps -eo pid,ppid,stat,etime,cmd | rg 'tracking_hub|lafan|jumps|walk|torch\.distributed|rsl_rl|resource_adjusted_ppo'
tmux ls
ls -lh res/tracking/stage1_multisource_motion_bundle_robot_joint_order/motions/lafan1_walk1_subject1/motion.npz
```

启动命令等价于：

```bash
BM_HUB_SINGLELEG_RUN_TAG=lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951 \
BM_HUB_SINGLELEG_TARGET_GPUS=5,6 \
BM_HUB_SINGLELEG_MOTION_NPZ=/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle_robot_joint_order/motions/lafan1_walk1_subject1/motion.npz \
BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=40960 \
BM_HUB_SINGLELEG_MAX_ITERATIONS=30000 \
BM_HUB_SINGLELEG_SAVE_INTERVAL=500 \
BM_HUB_SINGLELEG_SEED=20261051 \
BM_REFRESH_MOTION_COMMAND_TARGETS=1 \
BM_PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=16777216 \
BM_PHYSX_GPU_MAX_RIGID_PATCH_COUNT=1048576 \
BM_PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=4194304 \
BM_PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=33554432 \
BM_PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=4194304 \
BM_PHYSX_GPU_COLLISION_STACK_SIZE=134217728 \
BM_PHYSX_GPU_HEAP_CAPACITY=134217728 \
BM_PHYSX_GPU_TEMP_BUFFER_CAPACITY=33554432 \
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python -u reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py
```

## Results

- 新训练会话：`bm_walk1_teacher_repaired_env40960_iter30000_20260626_104951`
- Run tag：`lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951`
- Run root：`res/runs/hub_singleleg_paper_contract_ppo_training_lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951/resource_adjusted_ppo_20260626_024951_seed20261051`
- Log：`logs/tracking_hub_singleleg_paper_contract_ppo_training_run_lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951/tracking_g1_resource_adjusted_ppo_training_run.log`
- Motion：`lafan1_walk1_subject1`，来自 robot-joint-order 修复 bundle，`motion.npz` 约 23 MB，约 261.3 秒。
- 当前已通过 AppLauncher、PhysX buffer override、Tracking-Flat-G1-v0 env startup，并进入 PPO learning iteration。
- 40960 env/rank、2 rank 时 GPU 5/6 显存约 `34.4 GB / 34.5 GB`，满足本轮希望每卡约 30GB 的高吞吐目标。
- 初始迭代仍然很弱：到 iteration 4/30000 时 `Mean reward` 仍为负，`Mean episode length` 约 12，`error_joint_pos` 约 2.12，说明这只是刚开始训练，不可作为 downstream teacher。

## Verification

- 已确认 5/6 GPU 在启动前空闲。
- 已确认训练 worker 进程存活，`torch.distributed.run --nproc_per_node=2` 正在使用 GPU 5/6。
- 已确认日志包含：
  - `BM_SENTINEL:rank=0:after_app`
  - `BM_SENTINEL:rank=1:after_app`
  - `physx_buffer_overrides`
  - `Learning iteration 4/30000`
- 已新增 `monitor_walk1_teacher_training.sh` 用于只读监控当前 walk1 teacher。
- 已新增 `eval_walk1_teacher_checkpoint.sh`，用于 checkpoint 出现后按同一个 walk1 motion 做质量筛选。

## Failed / Blocked Items

- 当前还没有可用 teacher checkpoint。`model_0.pt` 只是初始化/早期权重，不可用于 VAE、diffusion 或视频。
- VAE 训练、state-latent diffusion 训练、guidance、joystick/waypoint/obstacle/inpainting MuJoCo 控制视频均仍 blocked by teacher quality。
- 当前不得把旧 `reference_action_control.mp4`、`teacher_policy_action_control.mp4`、`vae_reconstructed_action_control.mp4`、`diffusion_denoised_latent_action_control.mp4`、`guided_latent_action_control.mp4`、`guided_vs_unguided_action_control.mp4` 写成成功结果；旧版本已被判定包含跳变拼接或 teacher 质量不足问题。

## Effect on English Reading Report

本轮为报告提供了一个更诚实的实验叙述：旧 local video 结果质量差不是简单渲染问题，而是 Stage-1 teacher 质量不足导致 downstream VAE/diffusion/guidance 学不到完整 walking 姿态。新的 walk1-only line 是一个单动作隔离实验，用于验证整条 BeyondMimic-like pipeline 的实现正确性，但它仍不是 paper-level BeyondMimic full teacher。

## Next Step

1. 继续监控 walk1 teacher training。
2. `model_500.pt` 及以后 checkpoint 出现后，运行 walk1 quality eval。
3. 只有当 teacher eval 的 reward/body/joint/done gate 合理时，才进入 teacher rollout dataset collection。
4. 使用新的连续 walk1 teacher rollout 训练 DAgger VAE。
5. 再用 VAE latent 构建 state-latent trajectory dataset，训练 Transformer diffusion denoiser。
6. 最后生成 walk1-only 的 VAE / diffusion / guided / guided-vs-unguided MuJoCo action-control videos，并明确 claim level。

## Git Commit

Pending.
