# Progress Update

## Goal

修复 Stage-1 teacher 训练链路里会直接污染 PPO 质量的 PhysX GPU buffer 配置问题，为后续重新训练可信 single-leg / LAFAN1 teacher 做准备。当前目标仍是让 teacher/RL、VAE、diffusion、guidance 最终能通过真实闭环 action-control 学到 reference 动作；本轮不是生成成功视频。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/data/motion_file_manifest.csv`
- `/mnt/infini-data/test/BeyondMimic/report/video_index.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/launch_stage1_singleleg_robot_order_training.sh`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/launch_stage1_singleleg_robot_order_training.sh`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260625_085641_stage1_physx_buffer_teacher_gate.md`

## Commands Run

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'tracking_|rsl_rl|whole_body_tracking|resource_adjusted|singleleg|stage1'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
python3 -m py_compile reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py
bash -n reproduction/scripts/launch_stage1_singleleg_robot_order_training.sh
BM_HUB_SINGLELEG_MAX_ITERATIONS=1 BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=4096 BM_HUB_SINGLELEG_TARGET_GPUS=4,7 ... python -u reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py
```

## Results

- 当前没有正在运行的 Stage-1 PPO 训练进程。
- 8 张 H20 中 GPU 4/5/6/7 在本轮开始时基本空闲；GPU 3 有约 5GB 非本轮占用。
- 共享 PPO training/eval worker 现在支持并记录以下 PhysX buffer 环境变量：
  - `BM_PHYSX_GPU_MAX_RIGID_CONTACT_COUNT`
  - `BM_PHYSX_GPU_MAX_RIGID_PATCH_COUNT`
  - `BM_PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY`
  - `BM_PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY`
  - `BM_PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY`
  - `BM_PHYSX_GPU_COLLISION_STACK_SIZE`
  - `BM_PHYSX_GPU_HEAP_CAPACITY`
  - `BM_PHYSX_GPU_TEMP_BUFFER_CAPACITY`
- single-leg robot-joint-order launcher 默认加入高 buffer 设置，并默认使用 GPU 5,6。
- 4096 env/rank、2 rank、1 iteration probe 完成：
  - JSON: `/mnt/infini-data/test/BeyondMimic/res/tracking/hub_singleleg_paper_contract_ppo_training_run_rootxy0_robotjoint_physxbuf_env4096_probe1_20260625_085343/tracking_hub_singleleg_paper_contract_ppo_training_run.json`
  - log: `/mnt/infini-data/test/BeyondMimic/logs/tracking_hub_singleleg_paper_contract_ppo_training_run_rootxy0_robotjoint_physxbuf_env4096_probe1_20260625_085343/tracking_g1_resource_adjusted_ppo_training_run.log`
  - run dir: `/mnt/infini-data/test/BeyondMimic/res/runs/hub_singleleg_paper_contract_ppo_training_rootxy0_robotjoint_physxbuf_env4096_probe1_20260625_085343/resource_adjusted_ppo_20260625_005343_seed20260952`
- Probe status: `ok_hub_singleleg_paper_contract_ppo_training_completed`.
- Probe confirmed both ranks received `gpu_max_rigid_patch_count=1048576` and other buffer overrides.
- Probe log did not contain `Patch buffer overflow`.
- 32768 env/rank、2 rank、1 iteration probe also completed on GPU 5/6:
  - JSON: `/mnt/infini-data/test/BeyondMimic/res/tracking/hub_singleleg_paper_contract_ppo_training_run_rootxy0_robotjoint_physxbuf_env32768_probe1_20260625_090243/tracking_hub_singleleg_paper_contract_ppo_training_run.json`
  - log: `/mnt/infini-data/test/BeyondMimic/logs/tracking_hub_singleleg_paper_contract_ppo_training_run_rootxy0_robotjoint_physxbuf_env32768_probe1_20260625_090243/tracking_g1_resource_adjusted_ppo_training_run.log`
  - run dir: `/mnt/infini-data/test/BeyondMimic/res/runs/hub_singleleg_paper_contract_ppo_training_rootxy0_robotjoint_physxbuf_env32768_probe1_20260625_090243/resource_adjusted_ppo_20260625_010244_seed20260953`
  - peak GPU memory from telemetry: about `25.9 GB/card`
  - log marker check: no `Patch buffer overflow`

## Verification

- Python syntax check passed for the modified training/eval wrappers.
- Shell syntax check passed for `launch_stage1_singleleg_robot_order_training.sh`.
- The 4096 env/rank smoke training generated one checkpoint on rank 0 and metrics JSON for rank 0/rank 1.
- The 32768 env/rank smoke training generated one checkpoint on rank 0 and metrics JSON for rank 0/rank 1.
- These were probes only; the larger probe peaked at about 25.9GB/card, so neither is a formal 80GB/card high-throughput PPO training run.

## Failed / Blocked Items

- The previous corrected 32768 env/rank single-leg training used robot-joint-order motion, but its logs contained many PhysX `Patch buffer overflow` errors and strict eval did not pass reward quality gate. It should not be used as a downstream teacher.
- A background 65536 env/rank probe launched earlier produced an empty log and no JSON, so it is not valid evidence. It should be rerun foreground or through a monitored launcher if the next run needs to push closer to the requested 80GB/card target.
- Teacher quality remains the main blocker. VAE/diffusion/guidance should not be retrained from weak teacher rollouts.

## Effect on English Reading Report

This update strengthens the reproduction audit: it documents that poor learned-control videos are being traced back to Stage-1 teacher training quality and simulation-buffer correctness, rather than being hidden by better rendering or reference replay. The report should say this is an engineering repair step before claiming any learned single-leg/jumps control.

## Next Step

1. Rerun a monitored 32768 or 65536 env/rank high-buffer probe and confirm no `Patch buffer overflow`.
2. Start a fresh high-buffer single-leg teacher training only after that probe passes.
3. After teacher checkpoint passes reward/body/joint/action gates, collect continuous accepted teacher rollouts, then retrain VAE/diffusion/guidance.
4. Keep reference/replay videos separate from learned action-control videos.

## Git Commit

Not committed in this progress file at creation time.
