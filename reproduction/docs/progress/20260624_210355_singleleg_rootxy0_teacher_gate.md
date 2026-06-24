# Progress Update

## Goal

修复当前模型链“teacher/VAE/diffusion 都前倾、没有学到单腿站立姿态”的上游问题。本轮不生成成功视频；重点是确认 teacher 评估是否可信、motion 输入是否存在明显偏移混淆，并启动一条更可信的 single-leg teacher 训练线。

## Files Read

- `reproduction/paper/source/root.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`
- `res/tracking/stage1_multisource_motion_bundle/validate_motion_npz_contract_summary.json`
- existing MuJoCo/video summaries under `mujoco_mp4/res/`, `official_mp4/res/`, and `res/visualization/`

## Files Modified

- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
  - Added `BM_HUB_SINGLELEG_MOTION_NPZ` so the teacher training wrapper can use an audited recentered motion work copy.
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py`
  - Fixed checkpoint override so `BM_HUB_SINGLELEG_EVAL_CHECKPOINT` truly controls the checkpoint loaded by the shared eval harness.
  - Added `BM_HUB_SINGLELEG_EVAL_MOTION_NPZ` for matching eval to a recentered motion.
- `reproduction/scripts/tracking_stage1_short_motion_recenter_audit.py`
  - New script that writes root-XY-recentered work copies for short motions.

## New Results

- `res/tracking/stage1_short_motion_recentered_bundle/tracking_stage1_short_motion_recenter_audit.json`
- `res/tracking/stage1_short_motion_recentered_bundle/tracking_stage1_short_motion_recenter_audit.tsv`
- `res/tracking/stage1_short_motion_recentered_bundle/motions/hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz`
- `res/tracking/stage1_short_motion_recentered_bundle/motions/hub_swallow_balance_video_swift0322_rootxy0/motion.npz`
- `res/tracking/stage1_short_motion_recentered_bundle/motions/hub_squat_video_squat_4_rootxy0/motion.npz`
- `res/tracking/stage1_short_motion_recentered_bundle/motions/hub_squat_video_squat_18_rootxy0/motion.npz`
- `res/tracking/stage1_short_motion_recentered_bundle/motions/zenodo_tkd_skill_rootxy0/motion.npz`
- `res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_model1000_strict_overridefix_20260624_124800/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json`
- `res/tracking/hub_singleleg_paper_contract_ppo_training_run_rootxy0_smoke_20260624_125900/tracking_hub_singleleg_paper_contract_ppo_training_run.json`
- `res/tracking/hub_singleleg_paper_contract_ppo_training_run_rootxy0_highmem_probe_env16384_iter1_20260624_130500/tracking_hub_singleleg_paper_contract_ppo_training_run.json`
- Active training output root:
  - `res/runs/hub_singleleg_paper_contract_ppo_training_rootxy0_strict_env16384_20260624_131200/`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_stage1_short_motion_recenter_audit.py \
  reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py \
  reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py

python3 reproduction/scripts/tracking_stage1_short_motion_recenter_audit.py

BM_HUB_SINGLELEG_EVAL_RUN_TAG=model1000_strict_overridefix_20260624_124800 \
BM_HUB_SINGLELEG_EVAL_TARGET_GPUS=4 \
BM_HUB_SINGLELEG_EVAL_CHECKPOINT=.../model_1000.pt \
BM_HUB_SINGLELEG_EVAL_NUM_ENVS=256 \
BM_HUB_SINGLELEG_EVAL_STEPS=480 \
envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py

BM_HUB_SINGLELEG_RUN_TAG=rootxy0_smoke_20260624_125900 \
BM_HUB_SINGLELEG_TARGET_GPUS=4,7 \
BM_HUB_SINGLELEG_MOTION_NPZ=.../hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz \
BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=512 \
BM_HUB_SINGLELEG_MAX_ITERATIONS=1 \
BM_HUB_SINGLELEG_SAVE_INTERVAL=1 \
envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py

BM_HUB_SINGLELEG_RUN_TAG=rootxy0_highmem_probe_env16384_iter1_20260624_130500 \
BM_HUB_SINGLELEG_TARGET_GPUS=4,7 \
BM_HUB_SINGLELEG_MOTION_NPZ=.../hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz \
BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=16384 \
BM_HUB_SINGLELEG_MAX_ITERATIONS=1 \
BM_HUB_SINGLELEG_SAVE_INTERVAL=1 \
envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py

BM_HUB_SINGLELEG_RUN_TAG=rootxy0_strict_env16384_20260624_131200 \
BM_HUB_SINGLELEG_TARGET_GPUS=4,7 \
BM_HUB_SINGLELEG_MOTION_NPZ=.../hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz \
BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK=16384 \
BM_HUB_SINGLELEG_MAX_ITERATIONS=3000 \
BM_HUB_SINGLELEG_SAVE_INTERVAL=250 \
envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py
```

## Results

- The old strict eval wrapper was not trustworthy before this round: it recorded `BM_HUB_SINGLELEG_EVAL_CHECKPOINT`, but the shared base eval still selected an old checkpoint from the training summary.
- The eval override is now fixed and verified. The corrected strict eval loaded:
  - `model_1000.pt`
  - `loaded_iteration=1000`
- Corrected strict eval still failed:
  - `reward_mean=0.0390799459`
  - `error_body_pos_mean=0.1453408993`
  - `error_joint_pos_mean=0.8560752855`
  - `local_non_timeout_done_rate=0.2650065104`
  - `quality_gate.passed=false`
- The original Hub single-leg motion starts at nonzero global XY:
  - root XY offset approximately `(0.0283, 2.7947)`
- The recentered work copy starts at root XY `(0, 0)` while preserving relative motion.
- `rootxy0` smoke training succeeded and saved a checkpoint.
- `rootxy0` 16384 env/rank probe succeeded, using about 12-15 GB per H20 during rollout/learn.
- `rootxy0_strict_env16384_20260624_131200` is now running on GPUs 4 and 7.
- The previous relaxed training line on GPUs 5 and 6 is still running as a comparison line; it has not produced a usable teacher yet.

## Verification

Passed:

- Python syntax check for all modified/new scripts.
- Recenter audit status:
  - `ok_stage1_short_motion_recenter_audit`
- Rootxy0 smoke training status:
  - `ok_hub_singleleg_paper_contract_ppo_training_completed`
- Rootxy0 high-memory probe status:
  - `ok_hub_singleleg_paper_contract_ppo_training_completed`
- Corrected strict eval ran and loaded the intended checkpoint.

Not yet passed:

- Single-leg teacher quality gate.
- VAE closed-loop single-leg gate.
- Diffusion closed-loop single-leg gate.
- Guided single-leg video gate.

## Failed / Blocked Items

- The old `model_1000.pt` strict eval failed and must not be used for VAE/diffusion training.
- The previous non-recentered single-leg line appears to plateau around weak teacher behavior.
- Current rootxy0 strict line is promising but still has high `ee_body_pos` termination early in training.
- No successful single-leg teacher/VAE/diffusion/guidance video was produced this round.

## Effect on English Reading Report

This round improves the report's reproducibility story by separating three things:

1. Formula/code alignment: the public `whole_body_tracking` Stage-1 observation, reward, action, and termination contract is being used.
2. Evaluation correctness: checkpoint selection had a real bug and has been fixed.
3. Data preprocessing: short motions with large global XY offsets need audited recentered work copies before teacher training can be judged fairly.

The report should state that the project is actively repairing Stage-1 teacher quality and must not present downstream VAE/diffusion videos as successful until this teacher gate passes.

## Next Step

1. Let `rootxy0_strict_env16384_20260624_131200` reach at least `model_250.pt`.
2. Run strict checkpoint eval with:
   - `BM_HUB_SINGLELEG_EVAL_CHECKPOINT=<rootxy0 model_250.pt>`
   - `BM_HUB_SINGLELEG_EVAL_MOTION_NPZ=<rootxy0 motion.npz>`
3. If strict eval passes, collect clean teacher rollout.
4. Only then retrain VAE and diffusion and generate single-leg videos.

## Git Commit

Pending. Commit after verification and artifact manifest refresh.
