# Progress Update

## Goal

审查 BeyondMimic teacher/RL -> VAE -> diffusion -> guidance 控制链为什么在 single-leg、walk、jump 视频里退化成前倾/默认站姿，并确认是否可以继续生成成功版 MuJoCo VAE/diffusion/guidance 视频。

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
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `download/dependencies/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
- `mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.json`
- strict teacher eval JSONs for model 250, 500, and relaxed model 1000.

## Files Modified

- `reproduction/scripts/model_chain_teacher_gate_root_cause_audit.py`
- `reproduction/scripts/artifact_manifest.py`

## Commands Run

```bash
python3 reproduction/scripts/model_chain_teacher_gate_root_cause_audit.py
BM_HUB_SINGLELEG_EVAL_RUN_TAG=rootxy0_model500_strict_20260624_213700 \
  BM_HUB_SINGLELEG_EVAL_TARGET_GPUS=1 \
  BM_HUB_SINGLELEG_EVAL_CHECKPOINT=/mnt/infini-data/test/BeyondMimic/res/runs/hub_singleleg_paper_contract_ppo_training_rootxy0_strict_env16384_20260624_131200/resource_adjusted_ppo_20260624_125749_seed20260911/rank_0/model_500.pt \
  BM_HUB_SINGLELEG_EVAL_MOTION_NPZ=/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_short_motion_recentered_bundle/motions/hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz \
  BM_HUB_SINGLELEG_EVAL_NUM_ENVS=256 \
  BM_HUB_SINGLELEG_EVAL_STEPS=799 \
  envs/bm_tracking/bin/python reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py
```

## Results

- Official IsaacLab action contract was confirmed: `processed_action = raw_action * scale + offset`; clipping is optional and only applies if configured.
- Official `whole_body_tracking` public code contract was confirmed: policy obs contains command, anchor error, base velocity, joint position/velocity, and last action; reward terms and PPO config match the public Stage-1 tracking recipe.
- `rootxy0_model500_strict` loaded the intended checkpoint and failed the local teacher quality gate:
  - loaded iteration: `500`
  - reward mean: `0.04134666047598677`
  - local non-timeout done rate: `0.2632294274092616`
  - body position error mean: `0.13441340681235395`
  - joint position error mean: `0.882636540150911`
  - `ee_body_pos` termination mean per eval step: `67.38673341677097`
- Existing `rootxy0_model250_strict` also failed:
  - reward mean: `0.04151430909751503`
  - local non-timeout done rate: `0.27369661295369213`
- Existing relaxed model 1000 strict eval also failed:
  - reward mean: `0.03907994591475775`
  - local non-timeout done rate: `0.2650065104166667`
- New root-cause audit status: `blocked_by_teacher_quality_and_native_adapter_gap`.

## Verification

New audit outputs:

- `res/model_chain_teacher_gate_root_cause/model_chain_teacher_gate_root_cause_audit.json`
- `res/model_chain_teacher_gate_root_cause/model_chain_teacher_gate_root_cause_audit.tsv`

The audit explicitly sets `downstream_video_generation_allowed=false` because strict teacher gate and native MuJoCo adapter gate are not both passing.

## Failed / Blocked Items

- Stage-1 strict single-leg teacher gate still fails at model 500.
- Native MuJoCo PPO/VAE/diffusion observation-action adapter remains incomplete according to `mujoco_ppo_adapter_gap_audit.json`.
- Therefore, generating success-claimed VAE/diffusion/guidance videos from the current teacher remains blocked. Existing videos may stay as diagnostics, not success evidence.

## Effect on English Reading Report

This improves the report honesty boundary. The report can now explain that the public formulas and official tracking code were rechecked, but the current failure is upstream teacher quality plus MuJoCo adapter parity, not a verified paper-level BeyondMimic controller. This supports a rigorous limitation section instead of overclaiming failed videos.

## Next Step

Continue monitoring later strict checkpoints, especially model 750/1000+. Only if a checkpoint passes the local teacher gate should we collect teacher rollouts and regenerate VAE/diffusion/guidance videos. If later checkpoints plateau, the next technical step is a Stage-1 training adjustment around `ee_body_pos` curriculum/termination while preserving a strict final eval.

## Git Commit

Pending after verification.
