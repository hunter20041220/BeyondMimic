# Progress Update

## Goal

Return to the tracking mainline instead of adding more downstream proxy results. This round tested whether the current
robot-order FK tracking bottleneck can be fixed by reset-time command phase alignment before launching another full PPO
run.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/prompt06211658.txt`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic/robot_order_fk_ppo_tracking_quality_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_distribution_diagnostic/robot_order_fk_reset_state_action_distribution_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_termination_alignment_audit/robot_order_fk_reset_termination_alignment_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_reset_state_action_consistency_live_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_deterministic_reset_live_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_phase_alignment_live_probe.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Added this progress update.

## Commands Run

- `python3 -m py_compile reproduction/scripts/robot_order_fk_phase_alignment_live_probe.py`
- `BM_PHASE_ALIGNMENT_GPU=4 BM_PHASE_ALIGNMENT_NUM_ENVS=256 BM_PHASE_ALIGNMENT_SEED=20260762 envs/bm_analysis/bin/python reproduction/scripts/robot_order_fk_phase_alignment_live_probe.py`

## Results

The live probe completed successfully inside IsaacLab on physical GPU 4 with 256 environments, the official-importer G1
USDA, the robot-order FK-repaired public-motion bundle, and the latest local robot-order FK PPO checkpoint.

Output paths:

- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_phase_alignment_live_probe/robot_order_fk_phase_alignment_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_phase_alignment_live_probe/robot_order_fk_phase_alignment_live_probe.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_phase_alignment_live_probe/robot_order_fk_phase_alignment_live_probe.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_phase_alignment_live_probe/robot_order_fk_phase_alignment_live_probe_worker_metrics.json`
- `/mnt/infini-data/test/BeyondMimic/logs/tracking_robot_order_fk_phase_alignment_live_probe/robot_order_fk_phase_alignment_live_probe.log`

Key metrics:

- Baseline no-advance target refresh policy done rate: `0.38671875`.
- Baseline no-advance target refresh post-step joint-velocity error: `11.769760131835938`.
- `compute_dt0` reduced joint velocity to `11.086766242980957` but worsened done rate to `0.49609375`.
- manual `_update_command()` reduced joint velocity to `10.390115737915039` but worsened done rate to `0.5390625`.
- `phase_plus_1_target_only` reduced joint velocity to `6.945001602172852` but worsened done rate to `0.66015625`.
- No variant improved both done rate and joint velocity.
- Recommended full-eval variant: empty.

## Verification

Immediate verification:

- Script syntax compile passed.
- The live probe returned status `ok_robot_order_fk_phase_alignment_live_probe`.
- The worker returned zero and all zero/policy action variants were tested.
- The output explicitly keeps `does_not_claim_paper_level_tracking=true`, `does_not_claim_goal_complete=true`, and `does_not_train=true`.

Full project verification is run after this file so the new row and artifacts enter the manifest, comparison table, final report, and master audit.

## Failed / Blocked Items

This was a successful live diagnostic but a negative technical result. Reset-time phase alignment alone is not a safe
candidate for full checkpoint evaluation or PPO retraining because every variant that reduced joint-velocity error
worsened policy-step done rate. The tracking mainline is therefore still blocked by reset/target/termination semantics,
especially endpoint z termination and the stale-target/order of command update relative to termination computation.

## Effect on English Reading Report

This strengthens the reproduction narrative by showing that the project did not blindly extend training after a weak
teacher. It source-links the current tracking failure to IsaacLab step ordering and official `MotionCommand` behavior,
then records a concrete negative live experiment. The reading report can describe this as a reproducibility insight:
public-code recovery is not enough if reset/command/termination contracts are misaligned in the local asset/motion path.

## Next Step

Do not launch the next full PPO run from this phase-alignment variant. The next mainline step should inspect and patch
termination/body-target semantics more directly, likely by testing a reset-time command-target initialization hook or a
first-step termination-grace diagnostic before full eval. Only after a live gate improves both done rate and transient
consistency should the project proceed to full PPO, multi-seed eval, policy video, and downstream teacher rollout/VAE/
diffusion/guidance reruns.

## Git Commit

Pending after full verification and audit refresh.
