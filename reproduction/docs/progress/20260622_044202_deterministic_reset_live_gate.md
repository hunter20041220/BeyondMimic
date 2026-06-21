# Progress Update

## Goal

Move the tracking work back toward the paper reproduction mainline by testing whether the current robot-order FK tracking blocker can be resolved by a deterministic reset distribution before launching another full PPO run. The specific gate was: if a small live IsaacLab probe improves both policy-step done rate and post-step joint-velocity transient, then the next step can be a full same-seed checkpoint eval and then full PPO; otherwise do not promote the reset shortcut to full training.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic/robot_order_fk_ppo_tracking_quality_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_target_refresh_no_advance_live_probe/robot_order_fk_reset_target_refresh_no_advance_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_distribution_diagnostic/robot_order_fk_reset_state_action_distribution_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/managers/command_manager.py`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_deterministic_reset_live_probe.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`.
- Added this progress file.

## Commands Run

- `python3 -m py_compile reproduction/scripts/robot_order_fk_deterministic_reset_live_probe.py`
- `envs/bm_analysis/bin/python reproduction/scripts/robot_order_fk_deterministic_reset_live_probe.py`
- `jq '{status, metrics, checks, interpretation, outputs}' res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.json`
- Pending after this file: regenerate `paper_vs_reproduction`, `final_reproduction_report`, `artifact_manifest`, `reproduction_master_audit`, completion/status audits, and verification command audits.

## Results

The new live gate wrote:

- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe_worker.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe_worker_metrics.json`
- `/mnt/infini-data/test/BeyondMimic/logs/tracking_robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.log`

Key metrics:

- Official target-refresh policy done rate: `0.33203125`.
- Official target-refresh post-step joint-velocity error: `15.347245216369629`.
- Deterministic reset-target refresh done rate: `0.453125`.
- Deterministic reset-target refresh joint-velocity error: `11.510969161987305`.
- Deterministic-vs-official done-rate delta: `+0.12109375`.
- Deterministic-vs-official joint-velocity delta: `-3.836276054382324`.
- Deterministic motion-state done rate: `0.55078125`.
- Deterministic motion-state joint-velocity error: `14.773418426513672`.
- `recommended_full_eval_variant=""`.
- `any_variant_improves_done_and_joint_velocity=false`.

Interpretation: deterministic reset reduces the joint-velocity transient but worsens policy-step done rate. It should not be promoted to a full checkpoint evaluation, full PPO training run, teacher rollout dataset, VAE, diffusion, or guidance chain.

## Verification

The live IsaacLab worker returned zero and the JSON reports:

- `worker_status_ok=true`
- `checkpoint_loaded=true`
- `uses_official_importer_export_usd=true`
- `uses_robot_order_fk_repaired_bundle=true`
- `all_modes_zero_and_policy_tested=true`
- `does_not_train=true`
- `does_not_claim_paper_level_tracking=true`
- `does_not_claim_real_robot=true`

Full repository verification is still pending after registering this progress file and rerunning the audit scripts.

## Failed / Blocked Items

- No full eval was launched from deterministic reset because the gate did not improve both done rate and joint velocity.
- No PPO training was launched. This is intentional: the current evidence says deterministic reset is a harmful shortcut for termination even though it improves velocity transient.
- The main non-robot blocker remains tracking teacher quality, especially `ee_body_pos`/endpoint termination semantics and reset target consistency.

## Effect on English Reading Report

This strengthens the report's engineering narrative. It shows the project is not only accumulating failure audits; it is using live IsaacLab gates to decide whether a candidate fix deserves full training resources. The result supports the honest report claim: the project has a local virtual BeyondMimic-like pipeline and strong public-resource evidence, but it does not yet have a paper-level tracking teacher.

## Next Step

Repair the endpoint/termination side rather than the reset-randomization shortcut. The next high-value gate should compare official `ee_body_pos` z-only termination against a more diagnostic full-body/phase-aware endpoint condition, while keeping the same robot-order FK bundle and checkpoint. Only if that gate reduces done rate without hiding genuine tracking failure should a full same-seed eval and then full PPO be scheduled.

## Git Commit

This progress update is included in the round commit. The final commit hash is reported in the user-facing update after the commit is created.
