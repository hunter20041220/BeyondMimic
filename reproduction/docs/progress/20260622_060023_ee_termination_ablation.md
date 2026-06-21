# Progress Update

## Goal

Diagnose whether the current robot-order FK tracking bottleneck is dominated by the official z-only `ee_body_pos` endpoint termination after reset-target refresh. This round did not start PPO training and did not claim a paper-level tracking result.

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
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json`
- `res/tracking/robot_order_fk_reset_state_action_distribution_diagnostic/robot_order_fk_reset_state_action_distribution_diagnostic.json`
- `res/tracking/robot_order_fk_phase_alignment_live_probe/robot_order_fk_phase_alignment_live_probe.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/progress/20260622_060023_ee_termination_ablation.md`

## Commands Run

- `python3 -m py_compile reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.py`
- `BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_EE_ABLATION_NUM_ENVS=2048 BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_EE_ABLATION_SEED=20260721 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.py`

## Results

The ablation completed with status `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation_completed`.

Key same-seed results:

- Baseline robot-order FK eval done rate: `0.1782798129180602`.
- Reset-target-refresh/no-advance done rate: `0.22340745192307693`.
- Relaxed `ee_body_pos` termination done rate: `0.07152912050585285`.
- Relaxed ablation vs target-refresh done-rate delta: `-0.15187833141722407`.
- Relaxed ablation post-step0 done rate: `0.07176915111157718`.
- Manual original 0.25m endpoint-z violation proxy post-step0 rate: `0.4700460753984899`.

Interpretation: the official `ee_body_pos` endpoint termination is a dominant gate in the current weak-teacher evaluation, but relaxing it is only diagnostic. The high original-threshold violation proxy means this result must not be reported as improved paper-level tracking.

## Verification

Pending full refresh after this progress note:

- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Failed / Blocked Items

No command failed in this round before the verification refresh. The scientific blocker remains tracking quality: the local teacher still has high endpoint violations under the original termination contract and should not be used as a final DAgger/VAE/diffusion teacher.

## Effect on English Reading Report

This gives a concrete example of robotics reproduction debugging: an apparent policy-quality issue is partly a termination/data-quality contract issue. The reading report can now explain that the project did not merely rerun PPO, but isolated `body_pos_w`, body order, reset target refresh, phase alignment, and endpoint termination as distinct failure modes.

## Next Step

Refresh comparison, artifact manifest, final report, completion matrix, verification command audits, and master audit. Then decide whether to repair endpoint target semantics or termination thresholds before launching any new full PPO training.

## Git Commit

Pending.
