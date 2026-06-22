# Progress Update

## Goal

Test a conservative tracking data-quality repair candidate before launching another full PPO run: keep all official `ee_body_pos` endpoint bodies active, sweep only the z-only termination threshold, and record whether this reduces the robot-order FK PPO done rate without pretending it is a paper-level score.

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
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.json`
- `res/tracking/robot_order_fk_wrist_endpoint_source_full_diagnostic/robot_order_fk_wrist_endpoint_source_full_diagnostic.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/update_course_reports.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/update_course_reports.py
python3 reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep.py
python3 reproduction/scripts/update_course_reports.py
```

## Results

The endpoint threshold sweep completed successfully. It used the same robot-order FK PPO checkpoint, the same seed, and the same full 2048-env x 299-step evaluation scope as the current tracking checkpoint diagnostics.

Key results:

- Target-refresh baseline done rate: `0.22340745192307693`.
- Threshold `0.30`: done rate `0.16474184782608695`.
- Threshold `0.35`: done rate `0.13060951870819398`.
- Threshold `0.40`: done rate `0.11093782660953178`.
- Threshold `0.50`: done rate `0.08907621760033445`.
- Best threshold: `0.50`.
- Best done-rate delta vs target-refresh baseline: `-0.13433123432274247`.
- Moderate-threshold candidate count: `3`.

The sweep keeps all four official endpoint bodies active (`left_ankle_roll_link`, `right_ankle_roll_link`, `left_wrist_yaw_link`, `right_wrist_yaw_link`) and records the original 0.25 m manual violation proxy for every variant.

## Verification

Initial syntax checks and report regeneration passed. The full artifact/comparison/final-report/completion-matrix/verification/master-audit command suite is the next step in this same round.

## Failed / Blocked Items

The sweep changes the termination threshold, so it changes the evaluator. It is a diagnostic repair candidate, not a paper-level tracking metric. The original 0.25 m manual endpoint violation proxy remains high, especially at larger relaxed thresholds, so the underlying data/termination semantics are not fully fixed.

Remaining incomplete paper-level items include a high-quality official-equivalent tracking teacher, true DAgger rollout logs, official VAE/diffusion checkpoints, strict Fig. 5/Fig. 6 metrics and videos, TensorRT/asynchronous deployment, MuJoCo/ROS sim-to-sim logs, and real robot evidence.

## Effect on English Reading Report

The English and Chinese reports now explain a concrete next-step signal: before retraining PPO, the project should evaluate whether a calibrated endpoint threshold can reduce false or overly strict `ee_body_pos` termination while preserving the official endpoint body set. The reports also state clearly that a relaxed threshold is not an official BeyondMimic paper score.

## Next Step

Run the full verification suite, then decide whether to evaluate a threshold candidate in a controlled tracking run or continue repairing wrist endpoint target/body semantics directly. A new full PPO run should start only after this tracking gate produces a clearly better done/termination profile.

## Git Commit

Pending.
