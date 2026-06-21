# Progress Update

## Goal

Link the weak official-importer-export scaled PPO teacher behavior to the official tracking termination source, so the next reproduction step is a concrete tracking-debug target rather than a vague low-reward observation.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/reward_termination_diagnostic.json`
- `res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/termination_components.csv`
- `res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/motion_error_components.csv`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`

## Files Modified

- `reproduction/scripts/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Regenerated audit/report outputs under `res/artifact_manifest/`, `res/comparison/`, `res/docs/completion_matrix_status_audit/`, `res/final_report/`, `res/master_audit/`, and `res/verification_command_*`.

## Commands Run

- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/progress_report_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

New source-linked diagnostic assets:

- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/ee_body_pos_termination_source_audit.json`
- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/ee_body_pos_source_evidence.csv`
- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/ee_body_pos_termination_fraction.png`
- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/ee_body_pos_motion_error_context.png`
- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/ee_body_pos_termination_source_audit.md`
- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/README.md`

Main finding:

- Official `ee_body_pos` termination calls `bad_motion_body_pos_z_only`.
- Official threshold is `0.25` m.
- Termination body names are `left_ankle_roll_link`, `right_ankle_roll_link`, `left_wrist_yaw_link`, and `right_wrist_yaw_link`.
- Full public motion bundle shape is `body_pos_w = [11960, 40, 3]`.
- Local scaled PPO iteration-300 and iteration-999 checkpoint evals trip `ee_body_pos` for fractions `0.9985874137750836` and `0.9988405361622074` of env-steps.

## Verification

All required verification commands passed after integration:

- `artifact_manifest.py`: `ok`, `1319` artifacts, `0` missing.
- `paper_vs_reproduction_comparison.py`: `ok`, `207` rows.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `197` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `197` scripts, `0` failures.
- `verification_command_script_manifest.py`: `ok`, `197` scripts.
- `verification_command_coverage_audit.py`: `ok`, `205` commands.
- `required_artifact_absence_audit.py`: `ok`, `32` rows.
- `progress_report_audit.py`: `ok`.
- `reproduction_master_audit.py`: `ok`, `338/338` artifacts passed.

## Failed / Blocked Items

- No command failed in this round.
- This does not fix the local scaled PPO teacher. It identifies the concrete source-linked blocker: endpoint z tracking at the ankle/wrist termination gate.
- It does not produce an official BeyondMimic checkpoint, true DAgger rollout logs, paper-level Fig. 5/Fig. 6 videos, TensorRT deployment, or real robot evidence.

## Effect on English Reading Report

The English report now explains that the local teacher is failing a specific official z-axis endpoint tracking gate rather than simply having a low reward. This supports a stronger independent reflection about why unpublished teacher checkpoints and training curriculum matter for robotics reproducibility.

## Next Step

Inspect the four distal termination bodies in the motion/rollout traces: compare retargeted wrist/ankle z trajectories, robot body-index mapping, early-step policy tracking stability, and whether a termination curriculum or warm-started teacher is needed before collecting downstream VAE/diffusion rollouts.

## Git Commit

Pending at the time this progress file was written.
