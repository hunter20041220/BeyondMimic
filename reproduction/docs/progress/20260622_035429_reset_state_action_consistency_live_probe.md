# Progress Update

## Goal

Audit the current robot-order FK reset repair path before running another full checkpoint evaluation or PPO training job. The specific question was whether target refresh plus action-history reset, action-offset alignment, or motion-state rewrite produces a safe candidate that improves both reset done rate and the joint-velocity transient.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/current_project_reproduction_summary_20260622.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/tracking_env_cfg.py`
- IsaacLab local action/reset source under `/mnt/infini-data/test/BeyondMimic/download/isaaclab`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_reset_state_action_consistency_live_probe.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/current_project_reproduction_summary_20260622.md`.
- Added this progress file.

## Commands Run

- `envs/bm_analysis/bin/python reproduction/scripts/robot_order_fk_reset_state_action_consistency_live_probe.py`
- `python3 - <<'PY' ... inspect res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.json ... PY`
- `rg -n "reset|robot_order|tracking" reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py`
- Pending after this file: full verification suite including artifact manifest, paper-vs-reproduction comparison, final report, completion matrix audit, verification command audits, progress audit, required-artifact absence audit, and master audit.

## Results

The live probe completed and wrote:

- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe_worker.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe_worker_metrics.json`
- `/mnt/infini-data/test/BeyondMimic/logs/tracking_robot_order_fk_reset_state_action_consistency_live_probe/robot_order_fk_reset_state_action_consistency_live_probe.log`

Key result:

- Baseline policy first-step done rate: `0.55078125`.
- Target-refresh policy done rate: `0.28125`; post-step joint-velocity error: `14.182840347290039`.
- Target-refresh + action reset policy done rate: `0.4765625`; post-step joint-velocity error: `10.899185180664062`.
- Target-refresh + action-offset alignment policy done rate: `0.49609375`; post-step joint-velocity error: `10.263128280639648`.
- Target-refresh + motion-state rewrite + action-offset alignment policy done rate: `0.73828125`; post-step joint-velocity error: `8.305423736572266`.
- `any_variant_improves_done_and_joint_velocity=false`.
- `recommended_full_eval_variant=""`.

Interpretation: action reset, action-offset alignment, and motion-state rewrite reduce the joint-velocity transient but worsen done rate. This round therefore recommends no full eval and no PPO rerun from these variants.

## Verification

Verification passed after registering the new diagnostic:

- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/robot_order_fk_reset_state_action_consistency_live_probe.py reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py` -> `ok`, `228` rows.
- `python3 reproduction/scripts/final_reproduction_report.py` -> `ok`.
- `python3 reproduction/scripts/completion_matrix_status_audit.py` -> `ok`, `208` rows, invalid statuses `0`.
- `python3 reproduction/scripts/verification_command_syntax_audit.py` -> `ok`.
- `python3 reproduction/scripts/verification_command_script_manifest.py` -> `ok`.
- `python3 reproduction/scripts/verification_command_coverage_audit.py` -> `ok`.
- `python3 reproduction/scripts/progress_report_audit.py` -> `ok`, latest progress update is this file.
- `python3 reproduction/scripts/required_artifact_absence_audit.py` -> `ok`, `32` rows.
- `python3 reproduction/scripts/artifact_manifest.py` -> `ok`, `1496` artifacts, missing `0`.
- `python3 reproduction/scripts/reproduction_master_audit.py` -> `ok`.

## Failed / Blocked Items

- No full checkpoint eval was run because the live probe found no reset/action variant that improves both done rate and joint velocity.
- No PPO training was run. This was a diagnostic gate, not a formal GPU training experiment.
- The current local tracking teacher remains below paper-level quality.
- Official BeyondMimic DAgger logs, official VAE/diffusion checkpoints, paper Fig. 5/Fig. 6 closed-loop videos/metrics, TensorRT deployment, and real-robot evidence remain absent.

## Effect on English Reading Report

This result strengthens the report's reproducibility discussion. It shows that the project is not merely running scripts until something looks good: it is auditing why a plausible reset repair should not be promoted to a paper-level claim. The report should describe this as negative but useful simulation-side evidence for the current tracking-teacher bottleneck.

## Next Step

Inspect the coupled reset distribution more carefully: command target refresh, robot root/joint state, initial joint velocities, contact settling, action offset, last-action observation, and `ee_body_pos` termination. Only run another full eval or PPO job after a small live probe improves done rate without increasing joint/action transients.

## Git Commit

This progress file is included in the round commit reported in the user-facing update. The file itself does not self-reference a hash because the final commit hash is only stable after the commit is created.
