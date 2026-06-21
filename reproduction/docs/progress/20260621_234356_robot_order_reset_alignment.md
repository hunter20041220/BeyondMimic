# Progress Update

## Goal

Advance the main tracking-quality repair path for the robot-order FK-repaired official-importer-export baseline. The aim of this round was not to rerun PPO, but to determine whether the remaining step-0 all-done spike and endpoint-z termination behavior should be treated as a motion-bundle problem, a teacher-quality problem, or a reset/command/termination alignment problem.

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
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json`
- `res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic/robot_order_fk_ppo_tracking_quality_diagnostic.json`
- Official `whole_body_tracking` `commands.py`, `terminations.py`, and `tracking_env_cfg.py`
- IsaacLab `manager_based_rl_env.py` and `manager_based_env.py`

## Files Modified

- `reproduction/scripts/robot_order_fk_reset_termination_alignment_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- Regenerated comparison, manifest, final-report, completion-matrix, verification-manifest, and master-audit outputs under `res/`

## Commands Run

```bash
python3 reproduction/scripts/robot_order_fk_reset_termination_alignment_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- New audit status: `ok`.
- New evidence path: `res/tracking/robot_order_fk_reset_termination_alignment_audit/`.
- The robot-order FK motion bundle still looks plausible: 40 motions, 11960 frames, preserved named target z, and ankle mean z below 0.25 m.
- The split task remains not ready: 2166/11960 done signals.
- The PPO multiseed eval has a deterministic step-0 issue: 2048/2048 done at step 0 and about 43 m body-position error.
- Step-0 anchor error does not spike, and step-1 body error drops below 0.3 m, pointing away from wholesale motion-bundle corruption.
- Official source order shows `body_pos_relative_w` is zero-initialized, `ee_body_pos` is a z-only ankle/wrist termination with a 0.25 m threshold, and IsaacLab computes termination before `command_manager.compute()` inside `ManagerBasedRLEnv.step()`.

## Verification

- `artifact_manifest.py`: ok, 1427 artifacts.
- `paper_vs_reproduction_comparison.py`: ok, 222 rows.
- `final_reproduction_report.py`: ok.
- `completion_matrix_status_audit.py`: ok, 202 rows, 0 invalid statuses.
- `verification_command_syntax_audit.py`: ok, 199 scripts, 0 failures.
- `verification_command_script_manifest.py`: ok, 199 scripts.
- `verification_command_coverage_audit.py`: ok, 207 commands, 10/10 lightweight smoke pass.
- `required_artifact_absence_audit.py`: ok, 32 rows.
- `reproduction_master_audit.py`: ok, 352/352 artifacts passed.

## Failed / Blocked Items

No command failed in this round. The underlying paper-level blocker remains: current robot-order FK tracking is not yet a paper-level teacher because step-0 reset termination and post-step0 done rate remain too high.

## Effect on English Reading Report

This adds a concrete example for the report's reproducibility discussion: after repairing body order, the remaining issue is not simply "train longer"; source-level simulator semantics can still create misleading termination signals. It strengthens the project's claim as an audited local virtual BeyondMimic-like pipeline while preserving the boundary from official paper-level tracking.

## Next Step

Run a small live IsaacLab command-warmup probe on GPU 4: record endpoint z-errors and termination terms immediately after reset, after an explicit `command_manager.compute(dt=env.step_dt)` or equivalent `MotionCommand._update_command()` warmup, and after one zero-action step. If the step-0 all-done spike clears, patch local train/eval wrappers to warm up command targets after reset, then rerun full tracking eval before rebuilding downstream VAE/diffusion/guidance.

## Git Commit

Pending at the time this progress file was written.
