# Progress Update

## Goal

Advance the BeyondMimic reproduction from a single task-conditioned local latent-guidance rollout to a multi-seed local virtual closed-loop evaluation, while keeping the claim boundary explicit for the English reading report.

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
- Existing task-conditioned guidance scripts and summaries under `reproduction/scripts/` and `res/level_c/`.

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.py`
- `reproduction/scripts/official_csv_loop_task_conditioned_guidance_multiseed_report_assets.py`
- `reproduction/scripts/gpu_wangjc_process_guard.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- This progress update.

## Commands Run

- `python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.py`
- `python3 reproduction/scripts/gpu_wangjc_process_guard.py`
- `BM_TERMINATE_WANGJC_GPU_GUARD=1 python3 reproduction/scripts/gpu_wangjc_process_guard.py`
- Single-task gate after stopping the external guard, using `envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py`
- `envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_task_conditioned_guidance_multiseed_report_assets.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/final_deliverables_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- The first task-conditioned multi-seed attempts failed with return code `-15` during Kit startup. The logs reached `BM_SENTINEL:receding_latent_guidance:before_app` and then died without a Python stack trace.
- Process inspection found two external `wangjc` GPU out-of-bounds guard processes configured with `--blocked-gpus 4,5,6,7`. A narrow audit script stopped only those matching guard processes and saved records under `res/gpu_guard/`.
- After the guard was stopped, a single-task AppLauncher/rollout/render gate completed successfully.
- The full multi-seed task-conditioned latent-guidance evaluation completed with status `ok_official_csv_loop_task_conditioned_latent_guidance_multiseed_eval`.
- Summary metrics:
  - `row_count=12`
  - `task_count=4`
  - `seed_group_count=3`
  - `total_rollout_variant_steps=14352`
  - all rows ok
  - all rollouts 299 steps
  - all rows have MP4 paths
- Aggregate guided reward means:
  - joystick: `0.026750468158909645`
  - waypoint: `0.025070973195409695`
  - obstacle_avoidance: `0.02468773125543144`
  - composed: `0.027051134261332762`
- Aggregate guided target-body error means:
  - joystick: `0.08046085387468338`
  - waypoint: `0.08036413292090099`
  - obstacle_avoidance: `0.08021580427885056`
  - composed: `0.0780883530775706`

## Verification

- `required_artifact_absence_audit.py`: `ok`, 26 rows. The new local MP4s are classified as local reference/report videos, not paper-level Fig. 5/Fig. 6 reproduction videos.
- `artifact_manifest.py`: `ok`, 553 artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`, comparison artifacts regenerated.
- `final_reproduction_report.py`: `ok`, final report artifacts regenerated.
- `completion_matrix_status_audit.py`: `ok`, 170 rows, 0 invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, 185 scripts, 0 failed.
- `verification_command_script_manifest.py`: `ok`, 185 scripts.
- `verification_command_coverage_audit.py`: `ok`, 193 commands.
- `final_deliverables_audit.py`: `ok`, 38 rows.
- `reproduction_master_audit.py`: `ok`.

## Failed / Blocked Items

- The initial `-15` failure is retained under `res/failed_runs/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/`.
- The failure was traced to an external `wangjc` GPU out-of-bounds guard, not to a broken IsaacLab import path.
- This result still does not satisfy paper-level Fig. 5/Fig. 6 reproduction because the official BeyondMimic VAE/diffusion checkpoints, paper task logs, TensorRT deployment, and real-robot evidence are absent.
- Current status remains `goal_complete=false`; the project must not claim a complete BeyondMimic reproduction.

## Effect on English Reading Report

The report now has stronger code reproduction evidence: a multi-seed local virtual closed-loop guidance suite with videos, keyframes, plots, and aggregate task metrics. It also has a concrete reproducibility lesson about multi-user GPU guard processes masquerading as simulator instability.

## Next Step

Commit the code/small audit artifacts to GitHub, then continue toward more paper-facing closed-loop evaluation and deployment audits without claiming full reproduction.

## Git Commit

Pending.
