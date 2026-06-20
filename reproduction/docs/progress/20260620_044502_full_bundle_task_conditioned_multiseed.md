# Progress Update

## Goal

Extend the local closed-loop guidance evidence from single-seed full-bundle task-conditioned rollouts to a three-seed, four-task full-bundle audit, then wire the new evidence into the comparison table, manifest, master audit, final report, and English reading report without claiming paper-level Fig. 5/Fig. 6 reproduction.

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
- existing full-bundle and task-conditioned guidance scripts/audits under `reproduction/scripts/` and `res/level_c/`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py`
- `reproduction/scripts/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_report_assets.py`
- `reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/visual_media_inventory_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- refreshed generated audit/report outputs under `res/artifact_manifest/`, `res/comparison/`, `res/final_report/`, `res/master_audit/`, `res/report_assets/`, `res/required_artifact_absence/`, `res/verification_command_*`, and `res/visual_media_inventory/`.

## Commands Run

- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py`
- `python3 -m py_compile ...`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_report_assets.py`
- `python3 reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/visual_media_inventory_audit.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/visual_evidence_index.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/required_artifact_absence_audit.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/final_deliverables_audit.py`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

- New full-bundle task-conditioned latent-guidance multi-seed summary:
  - `res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json`
  - status: `ok_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval`
  - rows: `12`
  - seed groups: `seed_group_0_existing`, `seed_group_1`, `seed_group_2`
  - tasks: `joystick`, `waypoint`, `obstacle_avoidance`, `composed`
  - rollout steps per row: `299`
  - bundle motion count: `40`
- New report assets:
  - `res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets.json`
  - aggregate CSV, bar plot, seed scatter plot, and README.
- Guided-vs-unguided closed-loop matrix updated to:
  - row count: `35`
  - aggregate rows: `12`
  - full-bundle task-conditioned multiseed rows: `12`
  - video-linked rows: `35`
- `paper_vs_reproduction.json` updated to `168` rows with one new `qualitative_only` full-bundle multiseed entry.
- `artifact_manifest.json` updated to `768` artifacts.
- `reproduction_master_audit.json` remains `ok`.

## Verification

Passed:

- `py_compile` for edited scripts.
- `official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_report_assets.py`
- `guided_vs_unguided_closed_loop_matrix.py`
- `visual_media_inventory_audit.py`
- `visual_evidence_index.py`
- `required_artifact_absence_audit.py`
- `artifact_manifest.py`
- `paper_vs_reproduction_comparison.py`
- `final_reproduction_report.py`
- `completion_matrix_status_audit.py`
- `verification_command_syntax_audit.py`
- `verification_command_script_manifest.py`
- `verification_command_coverage_audit.py`
- `final_deliverables_audit.py`
- `reproduction_master_audit.py`

## Failed / Blocked Items

- One command failed when `python3 reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py` used the system Python without `matplotlib`. The fix was to rerun it with `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python`, which passed.
- The full-bundle multiseed run used GPU 4 with about 5.7 GB peak observed during the serial Isaac worker phases, so it is recorded as local virtual closed-loop evaluation/gate evidence, not a formal large-GPU training experiment.
- Still blocked/not claimed: official BeyondMimic VAE/diffusion checkpoints, official Fig. 5/Fig. 6 task success logs/videos, TensorRT deployment, and real robot results.

## Effect on English Reading Report

The English report now has a stronger code-reproduction section for the simulation-only part: it can cite a 12-row, three-seed, full-public-bundle task-conditioned closed-loop guidance audit. The report still explicitly says this is not official paper-level reproduction.

## Next Step

After committing this evidence, the next technical step is to continue from the strongest currently open blocker: official/paper-facing closed-loop rollout realism. A practical next move is to inspect whether the full-bundle task-conditioned bridge can be run through the actual official task entry with fewer local proxy substitutions, or to improve the TensorRT/CUDA deployment audit if the runtime stack supports it.

## Git Commit

Included in the round commit for this update. The final commit hash is reported in the user-facing summary.
