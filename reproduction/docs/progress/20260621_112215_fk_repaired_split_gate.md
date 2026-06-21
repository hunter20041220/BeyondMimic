# Progress Update

## Goal

Move the FK-repaired public-motion evidence from a single concatenated candidate bundle toward the stable per-motion IsaacLab task-evaluation path, while keeping paper-level claims separated from local preprocessing/debug evidence.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_split_task_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- Refreshed comparison, final-report, manifest, verification-command, completion-matrix, and master-audit outputs under `/mnt/infini-data/test/BeyondMimic/res/`.

## Commands Run

- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz.py reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_split_task_eval.py`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz.py`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_split_task_eval.py`
- `BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_LIMIT=1 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py`
- `BM_FK_REPAIRED_SPLIT_TASK_GPU=7 BM_FK_REPAIRED_SPLIT_TASK_LIMIT=1 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_split_task_eval.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

- Split the FK-repaired 40-motion full bundle into 40 isolated 299-frame MotionLoader NPZ files under `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz/motions/`.
- Split summary status: `ok_fk_repaired_split_motion_npz`.
- Split metrics: `40` motions, `11960` total frames, total split NPZ bytes `27723280`, per-motion mean z-spread range `1.087217926979065` to `1.247505784034729`, max left/right ankle mean z `0.1394355595111847` / `0.18895088136196136` m.
- Added a qualitative-only paper-vs-reproduction row for the FK-repaired per-motion split.
- Added final-report and master-audit coverage for the split NPZ preparation step.

## Verification

- `artifact_manifest.py`: passed, `1355` artifacts.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `199` rows and `0` invalid statuses.
- `verification_command_syntax_audit.py`: passed, `199` scripts and `0` failed syntax checks.
- `verification_command_script_manifest.py`: passed, `199` scripts.
- `verification_command_coverage_audit.py`: passed, `207` commands and `10` smoke-pass records.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- FK-repaired split task eval did not reach `env_created`; the wrapper was terminated by signal `15` after AppLauncher reached `BM_SENTINEL:after_app` and IsaacLab base environment information for `dance1_subject1`.
- A control run of the previously successful official-importer-export full-dataset task-eval script with `LIMIT=1` now shows the same `143`/SIGTERM behavior at the same env-construction stage.
- A GPU7 limited retry also terminated with `143` before task metrics. GPU4 and GPU7 were not occupied by target-card `wangjc` processes, so this is not currently explained by target GPU memory contention.
- Current interpretation: the active blocker has moved back to the live IsaacLab/Kit task-env construction gate on this host/session. The FK-repaired split files are validated preprocessing artifacts only; they are not yet task-eval, PPO, DAgger, VAE/diffusion, Fig. 5/Fig. 6, TensorRT, or real-robot evidence.

## Effect on English Reading Report

The split artifacts strengthen the reproduction section by turning the FK repair from a single diagnostic bundle into a reusable per-motion dataset candidate. The failure evidence also supports a clear limitations narrative: semantic motion repair is necessary but insufficient until the live IsaacLab task-env construction gate is stable again.

## Next Step

Repair the current live `Tracking-Flat-G1-v0` env-construction gate. The first diagnostic should compare the lightweight AppLauncher-only headless probe, a minimal `gym.make("Tracking-Flat-G1-v0")` probe without motion stepping, and the previously successful full-dataset worker under the same GPU/Kit settings to identify who sends SIGTERM and why.

## Git Commit

Pending at time of writing; this progress file should be committed with the FK-repaired split scripts and refreshed audit/report outputs.
