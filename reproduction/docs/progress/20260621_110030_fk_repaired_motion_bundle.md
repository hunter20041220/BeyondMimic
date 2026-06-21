# Progress Update

## Goal

Repair and audit the degenerate full-public-motion body-position bundle before the next official-importer-export tracking replay/task/PPO attempt, while preserving clear claim boundaries for the English reading report.

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
- Existing tracking comparison/report/master-audit generator sections around the full-bundle motion NPZ, official-importer-export task eval, and body-position degeneracy audit.

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.tsv`

## Commands Run

- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.py`
- `envs/bm_analysis/bin/python reproduction/scripts/validate_motion_npz_contract.py --npz res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/official_csv_loop_full_public_motion_bundle_fk_repaired.npz --summary-json res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/validate_motion_npz_contract_summary.json`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.py`
- `BM_FK_REPAIRED_FULL_TASK_MAX_STEPS=299 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.py`
- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.py reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

- Built a full 40-motion, 11,960-frame FK-repaired MotionLoader candidate:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/official_csv_loop_full_public_motion_bundle_fk_repaired.npz`.
- The FK-repaired bundle has `joint_pos` shape `[11960, 29]`, `body_pos_w` shape `[11960, 40, 3]`, and unit body quaternions within `1.19e-07`.
- The repaired body target is no longer degenerate: mean z spread is `1.2178876399993896 m`, max z spread is `1.2534520626068115 m`, left ankle mean z is `0.05472046881914139 m`, and right ankle mean z is `0.05676112323999405 m`.
- The previous official-loop bundle remains diagnostic only because its body positions are root-like for all 40 bodies. The FK-repaired bundle is a local repair candidate, not unmodified official output.
- Generated report assets under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_full_bundle_fk_repaired_motion_npz/`.
- Added a qualitative-only paper-vs-reproduction row and final-report section for the FK-repaired bundle.
- Added a failed-run summary for the FK-repaired full-bundle task-eval attempt under `/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval/`.

## Verification

- `artifact_manifest.py`: passed, `1350` artifacts.
- `paper_vs_reproduction_comparison.py`: passed after correcting the target-height field name from `mean_z_m` to `z_mean_m`.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `199` rows and `0` invalid statuses.
- `verification_command_syntax_audit.py`: passed, `199` scripts and `0` failed syntax checks.
- `verification_command_script_manifest.py`: passed, `199` scripts.
- `verification_command_coverage_audit.py`: passed, `207` commands and `10` smoke-pass records.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- The FK-repaired full-bundle task-eval wrapper did not produce a task metrics JSON. The first full attempt reached `BM_SENTINEL:after_app` and IsaacLab base environment info but did not reach `BM_SENTINEL:env_created`. A bounded 299-step retry exited with code `143` during early Isaac Sim/Vulkan startup.
- GPU4 and GPU7 were idle after failure, with no target-GPU `wangjc` process to kill, so the current evidence points to live Kit/native startup instability rather than target-GPU occupation.
- The FK-repaired bundle is therefore validated as a motion-preprocessing artifact only. It is not yet validated as a successful IsaacLab task-eval input, not PPO, not DAgger, not VAE/diffusion, not Fig. 5/Fig. 6, not TensorRT, and not real robot.

## Effect on English Reading Report

This round gives the report a concrete reproducibility lesson: a motion NPZ can satisfy outer shapes while still being semantically wrong if target body positions collapse to root height. The FK-repaired bundle and plots provide a clear, visualizable debugging story for why the local tracking chain had poor ankle endpoint behavior and what must be repaired before trusting downstream teacher rollouts or VAE/diffusion experiments.

## Next Step

Split the FK-repaired full bundle back into per-motion NPZs and reuse the already-stable 40-process official-importer-export full-dataset task-eval harness. If that passes, rerun PPO/checkpoint eval on the FK-repaired per-motion or compatible bundle path before collecting new teacher rollout data.

## Git Commit

Pending at time of writing; this progress file should be committed with the FK-repaired bundle audit/report integration.
