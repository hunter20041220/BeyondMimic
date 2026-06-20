# Progress Update

## Goal

Record and audit the full public-motion official `csv_to_npz.py` and `replay_npz.py` loop runs on the captured G1 USDA exported by the official Isaac Sim URDF importer, without overclaiming them as unmodified official converter success or paper-level tracking evaluation.

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
- `reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py`
- `reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`
- `reproduction/scripts/tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.py`
- `reproduction/scripts/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.py`

## Files Modified

- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py`
- `reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7
BM_HEADLESS_GATE_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_TERMINATE_WANGJC_GPU_GUARD=1 envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_HEADLESS_GATE_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py
BM_OFFICIAL_IMPORTER_CSV_FULL_DATASET_TARGET_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.py
BM_OFFICIAL_IMPORTER_REPLAY_FULL_DATASET_TARGET_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.py
```

Verification commands are listed after the final rerun in the user-facing response and the regenerated reports.

## Results

- Headless AppLauncher gate rerun status: `ok`, with `app_launcher_headless_success_sentinel=true`.
- Official-importer-export full `csv_to_npz.py` loop status: `ok_official_csv_to_npz_loop_full_dataset_with_official_importer_export`.
- Conversion rows: `40`; failed rows: `0`; total frames: `11960`; total joint values: `346840`; total NPZ bytes: `27723280`.
- Official-importer-export full `replay_npz.py` loop status: `ok_official_replay_npz_loop_full_dataset_with_official_importer_export`.
- Replay rows: `40`; failed rows: `0`; total replayed steps: `11960`; shutdown warning count: `0`.

## Verification

Pending at write time. The required audit chain will be rerun before commit:

```bash
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

- Early limit-1 runs exposed audit-predicate bugs and were retained under `res/failed_runs/`.
- The final full conversion and replay loops passed.
- The live unmodified official converter entry remains unresolved; this result uses a captured local USDA exported by the official Isaac Sim URDF importer.
- No trained-policy evaluation, true DAgger rollout dataset, official VAE/diffusion checkpoint, Fig. 5/Fig. 6 result, TensorRT deployment, or real robot result is claimed.

## Effect on English Reading Report

This gives the English reading report a cleaner tracking reproduction story: the full public-motion official script loop can now be described on the recovered official-importer-export asset path, not only on the generated enriched scaffold. The report must still state that this is local virtual reference preprocessing/replay evidence, not paper-level tracking reproduction.

## Next Step

Refresh the artifact manifest, paper-vs-reproduction table, final report, completion matrix audit, verification command audits, and master audit; then commit and push the code, documentation, and small summary artifacts.

## Git Commit

Pending at write time.
