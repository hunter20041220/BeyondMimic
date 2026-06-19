# Progress Update

## Goal

Move the tracking pipeline beyond a single smoke motion by revalidating the current IsaacLab headless gate and running the official `whole_body_tracking` `csv_to_npz.py` loop body over the full local public G1 LAFAN CSV bundle.

## Files Read

- `goal.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/english_reading_report.md`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- Existing official replay/conversion scripts and audits under `reproduction/scripts/` and `res/tracking/`.

## Files Modified

- `reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py`
- `reproduction/scripts/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- This progress update.

## Commands Run

- `CUDA_VISIBLE_DEVICES=4 python3 reproduction/scripts/isaaclab_current_headless_gate.py`
- `python3 -m py_compile reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py reproduction/scripts/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.py`
- `BM_OFFICIAL_CSV_FULL_DATASET_TARGET_GPU=4 BM_OFFICIAL_CSV_FULL_DATASET_LIMIT=2 python3 reproduction/scripts/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.py`
- `BM_OFFICIAL_CSV_FULL_DATASET_TARGET_GPU=4 python3 reproduction/scripts/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.py`

## Results

- The current IsaacLab `AppLauncher(headless=True)` gate passed on GPU 4 with status `ok`.
- The single-motion official `csv_to_npz.py` loop runner was parameterized for input CSV, output NPZ, metrics JSON, output name, target GPU, and log basename.
- A full-dataset wrapper now runs the same official loop body over all 40 local G1 LAFAN CSV files.
- The first long run exposed a Kit shutdown hang after `BM_SENTINEL:official_csv_loop_complete=299` and `BM_SENTINEL:simulation_app_close_called`; the runner now records `forced_after_success_sentinel=true` when the NPZ and success sentinel are present but Kit does not exit promptly.
- Full-dataset conversion result:
  - status: `ok_official_csv_to_npz_loop_full_dataset_with_enriched_usd`
  - rows: `40`
  - ok: `40`
  - failed: `0`
  - total frames: `11960`
  - total joint values: `346840`
  - max joint abs overall: `2.444481134414673`
  - max joint velocity abs overall: `11.793120384216309`

## Verification

- `artifact_manifest.py`: `ok`, 557 artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, 170 rows, 0 invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, 185 scripts, 0 failed.
- `verification_command_script_manifest.py`: `ok`, 185 scripts.
- `verification_command_coverage_audit.py`: `ok`, 193 commands.
- `required_artifact_absence_audit.py`: `ok`, 26 rows.
- `final_deliverables_audit.py`: `ok`, 38 rows.
- `reproduction_master_audit.py`: `ok`.

## Failed / Blocked Items

- No full-dataset conversion rows failed.
- This is still resource-adjusted because the official G1 config is patched in memory to use the enriched USD. It is not unpatched official converter output, not policy replay/evaluation, not PPO training, and not paper-level tracking.
- Current status remains `goal_complete=false`; the project must not claim a complete BeyondMimic reproduction.

## Effect on English Reading Report

The report can now say that motion preprocessing evidence is no longer a single-motion smoke: the official `csv_to_npz.py` loop body has been exercised over the full 40-motion public G1 LAFAN bundle, with explicit limitations.

## Next Step

Regenerate audits/reports, commit the code and small audit artifacts to GitHub, then continue from full conversion coverage toward full-bundle replay/evaluation and stronger policy-facing tracking artifacts.

## Git Commit

Pending.
