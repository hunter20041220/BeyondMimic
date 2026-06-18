# Progress Update

## Goal

Move beyond debug fixtures toward official motion replay evidence by converting an official downloaded G1 LAFAN CSV segment and replaying the resulting motion through the currently working resource-adjusted IsaacLab/enriched-USD path.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/completion_matrix.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py`
- `reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_csv_conversion_audit.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_csv_full_replay_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/progress/20260619_003039_resource_adjusted_csv_replay.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_conversion_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_full_replay_audit.py
```

## Results

The official downloaded G1 LAFAN `walk1_subject1.csv` frame range 1-180 was converted through the official interpolation/logging schema and generated enriched USD into a resource-adjusted 299-frame `motion.npz` contract.

The full replay gate then replayed all 299 frames through the enriched USD replay surface.

Key metrics:

- conversion status: `ok_resource_adjusted_csv_conversion`
- full replay status: `ok_resource_adjusted_csv_full_replay`
- input frames: `180`
- input columns: `36`
- output/replay steps: `299`
- joint shape: `[299, 29]`
- body position shape: `[299, 40, 3]`
- max body quaternion norm error from one: `4.76837158203125e-07`
- full replay max joint position write-read error: `0.0`
- full replay max root position write-read error: `0.0`

## Verification

Full verification bundle is run after this progress file is written:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/progress_report_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

This remains resource-adjusted evidence. It does not prove official `csv_to_npz.py`, official `replay_npz.py`, PPO tracking training/evaluation, DAgger rollout collection, teacher rollout dataset, trained tracking checkpoint, TensorRT deployment, Fig. 5/Fig. 6 videos, or real robot execution.

The generated `.npz` is intentionally treated as a local data artifact and is not committed to GitHub.

## Effect on English Reading Report

This gives the reading report a stronger and more nuanced reproduction story: the project can now process official-source motion data through the validated IsaacLab replay surface, while honestly identifying the remaining blocker as the official URDF/USD conversion path rather than the CSV data or replay schema itself.

## Next Step

Try wiring the official-CSV-derived resource-adjusted `motion.npz` into the official `Tracking-Flat-G1-v0` task eval, then decide whether a bounded PPO train-entry retry is meaningful.

## Git Commit

Pending.
