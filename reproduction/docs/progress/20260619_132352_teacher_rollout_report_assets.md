# Progress Update

## Goal

Add report-ready assets for the full official csv-loop teacher rollout dataset, so the English report/PPT can explain the DAgger/teacher-data stage with quantitative plots rather than only JSON audits.

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
- `res/tracking/g1_official_csv_loop_teacher_rollout_dataset/tracking_g1_official_csv_loop_teacher_rollout_dataset.json`
- Teacher rollout shards under `res/runs/tracking_g1_official_csv_loop_teacher_rollout_dataset/.../rank_*/teacher_rollout_shard.npz`

## Files Modified

- `reproduction/scripts/official_csv_loop_teacher_rollout_report_assets.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Generated/updated audit and report outputs under `res/`.

## Commands Run

- `envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_teacher_rollout_report_assets.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`
- Full verification sequence after this progress note.

## Results

- Added `res/report_assets/official_csv_loop_teacher_rollout_dataset/`.
- Generated reward/done timeseries, action distribution, motion-step coverage plots, shard summary CSV, action summary CSV, README, and asset JSON.
- Source data: two full local virtual teacher rollout shards, 299 steps, 1024 total envs, 306,176 total env steps.
- Summary metrics: reward mean over rollout steps `0.025913262011314893`, done count `26331`, timeout count `0`, action dimension `29`, motion-step coverage `299`.
- Artifact manifest increased to 378 artifacts.
- Master audit passed after adding the asset checks.

## Verification

The new asset JSON checks source rollout status, two shard loading, total env-step consistency, 29D action contract, 299-step rollout contract, PNG/CSV/README existence, and no overclaiming of official DAgger, closed-loop guidance, or real robot evidence.

## Failed / Blocked Items

- The teacher rollout shards do not contain robot body poses, only observations/actions/rewards/dones/timeouts/motion timesteps. Therefore this round does not create a policy robot video.
- This remains local virtual official-loop evidence, not the official BeyondMimic DAgger dataset.
- Closed-loop guided diffusion rollout, Fig. 5/Fig. 6 videos/metrics, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.

## Effect on English Reading Report

The English report now has a clearer bridge from motion tracking/PPO evaluation to the teacher-data stage: full rollout reward/done, action-distribution, and motion-step coverage plots can be used directly in the reproduction section or PPT.

## Next Step

Continue toward a true closed-loop guidance gate. If body-pose video is required before that, the simulator rollout script must be extended to record robot state/body poses during policy execution; the current teacher rollout shards are insufficient for an honest policy video.

## Git Commit

Pending at the time of writing this progress file.
