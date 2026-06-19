# Progress Update

## Goal

Create a report-facing guided-vs-unguided closed-loop evidence matrix from existing local virtual guidance rollouts, without claiming paper-level BeyondMimic Fig. 5/Fig. 6 reproduction.

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
- Existing closed-loop guidance result JSON/CSV files under `res/level_c/` and `res/visualization/`.

## Files Modified

- `reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Refreshed generated audit/report files under `res/artifact_manifest`, `res/comparison`, `res/final_report`, `res/master_audit`, `res/docs`, `res/verification_command_*`, `res/required_artifact_absence`, and `res/final_deliverables_audit`.

## Commands Run

```bash
python3 reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/final_deliverables_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Added `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_matrix.json`.
- Added matrix CSV/aggregate CSV/Markdown plus two PNG plots:
  - `guided_vs_unguided_closed_loop_matrix.csv`
  - `guided_vs_unguided_closed_loop_aggregate.csv`
  - `guided_vs_unguided_closed_loop_matrix.md`
  - `task_conditioned_multiseed_guided_deltas.png`
  - `task_conditioned_guidance_signal_strength.png`
- Matrix status: `ok`.
- Matrix rows: `19`.
- Multiseed rows: `12`.
- Aggregate task rows: `4`.
- Video-linked rows: `19`.
- Claim level: `local_virtual_guided_vs_unguided_closed_loop_report_matrix`.
- Artifact manifest now records `640` artifacts.
- Paper-vs-reproduction comparison now records the matrix as a `qualitative_only` report-evidence row.
- Master audit remains `ok`.

## Verification

All required verification commands completed successfully:

- `paper_vs_reproduction_comparison.py`: `ok`
- `artifact_manifest.py`: `ok`, `640` artifacts
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, `170` rows
- `verification_command_syntax_audit.py`: `ok`, `186` scripts, `0` failed
- `verification_command_script_manifest.py`: `ok`, `186` scripts
- `verification_command_coverage_audit.py`: `ok`, `194` commands, `10/10` lightweight smoke pass
- `required_artifact_absence_audit.py`: `ok`, `26` rows
- `final_deliverables_audit.py`: `ok`, `38` rows
- `reproduction_master_audit.py`: `ok`

## Failed / Blocked Items

- The first direct `python3 reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py` run failed because the system Python does not have `matplotlib`.
- The same script passed under the project `bm_analysis` environment, which is the correct analysis/report-generation environment for plotting.
- No new paper-level blocker was resolved by this matrix. Official unpatched replay, official BeyondMimic VAE/diffusion checkpoints, Fig. 5/Fig. 6 paper-level rollouts, TensorRT deployment, and real-robot evidence remain unavailable or incomplete.

## Effect on English Reading Report

The English reading report now has a compact citation point for the strongest local closed-loop guidance evidence. The new matrix groups the action-space bridge, receding-latent bridge, task-conditioned single-seed rollouts, and multi-seed task-conditioned rollouts while preserving conservative labels (`qualitative_only` / `approximately_comparable`).

## Next Step

Continue from this report-facing matrix toward a stronger simulation-side result: either a broader closed-loop guided rollout over the full public motion bundle, or a clearer official replay/converter blocker audit if the official unpatched path is retried.

## Git Commit

Pending at time of writing this progress file.
