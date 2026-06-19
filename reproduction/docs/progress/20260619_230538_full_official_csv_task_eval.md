# Progress Update

## Goal

Extend the official csv-loop tracking evidence from full conversion and replay into a full public-motion `Tracking-Flat-G1-v0` task diagnostic, then integrate the result into the audit chain and English reading-report evidence.

## Files Read

- `goal.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_ppo_checkpoint_eval.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.py`
- `res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json`
- `res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`

## Files Modified

- Added `reproduction/scripts/tracking_g1_official_csv_loop_full_dataset_task_eval.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Updated `reproduction/scripts/paper_vs_reproduction_comparison.py`
- Updated `reproduction/scripts/final_reproduction_report.py`
- Updated `reproduction/scripts/reproduction_master_audit.py`
- Updated `reproduction/docs/completion_matrix.md`
- Updated `reproduction/docs/english_reading_report.md`
- Refreshed generated audit/report outputs under `res/artifact_manifest`, `res/comparison`, `res/docs/completion_matrix_status_audit`, `res/final_deliverables_audit`, `res/final_report`, `res/master_audit`, `res/verification_command_coverage`, and `res/verification_command_script_manifest`.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_full_dataset_task_eval.py
BM_OFFICIAL_CSV_TASK_FULL_TARGET_GPU=4 BM_OFFICIAL_CSV_TASK_FULL_LIMIT=2 python3 reproduction/scripts/tracking_g1_official_csv_loop_full_dataset_task_eval.py
BM_OFFICIAL_CSV_TASK_FULL_TARGET_GPU=4 python3 reproduction/scripts/tracking_g1_official_csv_loop_full_dataset_task_eval.py
cp reproduction/docs/english_reading_report.md res/final_report/english_reading_report.md
python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_full_dataset_task_eval.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/final_deliverables_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
git status --short
git diff --stat
```

## Results

The new full task diagnostic ran all 40 official csv-loop NPZ motions through `Tracking-Flat-G1-v0`.

- Status: `ok_official_csv_loop_full_dataset_task_eval`
- Rows: `40/40` ok
- Failed rows: `0`
- Total task steps: `11960`
- Every motion reached step `299`
- Reward mean across motions: `0.024103513569571078`
- Anchor-position error mean: `0.12021495481021702`
- Body-position error mean: `0.11473371332976967`
- Joint-position error mean: `1.4624223172664643`
- Contract checks: action dim `29`, policy obs dim `160`, critic obs dim `286`, nine reward terms, four termination terms, `29` robot joints, `40` robot bodies.

This is a resource-adjusted, zero-action task diagnostic over the full public motion bundle. It is not trained PPO evaluation, not unpatched official asset execution, not DAgger, not Fig. 5/Fig. 6, and not real-robot evidence.

## Verification

All required verification commands completed with `status: ok`.

- `artifact_manifest.py`: `572` artifacts
- `paper_vs_reproduction_comparison.py`: `155` rows; the new task diagnostic is `qualitative_only`
- `final_reproduction_report.py`: report JSON and Markdown regenerated
- `completion_matrix_status_audit.py`: `170` rows, `0` invalid statuses
- `verification_command_syntax_audit.py`: `185` scripts, `0` failed
- `verification_command_script_manifest.py`: `185` scripts
- `verification_command_coverage_audit.py`: `193` commands, `10` smoke checks passed
- `required_artifact_absence_audit.py`: `26` rows
- `final_deliverables_audit.py`: `38` rows
- `reproduction_master_audit.py`: `status: ok`

## Failed / Blocked Items

- The first two-motion test completed the IsaacLab workers but failed during plotting because the outer Python process lacked `matplotlib`. The script was patched so plotting uses `envs/bm_analysis/bin/python`, and the task evaluation data were preserved.
- Full official unpatched G1 URDF converter execution remains unresolved.
- Official paper-level closed-loop tracking teacher evaluation remains incomplete.
- Official BeyondMimic VAE/diffusion checkpoints, true DAgger rollout logs, Fig. 5/Fig. 6 closed-loop videos/metrics, TensorRT deployment evidence, and real-robot results remain absent.

## Effect on English Reading Report

This round adds a stronger Level-B paragraph for the code reproduction section: the report can now state that the full public official-loop motion bundle drives the IsaacLab task layer consistently, with all 40 motions reaching 299 steps and validating observation/action/reward/termination/robot contracts. The report also explicitly marks the result as a zero-action, resource-adjusted diagnostic rather than paper-level PPO or guided diffusion reproduction.

## Next Step

Proceed from task-contract diagnostics toward trained-policy evidence: either run a longer official-csv-loop PPO training/evaluation pass if GPU time is available, or use the existing local teacher rollout dataset to continue VAE/state-latent downstream evaluation. The unpatched official URDF/asset gate remains the main Level-B technical blocker.

## Git Commit

Pending at time of writing. Intended message: `feat: add full official task-eval audit`.
