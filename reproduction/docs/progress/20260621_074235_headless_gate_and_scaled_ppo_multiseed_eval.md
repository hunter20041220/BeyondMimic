# Progress Update

## Goal

Confirm the current IsaacLab headless gate is no longer blocking the mainline, then advance tracking-side evidence beyond a single scaled PPO checkpoint evaluation by running a full multi-seed eval.

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
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json`
- `res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`
- `res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval.py`
- `reproduction/scripts/official_importer_export_scaled_ppo_multiseed_eval_report_assets.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/progress/20260621_074235_headless_gate_and_scaled_ppo_multiseed_eval.md`

## Commands Run

- `envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py`
- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval.py`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval.py`
- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_scaled_ppo_multiseed_eval_report_assets.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_multiseed_eval_report_assets.py`
- `envs/bm_analysis/bin/python reproduction/scripts/progress_report_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

The current AppLauncher headless gate rerun passed:

```text
res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json
status: ok
gate_ok: true
```

The new scaled PPO multi-seed checkpoint evaluation completed:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval/
status: ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_completed
seeds: 20260710, 20260711, 20260712
ok seeds: 3/3
envs per seed: 2048
steps per seed: 299
total env steps: 1837056
```

Aggregate metrics:

- reward mean: `0.02347703839973064 +/- 0.00016869219841160913`
- done count total: `611601 +/- 138.07485892321841`
- anchor position error mean: `0.05960486921063368 +/- 1.6167214224252344e-05`
- body position error mean: `0.7051227944617553 +/- 0.000989948130753293`
- joint position error mean: `0.9113616949339773 +/- 0.005944694571661651`

Report assets:

```text
res/report_assets/official_importer_export_scaled_ppo_checkpoint_multiseed_eval/
```

## Verification

The new multi-seed JSON checks all seeds completed, all evals used `2048` envs and `299` steps, all rows used the official-importer-export G1 USDA, all rows covered `40` motions and `11960` motion frames, and no row used the resource-adjusted enriched USD. Report assets generated a summary CSV plus reward/done and tracking-error PNGs.

Full verification passed:

- `artifact_manifest.py`: `ok`, `1260` artifacts, `missing_count=0`.
- `paper_vs_reproduction_comparison.py`: `ok`, `199` rows; comparison types are `58 exactly_comparable`, `19 approximately_comparable`, `109 qualitative_only`, `10 not_publicly_reproducible`, and `3 requires_real_robot`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `189` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `189` scripts, `0` failed command parses.
- `verification_command_script_manifest.py`: `ok`, `189` scripts.
- `verification_command_coverage_audit.py`: `ok`, `197` commands, `10` smoke-pass entries.
- `required_artifact_absence_audit.py`: `ok`, `32` rows.
- `reproduction_master_audit.py`: `ok`, `330/330` artifacts passed.

## Failed / Blocked Items

No new execution failure occurred in this round.

The main paper-level blockers remain: no official BeyondMimic teacher checkpoint, no official VAE/diffusion checkpoint, no real DAgger logs, no paper Fig.5/Fig.6 guided rollout protocol/video, no TensorRT/asynchronous deployment proof, and no real Unitree G1 robot result. The repeated scaled checkpoint eval confirms the current local policy is weak, so it must not be described as a successful paper teacher.

## Effect on English Reading Report

The reading report now has a stronger tracking-side robustness paragraph: the recovered official-importer-export path can run full local virtual evals repeatedly, but the current checkpoint remains weak and below paper-level.

## Next Step

Run the full verification chain, refresh generated audit/report artifacts, commit locally, attempt GitHub push, then continue toward stronger tracking teacher training/evaluation or downstream closed-loop guidance with clear claim boundaries.

## Git Commit

Pending at the time this progress note is written.
