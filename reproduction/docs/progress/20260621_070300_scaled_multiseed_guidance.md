# Progress Update

## Goal

Integrate the new official-importer-export scaled PPO task-conditioned latent-guidance multi-seed closed-loop audit into the reproducible BeyondMimic evidence chain without overstating it as paper-level Fig. 5/Fig. 6 reproduction.

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
- `res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json`
- `res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py`
- `reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py
CUDA_VISIBLE_DEVICES=4,7 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py
BM_METADATA_ONLY=1 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py
cp reproduction/docs/english_reading_report.md res/final_report/english_reading_report.md
```

## Results

- Added a scaled PPO wrapper around the existing official-importer-export task-conditioned latent-guidance multi-seed runner.
- Added scaled report-assets generation for the multi-seed summary.
- Produced `5` seed groups, `20` task/seed rows, `23920` rollout-variant steps, and `20` local MP4 paths.
- All rows are `ok`, all rows reach 299 rollout steps, all rows record MP4 paths, and all checks preserve `goal_complete=false`.
- The summary verifies the full 40-motion public bundle, official-importer-export G1 USDA path, and scaled PPO training/checkpoint-eval/VAE/denoiser/offline-guidance chain.
- New report assets include aggregate CSV, bar plot, seed-scatter plot, and README.

## Verification

Full verification is run after this report is written:

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/progress_report_audit.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
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

- The first full multi-seed command completed the rollout rows but exited at metadata rewrite because the wrapper initially expected the scaled canonical JSON before writing it. This was fixed with a metadata-only rewrite and report-assets rerun; no failed rollout rows were produced.
- The local MP4 files remain report/reference media and are intentionally not committed to GitHub.
- This does not resolve official BeyondMimic VAE/diffusion checkpoint absence, true Fig. 5/Fig. 6 paper protocol absence, TensorRT deployment absence, or real-robot absence.

## Effect on English Reading Report

The English report can now describe a stronger scaled PPO local virtual evidence ladder:

1. scaled PPO training/checkpoint evaluation;
2. scaled teacher-rollout VAE/state-latent/denoiser training;
3. full-split scaled offline guidance;
4. single-seed closed-loop scaled task-conditioned guidance;
5. multi-seed closed-loop scaled task-conditioned guidance with aggregate plots.

This improves the reproduction section while still explicitly stating that the project does not fully reproduce BeyondMimic at paper level.

## Next Step

Run full verification, refresh generated audit outputs, commit only code/docs/small audit artifacts, and then continue toward protocol-aligned Fig. 5/Fig. 6 proxy gates or TensorRT/asynchronous deployment auditing.

## Git Commit

Pending.
