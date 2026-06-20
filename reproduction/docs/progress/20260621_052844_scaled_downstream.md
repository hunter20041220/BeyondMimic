# Progress Update

## Goal

Retrain the local Level C downstream chain from the larger official-importer-export scaled PPO teacher rollout dataset, then wire the resulting evidence into the auditable reproduction reports without overclaiming paper-level reproduction.

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

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_downstream_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/completion_matrix_status_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Commands Run

```bash
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_VAE_SEED=20260701 envs/bm_analysis/bin/python reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_STATE_LATENT_SEED=20260702 envs/bm_analysis/bin/python reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_DIFFUSION_SEED=20260703 envs/bm_analysis/bin/python reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py
```

Verification commands are recorded below after the audit refresh.

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py reproduction/scripts/official_importer_export_full_bundle_downstream_report_assets.py reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/completion_matrix_status_audit.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/required_artifact_absence_audit.py
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

## Results

- VAE: `1224704` samples, `40` epochs, test action MSE `0.00019815583800664172`, test absolute action error mean `0.010908454074524343`.
- State-latent dataset: `1224704` source samples, `1142784` windows, sequence length `21`, token dimension `192`, weighted posterior reconstruction MSE `0.00019638959393456675`.
- Diffusion denoiser: `1142784` windows, `30` epochs, test pred-token MSE `0.013214186100023133`, noisy-token MSE `0.06736994787518467`, denoising improvement ratio `0.8038563704323348`.
- Report assets: VAE curve PNG, diffusion curve PNG, split metrics CSV, stage summary CSV, README, and summary JSON under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_downstream/`.

## Verification

All required verification commands passed. Logs are stored under:

```text
logs/verification/scaled_downstream_20260621/
```

Final refreshed audit status:

- `progress_report_audit.py`: ok, `38` rows.
- `required_artifact_absence_audit.py`: ok, `32` rows.
- `artifact_manifest.py`: ok, `1173` artifacts.
- `paper_vs_reproduction_comparison.py`: ok, `193` rows: exactly comparable `58`, approximately comparable `19`, qualitative only `103`, not publicly reproducible `10`, requires real robot `3`.
- `final_reproduction_report.py`: ok.
- `completion_matrix_status_audit.py`: ok, `191` parsed rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: ok, `189` scripts, `0` failed.
- `verification_command_script_manifest.py`: ok, `189` scripts.
- `verification_command_coverage_audit.py`: ok, `197` commands.
- `reproduction_master_audit.py`: ok, `320/320` artifacts passed; completion counts are complete `73`, partial `107`, blocked `3`, out_of_scope `1`.

## Failed / Blocked Items

- GPU peak memory stayed below the requested 10GB/card formal threshold, so this is not reported as a formal high-memory GPU experiment.
- The subsequent offline guidance and closed-loop guidance evaluations have not yet been rerun from this scaled denoiser.
- This is not official BeyondMimic DAgger data, not official VAE/diffusion checkpoints, not TensorRT/asynchronous deployment, not Fig.5/Fig.6 paper-level closed-loop evidence, and not real-robot evidence.

## Effect on English Reading Report

The English report now has a stronger local downstream training chain to discuss: the project has moved from a `306176`-sample importer-export teacher candidate to a `1224704`-sample scaled PPO teacher candidate for VAE/state-latent/denoiser training. The report still labels the result as local virtual engineering evidence rather than paper-level reproduction.

## Next Step

Run the full audit refresh and then rerun offline guidance plus closed-loop proxy guidance from the scaled denoiser if the audit remains clean.

## Git Commit

Pending until final `git diff --stat`/staging review.
