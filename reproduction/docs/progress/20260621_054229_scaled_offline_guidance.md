# Progress Update

## Goal

Move the main BeyondMimic reproduction line forward after the scaled PPO VAE/state-latent/denoiser run by evaluating full-split offline guidance on the scaled denoiser and wiring the evidence into the auditable report chain.

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
- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_guidance_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader,nounits
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_GUIDANCE_SEED=20260704 envs/bm_analysis/bin/python reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_scaled_ppo_guidance_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_guidance_report_assets.py
```

Verification commands are recorded after the audit refresh.

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py reproduction/scripts/official_importer_export_scaled_ppo_guidance_report_assets.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py
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

- Current AppLauncher gate evidence remains ok: `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json` has `status=ok` and `app_launcher_headless_success_sentinel=true`.
- Offline guidance status: `ok_official_importer_export_scaled_ppo_state_latent_guidance_eval`.
- Full validation/test coverage: `228557` windows total, `114279` validation and `114278` test.
- Task/scale table: `48` task/split/scale rows.
- All `4` proxy tasks have positive best-scale cost deltas and nonzero guidance gradients.
- Report assets: best-cost-delta PNG, scale-response PNG, best rows CSV, scale rows CSV, README, and JSON summary under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/`.

## Verification

All required verification commands passed. Logs are stored under:

```text
logs/verification/scaled_offline_guidance_20260621/
```

Final refreshed audit status:

- `progress_report_audit.py`: ok, `38` rows.
- `required_artifact_absence_audit.py`: ok, `32` rows.
- `artifact_manifest.py`: ok, `1183` artifacts.
- `paper_vs_reproduction_comparison.py`: ok, `194` rows: exactly comparable `58`, approximately comparable `19`, qualitative only `104`, not publicly reproducible `10`, requires real robot `3`.
- `final_reproduction_report.py`: ok.
- `completion_matrix_status_audit.py`: ok, `185` parsed rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: ok, `189` scripts, `0` failed.
- `verification_command_script_manifest.py`: ok, `189` scripts.
- `verification_command_coverage_audit.py`: ok, `197` commands.
- `reproduction_master_audit.py`: ok, `322/322` artifacts passed; completion counts are complete `73`, partial `108`, blocked `3`, out_of_scope `1`.

## Failed / Blocked Items

- This round does not produce a new robot-motion MP4 because it is offline guidance, not closed-loop IsaacLab rollout.
- The scaled denoiser has not yet been used for closed-loop task-conditioned guidance rollouts.
- This is not official BeyondMimic DAgger data, not official VAE/diffusion checkpoints, not TensorRT/asynchronous deployment, not Fig.5/Fig.6 paper-level closed-loop evidence, and not real-robot evidence.

## Effect on English Reading Report

The English reading report can now say that the scaled PPO downstream chain has advanced beyond denoiser training into full-split offline guidance. It should still describe the result as local virtual offline evidence and reserve robot-motion claims for the older closed-loop guidance videos until scaled closed-loop rollouts are rerun.

## Next Step

Rerun task-conditioned closed-loop guidance rollouts using the scaled PPO VAE/denoiser chain, then generate MP4/report assets if the simulator rollout succeeds.

## Git Commit

Pending until final `git diff --stat`/staging review.
