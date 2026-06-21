# Progress Update

## Goal

Refresh the scaled-PPO downstream diffusion and offline guidance chain after the newly refreshed scaled-PPO VAE/state-latent dataset. The purpose is to keep the evidence chain current from full teacher rollout samples to VAE latents, state/action-latent windows, diffusion denoiser training, and full validation/test offline guidance, while preserving the claim boundary that this is local virtual evidence rather than official BeyondMimic paper-level closed-loop control.

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
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_guidance_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260621_143552_scaled_ppo_diffusion_guidance_refresh.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_downstream/official_importer_export_full_bundle_downstream_report_assets.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_downstream/official_importer_downstream_diffusion_training_curve.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/README.md`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/official_importer_export_scaled_ppo_guidance_report_assets.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/scaled_ppo_guidance_best_cost_delta.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/scaled_ppo_guidance_best_rows.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/scaled_ppo_guidance_scale_response.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/scaled_ppo_guidance_scale_rows.csv`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.tsv`

Large denoiser checkpoints, sampled guidance NPZ files, worker scripts, and GPU metric CSVs were written under ignored `/mnt/infini-data/test/BeyondMimic/res/runs/` and are intentionally not staged for GitHub.

## Commands Run

- `git status --short && git log -3 --oneline`
- `wc -l goal.md README.md reproduction/PROGRESS.md reproduction/RUNBOOK.md reproduction/docs/final_reproduction_report.md reproduction/docs/known_limitations.md reproduction/docs/experiment_protocol.md res/comparison/paper_vs_reproduction.json res/artifact_manifest/artifact_manifest.json res/master_audit/reproduction_master_audit.json res/required_artifact_absence/required_artifact_absence_audit.json`
- `find reproduction/scripts -maxdepth 1 -type f | rg 'scaled_ppo.*(diffusion|guidance|rollout|video|report_assets|onnx|async)' | sort`
- `jq '{status, settings, duration_seconds, gpu_metrics_summary, checks, outputs, state_latent: .worker_summary.source_state_latent, eval: .worker_summary.evaluation}' res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json`
- `jq '{status, settings, duration_seconds, checks, outputs, summary: .summary, metrics: .metrics, input_statuses}' res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json`
- `rg -n "STATE_LATENT|DIFFUSION|CUDA_VISIBLE_DEVICES|CANDIDATE_GPUS|EPOCHS|BATCH|RUN_ROOT|checkpoint|DataParallel|gpu_metrics|kill_wangjc|SEED|TIMEOUT|def main|status|MAX|split|window" reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7`
- `python3 reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py`
- `rg -n "DIFFUSION|STATE_LATENT|CUDA_VISIBLE_DEVICES|CANDIDATE_GPUS|MAX_WINDOWS|SCALES|TASKS|RUN_ROOT|checkpoint|guidance|gpu_metrics|def main|parse_gpus|status|eval|split" reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`
- `python3 reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_guidance_report_assets.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- Diffusion status: `ok_official_importer_export_scaled_ppo_state_latent_diffusion_training`.
- Diffusion source: refreshed scaled-PPO state-latent dataset summary at `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json`.
- Diffusion settings: seed `20260703`, epochs `30`, batch windows `2048`, hidden dim `512`, denoising steps `20`, visible GPUs `4,7`.
- Diffusion dataset: `1,142,784` windows, split counts `914,227/114,279/114,278`, sequence length `21`, token dim `192`.
- Diffusion metrics: test noisy token MSE `0.06736994787518467`, test predicted token MSE `0.013214186100023133`, test denoising improvement ratio `0.8038563704323348`.
- Diffusion runtime: `430.419` seconds. GPU telemetry recorded `166` rows; peak memory was `2216` MiB on GPU4 and `1806` MiB on GPU7.
- Guidance status: `ok_official_importer_export_scaled_ppo_state_latent_guidance_eval`.
- Guidance settings: full validation/test split (`max_windows_per_split=0`), selected windows `228,557`, tasks `velocity_command`, `latent_smoothness`, `latent_magnitude`, and `composed`, scales `0,0.0005,0.001,0.002,0.005,0.01`.
- Guidance metrics: `4/4` tasks had positive best-cost improvement and nonzero guidance gradients. Mean best cost deltas were `1.2843274838048281e-07` for velocity command, `8.483058341881932e-08` for latent smoothness, `1.9472134248565028e-08` for latent magnitude, and `1.3423171747550724e-07` for composed.
- Guidance runtime: `24.596` seconds. GPU telemetry recorded `10` rows; peak memory was `636` MiB on GPU4 and `4` MiB on GPU7.
- Report-ready assets were refreshed under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_downstream/` and `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_guidance/`.

This refresh keeps the scaled-PPO local virtual chain internally consistent through the new state-latent dataset and denoiser checkpoint. It is not official BeyondMimic DAgger data, not the unreleased official diffusion checkpoint, and not paper-level closed-loop Fig. 5/Fig. 6 evidence.

## Verification

- `python3 reproduction/scripts/artifact_manifest.py`: passed, `artifact_count=1366` before adding this progress artifact.
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`: passed.
- `python3 reproduction/scripts/final_reproduction_report.py`: passed.
- `python3 reproduction/scripts/completion_matrix_status_audit.py`: passed, `rows=199`, `invalid_status_count=0`.
- `python3 reproduction/scripts/verification_command_syntax_audit.py`: passed, `scripts=199`, `failed=0`.
- `python3 reproduction/scripts/verification_command_script_manifest.py`: passed, `scripts=199`.
- `python3 reproduction/scripts/verification_command_coverage_audit.py`: passed, `commands=207`, `smoke_pass=10`.
- `python3 reproduction/scripts/reproduction_master_audit.py`: passed, `status=ok`.

After this progress file is added to the manifest list, `artifact_manifest.py` and `reproduction_master_audit.py` must be rerun once more before commit.

## Failed / Blocked Items

- The base resource-adjusted guidance wrapper printed an intermediate `status=failed` because its base checks expect resource-adjusted source status names. The scaled-PPO wrapper patched the source checks and final JSON correctly records `status=ok_official_importer_export_scaled_ppo_state_latent_guidance_eval`; process return code was `0`, final wrapper checks were all true, and worker status was `ok`.
- GPU memory was below 10GB per card. This is expected for the compact denoiser and offline guidance evaluator, so this is a full-split downstream diffusion/guidance gate rather than a formal high-memory PPO tracking training run.
- No new robot-motion MP4 was generated in this round because the work refreshed denoiser and offline guidance evidence rather than a new closed-loop rollout. Existing scaled-PPO closed-loop guidance MP4 paths remain recorded elsewhere in the report/manifest chain.
- Still missing official DAgger rollout logs, official BeyondMimic VAE/diffusion checkpoints, paper-level Fig. 5/Fig. 6 success/fall/collision metrics, TensorRT/CUDA deployment evidence, and real-robot results.

## Effect on English Reading Report

This round strengthens the report's reproduction narrative for the diffusion and classifier-guidance stages. It provides current full-split evidence that the local scaled-PPO denoiser improves noisy state-latent tokens and that offline task-cost gradients reduce local proxy costs across four task formulations. The English report should present these as local virtual, qualitative/approximately comparable evidence, not as official BeyondMimic paper-level reproduction.

## Next Step

Return to paper-facing closed-loop evidence: rerun or extend scaled-PPO task-conditioned guidance rollout with the refreshed diffusion checkpoint, then prioritize MP4 policy/guidance rollout videos, tracking-error plots, success/fall/collision proxy tables, and a visual-evidence index refresh. These assets are the next most useful items for the English report and PPT.

## Git Commit

Pending at time of writing. Commit message target: `feat: refresh scaled ppo diffusion guidance chain`.
