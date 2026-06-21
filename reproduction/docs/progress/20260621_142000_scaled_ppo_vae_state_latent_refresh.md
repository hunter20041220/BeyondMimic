# Progress Update

## Goal

Refresh the downstream scaled-PPO teacher-rollout chain after the full teacher rollout dataset refresh: retrain the local conditional action VAE on the 1,224,704-sample scaled official-importer-export teacher rollout dataset, rebuild the state/action-latent window dataset, regenerate report-ready downstream plots/tables, and rerun the audit chain without claiming official BeyondMimic DAgger/VAE or paper-level closed-loop evidence.

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
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260621_142000_scaled_ppo_vae_state_latent_refresh.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.tsv`

The downstream report-assets script was rerun successfully, but its PNG/CSV/README outputs were content-identical to the previous committed assets and therefore did not appear in the final Git diff.

Large generated checkpoints and latent shards were written under ignored `/mnt/infini-data/test/BeyondMimic/res/runs/` and are intentionally not staged for GitHub.

## Commands Run

- `git status --short && git log -5 --oneline`
- `wc -l goal.md README.md reproduction/PROGRESS.md reproduction/RUNBOOK.md reproduction/docs/final_reproduction_report.md reproduction/docs/known_limitations.md reproduction/docs/experiment_protocol.md res/comparison/paper_vs_reproduction.json res/artifact_manifest/artifact_manifest.json res/master_audit/reproduction_master_audit.json res/required_artifact_absence/required_artifact_absence_audit.json`
- `rg -n "DEFAULT|EPOCH|BATCH|CUDA|VISIBLE|DEVICE|DataParallel|RUN_ROOT|CANDIDATE|teacher_rollout|checkpoint|status|def main|argparse|seed|gpu|memory|summary|json" reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py`
- `jq '{status, config, metrics, checks, outputs, interpretation}' res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json`
- `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7`
- `rg -n "CUDA_VISIBLE_DEVICES|VISIBLE|GPU|DataParallel|torch\\.device|EPOCHS|BATCH_SIZE|SAMPLE|MAX|RUN_ROOT|TEACHER|dataset|gpu_metrics|subprocess|nvidia|checkpoint|npz|SEED|TRAIN|VAL|TEST" reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `python3 reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py`
- `rg -n "VAE|checkpoint|CUDA_VISIBLE_DEVICES|CANDIDATE_GPUS|RUN_ROOT|TEACHER|teacher_rollout|state_latent|window|split|npz|summary|def main|status|gpu|BATCH|SEED" reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `python3 -m py_compile reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `python3 reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py`
- `python3 reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- VAE refresh status: `ok_official_importer_export_scaled_ppo_teacher_rollout_vae_training`.
- VAE settings: seed `20260701`, epochs `40`, batch size `16384`, latent dim `32`, hidden dim `512`, visible GPUs `[4, 7]`.
- VAE dataset: `1,224,704` teacher-rollout samples, observation dim `160`, action dim `29`, train/validation/test split `979,763/122,470/122,471`.
- VAE metrics: train action MSE `0.00019605412793074115`, validation action MSE `0.00019741396863537375`, test action MSE `0.00019815583800664172`, test action absolute error mean `0.010908454074524343`.
- VAE runtime: `279.986` seconds. GPU telemetry recorded `108` rows; peak memory was `1720` MiB on GPU4 and `1670` MiB on GPU7.
- State-latent refresh status: `ok_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset`.
- State-latent dataset: `1,224,704` samples, `1,142,784` sequence windows, sequence length `21`, token dim `192`, split counts `914,227/114,279/114,278`.
- State-latent metric: weighted posterior reconstruction MSE `0.00019638959393456675`.
- State-latent runtime: `28.659` seconds. GPU telemetry recorded `12` rows; peak memory was `744` MiB on GPU4 and `4` MiB on GPU7.
- Downstream report assets were refreshed under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_downstream/`, including VAE/diffusion curves and split/stage tables.

These are full local downstream-data refreshes from the scaled official-importer-export teacher rollout. They are not official DAgger logs, not official BeyondMimic VAE/diffusion checkpoints, and not paper-level closed-loop Fig. 5/Fig. 6 results.

## Verification

- `python3 reproduction/scripts/artifact_manifest.py`: passed, `artifact_count=1365`, `missing_count=0` before adding this progress artifact.
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`: passed, comparison JSON/CSV refreshed.
- `python3 reproduction/scripts/final_reproduction_report.py`: passed.
- `python3 reproduction/scripts/completion_matrix_status_audit.py`: passed, `rows=199`, `invalid_status_count=0`.
- `python3 reproduction/scripts/verification_command_syntax_audit.py`: passed, `scripts=199`, `failed=0`.
- `python3 reproduction/scripts/verification_command_script_manifest.py`: passed, `scripts=199`.
- `python3 reproduction/scripts/verification_command_coverage_audit.py`: passed, `commands=207`, `smoke_pass=10`.
- `python3 reproduction/scripts/reproduction_master_audit.py`: passed, `status=ok`.

After this progress file is added to the manifest list, `artifact_manifest.py` and `reproduction_master_audit.py` must be rerun once more before commit.

## Failed / Blocked Items

- `python3 reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py` failed under the system Python because `matplotlib` was unavailable. The same script passed with `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python`, which is the restored analysis conda-pack environment containing NumPy, pandas, matplotlib, ONNX, and ONNXRuntime.
- GPU memory for VAE/state-latent refresh was far below 10GB per card. This is expected for the compact VAE encoder/decoder and latent extraction workload, so the run is reported as a full-data downstream training/data gate rather than a formal high-memory PPO tracking training run.
- No new robot-motion MP4 was generated in this round because this round refreshed VAE/state-latent downstream data and plots, not a new motion replay or closed-loop rollout. Existing scaled-PPO closed-loop guidance MP4 paths remain recorded in the report/manifest chain.
- Still missing paper-level official DAgger rollout logs, official BeyondMimic VAE/diffusion checkpoints, official Fig. 5/Fig. 6 success/fall/collision protocol results, TensorRT/CUDA ONNX deployment evidence, and real-robot results.

## Effect on English Reading Report

This round strengthens the reproducible code-reproduction section for the BeyondMimic reading report. It provides a clean evidence chain from the scaled official-importer-export PPO teacher rollout dataset into a conditional action VAE and state/action-latent sequence dataset, plus report-ready curves and metric tables. The report should present this as local virtual, approximately comparable methodological reproduction evidence, not as official paper-level reproduction.

## Next Step

Return to the main paper-facing control path: keep IsaacLab live headless gate green, refresh official replay/task evaluation evidence when needed, then push toward scaled-PPO checkpoint evaluation, policy rollout videos, and protocol-aligned Fig. 5/Fig. 6 local virtual metrics. For any new rollout or checkpoint evaluation milestone, generate or refresh MP4/screenshot/plot assets under `/mnt/infini-data/test/BeyondMimic/res/visualization/` or `/mnt/infini-data/test/BeyondMimic/res/report_assets/` and record claim level and limitations.

## Git Commit

Pending at time of writing. Commit message target: `feat: refresh scaled ppo vae state latent chain`.
