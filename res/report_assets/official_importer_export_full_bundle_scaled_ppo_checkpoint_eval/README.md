# Scaled official-importer-export PPO training curve Eval Assets

These plots and tables summarize the local virtual PPO checkpoint evaluation using the
official-importer GPU4 G1 USDA export and the 40-motion official-loop public bundle.

## Source

- Eval audit: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`
- Eval metrics: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260620_193435_seed20260697/eval_metrics.json`
- Training audit: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json`
- Training log: `/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.log`
- Timeseries: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260620_193435_seed20260697/eval_timeseries.csv`
- GPU telemetry: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260620_193435_seed20260697/gpu_metrics.csv`
- Checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training/resource_adjusted_ppo_20260620_183959_seed20260696/rank_0/model_999.pt`
- Status: `ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed`

## Assets

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_error_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/reward_done_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/gpu_usage_eval.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/training_curve.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/training_curve.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/ppo_checkpoint_eval_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/ppo_checkpoint_eval_gpu_summary.csv`

## Claim Level

local_virtual_official_importer_export_scaled_ppo_report_asset / qualitative engineering evidence. This is not a
released official BeyondMimic PPO checkpoint, not paper-scale teacher training, not Fig. 5/Fig. 6
guided diffusion, and not real-robot validation.
