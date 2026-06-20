# Official-Importer-Export VAE Closed-Loop Rollout Assets

These plots summarize a local virtual rollout where the full-bundle PPO teacher action is encoded
and decoded through the local official-importer-export conditional action VAE before stepping IsaacLab.

## Source

- Rollout summary: `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json`
- Status: `ok_official_importer_export_full_bundle_vae_closed_loop_rollout_eval`
- Total env steps: `918528`
- Teacher/VAE action MSE mean: `5.015458783269533e-05`
- Teacher/VAE action absolute-error mean: `0.005258061872471286`

## Assets

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_reward_done_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_action_reconstruction_error.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_action_magnitude.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_tracking_errors.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_gpu_memory.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_shard_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/official_importer_vae_closed_loop_gpu_summary.csv`

## Boundary

These assets summarize a local virtual VAE action-reconstruction rollout on the official-importer-export G1 USDA. They do not use unreleased official BeyondMimic VAE/diffusion checkpoints, do not evaluate receding-horizon guided diffusion, do not reproduce Fig. 5/Fig. 6, and do not contain real-robot evidence.
