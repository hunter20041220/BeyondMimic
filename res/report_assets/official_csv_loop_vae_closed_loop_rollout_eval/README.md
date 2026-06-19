# Official-Loop VAE Closed-Loop Rollout Assets

These plots summarize a local virtual rollout where the PPO teacher action is encoded and decoded
through the local official-csv-loop conditional action VAE before stepping IsaacLab.

## Source

- Rollout summary: `/mnt/infini-data/test/BeyondMimic/res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json`
- Status: `ok_official_csv_loop_vae_closed_loop_rollout_eval`
- Total env steps: `612352`
- Teacher/VAE action MSE mean: `0.004145793081027608`
- Teacher/VAE action absolute-error mean: `0.04706366988752399`

## Assets

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_reward_done_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_action_reconstruction_error.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_action_magnitude.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_tracking_errors.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_gpu_memory.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_shard_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_gpu_summary.csv`

## Boundary

These assets summarize a local virtual VAE action-reconstruction rollout. They do not use the unreleased official BeyondMimic VAE/diffusion checkpoints, do not evaluate receding-horizon guided diffusion, do not reproduce Fig. 5/Fig. 6, and do not contain real-robot evidence.
