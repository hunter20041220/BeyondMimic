# Official-Loop PPO Checkpoint Evaluation Assets

These plots and tables summarize the local virtual PPO checkpoint evaluation on the official-loop
motion chain. They are intended for the English reading report and presentation slides.

## Source

- Eval audit: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json`
- Eval metrics: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260619_025210_seed20260630/eval_metrics.json`
- Timeseries: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260619_025210_seed20260630/eval_timeseries.csv`
- GPU telemetry: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260619_025210_seed20260630/gpu_metrics.csv`
- Checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_ppo_training/resource_adjusted_ppo_20260618_224626_seed20260629/rank_0/model_299.pt`
- Status: `ok_official_csv_loop_ppo_checkpoint_eval_completed`

## Assets

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_eval/tracking_error_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_eval/reward_done_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_eval/gpu_usage_eval.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_eval/ppo_checkpoint_eval_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_eval/ppo_checkpoint_eval_gpu_summary.csv`

## Claim Level

local_virtual_resource_adjusted / approximately comparable engineering evidence. This is not
unpatched official BeyondMimic PPO evaluation, not paper-scale teacher training, not Fig. 5/Fig. 6
guided diffusion, and not real-robot validation.
