# Official-CSV-Loop PPO Multi-Seed Evaluation Assets

These plots and tables summarize three full 299-step local virtual evaluations of the
iteration-299 official-csv-loop PPO checkpoint.

## Source

- Multi-seed audit: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json`
- Rows CSV: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_rows.csv`
- Seeds: `[20260640, 20260641, 20260642]`
- GPU assignment: `[4, 7, 4]`
- Total env steps: `459264`

## Key Aggregate Metrics

- reward_mean: `0.025978426701298924` +/- `0.00010146760409522878`
- body_pos_error_mean: `0.18423418407697012` +/- `0.000271408645496586`
- joint_pos_error_mean: `1.2231450603159773` +/- `0.0027425904840304373`
- done_count_total: `13145.0` +/- `9.899494936611665`

## Assets

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/multiseed_reward_body_error_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/multiseed_eval_aggregate_bars.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/multiseed_eval_gpu_usage.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/multiseed_eval_aggregate_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/multiseed_eval_gpu_summary.csv`

## Claim Level

local_virtual_multiseed_tracking_eval. This is useful stability evidence for the reading report,
but it is not the unpatched official BeyondMimic tracking teacher, not paper-scale PPO evaluation,
not DAgger, not Fig. 5/Fig. 6 guided diffusion, and not real-robot validation.
