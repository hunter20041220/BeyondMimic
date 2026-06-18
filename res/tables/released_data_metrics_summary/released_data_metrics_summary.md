# Released-data metrics summary

This table is derived from already reproduced released-data processed CSV files under `res/released_figures`.
It is Level A released-data evidence, not a new policy training run.

- Status: `ok`
- Source CSV count: `10`
- Ablation summary rows: `30`
- GRF summary rows: `12`
- IMU summary rows: `10`
- Best global position ablation: `{'figure_id': 'ablation_orientation_representation', 'best_experiment': 'quat', 'best_mean': 0.2161546877936451, 'baseline_experiment': 'origin', 'baseline_mean': 0.2350160799679482}`
- Peak vertical GRF abs value: `2.316238655046181`
- IMU duration seconds: `6.318044000072405`

Outputs:
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_data_metrics_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_grf_metrics.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_imu_metrics.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/source_hashes.tsv`
