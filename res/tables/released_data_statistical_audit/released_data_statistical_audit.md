# Released-data statistical audit

This audit adds confidence intervals and effect-size style summaries to Level A released-data outputs.
It is released-data evidence only, not a new training or rollout result.

- Status: `ok`
- Source CSV count: `10`
- Ablation comparison rows: `30`
- GRF CI rows: `12`
- IMU CI rows: `11`
- Best relative ablation improvement: `{'figure_id': 'ablation_pd_gain', 'scope': 'local', 'metric': 'ori_err', 'baseline_experiment': 'origin', 'best_experiment': 'wn25', 'relative_improvement': 0.12405441378347314, 'range_based_effect_size': 6.661494579260983}`
- IMU paper-claim comparison: `{"duration_s": 6.318044000072405, "mean_ang_abs_error": 2.1557431036475476, "paper_mean_ang_norm_rad_s": 7.01, "paper_peak_acc_norm_m_s2": 31.0, "paper_peak_ang_norm_rad_s": 20.0, "peak_acc_abs_error": 5.186530611276872, "peak_ang_abs_error": 3.235054179413506, "released_mean_ang_norm_rad_s": 4.854256896352452, "released_peak_acc_norm_m_s2": 36.18653061127687, "released_peak_ang_norm_rad_s": 16.764945820586494}`

Outputs:
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_data_statistical_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_grf_confidence_intervals.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_imu_confidence_intervals.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_data_statistical_audit.md`
