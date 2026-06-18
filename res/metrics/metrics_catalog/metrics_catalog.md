# Metrics catalog

This catalog indexes current metric-bearing artifacts by evidence level.
It does not upgrade debug or blocked evidence into paper-level reproduction.

- Status: `ok`
- Source count: `23`
- Released-data source count: `4`
- Formula-API source count: `2`
- Debug-only source count: `11`
- Blocked-boundary source count: `2`
- Total indexed rows: `456`

| Metric group | Evidence level | Rows | Source |
|---|---:|---:|---|
| `released_tracking_ablation` | `released_data` | `30` | `res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv` |
| `released_grf` | `released_data` | `12` | `res/tables/released_data_metrics_summary/released_grf_metrics.csv` |
| `released_imu` | `released_data` | `10` | `res/tables/released_data_metrics_summary/released_imu_metrics.csv` |
| `released_data_statistical_audit` | `released_data` | `None` | `res/tables/released_data_statistical_audit/released_data_statistical_audit.json` |
| `paper_comparison` | `comparison` | `110` | `res/comparison/paper_vs_reproduction.csv` |
| `section12_coverage` | `coverage_audit` | `44` | `res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json` |
| `results_claims` | `coverage_audit` | `14` | `res/results_claims_audit/results_claims_audit.json` |
| `trial_failure_accounting` | `coverage_audit` | `14` | `res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json` |
| `goal_metric_api_contracts` | `formula_api` | `8` | `res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json` |
| `goal_metric_core_math` | `formula_api` | `23` | `res/tests/core_math_unit_tests/core_math_unit_tests.json` |
| `small_dataset_multiseed` | `debug_only` | `3` | `res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json` |
| `small_dataset_heldout_multiseed` | `debug_only` | `3` | `res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json` |
| `paper_state_heldout_multiseed` | `debug_only` | `3` | `res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json` |
| `vae_latent_heldout_multiseed` | `debug_only` | `3` | `res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json` |
| `guidance_task_scale_sweep` | `debug_only` | `40` | `res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json` |
| `guidance_task_metric_audit` | `debug_only` | `5` | `res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json` |
| `smoothness_latency` | `debug_only` | `None` | `res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json` |
| `diffusion_to_vae_action_smoothness` | `debug_only` | `3` | `res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json` |
| `direct_vs_latent_action_ablation` | `debug_only` | `3` | `res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json` |
| `state_latent_dataset_consistency` | `debug_only` | `84` | `res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json` |
| `reimpl_runtime_integration` | `debug_only` | `22` | `res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json` |
| `fig5_fig6_boundary` | `blocked_boundary` | `6` | `res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json` |
| `required_artifact_absence` | `blocked_boundary` | `16` | `res/required_artifact_absence/required_artifact_absence_audit.json` |

Outputs:
- `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.json`
- `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.csv`
- `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.md`
