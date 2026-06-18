# Phase 2 Released Figures

Input dataset:

- Raw immutable archive: `/mnt/infini-data/test/BeyondMimic/download/official/Dataset_beyondmimic.zip`
- Working extraction: `/mnt/infini-data/test/BeyondMimic/reproduction/data/Dataset_beyondmimic`
- Dataset inventory: `/mnt/infini-data/test/BeyondMimic/reproduction/docs/released_dataset_inventory.tsv`

Main outputs:

- Summary: `/mnt/infini-data/test/BeyondMimic/res/released_figures/released_figure_summary.tsv`
- Execution logs:
  `/mnt/infini-data/test/BeyondMimic/logs/data/reproduce_released_figures.log`,
  `/mnt/infini-data/test/BeyondMimic/logs/data/convert_adaptive_sampling.log`,
  `/mnt/infini-data/test/BeyondMimic/logs/data/plot_adaptive_sampling_released.log`

Current figure groups:

- `imu_orientation_accel_angular_velocity`
- `ablation_observation_history`
- `ablation_armature`
- `ablation_pd_gain`
- `ablation_orientation_representation`
- `ablation_latency`
- `grf_walk_human_reference`
- `grf_run_human_reference`
- `grf_walk_robot_real`
- `grf_run_robot_real`
- `adaptive_sampling_w`
- `adaptive_sampling_wo`
- `adaptive_sampling_probability_evolution`

Each group stores processed CSV data, source hashes, paper-panel mapping notes, a run log, and PDF/SVG/PNG outputs.

Coverage against `goal.md` Level A:

| Level A item | Current evidence |
|---|---|
| IMU orientation | `imu_orientation_accel_angular_velocity` |
| acceleration | `imu_orientation_accel_angular_velocity` |
| angular velocity | `imu_orientation_accel_angular_velocity` |
| local tracking error | ablation local panels |
| global tracking error | ablation global panels |
| orientation representation ablation | `ablation_orientation_representation` |
| observation history ablation | `ablation_observation_history` |
| armature ablation | `ablation_armature` |
| latency ablation | `ablation_latency` |
| PD gain sensitivity | `ablation_pd_gain` |
| adaptive sampling failure map | `adaptive_sampling_w`, `adaptive_sampling_wo` |
| adaptive sampling probability evolution | `adaptive_sampling_probability_evolution` |
| walking/running GRF | `grf_walk_*`, `grf_run_*` |

Current output count from the summary directory: 13 figure rows, 20 PDF files, 20 SVG files, and 20 PNG files.

Exact paper/source mapping:

- TSV: `/mnt/infini-data/test/BeyondMimic/reproduction/docs/paper_panel_map.tsv`
- Paper source extraction: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source`
- Key source evidence:
  - Fig. 3B IMU orientation/acceleration/angular velocity: `root.tex:206-212`.
  - Fig. 4C GRF walking/running human-vs-humanoid comparison: `root.tex:215-219`.
  - Fig. 8A MDP ablations: `root.tex:258-267`.
  - Fig. 8B adaptive sampling training matrix/failure/probability evolution: `root.tex:258-267`.
  - Fig. S2 PD gain ablation: `root.tex:616-620`.
