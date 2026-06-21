# Storage Cleanup Report

## Scope

This report records a conservative storage audit for the BeyondMimic reproduction workspace after the robot-order FK motion repair. It does not delete checkpoints, videos, rollout shards, datasets, raw downloads, `other/`, or environment directories.

## Current Disk Picture

- Project tree size from `du -h -d 2`: about `81G`.
- `envs/`: about `22G`.
- `other/`: about `15G`, preserved as the old-server backup.
- `download/`: about `7.8G`, treated as read-only source material.
- `res/runs/`: about `9.8G`, contains checkpoints and rollout shards.
- `res/level_c/`: about `1.6G`.
- `res/tracking/`: about `485M`.
- `res/visualization/`: about `263M`.

## Largest Candidates

The largest non-environment files found outside `download/`, `other/`, and `envs/` are:

| Approx Size | Path | Decision |
|---:|---|---|
| `1.9G` | `reproduction/data/Dataset_beyondmimic/rosbag_walk_and_run/run_rosbag2_2025_10_23-18_05_39/...mcap` | Keep for released dataset evidence unless space becomes critical. |
| `1.6G` | `reproduction/data/Dataset_beyondmimic/rosbag_ablation/5ms_rosbag2_2025_10_22-03_41_01/...mcap` | Keep for released dataset evidence unless space becomes critical. |
| `1.6G` | `reproduction/data/Dataset_beyondmimic/rosbag_walk_and_run/walk_rosbag2_2025_10_23-18_21_05/...mcap` | Keep for released dataset evidence unless space becomes critical. |
| `1.1G` | `reproduction/data/Dataset_beyondmimic/rosbag_ablation/10ms_rosbag2_2025_10_22-03_36_03/...mcap` | Keep for released dataset evidence unless space becomes critical. |
| `916M x 4` | duplicate scaled-PPO teacher rollout shards under `res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/` | Candidate for later deduplication after confirming which run is the current best source. |
| `289M x several` | LAFAN1 paper-architecture VAE/diffusion checkpoints under `res/runs/level_c_*` | Keep as completed local Level C evidence; do not delete before report export. |

## Safe Cleanup Performed

Only rebuildable Python/runtime caches were removed:

```bash
find res logs tmp cache reproduction -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
```

Post-cleanup cache directory count:

```text
0
```

## Not Deleted This Round

The following were deliberately not deleted:

- `download/` raw source material.
- `other/` old workspace backup.
- `envs/` conda environments.
- `res/runs/` checkpoints and rollout shards.
- `res/visualization/` videos.
- `reproduction/data/Dataset_beyondmimic/*.mcap` released dataset bags.
- robot-order FK `.npz` files, because they are the current next-PPO input candidate.

## Next Cleanup Step

Before the next large PPO run, make a deduplication decision for the two scaled-PPO teacher rollout runs with the same seed:

- `resource_adjusted_teacher_rollout_20260620_195754_seed20260700`
- `resource_adjusted_teacher_rollout_20260621_060339_seed20260700`

Keep the run that is referenced by the current report/manifest or has stronger downstream continuity, then preserve its JSON/CSV/hash summary before deleting duplicate shards.
