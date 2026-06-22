# Official Released Data MuJoCo MP4

This folder collects MuJoCo videos generated from the released `Dataset_beyondmimic/` package.

## Claim Boundary

The videos here are official released data visualizations in MuJoCo. They are not official BeyondMimic policy checkpoints, not diffusion/guidance closed-loop control, not IsaacLab rendered videos, and not real-robot deployment.

## Direct Video Sources

The direct G1 mesh video sources currently used are:

- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/ablation/tkd_skill.csv`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_agile_motion/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_ablation/*/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_walk_and_run/*/*.mcap`

The CSV file has 36 generalized-coordinate columns:

- root position, 3 columns
- root quaternion in xyzw order, 4 columns
- G1 joint positions, 29 columns

It can be rendered in MuJoCo by imposing the reference state frame-by-frame and calling `mj_forward`.

The MCAP files are parsed for `/joint_states` and `/odom`. They are rendered in MuJoCo by imposing the recorded robot state frame-by-frame and calling `mj_forward`.

## Other Released Data

The `GRF/` CSV files are ground reaction force data for paper plots and simulation-vs-real contact-force comparison. They are not robot joint trajectories.

The MuJoCo venv includes `mcap` and `mcap-ros2-support`, so the official agile, ablation, walk, and run MCAP files can be rendered as recorded joint/odom state replay videos. These are still kinematic replays of released records, not closed-loop controller rollouts.

The `rosbag_ablation/*/global/*.csv` files are Motive/global mocap rigid-body and marker CSVs. They are useful for tracking-error analysis and possible marker visualization, but they do not directly provide 29 G1 joint positions.

## Run

```bash
cd /mnt/infini-data/test/BeyondMimic
bash official_mp4/run_official_mp4.sh
```

## Outputs

- `official_mp4/videos/<motion>/`: MP4 and keyframe PNG
- `official_mp4/res/<motion>/`: metrics CSV and summary JSON
- `official_mp4/official_dataset_inventory.json`
- `official_mp4/official_dataset_inventory.tsv`
- `official_mp4/official_mp4_manifest.json`
- `official_mp4/official_mp4_manifest.tsv`

## Current Videos

Current generated videos: 22 MP4 files.

- `official_zenodo_tkd_skill`: 36-column released reference motion replay.
- `official_agile_C1970_tkd_skill_clip1`: released real-robot MCAP joint/odom replay.
- `official_agile_C1975_side_flip`: released real-robot MCAP joint/odom replay.
- `official_agile_C1980_double_high_kick`: released real-robot MCAP joint/odom replay.
- `official_agile_C1985_merge2`: released real-robot MCAP joint/odom replay.
- `official_ablation_*`: 15 released ablation MCAP joint/odom replays.
- `official_walk_rosbag2_2025_10_23_18_21_05`: released walking MCAP joint/odom replay.
- `official_run_rosbag2_2025_10_23_18_05_39`: released running MCAP joint/odom replay.

All generated MP4s are 1280x720. The MCAP videos are 450-frame, 15-second MuJoCo state replays. The `tkd_skill` CSV video is 332 frames, about 11.07 seconds.
