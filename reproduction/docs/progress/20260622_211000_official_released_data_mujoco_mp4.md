# Progress Update

## Goal

Create an `official_mp4/` package that uses all currently usable `Dataset_beyondmimic/` released data to generate MuJoCo videos and records the claim boundary.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/ablation/tkd_skill.csv`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_agile_motion/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_walk_and_run/*/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/GRF/**`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/base_imu/**`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_ablation/*/global/*.csv`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_common.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/configs/g1_joint_mapping.yaml`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/official_mp4/README_OFFICIAL_MP4.md`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/official_dataset_inventory.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/render_official_g1_csv_replay.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/render_official_mcap_joint_replay.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/official_mp4_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/run_official_mp4.sh`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_211000_official_released_data_mujoco_mp4.md`

## Commands Run

```bash
mujoco_mp4/.venv/bin/python official_mp4/scripts/official_dataset_inventory.py
MUJOCO_GL=osmesa mujoco_mp4/.venv/bin/python official_mp4/scripts/render_official_g1_csv_replay.py --csv Dataset_beyondmimic/ablation/tkd_skill.csv --motion-name official_zenodo_tkd_skill
MUJOCO_GL=osmesa mujoco_mp4/.venv/bin/python official_mp4/scripts/render_official_mcap_joint_replay.py --mcap <released agile/ablation/walk/run mcap> --frames 450 --stride 2-or-4
mujoco_mp4/.venv/bin/python official_mp4/scripts/official_mp4_manifest.py
ffprobe -v error ...
```

## Results

Generated 22 official released-data MuJoCo videos:

- `official_zenodo_tkd_skill`: 332 frames, 11.067 s, 1280x720.
- `official_agile_C1970_tkd_skill_clip1`: 450 frames, 15 s, 1280x720.
- `official_agile_C1975_side_flip`: 450 frames, 15 s, 1280x720.
- `official_agile_C1980_double_high_kick`: 450 frames, 15 s, 1280x720.
- `official_agile_C1985_merge2`: 450 frames, 15 s, 1280x720.
- `official_ablation_*`: 15 videos, each 450 frames, 15 s, 1280x720.
- `official_walk_rosbag2_2025_10_23_18_21_05`: 450 frames, 15 s, 1280x720.
- `official_run_rosbag2_2025_10_23_18_05_39`: 450 frames, 15 s, 1280x720.

## Verification

`official_mp4/official_mp4_manifest.json` reports `ok_official_mp4_manifest`, `mp4_count=22`, `row_count=88`, all indexed paths exist, all MP4s have ffprobe frame counts, and all rows avoid policy/closed-loop overclaims.

## Failed / Blocked Items

This is not policy control. GRF/IMU/adaptive-sampling/global-mocap files are inventoried as plot/metric or pending marker-analysis sources, not used as direct G1 qpos video input. The first EGL render attempt exited during rendering/teardown, so stable rendering uses `MUJOCO_GL=osmesa`.

## Effect on English Reading Report

Provides report-ready MuJoCo videos from official released data: one reference-motion replay plus 21 real-robot rosbag state replays. These can be described as official released-data MuJoCo visualization evidence and contrasted against future learned-policy closed-loop videos.

## Next Step

Use the generated videos in the report/PPT, then continue Stage 1 teacher training evaluation. Future extensions can add GRF overlay plots and MCAP-derived contact-force comparison, but not as policy reproduction claims.

## Git Commit

Pending.
