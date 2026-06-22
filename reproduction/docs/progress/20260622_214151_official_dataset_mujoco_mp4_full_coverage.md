# Progress Update

## Goal

Use the official `Dataset_beyondmimic/` released data that can be rendered as robot motion/state, generate MuJoCo MP4 videos under `official_mp4/`, and keep the claim boundary explicit.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/ablation/tkd_skill.csv`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_agile_motion/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_ablation/*/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/rosbag_walk_and_run/*/*.mcap`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/GRF/**`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/base_imu/**`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_common.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/official_mp4/README_OFFICIAL_MP4.md`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/official_dataset_inventory.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/render_official_mcap_joint_replay.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/official_mp4_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/run_official_mp4.sh`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_211000_official_released_data_mujoco_mp4.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_214151_official_dataset_mujoco_mp4_full_coverage.md`

## Commands Run

```bash
mujoco_mp4/.venv/bin/python -m pip install mcap mcap-ros2-support
mujoco_mp4/.venv/bin/python official_mp4/scripts/official_dataset_inventory.py
MUJOCO_GL=osmesa mujoco_mp4/.venv/bin/python official_mp4/scripts/render_official_g1_csv_replay.py --csv Dataset_beyondmimic/ablation/tkd_skill.csv --motion-name official_zenodo_tkd_skill
MUJOCO_GL=osmesa mujoco_mp4/.venv/bin/python official_mp4/scripts/render_official_mcap_joint_replay.py --mcap <official mcap> --frames 450 --stride 2-or-4
mujoco_mp4/.venv/bin/python official_mp4/scripts/official_mp4_manifest.py
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,duration,avg_frame_rate -of csv=p=0 <mp4>
```

## Results

Generated 22 official released-data MuJoCo videos under `/mnt/infini-data/test/BeyondMimic/official_mp4/videos/`.

- 1 released 36-column G1 reference replay: `official_zenodo_tkd_skill`.
- 4 agile-motion real-robot MCAP joint/odom state replays.
- 15 ablation real-robot MCAP joint/odom state replays.
- 2 walk/run real-robot MCAP joint/odom state replays.

All 21 MCAP videos are 450 frames, 15 seconds, and 1280x720. The `tkd_skill` CSV video is 332 frames, about 11.067 seconds, and 1280x720.

The inventory classifies 414 released dataset files. GRF, IMU, adaptive-sampling, and global-mocap CSV files are retained as plot/metric/analysis sources, not direct G1 motion-video sources.

## Verification

`official_mp4/official_mp4_manifest.json` reports:

```text
status: ok_official_mp4_manifest
mp4_count: 22
row_count: 88
all_mp4_have_ffprobe_frames: true
has_all_known_official_video_sources: true
has_fifteen_ablation_mcap_replays: true
has_four_agile_mcap_replays: true
has_walk_and_run_mcap_replays: true
has_tkd_skill_reference_replay: true
all_rows_avoid_policy_claim: true
```

## Failed / Blocked Items

This is not policy control. These videos impose released qpos or recorded joint/odom states and call `mj_forward`; they do not execute a learned policy, VAE decoder, diffusion model, guidance controller, or PD closed-loop controller.

Stable rendering used `MUJOCO_GL=osmesa`. Isaac rendered MP4 remains blocked on this H20 host by the Isaac Sim Kit/Hydra/Vulkan/Replicator rendering stack.

## Effect on English Reading Report

This provides report-ready official released-data MuJoCo visualization evidence. The report should describe these as official data replay videos, useful for qualitative inspection and presentation, while clearly separating them from learned closed-loop control results.

## Next Step

Continue the Stage 1 teacher-policy training/evaluation line. Once a stronger teacher policy is available, generate true MuJoCo closed-loop videos for PPO, VAE, unguided diffusion, guided diffusion, and guided-vs-unguided comparisons.

## Git Commit

Pending.
