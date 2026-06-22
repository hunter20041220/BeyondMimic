# IsaacLab Rendered Policy Rollout MP4

This directory is reserved for true IsaacLab/Isaac Sim rendered rollout videos. Successful videos
must be created by
`AppLauncher(headless=True, enable_cameras=True)`, `Tracking-Flat-G1-v0`, an Isaac Sim offscreen
camera render product, and a PPO policy stepped through the real IsaacLab physics environment.

Current gate status is recorded in the latest asset JSON. If the status is failed, no paper-facing
simulation MP4 should be claimed from this directory.

They are qualitative local virtual simulation-deployment evidence only. They are not official
BeyondMimic checkpoints, Fig. 5/Fig. 6 paper-level videos, TensorRT deployment evidence, or real
robot results.

Latest asset JSON: `/mnt/infini-data/test/BeyondMimic/res/visualization/isaac_mp4/isaaclab_rendered_policy_rollout_video_asset.json`
Latest MP4: `/mnt/infini-data/test/BeyondMimic/res/visualization/isaac_mp4/20260622_044239_seed20260779_10step_robot_order_policy.mp4`
Latest metrics CSV: `/mnt/infini-data/test/BeyondMimic/res/visualization/isaac_mp4/20260622_044239_seed20260779_10step_robot_order_policy_metrics.csv`
