# Progress Update

## Goal

Correct the semantics of the LAFAN1 reference video after observing that `reference_action_control.mp4` did not behave like a clean source-motion replay.

## Files Read

- `reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py`
- `mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- `mujoco_mp4/scripts/mujoco_trace_mesh_video.py`
- `res/visualization/lafan1_paper_contract_videos/reference_action_control/reference_action_control_summary.json`
- `res/visualization/lafan1_paper_contract_videos/reference_action_control/reference_action_control_metrics.csv`
- `res/visualization/lafan1_paper_contract_videos/lafan1_paper_contract_video_suite_summary.json`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz`

## Files Modified

- `reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`

## Commands Run

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= BM_LAFAN1_VIDEO_FRAMES=450 BM_LAFAN1_VIDEO_FPS=30 \
  /mnt/infini-data/test/BeyondMimic/mujoco_mp4/.venv/bin/python \
  reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py
```

Additional read-only diagnostics checked the selected teacher rollout `motion_time_steps`, reference-control metrics, MP4 duration, and keyframes.

## Results

The old `reference_action_control.mp4` was confirmed to be a PD-control diagnostic, not a clean source-dataset replay. Its selected teacher rollout time steps have 35 non-+1 jumps, 18 negative jumps, and large discontinuities from command resets/re-sampling. Therefore it can look broken even when the source motion itself is valid.

A new clean source-motion video was generated:

- `res/visualization/lafan1_paper_contract_videos/reference_pose_replay/reference_pose_replay.mp4`
- `res/visualization/lafan1_paper_contract_videos/reference_pose_replay/reference_pose_replay_keyframe.png`
- `res/visualization/lafan1_paper_contract_videos/reference_pose_replay/reference_pose_replay_keyframes.png`
- `res/visualization/lafan1_paper_contract_videos/reference_pose_replay/reference_pose_replay_metrics.csv`
- `res/visualization/lafan1_paper_contract_videos/reference_pose_replay/reference_pose_replay_summary.json`

This new video writes the continuous `walk1_subject1` root pose and 29 joint positions frame-by-frame, calls `mj_forward`, and renders the MuJoCo G1 mesh. It is a kinematic reference replay, not control evidence.

## Verification

- `reference_pose_replay.mp4`: 450 frames, 30 FPS, 15.0 seconds.
- Keyframe strip was visually inspected and shows a visible G1 walking sequence.
- Top-level video suite summary now distinguishes `reference_pose_replay` from `reference_action_control`.

## Failed / Blocked Items

This does not fix the weak Stage-1 teacher or downstream VAE/diffusion control quality. It fixes the reference-video interpretation error and adds a correct continuous-reference visualization.

## Effect on English Reading Report

The report can now separate:

- clean source-motion visualization: `reference_pose_replay`;
- MuJoCo PD/control diagnostic: `reference_action_control`;
- learned-controller chain evidence: teacher/VAE/diffusion/guidance action-control videos.

This makes the reproduction narrative more honest and avoids presenting a discontinuous teacher-rollout diagnostic as the original LAFAN1 reference motion.

## Next Step

Wait for the 5/6 multi-source PPO training to finish, run checkpoint sweep, and regenerate teacher/VAE/diffusion/guidance control videos only if the multi-source teacher improves.

## Git Commit

Pending.
