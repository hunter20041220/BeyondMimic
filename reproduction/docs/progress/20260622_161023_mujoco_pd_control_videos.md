# Progress Update

## Goal

Replace the misleading interpretation of `mujoco_mp4/res/rollout_videos` as control rollouts with actual MuJoCo `mj_step` control-form videos, while preserving honest claim boundaries. The target was to make longer, centered videos for reference/PPO/VAE/denoised/guided/guided-vs-unguided evidence.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_trace_mesh_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- Local PPO/VAE/guided trace `.npz` files under `/mnt/infini-data/test/BeyondMimic/res/runs/`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_control_video_summary.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_mp4_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_reference_replay_batch.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/run_mujoco_control_videos.sh`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/MIGRATION_AND_RESULTS_SUMMARY.md`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/mujoco_mp4_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/mujoco_mp4_manifest.tsv`

## Commands Run

```bash
mujoco_mp4/.venv/bin/python -m py_compile mujoco_mp4/scripts/mujoco_pd_control_video.py
MUJOCO_GL=osmesa BM_MUJOCO_CONTROL_SPECS=reference_control BM_MUJOCO_CONTROL_FRAMES=90 BM_MUJOCO_VIDEO_FPS=30 BM_MUJOCO_WIDTH=960 BM_MUJOCO_HEIGHT=540 mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_pd_control_video.py
MUJOCO_GL=osmesa BM_MUJOCO_CONTROL_SPECS=reference_control,ppo_policy_control,vae_base_control,denoised_latent_control,guided_latent_control BM_MUJOCO_CONTROL_FRAMES=450 BM_MUJOCO_VIDEO_FPS=30 BM_MUJOCO_WIDTH=960 BM_MUJOCO_HEIGHT=540 BM_MUJOCO_CONTROL_SUBSTEPS=4 BM_MUJOCO_CONTROL_SETTLE_STEPS=40 mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_pd_control_video.py
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_control_video_summary.py
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_mp4_manifest.py
```

## Results

Created six 15-second MuJoCo PD-control videos:

- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/reference_control/reference_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/ppo_policy_control/ppo_policy_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/vae_base_control/vae_base_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/denoised_latent_control/denoised_latent_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/guided_latent_control/guided_latent_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/guided_vs_unguided_control/guided_vs_unguided_control.mp4`

Each single video is 960x540, 450 frames, 30 fps, 15 seconds. The side-by-side video is 1920x540, 450 frames, 30 fps, 15 seconds.

## Verification

`mujoco_control_video_summary.py` passed with status `ok`, six videos present, all 450 frames and 15 seconds. Single-video summaries record:

- `uses_mj_step=true`
- `writes_qpos_each_frame=false`
- `uses_29_position_actuators=true`
- `uses_root_assist_controller=true`
- `fall_proxy_count=0`

`mujoco_mp4_manifest.py` passed with 379 rows.

## Failed / Blocked Items

Native MuJoCo PPO/VAE/guided rollout is still blocked by the missing faithful IsaacLab 160-D observation/action adapter and by the fact that existing video trace `.npz` files contain action statistics rather than full 29-D action sequences. The new videos are therefore MuJoCo PD target-tracking control visualizations with root assist, not native MuJoCo policy rollouts.

## Effect on English Reading Report

This round provides stronger report/PPT visual evidence than skeleton plots or qpos replay because the videos are produced through MuJoCo physics stepping with G1 mesh and actuators. The report must still state that these are local virtual control visualizations, not official IsaacLab rendered videos, not native MuJoCo PPO control, not paper-level Fig.5/Fig.6, and not real robot results.

## Next Step

Use the new videos in the report as simulation-side visual evidence, then decide whether to invest in a faithful MuJoCo adapter for the IsaacLab 160-D policy observation contract or return to improving the IsaacLab tracking teacher quality.

## Git Commit

Pending.
