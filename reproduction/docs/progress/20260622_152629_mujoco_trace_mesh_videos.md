# Progress Update

## Goal

Generate report-ready MuJoCo G1 mesh videos for the previously missing PPO, VAE, guided latent, and guided-vs-unguided rollout visualizations while honestly resolving the native MuJoCo PPO adapter blocker.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_task_smoke/tracking_g1_official_importer_export_task_smoke.json`
- `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture/policy_rollout_capture_20260621_131648_seed20260722/official_csv_loop_policy_rollout_body_pose_trace.npz`
- `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/joystick/receding_latent_guidance_rollout_20260621_064022_seed20260705/official_csv_loop_receding_latent_guidance_rollout_trace.npz`
- `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training/resource_adjusted_ppo_20260621_121940_seed20260720/rank_0/model_999.pt`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_trace_mesh_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_ppo_adapter_gap_audit.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/MIGRATION_AND_RESULTS_SUMMARY.md`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/mujoco_mp4_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/mujoco_mp4_manifest.tsv`

## Commands Run

```bash
MUJOCO_GL=osmesa BM_MUJOCO_TRACE_RENDER_MODE=root_qpos_replay BM_MUJOCO_TRACE_FRAMES=450 BM_MUJOCO_TRACE_SPECS=ppo_policy BM_MUJOCO_WIDTH=960 BM_MUJOCO_HEIGHT=540 mujoco_mp4/.venv/bin/python -u mujoco_mp4/scripts/mujoco_trace_mesh_video.py
MUJOCO_GL=osmesa BM_MUJOCO_TRACE_RENDER_MODE=root_qpos_replay BM_MUJOCO_TRACE_FRAMES=450 BM_MUJOCO_TRACE_SPECS=vae_base BM_MUJOCO_WIDTH=960 BM_MUJOCO_HEIGHT=540 mujoco_mp4/.venv/bin/python -u mujoco_mp4/scripts/mujoco_trace_mesh_video.py
MUJOCO_GL=osmesa BM_MUJOCO_TRACE_RENDER_MODE=root_qpos_replay BM_MUJOCO_TRACE_FRAMES=450 BM_MUJOCO_TRACE_SPECS=guided_latent BM_MUJOCO_WIDTH=960 BM_MUJOCO_HEIGHT=540 mujoco_mp4/.venv/bin/python -u mujoco_mp4/scripts/mujoco_trace_mesh_video.py
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_ppo_adapter_gap_audit.py
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_mp4_manifest.py
```

## Results

- PPO MuJoCo mesh video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/ppo_policy/ppo_policy.mp4`
- VAE base MuJoCo mesh video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/vae_base/vae_base.mp4`
- Guided latent MuJoCo mesh video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/guided_latent/guided_latent.mp4`
- Guided-vs-unguided MuJoCo side-by-side video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/guided_vs_unguided/guided_vs_unguided.mp4`
- Native MuJoCo PPO adapter gap audit: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.json`

## Verification

- `ffprobe` verified PPO, VAE base, and guided latent videos as 960x540, 450 frames, 30 fps, 15 seconds.
- `ffprobe` verified guided-vs-unguided as 1920x540, 450 frames, 30 fps, 15 seconds.
- Summary JSON files record `status=ok`, `frames_rendered=450`, and claim boundaries.
- MuJoCo manifest regenerated with 269 rows.

## Failed / Blocked Items

- EGL long-video rendering repeatedly aborted near 50 frames with empty logs. OSMesa was used for stable 450-frame rendering.
- Native MuJoCo PPO closed-loop control is still not claimed. The checkpoint and schema are audited, but the IsaacLab command manager, observation corruption/noise, normalization, reset, and termination semantics are not yet reimplemented in MuJoCo.

## Effect on English Reading Report

These videos are now suitable as report/PPT visual evidence for local virtual MuJoCo mesh rendering of existing closed-loop/proxy rollout traces. The report must state that they are not official Isaac rendered rollouts, not native MuJoCo controller deployments, not real robot evidence, and not paper-level Fig.5/Fig.6 reproduction.

## Next Step

If native MuJoCo control is required, implement and validate the full IsaacLab-to-MuJoCo observation/action adapter before loading the PPO actor for live MuJoCo `mj_step` control.

## Git Commit

Pending at time of writing.
