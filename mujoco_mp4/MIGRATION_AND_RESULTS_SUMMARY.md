# MuJoCo MP4 H20 Results Summary

## Purpose

This directory is an independent MuJoCo MP4 route for the BeyondMimic reproduction project. It was created because true Isaac rendered MP4 generation on this H20 server is blocked in the Isaac Sim Kit/Hydra/Vulkan/Replicator rendering startup path before `Tracking-Flat-G1-v0` is created.

Claim level: MuJoCo local virtual simulation evidence. These videos are not IsaacLab results, not real robot results, and not paper-level Fig.5/Fig.6 closed-loop reproduction.

## Environment

- Package root: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4`
- Python environment: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/.venv`
- Python: 3.10.12
- MuJoCo: 3.9.0
- Torch: 2.12.1+cu130
- Preferred render backend on H20: `MUJOCO_GL=egl`
- Fallback backend also tested successfully: `MUJOCO_GL=osmesa`
- Dependency lock: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/requirements-lock.txt`

The venv is intentionally local to `mujoco_mp4/` and should not be committed to GitHub.

## Successful Results

### Backend And Minimal Rendering Smoke

- Script: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_render_backend_probe.py`
- Summary: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/smoke/mujoco_render_backend_probe.json`
- EGL minimal MP4: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/smoke/minimal_scene_egl.mp4`
- OSMesa minimal MP4: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/smoke/minimal_scene_osmesa.mp4`
- Result: EGL and OSMesa both rendered RGB frames, keyframes, and MP4 files.

### G1 Asset Import And Render Smoke

- Script: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_g1_import_smoke.py`
- Selected XML: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- Summary: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/g1_import/mujoco_g1_import_smoke.json`
- MP4: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/g1_import/g1_mocap_29dof_egl_g1_import_smoke.mp4`
- Keyframe: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/g1_import/g1_mocap_29dof_egl_g1_import_keyframe.png`
- Model dimensions: `nq=36`, `nv=35`, `nu=29`, `nbody=39`, `njnt=30`, `ngeom=72`, `nmesh=35`
- Video: 1280x720, 120 frames, 30 fps, 4 seconds.

### Reference Replay Video

- Script: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- Motion source: `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz`
- Joint mapping: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/configs/g1_joint_mapping.yaml`
- Summary: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/reference_replay/walk1_subject1/reference_replay_summary.json`
- MP4: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/reference_replay/walk1_subject1/reference_replay.mp4`
- Metrics CSV: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/reference_replay/walk1_subject1/reference_replay_metrics.csv`
- Keyframe: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/reference_replay/walk1_subject1/reference_replay_keyframe.png`
- Video: 1280x720, 299 frames, 30 fps, 9.97 seconds.

Reference replay imposes root pose and 29 joint angles frame-by-frame with `mj_forward`; it is useful for report/PPT visualization, but it is not policy closed-loop control.

## MuJoCo PD-Control Videos

The first `rollout_videos/` assets were trace-to-mesh renderings: they turned existing local IsaacLab/proxy body traces into MuJoCo mesh videos. They are useful visualizations, but they are not MuJoCo control rollouts.

The newer `control_videos/` assets are different. They use a MuJoCo G1 model with 29 position-servo actuators, apply target joint trajectories, call `mj_step`, and render the resulting physics state. A marked pelvis root-assist stabilizer keeps the robot centered/upright in the fixed camera. This is a local MuJoCo PD closed-loop tracking-control visualization, not a native PPO/VAE/guided policy adapter and not a paper-level result.

Generated 15-second, 450-frame videos:

- Reference PD control: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/reference_control/reference_control.mp4`
- PPO-target PD control: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/ppo_policy_control/ppo_policy_control.mp4`
- VAE-base PD control: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/vae_base_control/vae_base_control.mp4`
- Denoised-latent PD control: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/denoised_latent_control/denoised_latent_control.mp4`
- Guided-latent PD control: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/guided_latent_control/guided_latent_control.mp4`
- Guided-vs-unguided side-by-side PD control: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/guided_vs_unguided_control/guided_vs_unguided_control.mp4`

Verification summary:

- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/mujoco_control_video_summary.json`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/mujoco_control_video_summary.tsv`

All six videos are 15.0 seconds and 450 frames. Single videos are 960x540; the guided-vs-unguided comparison is 1920x540. The single-video summaries record `uses_mj_step=true`, `writes_qpos_each_frame=false`, `uses_29_position_actuators=true`, `uses_root_assist_controller=true`, and `fall_proxy_count=0`.

Run command:

```bash
cd /mnt/infini-data/test/BeyondMimic
MUJOCO_GL=osmesa ./mujoco_mp4/run_mujoco_control_videos.sh
```

Remaining native-controller blocker: a trustworthy MuJoCo reconstruction of the IsaacLab 160-D policy observation manager and action semantics is still not complete. The available video trace files save action means/maxima, not complete 29-D action vectors for each frame. Therefore these control videos must be described as target-tracking PD control visualizations, not native MuJoCo PPO/VAE/guided policy rollouts.

## Commands

Run smoke:

```bash
cd /mnt/infini-data/test/BeyondMimic
MUJOCO_GL=egl ./mujoco_mp4/run_mujoco_smoke.sh
```

Run reference replay:

```bash
cd /mnt/infini-data/test/BeyondMimic
MUJOCO_GL=egl BM_MUJOCO_MOTION_NAME=walk1_subject1 ./mujoco_mp4/run_mujoco_reference_replay.sh
```

Try OSMesa fallback:

```bash
cd /mnt/infini-data/test/BeyondMimic
MUJOCO_GL=osmesa ./mujoco_mp4/run_mujoco_smoke.sh
```

## Manifest

- JSON: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/mujoco_mp4_manifest.json`
- TSV: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/mujoco_mp4_manifest.tsv`

Manifest count: 229 files at the time of this summary.

## Report Use

Safe wording for the English report:

> Because Isaac Sim rendering is blocked on the H20 server, I added an independent MuJoCo visualization route. It verifies that the server can render offscreen robot simulation videos with G1 assets and can replay the FK-repaired public G1 reference motion as a mesh video. This is local virtual visualization evidence, not an official IsaacLab rollout, not closed-loop policy control, and not real-robot validation.

Current report-ready evidence:

- MuJoCo minimal rendering backend proof.
- MuJoCo G1 mesh import/render proof.
- MuJoCo G1 reference replay visualization for `walk1_subject1`.
- MuJoCo PD closed-loop tracking-control videos for reference, PPO-target, VAE-base, denoised-latent, guided-latent, and guided-vs-unguided comparison. These are local virtual control visualizations with root assist, not native PPO/VAE/guided policy control.

## 15-Second Trace-To-Mesh Rollout Videos

The H20 server's MuJoCo EGL backend can render short clips but repeatedly aborted near 50 frames for longer trace-to-mesh videos. The stable route for these videos is:

```bash
MUJOCO_GL=osmesa BM_MUJOCO_TRACE_RENDER_MODE=root_qpos_replay BM_MUJOCO_TRACE_FRAMES=450 BM_MUJOCO_WIDTH=960 BM_MUJOCO_HEIGHT=540 ...
```

This OSMesa route generated 450-frame, 30 fps, 15-second videos:

- PPO policy trace mesh video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/ppo_policy/ppo_policy.mp4`
- VAE base trace mesh video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/vae_base/vae_base.mp4`
- Guided latent trace mesh video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/guided_latent/guided_latent.mp4`
- Guided-vs-unguided side-by-side video: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/rollout_videos/guided_vs_unguided/guided_vs_unguided.mp4`

Each single video is 960x540, 450 frames, 15 seconds. The side-by-side video is 1920x540, 450 frames, 15 seconds.

Claim boundary: these are MuJoCo G1 mesh renderings of existing local IsaacLab closed-loop/proxy rollout traces. They are not native MuJoCo PPO/VAE/guided controllers, not Isaac rendered MP4, not real robot, and not paper-level Fig.5/Fig.6.

Native MuJoCo PPO adapter gap audit:

- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.json`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.tsv`

The audit confirms that the PPO checkpoint actor has input/output dimensions `160 -> 29`, and the official schema audit derives a 160-D policy observation. It also records that the native MuJoCo command/observation/action/normalization adapter is not complete, so native MuJoCo PPO closed-loop control is not claimed.

Next meaningful step: implement and audit a MuJoCo adapter for the IsaacLab tracking observation/action contract before attempting PPO, VAE, or guided closed-loop videos.
