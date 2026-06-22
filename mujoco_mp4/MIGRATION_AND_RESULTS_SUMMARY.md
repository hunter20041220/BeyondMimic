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

## Not Yet Completed

- MuJoCo PPO closed-loop rollout video is not complete.
- MuJoCo VAE rollout video is not complete.
- MuJoCo denoised/guided latent rollout video is not complete.
- Guided-vs-unguided MuJoCo comparison video/contact sheet is not complete.

The current blocker is not MuJoCo rendering. Rendering works. The remaining blocker is building a trustworthy MuJoCo obs/action/controller adapter for the IsaacLab-trained tracking policy. The current best PPO checkpoints exist under `/mnt/infini-data/test/BeyondMimic/res/runs/`, but their observation contract is IsaacLab-specific (`obs_dim=160`, `action_dim=29`) and must be reconstructed carefully before claiming closed-loop policy rollout.

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

Next meaningful step: implement and audit a MuJoCo adapter for the IsaacLab tracking observation/action contract before attempting PPO, VAE, or guided closed-loop videos.
