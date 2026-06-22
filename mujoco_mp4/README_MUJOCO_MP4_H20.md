# MuJoCo MP4 Experiments On H20

This directory is an independent MuJoCo video reproduction package for the BeyondMimic project. It is intentionally separate from the IsaacLab/Isaac Sim reproduction code and does not overwrite existing artifacts.

## Why This Exists

The H20 server cannot currently produce true IsaacLab/Isaac Sim rendered MP4s. The recorded failure happens during Isaac Sim Kit/Hydra/Vulkan/Replicator rendering startup, before `Tracking-Flat-G1-v0` is created and before any PPO checkpoint, policy rollout, or physics rollout is evaluated.

MuJoCo is a reasonable fallback route for report-ready local virtual simulation evidence because it supports lightweight headless/offscreen rendering through EGL or OSMesa and has multiple available Unitree G1 MJCF/XML assets in the local `download/` tree.

Claim level for this directory:

```text
MuJoCo local virtual simulation evidence.
Not official IsaacLab evidence.
Not real robot evidence.
Not paper-level Fig.5/Fig.6 reproduction.
```

## Environment

The isolated Python environment is:

```bash
/mnt/infini-data/test/BeyondMimic/mujoco_mp4/.venv
```

It was created with:

```bash
cd /mnt/infini-data/test/BeyondMimic
python3 -m venv mujoco_mp4/.venv
mujoco_mp4/.venv/bin/python -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR=$PWD/mujoco_mp4/cache/pip mujoco_mp4/.venv/bin/python -m pip install \
  mujoco numpy scipy pandas matplotlib imageio imageio-ffmpeg opencv-python-headless \
  tqdm pyyaml torch onnx onnxruntime gymnasium trimesh lxml
```

Installed key packages:

- `mujoco==3.9.0`
- `numpy==2.2.6`
- `scipy==1.15.3`
- `pandas==2.3.3`
- `matplotlib==3.10.9`
- `imageio==2.37.3`
- `opencv-python-headless==4.13.0.92`
- `torch==2.12.1+cu130`
- `onnx==1.22.0`
- `onnxruntime==1.23.2`

The lock file is:

```bash
mujoco_mp4/requirements-lock.txt
```

## System Checks

System packages checked or installed:

- `ffmpeg`
- `vulkan-tools`
- `mesa-utils`
- `libegl1`
- `libgl1`
- `libglvnd0`
- `libgles2`
- `libosmesa6`
- `libosmesa6-dev`
- `libglfw3`
- `libx11-6`
- `libxext6`
- `libxcb1`
- `libxkbcommon0`
- `libxkbcommon-x11-0`

Useful diagnostics:

```bash
nvidia-smi
eglinfo
glxinfo -B
vulkaninfo --summary
ffmpeg -version
```

## Rendering Backends

MuJoCo supports several OpenGL backends:

- `MUJOCO_GL=egl`: preferred on this H20 server. Uses EGL/headless GPU OpenGL when available.
- `MUJOCO_GL=osmesa`: CPU/software fallback. Usually slower but can work when EGL fails.
- `MUJOCO_GL=glfw`: interactive/windowing backend. Useful on a desktop, not preferred for headless servers.

Backend probe:

```bash
cd /mnt/infini-data/test/BeyondMimic
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_render_backend_probe.py
```

Minimal EGL smoke:

```bash
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_minimal_video_smoke.py
```

OSMesa fallback:

```bash
MUJOCO_GL=osmesa mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_minimal_video_smoke.py
```

Expected minimal outputs:

```text
mujoco_mp4/res/smoke/minimal_scene_<backend>.mp4
mujoco_mp4/res/smoke/minimal_scene_<backend>_keyframe.png
mujoco_mp4/res/smoke/minimal_scene_<backend>_summary.json
mujoco_mp4/logs/smoke/backend_probe_<backend>.log
```

## G1 Asset Candidates

Candidate G1 MuJoCo/MJCF/XML assets found in the read-only `download/` tree include:

```text
download/reference_code/GMR/assets/unitree_g1/g1_mocap_29dof.xml
download/reference_code/GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml
download/reference_code/PBHC/description/robots/g1/g1_29dof_rev_1_0.xml
download/reference_code/PBHC/description/robots/g1/g1_29dof_rev_1_0_with_toe.xml
download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml
download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml
```

Work copies for MuJoCo experiments should go under:

```text
mujoco_mp4/assets/work_g1/
```

The original `download/` tree must remain read-only.

## Common Failures

`mujoco.FatalError: an OpenGL platform library has not been loaded`

- Set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before importing `mujoco`.
- Confirm EGL/OSMesa system libraries exist.

`GLFWError` or display-related errors:

- Do not use `glfw` on the headless server unless diagnosing a desktop/X11 session.
- Prefer `egl`, then `osmesa`.

Blank or black video:

- Check keyframe PNG first.
- Verify camera name and scene bounds.
- Make sure frames are rendered after `mj_forward` or `mj_step`.

G1 asset import fails:

- Copy the XML and mesh directory into `mujoco_mp4/assets/work_g1/`.
- Check relative mesh paths.
- Try a simpler scene XML before policy rollout.

Policy rollout mismatch:

- Treat this as an observation/action contract problem, not a MuJoCo rendering failure.
- Record approximate observation gaps honestly.
- Do not claim official IsaacLab tracking reproduction.

## Next Stages

1. Minimal MuJoCo MP4 smoke.
2. G1 asset inventory and import smoke.
3. G1 reference replay video.
4. Approximate policy rollout video only after observation/action mapping is documented.
5. VAE/guidance video only after policy/reference pipelines are stable.
