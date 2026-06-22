# Isaac Rendered MP4 RTX Migration Package

This folder is the migration package for generating **true IsaacLab/Isaac Sim rendered rollout MP4s** on an RTX/RT-core machine. It is intentionally broader than a minimal script dump: it includes the current rendered MP4 gate, legacy diagnostic video/contact-sheet helpers, selected local checkpoints, selected motion/config metadata, small ONNX exports, and manifests for large dependencies that should be copied or mounted on the RTX host.

## Current H20 Blocker

The current H20 server cannot produce the requested true Isaac rendered MP4. The failure happens during Isaac Sim Kit/Hydra/Vulkan rendering startup, before `Tracking-Flat-G1-v0` is created and before any PPO checkpoint or closed-loop physics rollout is evaluated.

Recorded blocker:

- classification: `blocked_h20_isaac_sim_rendering_stack`
- no MP4, keyframes, metrics CSV, or summary markdown from the true-rendered gate
- failure boundary: Kit/Hydra/Vulkan/AppLauncher startup, not policy loading, not `env.step`, not robot physics
- claim level: server rendering-stack blocker audit

This package exists so the same MP4 generation path can be moved to a machine with supported graphics rendering hardware instead of repeatedly hitting the H20 rendering boundary.

## Package Contents

Manifest summary generated at `2026-06-22T05:50:04.997415+00:00`:

- total manifest entries: `2071`
- copied files: `2063`
- referenced files/directories: `8`
- referenced large files/directories: `8`
- scripts: `57`
- checkpoints: `34`
- configs: `0`
- metadata: `1185`
- assets: `106`
- motions: `3`
- results: `652`
- logs: `34`

Key files:

- `run_rtx_smoke.sh`: short RTX smoke runner, default 20 steps
- `run_rtx_full_videos.sh`: full runner, default 300 steps
- `isaac_mp4_need_manifest.json`: full machine-readable package inventory
- `isaac_mp4_need_manifest.tsv`: TSV inventory for spreadsheet review
- `scripts/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py`: true IsaacLab rendered policy rollout MP4 gate

## What This Package Can Claim

If the RTX smoke/full runners succeed, the output can be described as:

`true Isaac rendered simulation video / local virtual evidence`

It must **not** be described as:

- real robot evidence
- official paper-level Fig. 5/Fig. 6 reproduction
- official BeyondMimic VAE/diffusion checkpoint reproduction
- proof that DAgger rollout data is available

Legacy skeleton/contact-sheet scripts are included for comparison and report assets, but they remain diagnostic visualizations unless they are adapted to true Isaac rendering.

## Recommended RTX Host

Hardware:

- RTX 4090, RTX 4090D, RTX 6000 Ada, L40/L40S, or another NVIDIA GPU with RTX/RT-core rendering support suitable for Isaac Sim
- at least 24 GB VRAM recommended for comfortable IsaacLab rendering
- local SSD storage for Omniverse/Kit caches

System:

- Ubuntu 20.04 or 22.04
- recent NVIDIA driver compatible with Isaac Sim 4.5.0
- Vulkan loader and NVIDIA Vulkan ICD working
- EGL/headless rendering available
- X11/Xvfb available for fallback diagnostics
- `ffmpeg` available for video encoding

System packages commonly needed:

```bash
sudo apt-get update
sudo apt-get install -y \
  vulkan-tools mesa-utils ffmpeg xvfb x11-utils \
  libgl1 libegl1 libglvnd0 libx11-6 libxext6 libxrender1 libxi6 \
  libxrandr2 libxcursor1 libxinerama1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-cursor0
```

## Python / Conda Environment

Prefer reusing the project conda/venv layout:

- `envs/bm_tracking`: Isaac Sim + IsaacLab + whole_body_tracking + RSL-RL environment
- `envs/bm_diffusion`: PyTorch CUDA environment for VAE/diffusion/guidance
- `envs/bm_analysis`: numpy/pandas/matplotlib/onnxruntime report/audit environment

Core Python packages expected by the runners:

- `isaacsim==4.5.0.0`
- local editable `isaaclab`, `isaaclab_assets`, `isaaclab_rl`, `isaaclab_tasks`, `isaaclab_mimic`
- local editable `whole_body_tracking`
- `torch`, `torchvision`
- `numpy`, `pandas`, `matplotlib`
- `imageio`, `pillow`, `opencv-python`
- `gymnasium`, `rsl-rl`
- `onnx`, `onnxruntime`, `onnxscript`

## Environment Variables

The runner scripts set the important variables, but these are the knobs to know:

```bash
export BM_ROOT=/path/to/BeyondMimic
export BM_TRACKING_PY=$BM_ROOT/envs/bm_tracking/bin/python
export BM_GPU_ID=0
export BM_ISAAC_MP4_CANDIDATE_GPUS=0
export BM_ISAAC_MP4_STEPS=300
export BM_ISAAC_MP4_SEED=20260780
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export XDG_RUNTIME_DIR=$BM_ROOT/tmp/xdg-runtime-$(id -u)
export OMNI_USER_DIR=$BM_ROOT/cache/ov/user
export OMNI_LOGS_DIR=$BM_ROOT/logs/omniverse
export OMNI_CACHE_DIR=$BM_ROOT/cache/ov/cache
export PIP_CACHE_DIR=$BM_ROOT/cache/pip
export TORCH_HOME=$BM_ROOT/cache/torch
```

The true rendered MP4 script now honors `BM_ROOT`, so the RTX machine does not need to use the original H20 absolute path.

## First RTX Checks

From the RTX machine:

```bash
cd /path/to/BeyondMimic
nvidia-smi
vulkaninfo --summary
BM_ROOT=$PWD BM_GPU_ID=0 ./isaac_mp4_need/run_rtx_smoke.sh
```

The smoke runner verifies:

- Python/Isaac Sim/IsaacLab/whole_body_tracking imports
- AppLauncher with `headless=True, enable_cameras=True`
- true Isaac render product/camera path
- `Tracking-Flat-G1-v0` creation
- 10-30 step physics rollout through `env.step(action)`
- short MP4/keyframe/metrics path if rendering succeeds

## 300-Step True Isaac Rendered MP4

After smoke succeeds:

```bash
cd /path/to/BeyondMimic
BM_ROOT=$PWD BM_GPU_ID=0 BM_ISAAC_MP4_STEPS=300 ./isaac_mp4_need/run_rtx_full_videos.sh
```

Primary true-rendered expected outputs:

- `$BM_ROOT/res/visualization/isaac_mp4/*.mp4`
- `$BM_ROOT/res/visualization/isaac_mp4/*keyframes*.png`
- `$BM_ROOT/res/visualization/isaac_mp4/*metrics*.csv`
- `$BM_ROOT/res/visualization/isaac_mp4/*summary*.md` or asset JSON
- `$BM_ROOT/logs/isaac_mp4_rtx_package/*.log`
- `$BM_ROOT/res/failed_runs/isaac_mp4/*.json` if a gate fails

## Reference / Policy / VAE / Guidance Videos

The full runner starts with the true Isaac rendered policy rollout. It also optionally runs legacy diagnostic asset scripts:

- reference replay asset: `official_importer_export_full_dataset_reference_replay_video_asset.py`
- current PPO/policy rollout diagnostic asset: `tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture.py`
- VAE base rollout diagnostic asset: `tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture.py`
- guidance contact sheet: `official_importer_export_full_bundle_guidance_video_contact_sheet.py`

Set `BM_RUN_LEGACY_DIAGNOSTIC_VIDEO_ASSETS=0` if you only want the true Isaac rendered MP4 gate.

## Checkpoints And Motion Data

The package copies small/current checkpoints when available:

- robot-order FK-repaired G1 PPO `model_999.pt`
- endpoint-threshold candidate PPO `model_999.pt`
- scaled/full-bundle PPO `model_999.pt`
- teacher-rollout VAE checkpoints
- state-latent denoiser checkpoints
- small ONNX encoder/decoder/denoiser exports

Large paper-architecture LAFAN1 checkpoints and full run directories are referenced in the manifest when they exceed the copy limit. Copy them manually only if your RTX experiment needs them.

Motion/asset handling:

- the FK-repaired robot-order motion bundle is copied if below the copy limit
- the official G1 USDA is referenced if it is above the copy limit
- whole_body_tracking G1 task configs and small robot description files are copied
- full raw datasets under `download/` are referenced, not duplicated

## Common Failures

`No device could be created` / `GPU Foundation is not initialized`:

- check `nvidia-smi` and `vulkaninfo --summary`
- verify the NVIDIA Vulkan ICD is present under `/etc/vulkan/icd.d/`
- avoid CPU llvmpipe for Isaac Sim rendering

`VkResult: ERROR_DEVICE_LOST` or Hydra/Replicator initialization failure:

- reduce GPU contention
- try a single visible GPU
- clear Omniverse caches under `$BM_ROOT/cache/ov`
- update/downgrade driver to the Isaac Sim supported range

`GLFW initialization failed` / `GLXBadFBConfig`:

- verify EGL/headless setup
- try a real display, X11 session, or Xvfb only as a diagnostic fallback
- make sure XCB/XKB packages are installed

Camera/render product creates but video is empty:

- confirm `enable_cameras=True`
- inspect keyframe PNGs
- check that the camera prim is pointed at the robot root
- verify `simulation_app.update()` is called while frames are captured

Env creation fails:

- confirm `whole_body_tracking.tasks` imports
- confirm `Tracking-Flat-G1-v0` is registered
- verify the G1 USDA path and motion NPZ path in the runner summary

Checkpoint shape mismatch or obs/action dim mismatch:

- use the robot-order FK-repaired PPO checkpoint with the robot-order FK-repaired motion bundle
- compare observation/action schema JSONs in `metadata/`
- do not mix CSV-loop, scaled-PPO, and robot-order checkpoints without checking dimensions

## Manual Large Files To Copy

See `isaac_mp4_need_manifest.json` entries with:

- `copied_or_referenced = referenced_large_file`
- `large_file = true`

Those are intentionally not duplicated into this folder. The most important likely manual dependencies are:

- `download/`
- `reproduction/third_party/official/IsaacLab-v2.1.0`
- `reproduction/third_party/official/whole_body_tracking`
- full `res/runs/` if you need non-selected checkpoints
- official G1 USDA if not copied because it exceeds the local copy limit

## Audit Boundary

This package is an engineering migration artifact. It does not change the reproduction completion state by itself. The project remains incomplete at paper level until required paper-level gates pass, and real robot deployment remains unavailable unless real Unitree G1 hardware is explicitly provided.
