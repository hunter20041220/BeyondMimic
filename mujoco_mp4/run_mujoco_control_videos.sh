#!/usr/bin/env bash
set -euo pipefail

ROOT="${BM_ROOT:-/mnt/infini-data/test/BeyondMimic}"
VENV="${ROOT}/mujoco_mp4/.venv"

cd "${ROOT}"

export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export BM_MUJOCO_CONTROL_SPECS="${BM_MUJOCO_CONTROL_SPECS:-reference_control,ppo_policy_control,vae_base_control,denoised_latent_control,guided_latent_control}"
export BM_MUJOCO_CONTROL_FRAMES="${BM_MUJOCO_CONTROL_FRAMES:-450}"
export BM_MUJOCO_VIDEO_FPS="${BM_MUJOCO_VIDEO_FPS:-30}"
export BM_MUJOCO_WIDTH="${BM_MUJOCO_WIDTH:-960}"
export BM_MUJOCO_HEIGHT="${BM_MUJOCO_HEIGHT:-540}"
export BM_MUJOCO_CONTROL_SUBSTEPS="${BM_MUJOCO_CONTROL_SUBSTEPS:-4}"
export BM_MUJOCO_CONTROL_SETTLE_STEPS="${BM_MUJOCO_CONTROL_SETTLE_STEPS:-40}"
export BM_MUJOCO_ROOT_ASSIST="${BM_MUJOCO_ROOT_ASSIST:-1}"

"${VENV}/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_pd_control_video.py"
"${VENV}/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_control_video_summary.py"
"${VENV}/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_mp4_manifest.py"
