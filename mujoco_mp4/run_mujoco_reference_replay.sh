#!/usr/bin/env bash
set -euo pipefail

ROOT="${BM_ROOT:-/mnt/infini-data/test/BeyondMimic}"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export BM_MUJOCO_MOTION_NAME="${BM_MUJOCO_MOTION_NAME:-walk1_subject1}"
export BM_MUJOCO_MOTION_NPZ="${BM_MUJOCO_MOTION_NPZ:-${ROOT}/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz}"
export BM_MUJOCO_REPLAY_FRAMES="${BM_MUJOCO_REPLAY_FRAMES:-0}"

"${ROOT}/mujoco_mp4/.venv/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_reference_replay_video.py"
"${ROOT}/mujoco_mp4/.venv/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_mp4_manifest.py"
