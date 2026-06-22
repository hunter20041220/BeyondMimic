#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BM_ROOT:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
PY="${BM_TRACKING_PY:-${ROOT}/envs/bm_tracking/bin/python}"
GPU="${BM_GPU_ID:-0}"
STEPS="${BM_ISAAC_MP4_STEPS:-20}"
SEED="${BM_ISAAC_MP4_SEED:-20260780}"
LOG_DIR="${ROOT}/logs/isaac_mp4_rtx_package"

mkdir -p "${LOG_DIR}" "${ROOT}/tmp" "${ROOT}/cache/ov" "${ROOT}/cache/pip" "${ROOT}/cache/torch"
export BM_ROOT="${ROOT}"
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${ROOT}/tmp/xdg-runtime-$(id -u)}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" || true
export OMNI_USER_DIR="${OMNI_USER_DIR:-${ROOT}/cache/ov/user}"
export OMNI_LOGS_DIR="${OMNI_LOGS_DIR:-${ROOT}/logs/omniverse}"
export OMNI_CACHE_DIR="${OMNI_CACHE_DIR:-${ROOT}/cache/ov/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT}/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export BM_ISAAC_MP4_CANDIDATE_GPUS="${BM_ISAAC_MP4_CANDIDATE_GPUS:-${GPU}}"
export BM_ISAAC_MP4_STEPS="${STEPS}"
export BM_ISAAC_MP4_SEED="${SEED}"
export BM_ISAAC_MP4_TIMEOUT_SECONDS="${BM_ISAAC_MP4_TIMEOUT_SECONDS:-600}"

echo "[smoke] ROOT=${ROOT}"
echo "[smoke] PY=${PY}"
echo "[smoke] GPU=${BM_ISAAC_MP4_CANDIDATE_GPUS}"
echo "[smoke] steps=${BM_ISAAC_MP4_STEPS}"

nvidia-smi | tee "${LOG_DIR}/nvidia_smi_smoke.log"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary | tee "${LOG_DIR}/vulkaninfo_summary_smoke.log" || true
else
  echo "[smoke] vulkaninfo not found; install vulkan-tools before diagnosing render failures." | tee "${LOG_DIR}/vulkaninfo_missing_smoke.log"
fi

"${PY}" - <<'PY' | tee "${LOG_DIR}/python_import_smoke.log"
import importlib
mods = ["torch", "isaacsim", "isaaclab", "isaaclab_rl", "whole_body_tracking", "gymnasium", "imageio", "PIL"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"IMPORT_OK {name} {getattr(mod, '__version__', '')}")
PY

"${PY}" "${ROOT}/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py" 2>&1 | tee "${LOG_DIR}/rtx_true_isaac_rendered_mp4_smoke.log"
echo "[smoke] Done. Check ${ROOT}/res/visualization/isaac_mp4 and ${ROOT}/res/failed_runs/isaac_mp4."
