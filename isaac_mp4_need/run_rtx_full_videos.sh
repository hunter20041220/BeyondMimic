#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BM_ROOT:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
PY="${BM_TRACKING_PY:-${ROOT}/envs/bm_tracking/bin/python}"
GPU="${BM_GPU_ID:-0}"
STEPS="${BM_ISAAC_MP4_STEPS:-300}"
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
export BM_ISAAC_MP4_TIMEOUT_SECONDS="${BM_ISAAC_MP4_TIMEOUT_SECONDS:-1800}"

echo "[full] ROOT=${ROOT}"
echo "[full] PY=${PY}"
echo "[full] GPU=${BM_ISAAC_MP4_CANDIDATE_GPUS}"
echo "[full] steps=${BM_ISAAC_MP4_STEPS}"
echo "[full] First generating true IsaacLab/Isaac Sim rendered policy rollout MP4."
"${PY}" "${ROOT}/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py" 2>&1 | tee "${LOG_DIR}/rtx_true_isaac_rendered_policy_rollout_300step.log"

run_optional() {
  local script="$1"
  local log_name="$2"
  if [[ -f "${ROOT}/${script}" ]]; then
    echo "[full] Running optional diagnostic asset script: ${script}"
    "${PY}" "${ROOT}/${script}" 2>&1 | tee "${LOG_DIR}/${log_name}.log" || {
      echo "[full] Optional diagnostic script failed: ${script}. This does not invalidate the true Isaac rendered MP4 if it already succeeded."
    }
  else
    echo "[full] Missing optional script: ${script}"
  fi
}

if [[ "${BM_RUN_LEGACY_DIAGNOSTIC_VIDEO_ASSETS:-1}" == "1" ]]; then
  run_optional "reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py" "reference_replay_video_asset"
  run_optional "reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture.py" "policy_rollout_video_asset"
  run_optional "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture.py" "vae_base_rollout_video_asset"
  run_optional "reproduction/scripts/official_importer_export_full_bundle_guidance_video_contact_sheet.py" "guidance_contact_sheet_asset"
fi

echo "[full] Done. True rendered outputs should be under ${ROOT}/res/visualization/isaac_mp4."
echo "[full] Diagnostic skeleton/contact-sheet outputs remain diagnostic evidence, not true Isaac rendered videos."
