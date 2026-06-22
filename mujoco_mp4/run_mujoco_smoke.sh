#!/usr/bin/env bash
set -euo pipefail

ROOT="${BM_ROOT:-/mnt/infini-data/test/BeyondMimic}"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${ROOT}/mujoco_mp4/.venv/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_render_backend_probe.py"
"${ROOT}/mujoco_mp4/.venv/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_g1_asset_inventory.py"
MUJOCO_GL="${MUJOCO_GL}" "${ROOT}/mujoco_mp4/.venv/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_g1_import_smoke.py"
"${ROOT}/mujoco_mp4/.venv/bin/python" "${ROOT}/mujoco_mp4/scripts/mujoco_mp4_manifest.py"
