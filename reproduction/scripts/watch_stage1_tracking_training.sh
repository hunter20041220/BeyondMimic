#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/infini-data/test/BeyondMimic"
cd "${ROOT}"
export PYTHONNOUSERSITE=1
exec python3 reproduction/scripts/watch_stage1_tracking_training.py --interval "${1:-3}"
