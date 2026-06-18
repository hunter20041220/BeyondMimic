#!/usr/bin/env bash
# Capture Phase 0 system evidence without mutating the host.
set -u

ROOT=/mnt/infini-data/test/BeyondMimic
source "$ROOT/reproduction/scripts/project_env.sh"
OUT="$LOG_ROOT/setup/system_audit.txt"

{
  echo "# BeyondMimic system audit"
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo

  run() {
    echo "## $*"
    "$@" 2>&1 || echo "[command failed: $*]"
    echo
  }

  run nvidia-smi
  run nvidia-smi -L
  run nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv
  run nvcc --version
  run cat /etc/os-release
  run uname -a
  run python --version
  run which python
  run conda info
  run mamba --version
  run df -h
  run df -h /shared_disk
  run free -h
  run lscpu

  echo "## Tool discovery"
  for cmd in conda mamba micromamba python python3 pip pip3 nvcc nvidia-smi isaac-sim ros2 mujoco; do
    printf '%s\t' "$cmd"
    command -v "$cmd" 2>/dev/null || true
  done
} | tee "$OUT"
