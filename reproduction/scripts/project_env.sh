#!/usr/bin/env bash
# Shared project paths and cache locations for BeyondMimic reproduction.
set -euo pipefail

export ROOT=/mnt/infini-data/test/BeyondMimic
export DOWNLOAD_ROOT="$ROOT/download"
export WORKSPACE="$ROOT/reproduction"
export ENV_ROOT="$ROOT/envs"
export CACHE_ROOT="$ROOT/cache"
export TMP_ROOT="$ROOT/tmp"
export LOG_ROOT="$ROOT/logs"
export RES_ROOT="$ROOT/res"

export CONDA_PKGS_DIRS="$CACHE_ROOT/conda_pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export WANDB_DIR="$LOG_ROOT/wandb"
export TMPDIR="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export TMP="$TMP_ROOT"

mkdir -p \
  "$ENV_ROOT" \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$WANDB_DIR" \
  "$TMP_ROOT" \
  "$LOG_ROOT/setup" \
  "$LOG_ROOT/gpu" \
  "$RES_ROOT"
