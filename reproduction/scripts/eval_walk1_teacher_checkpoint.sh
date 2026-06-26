#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/infini-data/test/BeyondMimic}"
CHECKPOINT="${1:-}"
EVAL_GPUS="${BM_WALK1_EVAL_GPUS:-2,3}"
TAG_SUFFIX="${BM_WALK1_EVAL_TAG_SUFFIX:-latest}"

cd "${ROOT}"

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find res/runs -path '*hub_singleleg_paper_contract_ppo_training_lafan_walk1_subject1_repaired*' \
    -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint found. Pass /path/to/model_*.pt explicitly." >&2
  exit 2
fi

iter="$(basename "${CHECKPOINT}" .pt | sed 's/model_//')"
stamp="$(date +%Y%m%d_%H%M%S)"
run_tag="lafan_walk1_subject1_eval_${TAG_SUFFIX}_model${iter}_${stamp}"

echo "checkpoint: ${ROOT}/${CHECKPOINT}"
echo "eval_gpus: ${EVAL_GPUS}"
echo "run_tag: ${run_tag}"

env \
  BM_HUB_SINGLELEG_EVAL_RUN_TAG="${run_tag}" \
  BM_HUB_SINGLELEG_EVAL_TARGET_GPUS="${EVAL_GPUS}" \
  BM_HUB_SINGLELEG_EVAL_MOTION_NPZ="${ROOT}/res/tracking/stage1_multisource_motion_bundle_robot_joint_order/motions/lafan1_walk1_subject1/motion.npz" \
  BM_HUB_SINGLELEG_EVAL_CHECKPOINT="${ROOT}/${CHECKPOINT}" \
  BM_HUB_SINGLELEG_EVAL_NUM_ENVS="${BM_WALK1_EVAL_NUM_ENVS:-512}" \
  BM_HUB_SINGLELEG_EVAL_STEPS="${BM_WALK1_EVAL_STEPS:-799}" \
  BM_HUB_SINGLELEG_EVAL_SEED="${BM_WALK1_EVAL_SEED:-20261052}" \
  BM_REFRESH_MOTION_COMMAND_TARGETS=1 \
  "${ROOT}/envs/bm_tracking/bin/python" -u reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py
