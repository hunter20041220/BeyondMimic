#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/infini-data/test/BeyondMimic}"
TAG="${1:-lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951}"
ITERATION="${BM_WALK1_WATCH_ITERATION:-500}"
POLL_SECONDS="${BM_WALK1_WATCH_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${BM_WALK1_WATCH_TIMEOUT_SECONDS:-21600}"
EVAL_GPUS="${BM_WALK1_EVAL_GPUS:-4,7}"

cd "${ROOT}"

run_root="res/runs/hub_singleleg_paper_contract_ppo_training_${TAG}"
checkpoint="${run_root}/resource_adjusted_ppo_20260626_024951_seed20261051/rank_0/model_${ITERATION}.pt"
watch_log="logs/walk1_teacher_eval_watch_${TAG}_model${ITERATION}.log"
mkdir -p logs

start_ts="$(date +%s)"
{
  echo "watch_start=$(date -Iseconds)"
  echo "tag=${TAG}"
  echo "iteration=${ITERATION}"
  echo "checkpoint=${ROOT}/${checkpoint}"
  echo "poll_seconds=${POLL_SECONDS}"
  echo "timeout_seconds=${TIMEOUT_SECONDS}"
  echo "eval_gpus=${EVAL_GPUS}"
} | tee "${watch_log}"

while [[ ! -f "${checkpoint}" ]]; do
  now="$(date +%s)"
  elapsed=$((now - start_ts))
  if (( elapsed > TIMEOUT_SECONDS )); then
    echo "timeout_waiting_for_checkpoint elapsed=${elapsed}" | tee -a "${watch_log}"
    exit 124
  fi
  latest_iter="$(find "${run_root}" -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
    | sed -E 's/model_([0-9]+)\\.pt/\\1/' | sort -n | tail -1 || true)"
  echo "waiting elapsed=${elapsed}s latest_iter=${latest_iter:-none}" | tee -a "${watch_log}"
  sleep "${POLL_SECONDS}"
done

echo "checkpoint_found=$(date -Iseconds)" | tee -a "${watch_log}"
BM_WALK1_EVAL_GPUS="${EVAL_GPUS}" \
BM_WALK1_EVAL_TAG_SUFFIX="watch" \
reproduction/scripts/eval_walk1_teacher_checkpoint.sh "${checkpoint}" 2>&1 | tee -a "${watch_log}"
