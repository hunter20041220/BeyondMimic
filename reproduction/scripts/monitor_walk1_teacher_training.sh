#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/infini-data/test/BeyondMimic}"
TAG="${1:-}"

cd "${ROOT}"

if [[ -z "${TAG}" ]]; then
  latest_run_dir="$(find res/runs -maxdepth 2 -type d -name 'resource_adjusted_ppo_*' \
    -path '*hub_singleleg_paper_contract_ppo_training_lafan_walk1_subject1_repaired*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
  if [[ -n "${latest_run_dir:-}" ]]; then
    parent="$(basename "$(dirname "${latest_run_dir}")")"
    TAG="${parent#hub_singleleg_paper_contract_ppo_training_}"
  fi
fi

if [[ -z "${TAG}" ]]; then
  echo "No walk1 teacher training run found. Pass a run tag explicitly." >&2
  exit 2
fi

run_root="res/runs/hub_singleleg_paper_contract_ppo_training_${TAG}"
log_dir="logs/tracking_hub_singleleg_paper_contract_ppo_training_run_${TAG}"
summary_json="res/tracking/hub_singleleg_paper_contract_ppo_training_run_${TAG}/tracking_hub_singleleg_paper_contract_ppo_training_run.json"

echo "=== walk1 teacher run ==="
echo "tag: ${TAG}"
echo "run_root: ${ROOT}/${run_root}"
echo "log_dir: ${ROOT}/${log_dir}"
echo

echo "=== tmux ==="
tmux ls 2>/dev/null | grep -E "walk1|${TAG}" || true
echo

echo "=== GPU 5/6 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -i 5,6 || true
echo

echo "=== processes ==="
ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd \
  | grep -F "${TAG}" \
  | grep -v grep || true
echo

echo "=== checkpoints ==="
find "${run_root}" -name 'model_*.pt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort -V | tail -20 || true
echo

echo "=== latest GPU metrics ==="
find "${run_root}" -name 'gpu_metrics.csv' -print -quit 2>/dev/null \
  | while read -r csv; do tail -10 "${csv}"; done
echo

echo "=== latest learning log ==="
log="${log_dir}/tracking_g1_resource_adjusted_ppo_training_run.log"
if [[ -f "${log}" ]]; then
  grep -E 'Learning iteration|Mean reward|Mean episode length|Total timesteps|ETA|error_body_pos|error_joint_pos|Episode_Termination' "${log}" \
    | tail -80 || true
else
  echo "missing log: ${ROOT}/${log}"
fi
echo

echo "=== summary JSON ==="
if [[ -f "${summary_json}" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p = Path("${summary_json}")
d = json.loads(p.read_text())
print("status:", d.get("status"))
print("checkpoint_count:", d.get("run", {}).get("checkpoint_count"))
print("config:", d.get("config", {}))
PY
else
  echo "summary not written yet; training is still running or has not finalized."
fi
