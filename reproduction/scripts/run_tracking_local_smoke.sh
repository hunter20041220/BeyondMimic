#!/usr/bin/env bash
# Run the official-code local tracking smoke sequence.
# This still launches IsaacLab/Kit and therefore requires sufficient inotify
# limits before it can pass on this host.
set -euo pipefail

source /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh

ISAACLAB=/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0
WBT=/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking
GEN=/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local
PY="$ISAACLAB/isaaclab.sh -p"

MOTION_CSV=${MOTION_CSV:-/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv}
MOTION_NPZ=${MOTION_NPZ:-/mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke/walk1_subject1_50hz.npz}
LOG_DIR=/mnt/infini-data/test/BeyondMimic/logs/tracking_local_smoke

mkdir -p "$LOG_DIR" "$(dirname "$MOTION_NPZ")"

cd "$WBT"
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/prepare_tracking_local_smoke.py

TERM=xterm timeout 180s $PY "$GEN/csv_to_npz_local.py" \
  --input_file "$MOTION_CSV" \
  --input_fps 30 \
  --output_name walk1_subject1_local \
  --output_file "$MOTION_NPZ" \
  --headless \
  --device cuda:0 2>&1 | tee "$LOG_DIR/csv_to_npz_local.log"

python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_motion_npz_contract.py \
  "$MOTION_NPZ" \
  --summary-json /mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke/walk1_subject1_50hz_contract.json \
  2>&1 | tee "$LOG_DIR/validate_motion_npz_contract.log"

TERM=xterm timeout 120s $PY "$GEN/replay_npz_local.py" \
  --motion_file "$MOTION_NPZ" \
  --max_steps 50 \
  --headless \
  --device cuda:0 2>&1 | tee "$LOG_DIR/replay_npz_local.log"

TERM=xterm timeout 240s $PY "$GEN/rsl_rl/train_local.py" \
  --task=Tracking-Flat-G1-v0 \
  --motion_file "$MOTION_NPZ" \
  --num_envs=2 \
  --max_iterations=1 \
  --headless \
  --device cuda:0 \
  --logger tensorboard \
  --run_name local_smoke 2>&1 | tee "$LOG_DIR/train_local_smoke.log"
