#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/infini-data/test/BeyondMimic"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

"${ROOT}/mujoco_mp4/.venv/bin/python" official_mp4/scripts/official_dataset_inventory.py
"${ROOT}/mujoco_mp4/.venv/bin/python" official_mp4/scripts/render_official_g1_csv_replay.py \
  --csv "${ROOT}/Dataset_beyondmimic/ablation/tkd_skill.csv" \
  --motion-name "official_zenodo_tkd_skill" \
  --fps 50 \
  --width 1280 \
  --height 720

for mcap in \
  "${ROOT}/Dataset_beyondmimic/rosbag_agile_motion/C1970_tkd_skill_clip1.mcap" \
  "${ROOT}/Dataset_beyondmimic/rosbag_agile_motion/C1975_side_flip.mcap" \
  "${ROOT}/Dataset_beyondmimic/rosbag_agile_motion/C1980_double_high_kick.mcap" \
  "${ROOT}/Dataset_beyondmimic/rosbag_agile_motion/C1985_merge2.mcap"
do
  name="$(basename "${mcap}" .mcap)"
  "${ROOT}/mujoco_mp4/.venv/bin/python" official_mp4/scripts/render_official_mcap_joint_replay.py \
    --mcap "${mcap}" \
    --motion-name "official_agile_${name}" \
    --frames 450 \
    --stride 2 \
    --width 1280 \
    --height 720
done

find "${ROOT}/Dataset_beyondmimic/rosbag_ablation" -type f -name '*.mcap' | sort | while read -r mcap
do
  parent="$(basename "$(dirname "${mcap}")")"
  safe="$(printf '%s' "${parent}" | sed 's/[^A-Za-z0-9_]/_/g')"
  "${ROOT}/mujoco_mp4/.venv/bin/python" official_mp4/scripts/render_official_mcap_joint_replay.py \
    --mcap "${mcap}" \
    --motion-name "official_ablation_${safe}" \
    --frames 450 \
    --stride 2 \
    --width 1280 \
    --height 720
done

for mcap in \
  "${ROOT}/Dataset_beyondmimic/rosbag_walk_and_run/walk_rosbag2_2025_10_23-18_21_05/rosbag2_2025_10_23-18_21_05_0.mcap" \
  "${ROOT}/Dataset_beyondmimic/rosbag_walk_and_run/run_rosbag2_2025_10_23-18_05_39/rosbag2_2025_10_23-18_05_39_0.mcap"
do
  parent="$(basename "$(dirname "${mcap}")" | sed 's/[^A-Za-z0-9_]/_/g')"
  kind="$(basename "$(dirname "${mcap}")" | cut -d_ -f1)"
  "${ROOT}/mujoco_mp4/.venv/bin/python" official_mp4/scripts/render_official_mcap_joint_replay.py \
    --mcap "${mcap}" \
    --motion-name "official_${kind}_${parent}" \
    --frames 450 \
    --stride 4 \
    --width 1280 \
    --height 720
done

"${ROOT}/mujoco_mp4/.venv/bin/python" official_mp4/scripts/official_mp4_manifest.py
