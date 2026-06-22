# Progress Update

## Goal

Complete the local source/data inventory for the datasets and repositories named by `HybridRobotics/whole_body_tracking`, especially the short-sequence motions that were previously unclear: KungfuBot side kick, ASAP Cristiano Ronaldo celebration, and HuB balance motions.

The active paper-contract PPO training was not stopped.

## Files Read

- `download/official/whole_body_tracking/README.md`
- `download/README_download_scope.md`
- `reproduction/docs/source_ledger.md`
- `download/_supplemental/hub_data/hub_project_page.html`
- `download/reference_code/PBHC/example/motion_data/Side_kick.pkl`
- `download/reference_code/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl`
- `download/_supplemental/hub_data/drive_folder/singleleg.pkl`
- `download/_supplemental/hub_data/drive_folder/swallow_balance.pkl`

## Files Modified

- Added `reproduction/scripts/motion_tracking_required_sources_inventory.py`.
- Added this progress record.
- Installed lightweight download/inspection helpers into `envs/bm_analysis`: `gdown`, `joblib`.

## Commands Run

```bash
curl -L --max-time 60 \
  -o download/_supplemental/hub_data/hub_project_page.html \
  https://hub-robot.github.io/
```

```bash
envs/bm_analysis/bin/python -m pip install gdown joblib
```

```bash
envs/bm_analysis/bin/gdown --folder \
  'https://drive.google.com/drive/folders/1ZrF8HFMzJ7jBcHP6qFKgCyErvFDWlkmc?usp=drive_link' \
  --output download/_supplemental/hub_data/drive_folder
```

```bash
envs/bm_analysis/bin/python reproduction/scripts/motion_tracking_required_sources_inventory.py
```

## Results

The public source/data inventory for the `whole_body_tracking` README is now materially complete locally.

Available source repositories:

- `HybridRobotics/whole_body_tracking`
- `HybridRobotics/motion_tracking_controller`
- `mujocolab/mjlab`
- `TeleHuman/PBHC`
- `LeCAR-Lab/ASAP`
- `Unitree-retargeted LAFAN1 Dataset`
- `Ubisoft LaForge LAFAN1 original dataset`

Available named short-sequence motion sources:

- KungfuBot/PBHC side kick: `download/reference_code/PBHC/example/motion_data/Side_kick.pkl`
- ASAP Cristiano Ronaldo / CR7: `download/reference_code/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl`
- HuB single-leg balance: `download/_supplemental/hub_data/drive_folder/singleleg.pkl`
- HuB swallow balance: `download/_supplemental/hub_data/drive_folder/swallow_balance.pkl`
- Additional HuB motions: `bruce_lee.pkl`, `nezha.pkl`, `squat.pkl`

The downloaded HuB files are SHA256-recorded in:

```text
download/_supplemental/manifests/hub_data_sha256.txt
```

The inventory summary reports:

```text
row_count: 12
available_count: 12
unitree_lafan_g1_csv_count: 40
asap_g1_29dof_single_pkl_count: 52
pbhc_example_motion_pkl_count: 7
hub_downloaded_pkl_count: 5
needs_conversion_count: 5
```

## Verification

Generated:

```text
res/download_audit/motion_tracking_required_sources_inventory/motion_tracking_required_sources_inventory.json
res/download_audit/motion_tracking_required_sources_inventory/motion_tracking_required_sources_inventory.tsv
```

Field inspection confirms PBHC/ASAP/HuB motion files are joblib-readable dictionaries with motion fields such as:

```text
root_trans_offset
pose_aa
dof
root_rot
smpl_joints
fps
```

## Failed / Blocked Items

The non-LAFAN short-sequence motions are downloaded and readable, but they are not yet converted to the generalized-coordinate CSV/NPZ convention consumed by `whole_body_tracking/scripts/csv_to_npz.py`.

Therefore they are not yet part of the active PPO training dataset.

## Effect on English Reading Report

This strengthens the reproduction evidence: the project now has local copies of the public source repositories and named short-sequence motion sources referenced by the motion-tracking README. The report should still clearly separate:

- downloaded source/data availability;
- conversion/replay readiness;
- actual PPO training dataset membership;
- paper-level tracking results.

## Next Step

Build explicit converters from PBHC/ASAP/HuB joblib motion dicts to the `whole_body_tracking` CSV/NPZ convention, run `csv_to_npz.py` / replay gates for each, then decide whether to add these short sequences to a new PPO training run.

## Git Commit

Pending.
