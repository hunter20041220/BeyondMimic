# Progress Update

## Goal

Fix the reproduction route around one main Stage 1 line: use `HybridRobotics/whole_body_tracking` to train motion tracking teacher policies, while starting a second non-interfering training line on GPUs 5/6 using all currently train-ready motion sources.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/*.csv`
- `/mnt/infini-data/test/BeyondMimic/download/_supplemental/hub_data/drive_folder/*.pkl`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/.gitignore`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_motion_bundle.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_194337_stage1_multisource_teacher.md`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
python3 -m py_compile reproduction/scripts/tracking_stage1_multisource_motion_bundle.py reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_stage1_multisource_motion_bundle.py
python3 reproduction/scripts/validate_motion_npz_contract.py res/tracking/stage1_multisource_motion_bundle/stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz --summary-json res/tracking/stage1_multisource_motion_bundle/validate_motion_npz_contract_summary.json
envs/bm_analysis/bin/python reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py
```

## Results

The existing 4/7 paper-contract PPO line was left running. It uses the 40-motion public LAFAN/G1 bundle and reached at least `model_7000.pt`.

The new multi-source bundle contains `49` train-ready motions:

- `40` Unitree-retargeted LAFAN1 motions.
- `1` BeyondMimic Zenodo generalized-coordinate reference, `tkd_skill.csv`.
- `8` HuB 29-DoF pkl motions converted to the official 36-column CSV contract.

The bundle has `448358` frames at 50 Hz, about `2.4908777777777775` hours. It is close to the paper's reported `~2.5 h` motion pool, but it is not the complete official paper motion set.

## Verification

`validate_motion_npz_contract.py` passed:

- `joint_pos`: `[448358, 29]`
- `joint_vel`: `[448358, 29]`
- `body_pos_w`: `[448358, 40, 3]`
- `body_quat_w`: `[448358, 40, 4]`
- `fps`: `50`

The new 5/6 training line reached:

- `BM_SENTINEL:rank=0:env_created:num_envs=2048`
- `BM_SENTINEL:rank=1:env_created:num_envs=2048`
- `BM_SENTINEL:rank=0:runner_created`
- `BM_SENTINEL:rank=1:runner_created`
- learning iteration `0/30000`
- `model_0.pt` saved

## Failed / Blocked Items

PBHC/KungfuBot sidekick and ASAP Cristiano Ronaldo are present locally, but currently only as 23-DoF pkl files. They were not padded or guessed into 29-DoF G1 format. They require an audited 23-to-29 joint mapping before use in `whole_body_tracking` Stage 1 training.

Most `Dataset_beyondmimic/` content is rosbag/mcap/GRF/ablation/plotting data and remains paper-result analysis material, not bulk Stage 1 training data.

## Effect on English Reading Report

This gives the reading report a clearer, defensible Stage 1 story: the current reproduction now has one LAFAN-only teacher-training line and one expanded multi-source teacher-training line close to the paper's reported motion duration. The report must still state that these are local training candidates, not official BeyondMimic teacher checkpoints.

## Next Step

Let both PPO lines continue, then evaluate checkpoints by reward, done/fall count, target-body error, joint error, and rollout video quality before using either teacher to collect VAE/diffusion training data.

## Git Commit

Pending verification and commit.
