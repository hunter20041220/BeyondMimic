# Data Sources and Processing Report

## Summary

The latest 5/6-GPU Stage 1 run used a local multi-source motion bundle recorded in:

`/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`

Current bundle metrics:

- Motion count: `49`
- Total frames: `None`
- Total duration: `2.4908777777777775` h
- Source counts: `{"BeyondMimic Zenodo ablation reference CSV": 1, "HuB supplemental 29-DoF pkl": 8, "Unitree-retargeted LAFAN1": 40}`
- Checks: `{"all_arrays_finite": true, "asap_ronaldo_not_silently_padded": true, "body_shape_40": true, "candidate_count_ge_40": true, "does_not_claim_2_5h_complete_paper_motion_set": true, "does_not_claim_official_beyondmimic_teacher_dataset": true, "does_not_claim_real_robot": true, "fps_50": true, "hub_29dof_candidates_included": true, "joint_shape_29": true, "lafan1_40_csvs_included": true, "npz_written": true, "pb_hc_sidekick_not_silently_padded": true, "zenodo_tkd_skill_included": true}`

## Important Boundary

The paper describes about 2.5 hours of diverse motions from prior works, Unitree-retargeted LAFAN1, and online animation data. The current project should not claim it has exactly reconstructed the authors' private curated collection. What exists locally is:

- G1-retargeted LAFAN1 CSVs at `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1`;
- one train-ready BeyondMimic Zenodo reference CSV: `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic/ablation/tkd_skill.csv`;
- several HuB supplemental 29-DoF pkl candidates;
- PBHC sidekick / ASAP Ronaldo-like sources that were skipped because their 23-DoF mapping is not audited;
- Zenodo GRF/IMU/MCAP/ablation data, which is mainly released-result evidence rather than the full diffusion training dataset.

## Processing Flow

```text
raw BVH / downloaded CSV / pkl candidates
    -> audited G1 generalized coordinates
    -> FK-repaired body positions and velocities
    -> stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz
    -> IsaacLab motion-tracking teacher training
    -> teacher rollout shards
    -> VAE and state-latent diffusion training data
```

## Generated Tables

- `report/tables/dataset_inventory.csv`
- `report/data/motion_duration_summary.csv`
- `report/data/motion_file_manifest.csv`
