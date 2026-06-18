# Source Ledger

Root: `/mnt/infini-data/test/BeyondMimic`

This ledger records the evidence sources selected for the reproduction. The immutable raw material area is
`/mnt/infini-data/test/BeyondMimic/download`; working copies and generated outputs must stay outside that directory.

## Primary Paper Sources

| Source | Path | Use | Status |
|---|---|---|---|
| BeyondMimic PDF | `/mnt/infini-data/test/BeyondMimic/download/papers/BeyondMimic_2508.08241.pdf` | Main paper verification | present |
| BeyondMimic source tar | `/mnt/infini-data/test/BeyondMimic/download/papers/BeyondMimic_2508.08241_source.tar` | Paper figures and TeX parameter extraction | present |

## Official / Author Sources

| Item | Path | Remote | Commit | Use |
|---|---|---|---|---|
| whole_body_tracking | `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking` | `https://github.com/HybridRobotics/whole_body_tracking.git` | `cd65172032893724b445448818c34165846d847d` | Official Unitree G1 motion tracking training |
| motion_tracking_controller | `/mnt/infini-data/test/BeyondMimic/download/official/motion_tracking_controller` | `https://github.com/HybridRobotics/motion_tracking_controller.git` | `cbdb4a80d5ea506b2045bdd39cdfb4058084aeb4` | Official C++ ONNX inference / sim-to-sim reference |
| LAFAN1_Retargeting_Dataset | `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset` | `https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset.git` | `ce1572906efe6157840e8474d5a0d7aa87481e74` | Retargeted G1/H1/H1_2 motion CSVs |
| Dataset_beyondmimic.zip | `/mnt/infini-data/test/BeyondMimic/download/official/Dataset_beyondmimic.zip` | Zenodo metadata in `download/manifests/zenodo_17529720_metadata.json` | archive | Released-data figure reproduction |
| unitree_description.tar.gz | `/mnt/infini-data/test/BeyondMimic/download/official/unitree_description.tar.gz` | GCS mirror noted by official README | archive | G1 robot description/assets |

## Dependencies

| Item | Path | Remote | Commit | Required By |
|---|---|---|---|---|
| IsaacLab-v2.1.0 | `/mnt/infini-data/test/BeyondMimic/download/dependencies/IsaacLab-v2.1.0` | `https://github.com/isaac-sim/IsaacLab.git` | `21f7136325136ca3f6ca4e0a8125edffe5c24f7e` | tracking / Isaac Sim |
| rsl_rl | `/mnt/infini-data/test/BeyondMimic/download/dependencies/rsl_rl` | `https://github.com/leggedrobotics/rsl_rl.git` | `016c7ede710e358b7d6c205642e2540804d6281f` | PPO |
| unitree_bringup | `/mnt/infini-data/test/BeyondMimic/download/dependencies/unitree_bringup` | `https://github.com/qiayuanl/unitree_bringup.git` | `1b2c83dd846e92eee5b070e9551b6845257b0785` | ROS deployment/sim-to-sim |
| unitree_buildfarm | `/mnt/infini-data/test/BeyondMimic/download/dependencies/unitree_buildfarm` | `https://github.com/qiayuanl/unitree_buildfarm.git` | `847492ee91d7952992ebe0569e472e985b1b93dc` | ROS deployment packages |

## Reference Implementations

The following are reference-only unless an implementation decision explicitly cites them:

`ASAP`, `GMR`, `PBHC`, `diffuser`, `diffusion-motion-inbetweening`, `guided-motion-diffusion`,
`latent-diffusion`, `legged_template_controller`, `mjlab`, `motion-diffusion-model`, `unitree_rl_lab`,
and `unitree_rl_mjlab` under `/mnt/infini-data/test/BeyondMimic/download/reference_code`.

Current status: none of these reference repositories is treated as an official BeyondMimic VAE/diffusion implementation
or checkpoint source.

The Level C artifact audit is recorded at
`/mnt/infini-data/test/BeyondMimic/res/level_c/official_artifact_audit/level_c_official_artifact_audit.json`. It scanned the
local official/reference areas and found no official BeyondMimic-specific VAE/diffusion implementation, checkpoint,
TensorRT engine, or Level C deployment model; diffusion-related hits are reference repositories or already-audited
tracking/controller code.

## Inventory

The complete machine-readable inventory is:

`/mnt/infini-data/test/BeyondMimic/reproduction/docs/local_inventory.tsv`
