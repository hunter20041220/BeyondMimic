# Progress Update

## Goal

Start a clean G1 tracking PPO retraining run that follows the BeyondMimic/official `whole_body_tracking` parameter contract instead of trying to fix poor MuJoCo control videos with ad-hoc adapters.

This run is intended to answer the current reproduction question: if we use the paper/official motor parameters, reward terms, termination thresholds, observation/action contract, and PPO hyperparameters, can the local public-resource teacher become strong enough to support later rollout videos, DAgger-style data, VAE, diffusion, and guidance experiments?

## Files Read

- `prompt06211658.txt`
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/docs/english_reading_report.md`
- `.gitignore`

## Files Modified

- Added `reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py`.
- Added `reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.py`.
- Added this progress record.

## Commands Run

```bash
python3 -m py_compile \
  reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py \
  reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.py
```

```bash
BM_PAPER_CONTRACT_TARGET_GPUS=4,7 \
BM_PAPER_CONTRACT_TOTAL_ENVS=4096 \
BM_PAPER_CONTRACT_MAX_ITERATIONS=30000 \
BM_PAPER_CONTRACT_SAVE_INTERVAL=500 \
BM_PAPER_CONTRACT_SEED=20260801 \
python3 reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py
```

Monitoring commands:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -i 4,7
```

```bash
find res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training/ \
  -maxdepth 3 -type f -name 'model_*.pt' | sort -V
```

## Results

The paper-contract PPO run is active and has reached the first official save interval.

Current run directory:

```text
res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training/resource_adjusted_ppo_20260622_084243_seed20260801
```

Current local checkpoints:

```text
rank_0/model_0.pt
rank_0/model_500.pt
```

Lightweight parameter snapshot:

```text
res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/paper_contract_tracking_parameters.json
```

The parameter snapshot records:

- simulation `dt=0.005`, decimation `4`, 50 Hz policy control, 10 s episodes;
- 4096 total environments over GPUs 4 and 7;
- official G1 armature/stiffness/damping/action-scale formulas;
- policy obs dim `160`, critic obs dim `286`, action dim `29`;
- official tracking reward weights/stds;
- official termination thresholds;
- official RSL-RL PPO hyperparameters: 24 steps/env, 30000 iterations, save interval 500, actor/critic hidden dims `[512, 256, 128]`, ELU, LR `1e-3`, adaptive schedule, entropy `0.005`, gamma `0.99`, lambda `0.95`.

Early training trend through about 620 iterations:

```text
iter[0,50):      reward_mean=-0.3014, body_pos_mean=0.3669, joint_pos_mean=1.8953, ee_term_mean=324.81
iter[50,100):    reward_mean=-0.1080, body_pos_mean=0.3274, joint_pos_mean=1.8809, ee_term_mean=328.48
iter[100,200):   reward_mean=-0.0266, body_pos_mean=0.3244, joint_pos_mean=1.8700, ee_term_mean=330.33
iter[200,400):   reward_mean=0.0392,  body_pos_mean=0.3270, joint_pos_mean=1.8403, ee_term_mean=331.18
iter[400,620+):  reward_mean=0.0719,  body_pos_mean=0.3277, joint_pos_mean=1.8064, ee_term_mean=338.26
```

Interpretation: reward and joint error are improving, but body-position error and endpoint termination remain weak in early training. The first checkpoint is not a final teacher.

## Verification

Static verification passed:

```text
python3 -m py_compile ...  # passed
```

Live verification:

- IsaacLab/whole_body_tracking/RSL-RL training worker started successfully.
- Two worker processes are active through `torch.distributed.run`.
- GPUs 4 and 7 are being used.
- `model_500.pt` was written.

Full audit refresh is intentionally deferred until this long PPO run finishes and the checkpoint evaluation JSON is produced. The training process is still active.

## Failed / Blocked Items

- No final PPO checkpoint evaluation yet.
- No new policy rollout video yet.
- No MuJoCo PPO/VAE/guidance video is promoted from this run yet.
- Current result is not paper-level tracking. It is a local public-resource retraining attempt using paper/official-code formulas.

## Effect on English Reading Report

This adds a stronger and cleaner story for the reproduction section: after poor MuJoCo control videos suggested a weak teacher, the project returned to the official IsaacLab tracking contract and launched a full 4096-env, 30000-iteration PPO retraining run using the paper/official parameters.

The report should still say:

```text
This project does not fully reproduce BeyondMimic at paper-level.
```

## Next Step

Let the 30000-iteration PPO run continue. After completion:

1. run `tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.py`;
2. compare final done rate, body-position error, joint-position error, and endpoint termination against previous robot-order FK checkpoints;
3. only if the teacher is credible, generate policy rollout videos and rebuild downstream DAgger/VAE/diffusion/guidance artifacts.

## Git Commit

Pending at the time this progress record was written.
