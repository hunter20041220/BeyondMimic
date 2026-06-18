# Level B Official Tracking Protocol

This file records the executable official-code path for Level B tracking reproduction and the current gate status.

## Local Official Sources

- Official tracking repository work copy:
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking`
- Isaac Lab v2.1.0 work copy:
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0`
- Isaac Sim 4.5.0 standalone:
  `/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0`
- Retargeted LAFAN1 data:
  `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset`

## Official Commands

The official README states that retargeted CSV motions are first converted to motion NPZ artifacts through Isaac Sim:

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

The official script writes `/tmp/motion.npz` and uploads it to a WandB registry. On this project the cache/tmp policy
requires replacing `/tmp` with `/mnt/infini-data/test/BeyondMimic/tmp` before any long preprocessing run.

Training command from the official README:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

The official `train.py` requires `--registry_name`, downloads `motion.npz` from WandB, creates the IsaacLab gym
environment, wraps it with RSL-RL, saves `env.yaml` and `agent.yaml`, and launches PPO.

Evaluation/export command from the official README:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The official `play.py` loads the latest checkpoint, creates the IsaacLab environment, exports `policy.onnx`, attaches
metadata, and then steps the environment until the simulator exits or the requested video length is reached.

## Current Verified Evidence

- Isaac Sim Python import smoke passed:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/isaacsim_python_smoke.log`
- IsaacLab package import smoke passed for `isaaclab`:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/isaaclab_import_via_script.log`
- `whole_body_tracking` non-Kit package/asset smoke passed:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/whole_body_tracking_nokit_smoke.log`
- Non-Kit tracking config audit passed:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/smoke_config_audit/tracking_config_audit.json`

The audit verified:

- PPO actor and critic hidden dimensions `[512, 256, 128]`.
- PPO steps per env 24, max iterations 30000, LR `1e-3`, clip `0.2`, entropy `0.005`, gamma `0.99`,
  GAE lambda `0.95`, desired KL `0.01`, epochs 5, mini-batches 4.
- `sim.dt=0.005`, `decimation=4`, control rate `50 Hz`.
- 4096 training environments.
- G1 URDF and mesh references resolve.
- A retargeted LAFAN1 G1 CSV sample is present and readable.

## Current Gate

Full IsaacLab/Kit smoke is blocked by host inotify limits, not disk capacity. The failing logs repeatedly report
`Failed to create change watch ... errno=28`, while `/shared_disk` has ample free space. The current limits are recorded
in `/mnt/infini-data/test/BeyondMimic/logs/setup/inotify_status.txt`:

- `fs.inotify.max_user_watches=8192`
- `fs.inotify.max_user_instances=128`

Required host-side mitigation before running `csv_to_npz.py`, `replay_npz.py`, PPO training, evaluation, or ONNX export:

```bash
sudo sysctl fs.inotify.max_user_watches=524288
sudo sysctl fs.inotify.max_user_instances=1024
```

For persistence after reboot, the same values should be placed in a sysctl drop-in such as
`/etc/sysctl.d/99-beyondmimic-inotify.conf` by an administrator.

Current user permission check:

```text
sysctl: permission denied on key "fs.inotify.max_user_watches", ignoring
sysctl: permission denied on key "fs.inotify.max_user_instances", ignoring
```

## Prepared Local Smoke Path

To remove nonessential WandB and `/tmp` dependencies from the first tracking smoke, a generator now creates local
variants of the official scripts:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/prepare_tracking_local_smoke.py
```

Generated files:

- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/replay_npz_local.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/rsl_rl/cli_args.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/manifest.tsv`

The manifest records SHA256 hashes of the official source scripts and generated local variants.

After the inotify gate is fixed, run the full local smoke sequence with:

```bash
/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_tracking_local_smoke.sh
```

Default inputs/outputs:

- Input CSV:
  `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv`
- Local motion NPZ:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke/walk1_subject1_50hz.npz`
- Logs:
  `/mnt/infini-data/test/BeyondMimic/logs/tracking_local_smoke`

The generated NPZ should be validated with:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_motion_npz_contract.py \
  /mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke/walk1_subject1_50hz.npz \
  --summary-json /mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke/walk1_subject1_50hz_contract.json
```

`run_tracking_local_smoke.sh` now invokes this validator immediately after `csv_to_npz_local.py`, before replay or
training smoke.

The static producer/consumer contract can be audited without launching Kit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/motion_preprocessing_contract_audit.py
```

Output:

`/mnt/infini-data/test/BeyondMimic/res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json`

## Do-Not-Run Gate

Do not start long PPO training until all of the following pass:

- IsaacLab/Kit headless smoke with `SimulationApp`/`AppLauncher`.
- One-motion `csv_to_npz.py` conversion using project-local tmp/cache paths.
- `replay_npz.py` on the converted motion.
- A short `train.py` smoke with very small `--num_envs` and `--max_iterations`, only after the official script has been
  adapted or wrapped so it can use a local `motion.npz` without requiring WandB upload/download.
- `validate_motion_npz_contract.py` on the generated `motion.npz`.
