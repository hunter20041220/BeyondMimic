# Environment Status

## Shared Cache Configuration

All scripts source `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh`.

Verified in `bm_analysis`:

- `PIP_CACHE_DIR=/mnt/infini-data/test/BeyondMimic/cache/pip`
- `HF_HOME=/mnt/infini-data/test/BeyondMimic/cache/huggingface`
- `TORCH_HOME=/mnt/infini-data/test/BeyondMimic/cache/torch`
- `XDG_CACHE_HOME=/mnt/infini-data/test/BeyondMimic/cache/xdg`
- `TMPDIR=/mnt/infini-data/test/BeyondMimic/tmp`

## Environment Manager

Project-local micromamba:

`/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba`

Version: 2.8.1.

Install log:

`/mnt/infini-data/test/BeyondMimic/logs/setup/install_micromamba.log`

## bm_analysis

Prefix:

`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis`

Purpose:

- Released-data inspection.
- CSV / pandas / scipy analysis.
- Headless matplotlib plotting to PDF, SVG, PNG.

Installed with:

```bash
micromamba create -y -p /mnt/infini-data/test/BeyondMimic/envs/bm_analysis \
  -c conda-forge python=3.10 pip numpy pandas scipy matplotlib-base pyyaml tqdm zstandard
```

Smoke test:

`/mnt/infini-data/test/BeyondMimic/logs/setup/analysis_smoke.log`

Evidence:

- Loaded `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv`.
- Data shape: `(7839, 36)`.
- Generated:
  - `/mnt/infini-data/test/BeyondMimic/res/released_figures/smoke/analysis_smoke.pdf`
  - `/mnt/infini-data/test/BeyondMimic/res/released_figures/smoke/analysis_smoke.svg`
  - `/mnt/infini-data/test/BeyondMimic/res/released_figures/smoke/analysis_smoke.png`

Lock files:

- `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/environment.yml`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/requirements-lock.txt`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/pip-freeze.txt`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/conda-list-explicit.txt`

## bm_tracking

Status: partially complete.

Official requirements from local sources:

- Isaac Sim 4.5.0.
- Isaac Lab 2.1.0.
- Python 3.10.
- Official `whole_body_tracking` extension.
- RSL-RL.
- Unitree G1 assets.

Host facts:

- Ubuntu 20.04.5.
- No local Isaac Sim installation discovered.
- IsaacLab documentation recommends binary Isaac Sim installation for Ubuntu 20.04.
- No Isaac Sim binary archive was found in `/mnt/infini-data/test/BeyondMimic/download`.

Created `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` with Python 3.10.20, pip, git, and git-lfs.

Lock files:

- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/environment.yml`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/requirements-lock.txt`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/pip-freeze.txt`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/conda-list-explicit.txt`

PyTorch smoke status:

- Installed `torch==2.5.1+cu121` and `torchvision==0.20.1+cu121`.
- `torch.cuda.is_available() == True`.
- `torch.cuda.device_count() == 8`.
- CUDA tensor operation passed with sum `4.0`.
- Log: `/mnt/infini-data/test/BeyondMimic/logs/setup/install_bm_tracking_torch.log`.

Isaac Sim pip attempt:

- Command: `pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com`.
- Failed because the placeholder package could not find a compatible real wheel for this Ubuntu 20.04 host.
- Log: `/mnt/infini-data/test/BeyondMimic/logs/setup/install_bm_tracking_isaacsim.log`.

Isaac Sim binary route:

- Downloaded official standalone binary:
  `/mnt/infini-data/test/BeyondMimic/download/_supplemental/isaac-sim-standalone-4.5.0-linux-x86_64.zip`
- SHA256:
  `c33a452a39bccd2a8a7917b06b5812c6a9d08352e0f03ee42b59d4fdc81b7fb9`
- Extracted to:
  `/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0`
- Isaac Sim Python smoke passed:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/isaacsim_python_smoke.log`
- Isaac Sim standalone Python package list:
  `/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0/pip-freeze.txt`

IsaacLab v2.1.0:

- Work copy:
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0`
- `_isaac_sim` symlink points to:
  `/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0`
- Installed editable packages: `isaaclab`, `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks`.
- Installed `rsl-rl-lib==2.3.1` and `warp-lang==1.14.0`.
- Plain `import isaaclab` passes under Isaac Sim Python.
- `isaaclab_tasks` and full tracking config imports require Isaac Sim extension modules such as `isaacsim.core`; these are normally managed by Kit/SimulationApp rather than plain Python.

Current IsaacLab blocker:

- Headless Kit smoke starts but repeatedly reports `Failed to create change watch ... errno=28/No space left on device`.
- Disk space is sufficient; the actionable host limit is inotify:
  `fs.inotify.max_user_watches=8192`, `fs.inotify.max_user_instances=128`.
- Logs:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/isaaclab_headless_smoke.log`
  `/mnt/infini-data/test/BeyondMimic/logs/setup/isaaclab_headless_smoke_retry.log`
  `/mnt/infini-data/test/BeyondMimic/logs/setup/inotify_status.txt`

whole_body_tracking:

- Work copy:
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking`
- Unitree assets extracted under:
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets`
- Installed editable into Isaac Sim Python with `--no-deps`.
- Non-Kit package/asset smoke script:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/whole_body_tracking_nokit_smoke.py`
- Non-Kit tracking configuration audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_config_audit.py`
- Tracking audit output:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/smoke_config_audit/tracking_config_audit.json`
- Verified by this audit:
  PPO `[512, 256, 128]` actor/critic hidden dimensions, `sim.dt=0.005`, `decimation=4`,
  `50.0 Hz` control frequency, 4096 envs, complete G1 URDF mesh references, and a LAFAN1 G1 CSV sample.

## bm_diffusion

Status: created for Level C VAE/diffusion smoke and debug training mechanics.

Prefix:

`/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion`

Purpose:

- Conditional VAE and diffusion smoke/probe execution.
- Torch CUDA tensor checks for Level C training code paths.
- NumPy/SciPy/PyYAML/tqdm formula and dataset utilities without IsaacLab/Kit dependencies.

Created with:

```bash
micromamba create -y -p /mnt/infini-data/test/BeyondMimic/envs/bm_diffusion \
  -c conda-forge python=3.10 pip numpy scipy pyyaml tqdm
```

Installed with pip from the CUDA 12.1 PyTorch wheel index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121
```

Smoke/audit evidence:

- `/mnt/infini-data/test/BeyondMimic/res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json`
- Python `3.10.20`, NumPy `2.2.6`, SciPy `1.15.2`, PyYAML `6.0.3`, tqdm `4.68.2`.
- Torch `2.5.1+cu121`, torchvision `0.20.1+cu121`.
- `torch.cuda.is_available() == True`, `torch.cuda.device_count() == 8`, CUDA tensor sum smoke passed.

Lock files:

- `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/environment.yml`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/requirements-lock.txt`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/pip-freeze.txt`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/conda-list-explicit.txt`

Setup logs:

- `/mnt/infini-data/test/BeyondMimic/logs/setup/create_bm_diffusion.log`
- `/mnt/infini-data/test/BeyondMimic/logs/setup/install_bm_diffusion_torch.log`
- `/mnt/infini-data/test/BeyondMimic/logs/setup/export_bm_diffusion_locks.log`

Note: the pip install wrapper was interrupted after a prolonged shared-filesystem IO wait, so its log does not contain
the usual final success line. The exported locks and runtime smoke audit prove the installed environment is usable for
Torch/CUDA smoke checks. This environment does not imply that VAE/diffusion paper training or metrics have been
completed.

## Released-Data Plotting

The `bm_analysis` environment is sufficient for non-torch released-data plots:

```bash
source /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh
/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba run \
  -p /mnt/infini-data/test/BeyondMimic/envs/bm_analysis \
  python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduce_released_figures.py
```

Adaptive sampling source files are torch-save `.npy` files. They are converted with the `bm_tracking` Python/Torch
environment and plotted with `bm_analysis`:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/convert_adaptive_sampling.py

/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba run \
  -p /mnt/infini-data/test/BeyondMimic/envs/bm_analysis \
  python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/plot_adaptive_sampling_released.py
```
