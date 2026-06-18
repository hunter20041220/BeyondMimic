# Environment Plan

## Current Host Audit

Evidence: `/mnt/infini-data/test/BeyondMimic/logs/setup/system_audit.txt`

- OS: Ubuntu 20.04.5 LTS.
- GPUs: 8 x NVIDIA GeForce RTX 4090 D, 24564 MiB each.
- Driver: 560.35.03, CUDA driver capability reported by `nvidia-smi`: 12.6.
- `nvcc`: not found on host.
- `conda`, `mamba`, `micromamba`: not found on host.
- Default `python`: `/usr/bin/python`, Python 2.7.18.
- `python3`: `/usr/bin/python3`.
- `/shared_disk`: 10P total, 5.1P available.

GPU 6 reported about 3988 MiB used during audit while `nvidia-smi` listed no process; this must be checked again before scheduling long jobs.

## Required Versions From Official Sources

- `whole_body_tracking/README.md`: Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10, Linux x86_64.
- `IsaacLab-v2.1.0/environment.yml`: `python=3.10`, `importlib_metadata`.
- `whole_body_tracking/pyproject.toml`: Python 3.10 toolchain assumptions.
- `motion_tracking_controller/README.md`: ROS 2 Jazzy and `legged_control2` for C++ inference/sim-to-sim. This likely requires system-level packages and is deferred until the Python/Isaac tracking environment is proven.

## Planned Prefix Environments

Because no conda/mamba executable exists on the host, Phase 1 will install a project-local micromamba executable under
`/mnt/infini-data/test/BeyondMimic/envs/_micromamba` and create prefix environments:

- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis`

Current update: micromamba 2.8.1 is installed and `bm_analysis` exists. Its first attempt with `matplotlib`
pulled Qt and was interrupted because `qt6-main` extraction was too slow on the project filesystem; the successful
environment uses `matplotlib-base` for headless plotting.

All package, pip, Hugging Face, torch, XDG, WandB, Isaac, and temporary caches are redirected by
`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh`.

## Installation Order

1. Install micromamba to project disk if still absent.
2. Create `bm_tracking` with Python 3.10 from IsaacLab environment requirements.
3. Install or expose Isaac Sim 4.5.0, then Isaac Lab 2.1.0 from the local copy.
4. Create readonly-preserving working copies of official repos in `reproduction/third_party/official`.
5. Install `whole_body_tracking` editable from the working copy, not the download source.
6. Install `rsl_rl` from local dependency copy.
7. Create lighter `bm_analysis` for released-data plotting.
8. Create `bm_diffusion` after paper/source extraction clarifies exact packages.

## Smoke Test Gates

Phase 1 is not complete until the tests listed in `goal.md` Section 5.5 pass or are documented as blocked with concrete evidence.
