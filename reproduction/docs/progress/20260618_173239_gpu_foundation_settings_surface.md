# Progress Update

## Goal

Continue repairing the IsaacLab live headless gate by testing earlier SimulationApp GPU configuration paths and auditing local `gpu.foundation` / `carb.cudainterop` settings surfaces.

## Files Read

- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/setup/cuda_p2p_runtime_probe/cuda_p2p_runtime_probe.json`
- `res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/artifact_manifest/artifact_manifest.json`
- Isaac Sim `simulation_app.py`
- IsaacLab `app_launcher.py`
- Isaac Sim / IsaacLab `.kit`, `.toml`, `.json`, and Python settings surfaces

## Files Modified

- `reproduction/scripts/isaaclab_live_gate_probe.py`
- `reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- refreshed audit outputs under `res/setup/`, `res/artifact_manifest/`, `res/final_report/`, and `res/master_audit/`

## Commands Run

- `python3 reproduction/scripts/isaaclab_live_gate_probe.py`
- `python3 reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- Confirmed that `AppLauncher` supports passing `multi_gpu=False` into `SimulationApp` earlier than raw `kit_args`.
- Added an AppLauncher candidate using `BM_ISAACLAB_MULTI_GPU=false` plus EGL ICD and single-GPU renderer args.
- The `multi_gpu=False` candidate still reaches `after_app` but not `after_close`; it limits active rendering to GPU 6 but does not stop IOMMU P2P validation.
- Added a CPU-device candidate. It still loads `gpu.foundation`, selects an active GPU internally, and fails on the same IOMMU P2P validation path.
- Added `isaaclab_gpu_foundation_settings_audit.py` to preserve settings-surface evidence and attempted repair candidates.
- `artifact_manifest` increased to 230 artifacts.
- `master_audit` remains `ok`.

## Verification

Targeted verification passed:

- `python3 reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Failed / Blocked Items

- `isaaclab_live_headless_gate_ok=false`.
- Current blocker remains `cuda_p2p_iommu_validation`.
- Official replay, tracking task smoke/eval, PPO training/evaluation, teacher rollout collection, and closed-loop VAE/diffusion evaluation remain blocked.

## Effect on English Reading Report

This adds a clean narrative for the limitations section: the project tested progressively earlier configuration routes, including SimulationApp `multi_gpu=False` and CPU-device mode, but the blocker is still inside Isaac Sim `gpu.foundation` startup on an IOMMU-enabled multi-H20 host.

## Next Step

Either identify a supported `gpu.foundation`/`carb.cudainterop` switch for the IOMMU P2P validation path, or document that this host requires an administrator/runtime change before live IsaacLab replay can be attempted.

## Git Commit

Pending at the time this progress file was written.
