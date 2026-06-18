# Progress Update

## Goal

Refine the IsaacLab live headless gate blocker after the EGL ICD workaround moved the failure from Vulkan initialization to CUDA P2P/IOMMU validation.

## Files Read

- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json`
- `res/blocked_gates/blocked_gate_audit.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/comparison/paper_vs_reproduction.json`
- `logs/setup/isaaclab_live_gate_probe/app_launcher_project_egl_icd_no_cuda_visible_devices_device_cuda6.log`
- `reproduction/docs/progress/20260618_170858_vulkan_egl_icd_and_cuda_p2p_gate.md`
- Isaac Sim / IsaacLab `.kit`, `.toml`, Python, and native-plugin string surfaces related to `renderer.multiGpu`, `gpu.foundation`, and `carb.cudainterop`

## Files Modified

- `reproduction/scripts/cuda_p2p_runtime_probe.py`
- `reproduction/scripts/isaaclab_live_gate_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/blocked_gate_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- refreshed audit outputs under `res/setup/`, `res/blocked_gates/`, `res/artifact_manifest/`, `res/final_report/`, and `res/master_audit/`

## Commands Run

- Searched Kit/native plugin settings with `rg` and `strings`
- `python3 reproduction/scripts/cuda_p2p_runtime_probe.py`
- `python3 reproduction/scripts/isaaclab_live_gate_probe.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- Added `cuda_p2p_runtime_probe.py` to test CUDA peer-access behavior without launching Kit or training.
- Direct CUDA runtime peer-access calls on GPU pairs 6/7 did not reproduce the `peer access is already enabled` error; ordinary `cudaDeviceEnablePeerAccess` succeeds for 6->7 and 7->6 and reports unsupported for 6->6.
- Added a single-GPU renderer AppLauncher candidate using the EGL ICD and explicit `--/renderer/multiGpu/enabled=false`, `--/renderer/multiGpu/autoEnable=false`, `--/renderer/multiGpu/maxGpuCount=1`, `activeGpu=6`, and `physics/cudaDevice=6`.
- The single-GPU renderer candidate does limit Kit to GPU 6, but `gpu.foundation.plugin` still detects IOMMU and runs the CUDA P2P bandwidth/latency validation, which fails with `peer access is already enabled`.
- A `CUDA_VISIBLE_DEVICES=6` single-GPU candidate avoids the P2P signature but is not viable: `gpu.foundation.plugin` reports CUDA bad state / no device could be created / activeGpu incompatible.
- The current blocker remains `cuda_p2p_iommu_validation`.
- `artifact_manifest` increased to 229 artifacts.
- `master_audit` remains `ok`.

## Verification

Targeted verification passed:

- `python3 reproduction/scripts/cuda_p2p_runtime_probe.py`
- `python3 reproduction/scripts/isaaclab_live_gate_probe.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Failed / Blocked Items

- `isaaclab_live_headless_gate_ok=false`.
- Official `whole_body_tracking` replay, tracking task smoke/eval, PPO training/evaluation, teacher rollout collection, closed-loop VAE/diffusion evaluation, and Fig. 5/Fig. 6 paper-level videos/metrics remain blocked until the AppLauncher reaches a clean success sentinel.

## Effect on English Reading Report

This round adds useful negative evidence for the report: the remaining failure is not caused by Python imports, not by the original GLX Vulkan ICD once the EGL ICD workaround is applied, not by broad multi-GPU renderer activation alone, and not reproducible as a simple CUDA peer-access API failure outside Kit. It is currently localized to Isaac Sim's `gpu.foundation` / `carb.cudainterop` IOMMU validation path on this multi-H20 host.

## Next Step

Continue looking for a supported Kit/gpu.foundation setting or host-side configuration to skip or repair the IOMMU P2P validation. If none exists locally, record it as an external host-runtime blocker and avoid claiming live IsaacLab reproduction.

## Git Commit

Pending at the time this progress file was written.
