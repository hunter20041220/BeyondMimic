# Progress Update

## Goal

Continue repairing the IsaacLab live headless gate by moving below the Python/IsaacLab layer into Vulkan loader, NVIDIA ICD, and CUDA interop diagnostics.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/known_limitations.md`
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/blocked_gates/blocked_gate_audit.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Isaac Sim / IsaacLab Kit app files and NVIDIA Vulkan ICD files under `/etc/vulkan`

## Files Modified

- `reproduction/scripts/vulkan_runtime_probe.py`
- `reproduction/scripts/isaaclab_live_gate_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/blocked_gate_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/final_reproduction_report.md`
- refreshed lightweight JSON/TSV audit outputs under `res/setup/`, `res/blocked_gates/`, `res/artifact_manifest/`, `res/final_report/`, and `res/master_audit/`

## Commands Run

- Inspected `/etc/vulkan/icd.d/nvidia_icd.json` and `/etc/vulkan/implicit_layer.d/nvidia_layers.json`
- Checked NVIDIA Vulkan symbols in `libGLX_nvidia.so.0` and `libEGL_nvidia.so.0`
- Ran minimal Vulkan `vkCreateInstance` probes through `ctypes`
- Ran `VK_LOADER_DEBUG` diagnostics and saved local logs under `logs/setup/`
- `python3 reproduction/scripts/vulkan_runtime_probe.py`
- `python3 reproduction/scripts/isaaclab_live_gate_probe.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- The default system Vulkan loader is missing: `libvulkan.so.1` does not resolve from the system library path.
- The host NVIDIA ICD points to `libGLX_nvidia.so.0`; that library exports `vk_icdGetInstanceProcAddr`, but returns null for core Vulkan entry points in the direct probe.
- A project-local ICD pointing to `libEGL_nvidia.so.0` was generated at `res/setup/vulkan_runtime_probe/nvidia_egl_icd.json`.
- Isaac bundled Vulkan loader plus the project EGL ICD succeeds on minimal `vkCreateInstance`.
- The IsaacLab AppLauncher candidate using the project EGL ICD removes the previous `ERROR_INCOMPATIBLE_DRIVER` / `carb::graphics::createInstance failed` error.
- The live gate still does not pass: the current blocker has advanced to `cuda_p2p_iommu_validation`, with Kit logging IOMMU enabled and CUDA P2P bandwidth/latency validation failing at `peer access is already enabled`.
- Candidate Kit args for disabling P2P validation did not bypass this check.
- `artifact_manifest` increased to 228 artifacts.
- `master_audit` remains `ok`.

## Verification

Targeted verification passed before the full required bundle:

- `python3 reproduction/scripts/vulkan_runtime_probe.py`
- `python3 reproduction/scripts/isaaclab_live_gate_probe.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Failed / Blocked Items

- `isaaclab_live_headless_gate_ok=false`.
- Current blocker: `cuda_p2p_iommu_validation`.
- Official `whole_body_tracking` replay, tracking task smoke/eval, PPO training/evaluation, teacher rollout collection, closed-loop VAE/diffusion evaluation, and Fig. 5/Fig. 6 paper-level videos/metrics remain blocked until the AppLauncher reaches a clean success sentinel.

## Effect on English Reading Report

This round improves the report's reproducibility discussion: the failure is no longer a vague IsaacLab setup issue. The evidence now separates three layers: Python packages are importable, Vulkan can be made to initialize through an EGL ICD workaround, and the remaining blocker is CUDA peer-to-peer/IOMMU validation inside the Isaac Sim graphics stack.

## Next Step

Find the correct Kit/carb setting or host configuration to skip or repair CUDA P2P validation on this multi-H20 IOMMU-enabled server, then rerun `python3 reproduction/scripts/isaaclab_live_gate_probe.py`.

## Git Commit

Pending at the time this progress file was written.
