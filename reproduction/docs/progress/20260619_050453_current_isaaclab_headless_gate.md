# Progress Update

## Goal

Refresh the current IsaacLab `AppLauncher(headless=True)` startup gate and remove stale audit/report wording that still described the live headless gate as blocked.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/setup/env_probe/env_import_probe.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_live_gate_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/env_import_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_current_headless_gate.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/env_import_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260619_050453_current_isaaclab_headless_gate.md`

## Commands Run

- `nvidia-smi -i 4,7,6`
- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/isaaclab_current_headless_gate.py`
- `envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py`
- `tail -120 logs/setup/isaaclab_current_headless_gate/...`

## Results

- The current headless gate passed on physical GPU 4 with no `CUDA_VISIBLE_DEVICES` hiding.
- Status: `ok`.
- `app_launcher_headless_success_sentinel=true`.
- Payload reports `device=cuda:4`, `headless=true`, and `is_running=true`.
- CUDA P2P/IOMMU warning remains recorded, but no fatal Vulkan/GPU-foundation/inotify error blocked startup.
- A failed first attempt using `CUDA_VISIBLE_DEVICES=4` reproduced the known Omniverse/CUDA enumeration mismatch; the passing configuration keeps all GPUs visible and selects physical GPU 4 directly.
- GPU6 had a non-wangjc VLLM process and was not touched.

## Verification

Full verification bundle is scheduled after refreshing `env_import_probe`, final report, manifest, and master audit.

## Failed / Blocked Items

- Official G1 conversion/replay is still blocked.
- This gate does not prove official replay, PPO training, DAgger, VAE/diffusion, Fig.5/Fig.6, TensorRT deployment, or real robot behavior.

## Effect on English Reading Report

This cleans up the environment narrative: the report can now say IsaacLab headless startup is operational on the current host, while the remaining Level B blocker is the official G1 conversion/replay path rather than AppLauncher startup.

## Next Step

Refresh audits and commit/push the current gate evidence. Then continue mainline official replay recovery or use the already-audited resource-adjusted stack for the next downstream full-data experiment with explicit boundaries.

## Git Commit

Pending at time of writing.
