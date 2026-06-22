# Progress Update

## Goal

Repair the true IsaacLab/Isaac Sim rendered MP4 gate for a real `Tracking-Flat-G1-v0` G1 policy rollout, and determine whether the current H20 server can produce an Isaac-rendered MP4 rather than matplotlib skeleton media.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/isaac_mp4/isaaclab_rendered_policy_rollout_video_asset.json`
- `/mnt/infini-data/test/BeyondMimic/res/failed_runs/isaac_mp4/isaaclab_rendered_policy_rollout_video_failed_gate.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.rendering.kit`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaac_render_stack_repair_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- refreshed generated audit/report outputs under `/mnt/infini-data/test/BeyondMimic/res/`

## Commands Run

- Installed or verified graphics diagnostics/runtime packages: `vulkan-tools`, `mesa-utils`, `mesa-vulkan-drivers`, `xvfb`, `xauth`, and XCB/XKB helper libraries.
- Ran `vulkaninfo --summary` under default ICD, system NVIDIA ICD, project EGL ICD, Xvfb, and llvmpipe ICD.
- Ran full `vulkaninfo` for system NVIDIA ICD and checked ray-tracing extension strings.
- Ran a 10-step true IsaacLab rendered MP4 gate with system NVIDIA ICD.
- Ran a llvmpipe/PXR AppLauncher probe.
- Re-ran the MP4 gate under Xvfb after dependency installation.
- Re-ran the non-Xvfb MP4 gate to preserve the canonical Vulkan/Hydra failure artifact.

## Results

The gate is still blocked. The repaired script now defaults to `/etc/vulkan/icd.d/nvidia_icd.json`, sets project-local `XDG_RUNTIME_DIR`, disables the NVIDIA Optimus layer, records actual runtime settings in the worker log, and classifies H20 rendering failures explicitly.

The latest canonical MP4 gate still fails before `BM_SENTINEL:isaac_mp4:after_app`, before `Tracking-Flat-G1-v0` creation, before render-product creation, before `env.step`, and before any MP4/keyframes/metrics CSV are written. The failure is classified as `h20_isaac_rendering_hardware_blocker=true`, `server_rendering_stack_blocker=true`, `policy_or_checkpoint_failure=false`, and `physics_rollout_failed=false`.

## Verification

Passed:

- `envs/bm_analysis/bin/python reproduction/scripts/isaac_render_stack_repair_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

`artifact_manifest` now records `1578` artifacts. `reproduction_master_audit` is `ok`.

## Failed / Blocked Items

- No true Isaac rendered MP4 exists.
- No Isaac-rendered keyframes exist.
- No Isaac-rendered rollout metrics CSV exists.
- The current blocker is Isaac Sim Kit/Hydra/Vulkan rendering startup on NVIDIA H20, not the PPO checkpoint and not the closed-loop physics rollout.
- Xvfb does not repair the gate; the post-dependency Xvfb rerun fails with `GLXBadFBConfig`.
- llvmpipe/PXR does not repair the gate; Isaac GPU Foundation does not accept the CPU Vulkan device as a stable rendering fallback.

## Effect on English Reading Report

This gives the report a stronger and more honest simulation-media boundary: the project can describe non-rendering IsaacLab/G1 task execution and local virtual policy/VAE/guidance evidence, but true Isaac-rendered MP4 capture is blocked by server rendering hardware/driver constraints on this H20 host. The report should not present matplotlib skeleton videos as Isaac deployment video.

## Next Step

Use one of these paths:

1. Run the same script on a machine with supported RTX/RT-core graphics hardware and Isaac Sim display/rendering support.
2. Keep this H20 host for non-rendering IsaacLab physics/evaluation/training and generate report media on a separate RTX workstation.
3. If no RTX host is available, continue with existing local virtual evidence and clearly label all videos that are not true Isaac-rendered MP4.

## Git Commit

Pending at the time of this progress note.
