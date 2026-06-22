# Progress Update

## Goal

Attempt the current minimal true IsaacLab rendered-media target: launch IsaacLab/Isaac Sim with cameras enabled, load the current best local robot-order FK-repaired PPO checkpoint, and record a 300-step G1 policy rollout MP4 from the real `Tracking-Flat-G1-v0` simulator rather than a matplotlib skeleton plot. The execution was first bounded to 10 steps to recover the rendering gate safely before expanding to 300 steps.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Existing policy/skeleton video scripts under `reproduction/scripts/`
- IsaacLab `AppLauncher`, camera/rendering experience files, and `Tracking-Flat-G1-v0` task config under the local official IsaacLab/whole-body-tracking copies

## Files Modified

- `reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py`
- `res/visualization/isaac_mp4/README.md`
- `res/visualization/isaac_mp4/isaaclab_rendered_policy_rollout_video_asset.json`
- `res/visualization/isaac_mp4/isaaclab_rendered_policy_rollout_worker.py`
- `res/failed_runs/isaac_mp4/isaaclab_rendered_policy_rollout_video_failed_gate.json`
- `reproduction/docs/progress/20260622_124556_isaaclab_rendered_mp4_gate.md`

## Commands Run

- `BM_ISAAC_MP4_STEPS=10 BM_ISAAC_MP4_SEED=20260779 BM_ISAAC_MP4_TIMEOUT_SECONDS=180 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py`
- `tail -n 220 logs/isaac_mp4/20260622_044057_seed20260779_10step_robot_order_policy.log`
- `tail -n 120 logs/isaac_mp4/20260622_044239_seed20260779_10step_robot_order_policy.log`
- `nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,display_active,display_mode --format=csv,noheader,nounits`
- `ps -eo pid,ppid,stat,etime,cmd | rg -i 'isaaclab_rendered_policy_rollout_worker|isaacsim|kit|omni|python.*isaac_mp4' || true`

## Results

- New script attempts a real IsaacLab/Isaac Sim rendered rollout path using `AppLauncher(headless=True, enable_cameras=True)`, official `isaaclab.python.headless.rendering.kit`, `Tracking-Flat-G1-v0`, local G1 official-importer USDA, local robot-order FK-repaired motion bundle, and the current best local robot-order PPO checkpoint.
- The script defaults to candidate GPUs `5,6` and selected physical GPU `5` on the latest run.
- The latest gate status is `failed_isaaclab_rendered_policy_rollout_mp4`.
- No MP4, keyframe PNG, metrics CSV, or worker summary was produced.
- The process exited with return code `-11` after a Vulkan/Kit segmentation fault before the `after_app`, `env_created`, `render_product_created`, or `env.step` sentinels.
- Latest failure classification records `startup_failed=true`, `server_rendering_stack_blocker=true`, and `replicator_hydra_vulkan_device_lost=true`.
- Detected log patterns include `GLFW initialization failed`, `GPU crash occurred`, `VkResult: ERROR_DEVICE_LOST`, and `Segmentation fault`.

## Verification

This progress update records the failed rendering gate. The follow-up verification round must run:

- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Failed / Blocked Items

- The requested true IsaacLab rendered policy rollout MP4 is blocked by the server-side Isaac Sim Vulkan/Hydra/Replicator rendering stack.
- This is not a PPO checkpoint failure and not a simulated policy rollout quality failure, because the crash occurs before the IsaacLab tracking environment is created.
- A 300-step video was not attempted after the 10-step rendering gate failed.
- The existing matplotlib/keypoint videos remain diagnostic visualizations only and must not be described as true IsaacLab rendered deployment videos.

## Effect on English Reading Report

This adds a precise limitation for the reproduction/media section: the project has local virtual policy and guidance rollouts plus skeleton/diagnostic MP4s, but the stronger Isaac Sim mesh-rendered MP4 evidence is currently blocked by the host rendering stack. The report can honestly state that the attempted true IsaacLab rendered deployment path reached Isaac Sim rendering startup but failed before environment creation due `VkResult: ERROR_DEVICE_LOST`.

## Next Step

Resolve the host rendering gate before retrying 300-step media: install/run `vulkaninfo`, verify NVIDIA Vulkan/RTX support for Isaac Sim 4.5 on this H20 server, check X/GLFW/headless EGL requirements, and rerun the same script. Once the 10-step gate produces a nonblank MP4, expand to 300 steps, then add reference replay, teacher/policy, VAE, and guided-vs-unguided videos.

## Git Commit

Pending. This update should be committed together with the Isaac MP4 script and failed-gate audit after verification.
