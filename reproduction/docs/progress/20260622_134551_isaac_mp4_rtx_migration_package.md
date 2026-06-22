# Progress Update

## Goal

Build `/mnt/infini-data/test/BeyondMimic/isaac_mp4_need/` as a migration package for generating true IsaacLab/Isaac Sim rendered rollout MP4s on an RTX/RT-core machine. This round does not claim that the H20 server generated an MP4; it records the H20 blocker and packages the code/data needed to move the run.

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
- `res/failed_runs/isaac_mp4/isaaclab_rendered_policy_rollout_video_failed_gate.json`
- `res/setup/isaac_render_stack_repair_audit/isaac_render_stack_repair_audit.json`

## Files Modified

- `reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py`
- `reproduction/scripts/isaac_render_stack_repair_audit.py`
- `reproduction/scripts/build_isaac_mp4_need_package.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/progress/20260622_134551_isaac_mp4_rtx_migration_package.md`

## Commands Run

- `find reproduction/scripts ...`
- `find res/runs res/level_c res/tracking res/visualization res/report_assets res/failed_runs res/setup ...`
- `find reproduction/third_party/official/whole_body_tracking reproduction/third_party/official/IsaacLab-v2.1.0 ...`
- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/build_isaac_mp4_need_package.py reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py reproduction/scripts/isaac_render_stack_repair_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/build_isaac_mp4_need_package.py`
- `du -sh isaac_mp4_need`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

- Created `isaac_mp4_need/`.
- Created `isaac_mp4_need/README_ISAAC_MP4_RTX.md`.
- Created `isaac_mp4_need/isaac_mp4_need_manifest.json`.
- Created `isaac_mp4_need/isaac_mp4_need_manifest.tsv`.
- Created `isaac_mp4_need/run_rtx_smoke.sh`.
- Created `isaac_mp4_need/run_rtx_full_videos.sh`.
- Package inventory status: `ok_isaac_mp4_rtx_migration_package`.
- Manifest rows: `2071`.
- Copied files: `2063`.
- Referenced large files/directories: `8`.
- Local package size: about `759M`.

## Verification

- Package build and Python syntax checks passed.
- `artifact_manifest.py`: passed, `1586` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed and now records `isaac_mp4_need_package`.
- `completion_matrix_status_audit.py`: passed, `209` rows, invalid statuses `0`.
- `verification_command_syntax_audit.py`: passed, failed `0`.
- `verification_command_script_manifest.py`: passed, `199` scripts.
- `verification_command_coverage_audit.py`: passed, `207` commands.
- `reproduction_master_audit.py`: passed, `406/406` artifacts.

## Failed / Blocked Items

- True Isaac rendered MP4 remains blocked on the H20 host as `blocked_h20_isaac_sim_rendering_stack`.
- The true rendered gate still has no MP4, keyframes, metrics CSV, or successful `Tracking-Flat-G1-v0` creation on H20.
- This package is intended for RTX/RT-core hardware and does not resolve the H20 Vulkan/Kit/Hydra rendering startup blocker.

## Effect on English Reading Report

This round adds a clear engineering handoff story for the code reproduction section: diagnostic skeleton/contact-sheet videos already exist, but true Isaac rendered simulation video requires a supported RTX rendering host. The report can cite this package as reproducibility infrastructure rather than as completed paper-level video evidence.

## Next Step

Copy the project plus `isaac_mp4_need/` to an RTX/RT-core machine and first run:

```bash
cd /path/to/BeyondMimic
BM_ROOT=$PWD BM_GPU_ID=0 ./isaac_mp4_need/run_rtx_smoke.sh
```

After smoke succeeds, run:

```bash
BM_ROOT=$PWD BM_GPU_ID=0 BM_ISAAC_MP4_STEPS=300 ./isaac_mp4_need/run_rtx_full_videos.sh
```

## Git Commit

Pending until the final git commit for this round is created.
