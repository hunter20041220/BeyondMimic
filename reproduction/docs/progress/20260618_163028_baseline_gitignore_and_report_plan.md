# Progress Update

## Goal

Establish the new phase baseline for GitHub-tracked code/report artifacts, verify `.gitignore` coverage for large local assets, and document the immediate plan before continuing IsaacLab live headless gate work.

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

## Files Modified

- `.gitignore`
- `reproduction/docs/progress/20260618_163028_baseline_gitignore_and_report_plan.md`

## Commands Run

- `git status --short`
- `git diff --stat`
- `git ls-files -o --exclude-standard`
- `git ls-files | while read ... wc -c ...`
- `git rm --cached --ignore-unmatch ...`

## Results

- Confirmed the repository is on `main` and tracking `origin/main`.
- Confirmed the active evidence baseline remains:
  - master audit: `ok`
  - master artifacts: `188/188`
  - artifact manifest: `226`
  - paper-vs-reproduction rows: `122`
- Confirmed the current environment baseline:
  - `bm_analysis` import stack is usable.
  - `bm_diffusion` has CUDA PyTorch and sees GPUs 5/6 under `CUDA_VISIBLE_DEVICES=5,6`.
  - `bm_tracking` imports pip Isaac Sim 4.5 and local editable IsaacLab/whole_body_tracking packages, but live headless Kit remains blocked.
- Removed already-tracked large generated/local files from Git tracking while keeping them on disk.
- Added `.gitignore` rules for large full-split TSV files and failed venv-wrapper directories.

## Verification

Full verification will be run after this baseline documentation update and index cleanup. This update does not change reproduction logic, model code, or experiment metrics.

## Failed / Blocked Items

- `isaaclab_live_headless_gate_ok=false` remains the key technical blocker.
- Live AppLauncher did not reach the success sentinel in the latest environment probe because host Kit/Vulkan startup failed.
- Historical inotify saturation remains documented in existing audits.
- No official replay, PPO tracking smoke, teacher rollout, VAE closed-loop rollout, diffusion closed-loop rollout, or TensorRT deployment was run in this baseline step.

## Effect on English Reading Report

This update improves the future English reading report by establishing a clean, auditable GitHub baseline. It supports a transparent reproduction section that separates tracked code/docs/small audit evidence from local-only environments, datasets, checkpoints, videos, and large logs.

## Next Step

Continue with the IsaacLab live headless gate: inspect the Kit/Vulkan startup log, run a minimal sentinel-based AppLauncher probe, and record whether the blocker is Vulkan, inotify, or both on the current host.

## Git Commit

Pending at time of writing. Intended commit message: `chore: establish github hygiene baseline`.
