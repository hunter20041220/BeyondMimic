# Progress Update

## Goal

Repair the current IsaacLab live headless gate enough to support the next tracking mainline step, and preserve a clean audit trail for GitHub and the English reading report.

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
- `reproduction/scripts/isaaclab_current_headless_gate.py`
- `reproduction/scripts/gpu_wangjc_process_guard.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`

## Files Modified

- `reproduction/scripts/tracking_g1_current_task_env_construction_gate.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/progress/20260621_114600_current_task_env_gate.md`

## Commands Run

```bash
ps -ef | rg 'tracking_g1_current_task_env_construction_gate|current_task_env_construction_gate_worker' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi pmon -c 1
envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_TERMINATE_WANGJC_GPU_GUARD=1 envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_HEADLESS_GATE_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py
BM_TASK_ENV_GATE_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_current_task_env_construction_gate.py
```

## Results

- The first staged task-env diagnostic reproduced the historical `SIGTERM` pattern during Isaac/Kit startup.
- Process audit found one matching `/mnt/infini-data/test/wangjc/.../gpu_out_of_bounds_guard.py --blocked-gpus 4,5,6,7` process.
- The guard script recorded a dry-run audit at `res/gpu_guard/20260621_033729_gpu47_wangjc_process_guard.json`.
- The authorized termination run recorded `res/gpu_guard/20260621_033745_gpu47_wangjc_process_guard.json` with `remaining=0`.
- `reproduction/scripts/isaaclab_current_headless_gate.py` then passed on physical GPU 4 with status `ok`.
- The new staged `Tracking-Flat-G1-v0` construction gate passed on physical GPU 4 using the official-importer G1 USDA and the FK-repaired split motion bundle.
- The task gate records `action_dim=29`, `robot_num_joints=29`, `robot_num_bodies=40`, and `num_envs=1`.
- The legacy missing motion path is now explicitly recorded as a migration artifact and no longer blocks the current task construction gate.

## Verification

Final verification chain for this round passed:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

Observed results:

- `artifact_manifest.py`: `ok`, `1359` artifacts after adding the task gate, progress report, and two explicit guard records.
- `paper_vs_reproduction_comparison.py`: `ok`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `199` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `199` scripts, `0` failures.
- `verification_command_script_manifest.py`: `ok`, `199` scripts.
- `verification_command_coverage_audit.py`: `ok`, `207` commands, `10` smoke pass references.
- `reproduction_master_audit.py`: `ok`, `343/343` master-audit artifacts passed, `0` failed.

## Failed / Blocked Items

- This is still a task construction gate, not official replay-loop success, PPO training, DAgger data collection, VAE/diffusion closed-loop evaluation, TensorRT deployment, Fig. 5/Fig. 6 reproduction, or real-robot evidence.
- No video was generated in this round because no replay or rollout was executed; the next successful reset/step or replay run should generate report media under `res/visualization/` or `res/report_assets/`.
- The external GPU guard can reappear and kill GPU 4/7 Isaac runs; future mainline scripts should run the narrow guard audit before starting GPU 4/7 Isaac work.

## Effect on English Reading Report

This round provides a stronger reproducibility story for the report: a failure that looked like IsaacLab instability was traced to an external GPU guard, then the same runtime path reached the headless AppLauncher sentinel and constructed the G1 tracking task with the expected 29-action / 29-joint / 40-body contract. It supports the code reproduction section while preserving the claim boundary that this is not a paper-level closed-loop result.

## Next Step

Run the next tracking mainline gate: reset/step smoke or official replay on the FK-repaired split motion bundle, then move to full-motion replay/eval only after the single-motion gate writes metrics cleanly.

## Git Commit

Pending at the time this file was written; final commit hash is recorded in the user-facing round summary.
