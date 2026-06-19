# Progress Update

## Goal

Recheck the current IsaacLab headless gate on the required GPU policy, rerun the official `replay_npz.py` loop body, and preserve the current GPU4/7 task-eval rerun result without overwriting earlier successful canonical evidence.

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
- `prompt06181626.txt`
- `reproduction/scripts/isaaclab_current_headless_gate.py`
- `reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py`

## Files Modified

- `reproduction/scripts/isaaclab_current_headless_gate.py`
- `reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/known_limitations.md`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/tracking/official_replay_npz_loop_with_enriched_usd/tracking_official_replay_npz_loop_with_enriched_usd_audit.json`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py
BM_CSV_TASK_EVAL_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py
BM_CSV_TASK_EVAL_GPU=7 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/isaaclab_current_headless_gate.py reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py
```

## Results

- Current headless `AppLauncher(headless=True)` gate passed on physical GPU 4: `status=ok`, `gate_ok=true`.
- Official `replay_npz.py` loop body completed the 299-step bound under the enriched-USD runtime patch: `status=ok_official_replay_loop_with_enriched_usd_patch`, `returncode=0`.
- The task-eval script now writes GPU4/7 candidate reruns into timestamped candidate directories and only promotes a rerun to the canonical artifact if `BM_CSV_TASK_EVAL_PROMOTE=1` and the candidate completes all 299 steps.
- Current GPU4/7 task-eval reruns reached AppLauncher, environment creation, and environment reset, then were killed with return code `-9` before step metrics were written. The failed rerun is retained under `res/failed_runs/tracking_g1_resource_adjusted_csv_task_eval_gpu47_20260619_124125/`.
- The earlier canonical GPU6 resource-adjusted 299-step task-eval artifact remains preserved and is not overwritten by failed current-GPU candidate reruns.

## Verification

Full verification is run after this progress file is written. A syntax-only check for the touched scripts passed.

## Failed / Blocked Items

- The unpatched official G1 URDF/USD converter path remains blocked.
- The current required-GPU task-eval candidate on GPUs 4/7 is not stable yet; it reaches reset but is killed before 299-step metrics.
- No new official PPO paper-scale training, closed-loop guided diffusion rollout, Fig. 5/Fig. 6 video, TensorRT deployment, or real-robot result was produced.

## Effect on English Reading Report

This round improves the report's reproduction narrative in two directions: it gives a fresh current-machine headless gate pass and official replay-loop completion, while also documenting that current GPU4/7 task-eval reruns are not yet stable enough to claim as a fresh pass. This supports a careful distinction between runnable official loop bodies, resource-adjusted canonical evidence, and current runtime blockers.

## Next Step

Run the full audit bundle, commit the updated scripts/reports, then decide whether to stabilize GPU4/7 task stepping or move directly to report-ready visualization from already successful PPO checkpoint evaluation.

## Git Commit

Pending.
