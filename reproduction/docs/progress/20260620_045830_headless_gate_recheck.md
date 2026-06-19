# Progress Update

## Goal

Recheck the current IsaacLab `AppLauncher(headless=True)` live gate on the mainline GPU policy before continuing
tracking replay/evaluation work, and make sure the result remains audit-visible.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_full_dataset_task_eval/tracking_g1_official_csv_loop_full_dataset_task_eval.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260620_045830_headless_gate_recheck.md`

The live-gate log was written under `/mnt/infini-data/test/BeyondMimic/logs/setup/isaaclab_current_headless_gate/`
and remains intentionally outside Git.

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
BM_HEADLESS_GATE_GPU=4 BM_HEADLESS_GATE_TIMEOUT=240 envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py
```

## Results

- GPU 4 and GPU 7 were idle before the gate rerun (`1 MiB` used, `0%` util each).
- The current headless gate returned `status=ok` and `gate_ok=true`.
- The gate reached the `AppLauncher(headless=True)` success sentinel on physical GPU 4.
- The payload reports `is_running=true`, `device=cuda:4`, `headless=true`, and `multi_gpu=false`.
- This confirms that the current active tracking blocker is not basic AppLauncher startup.

## Verification

Full verification was run after this progress note in the same round. See the final user report for pass/fail status.

## Failed / Blocked Items

No new gate failure was introduced in this round. The remaining official tracking blocker is still the unpatched
official G1 URDF/USD conversion and replay path, not the current headless AppLauncher sentinel. The existing
resource-adjusted official-loop conversion/replay/task-eval results remain useful but must not be reported as
unpatched official paper-level tracking.

## Effect on English Reading Report

This supports a cleaner reproduction narrative: the environment is now strong enough to launch IsaacLab headless and
run resource-adjusted full-dataset tracking diagnostics, while the honest limitation is the official asset conversion
path and the absence of paper-scale unpatched PPO/DAgger/diffusion closed-loop evidence.

## Next Step

Use the passing headless gate to continue mainline virtual reproduction work: prioritize stable full-dataset
policy/closed-loop evaluation and report-ready videos/plots, while keeping unpatched official replay claims separate
from resource-adjusted evidence.

## Git Commit

Pending at the time this progress file was written.
