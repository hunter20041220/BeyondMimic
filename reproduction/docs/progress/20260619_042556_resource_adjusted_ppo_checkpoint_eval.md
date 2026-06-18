# Progress Update

## Goal

Evaluate the locally trained resource-adjusted G1 PPO checkpoint through the official tracking task and RSL-RL inference
API, using the user-requested GPU 4/7 policy and preserving the official-vs-resource-adjusted boundary.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Official `whole_body_tracking/scripts/rsl_rl/play.py`
- Official RSL-RL `OnPolicyRunner.load()` / `get_inference_policy()` implementation

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/blocked_gate_audit.py`
- `reproduction/PROGRESS.md`
- `reproduction/docs/progress/20260619_042556_resource_adjusted_ppo_checkpoint_eval.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits -i 4,7
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py
```

## Results

The first eval attempt reached the official task stack but failed on a local RSL-RL API mismatch:
`OnPolicyRunner.load()` does not accept `map_location` in this environment. The script was fixed to use the local
signature, then rerun.

Before the rerun, GPU 4 was occupied by a `/mnt/infini-data/test/wangjc/` Python process. Per the user's updated GPU
policy, that process was terminated and the action was recorded in
`res/gpu_guard/20260619_gpu47_wangjc_kill_summary.json`.

The rerun completed successfully:

- Status: `ok_resource_adjusted_ppo_checkpoint_eval_completed`
- Checkpoint: `model_99.pt`
- Selected physical GPUs: `[4, 7]`
- Active rollout device: physical GPU 4 via `cuda:0`
- Environments: `512`
- Steps: `299`
- Total environment steps: `153088`
- Loaded iteration: `99`
- Runtime: `1193.852` seconds
- GPU 4 peak memory: `54692` MiB
- GPU 4 mean utilization: `98.2%`

Key metrics:

- Mean reward over steps: `0.025898515209431035`
- Mean anchor position error: `0.10595783163921091`
- Mean body position error: `0.18350737062859096`
- Mean joint position error: `1.2326450995776965`
- Done count total: `13172`
- Timeout count total: `0`

## Verification

Final verification passed. The new result is wired into artifact manifest, paper-vs-reproduction comparison,
blocked-gate audit, final report, required-artifact absence audit, and master audit.

- `required_artifact_absence_audit`: `ok`, 18 rows.
- `artifact_manifest`: `ok`, 293 artifacts.
- `paper_vs_reproduction`: `ok`, 128 rows.
- `completion_matrix_status_audit`: `ok`, 161 rows, 0 invalid statuses.
- `verification_command_syntax_audit`: `ok`, 179 scripts, 0 failed syntax checks.
- `verification_command_script_manifest`: `ok`, 179 scripts.
- `verification_command_coverage_audit`: `ok`, 187 commands.
- `progress_report_audit`: `ok`, 38 checklist rows.
- `reproduction_master_audit`: `ok`.
- `git diff --check`: passed.

## Failed / Blocked Items

- The first eval attempt failed due to `OnPolicyRunner.load(map_location=...)` API mismatch; the failure is retained in
  the log history.
- Official G1 USD conversion/replay remains blocked.
- This is not official `csv_to_npz.py`/`replay_npz.py` evaluation.
- No paper-scale PPO benchmark, true DAgger rollout logs, Fig. 5/Fig. 6 videos, TensorRT deployment, or real robot
  result was produced.

## Effect on English Reading Report

This provides a stronger reproduction narrative: after task smoke and resource-adjusted PPO training, the project now
evaluates the trained checkpoint through the official task and RSL-RL inference API. The report can use this as honest
virtual policy-evaluation evidence while explicitly stating that it is resource-adjusted rather than official
paper-level BeyondMimic tracking evaluation.

## Next Step

Use the evaluated checkpoint as a clearly labeled resource-adjusted teacher candidate for rollout dataset collection, or
continue the official converter/replay path if a lower-level USD conversion workaround becomes available.

## Git Commit

Pending before commit; final commit hash is recorded in the user-facing turn report.
