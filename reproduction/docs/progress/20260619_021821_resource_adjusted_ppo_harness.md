# Progress Update

## Goal

Move beyond the four-step train-entry smoke by adding and running a reusable PPO training harness for the
resource-adjusted G1 tracking path, while selecting free GPUs from the current shared server.

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
- Official `whole_body_tracking/scripts/rsl_rl/train.py`
- Official G1 PPO config `G1FlatPPORunnerCfg`
- RSL-RL `OnPolicyRunner` distributed-training implementation

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/blocked_gate_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/progress/20260619_021821_resource_adjusted_ppo_harness.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/blocked_gate_audit.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/progress_report_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
git diff --check
```

## Results

The harness selects available GPUs from physical GPUs 4-7 and launched `torch.distributed` on physical GPUs `[4, 7]`
with world size 2, total `1024` environments, PPO rollout length `24`, and `100` iterations. The run completed with
status `ok_resource_adjusted_ppo_training_completed`.

Key local outputs:

- Summary JSON: `res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.json`
- Runtime log: `logs/tracking_g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.log`
- Local ignored run directory: `res/runs/tracking_g1_resource_adjusted_ppo_training/resource_adjusted_ppo_20260618_182241_seed20260619`
- Checkpoints: `rank_0/model_0.pt`, `rank_0/model_50.pt`, `rank_0/model_99.pt`

Metrics: duration `4887.514` seconds, rank 0 total timesteps `2457600`, rank metrics written for ranks 0 and 1,
checkpoint count `3`.

## Verification

The result is integrated into artifact manifest, paper-vs-reproduction comparison, blocked-gate audit, final report, and
master audit. The master audit accepts it only as a resource-adjusted PPO training run, not as official paper-level
tracking training.

Final verification passed:

- `required_artifact_absence_audit`: `ok`, 18 rows; 3 resource-adjusted tracking checkpoints are recorded as
  `present_but_not_required_artifact`.
- `artifact_manifest`: `ok`, 289 artifacts.
- `paper_vs_reproduction`: `ok`, 127 rows.
- `completion_matrix_status_audit`: `ok`, 161 rows, 0 invalid statuses.
- `verification_command_syntax_audit`: `ok`, 179 scripts, 0 failed syntax checks.
- `verification_command_script_manifest`: `ok`, 179 scripts.
- `verification_command_coverage_audit`: `ok`, 187 commands.
- `progress_report_audit`: `ok`, 38 rows.
- `reproduction_master_audit`: `ok`.
- `git diff --check`: passed.

## Failed / Blocked Items

- No official converted G1 replay asset or official `csv_to_npz.py`/`replay_npz.py` motion replay was produced.
- No paper-scale PPO evaluation or trained official teacher claim is made.
- Official G1 USD conversion/replay remains blocked.
- Teacher rollout dataset, DAgger, VAE/diffusion closed-loop evaluation, Fig. 5/Fig. 6 videos, TensorRT deployment, and real robot remain incomplete.

## Effect on English Reading Report

This shows that the reproduction has moved from smoke-level train-entry checks to a completed resource-adjusted PPO
training run through the official task and runner stack. The English report can use this as honest virtual-training
evidence while explicitly separating it from official BeyondMimic paper-level tracking results.

## Next Step

Evaluate the produced `model_99.pt` checkpoint in the same resource-adjusted task, then decide whether to collect
clearly labeled resource-adjusted teacher rollouts.

## Git Commit

Pending before commit; final commit hash is recorded in the user-facing turn report.
