# Progress Update

## Goal

Advance the main tracking-quality repair path before another full PPO run. The specific question was whether the robot-order FK reset/step-0 termination spike is caused by stale or zero motion command targets immediately after environment reset.

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
- `reproduction/scripts/tracking_g1_current_task_env_construction_gate.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.py`
- `reproduction/scripts/robot_order_fk_reset_termination_alignment_audit.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/completion_matrix.md`

## Files Modified

- `reproduction/scripts/robot_order_fk_reset_command_warmup_live_probe.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/progress/20260622_002628_reset_command_warmup_live_probe.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/robot_order_fk_reset_command_warmup_live_probe.py
python3 reproduction/scripts/robot_order_fk_reset_command_warmup_live_probe.py
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

## Results

The new live probe succeeded on the configured IsaacLab tracking path:

```text
status: ok_robot_order_fk_reset_command_warmup_live_probe
device: cuda:4
num_envs: 256
diagnosis: command_warmup_partially_reduces_reset_endpoint_z_spike
```

Key measured values:

```text
after reset before warmup:
  manual endpoint-z done rate: 1.0
  endpoint-z mean error: 0.5298784375190735 m
  body error mean: 15.289739608764648 m
  body_pos_relative_w abs max: 0.0

after manual command_manager.compute():
  manual endpoint-z done rate: 0.2734375
  endpoint-z mean error: 0.10452914237976074 m
  body error mean: 0.15273417532444 m

after one zero-action step following warmup:
  manual endpoint-z done rate: 0.06640625
  actual step done rate: 0.26953125
  endpoint-z mean error: 0.0917641893029213 m
  body error mean: 0.1462995857000351 m
```

Interpretation: command warmup is a real part of the reset/step-0 spike, but not the whole tracking-quality fix. The next mainline step should patch local train/eval wrappers to warm command targets after reset and rerun the full robot-order FK evaluation before launching another full PPO/downstream chain.

## Verification

This progress file was written after the live probe and before the full audit-refresh chain. The expected verification commands for this round are:

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

## Failed / Blocked Items

- The live probe did not fully clear endpoint-z termination. Post-warmup manual endpoint-z done rate was still `0.2734375`, and the actual zero-action step done rate was `0.26953125`.
- This is therefore not sufficient evidence to start another downstream teacher rollout, VAE, denoiser, or guided-control chain.
- No paper-level tracking teacher, DAgger dataset, Fig. 5/Fig. 6 result, TensorRT deployment, or real-robot result is claimed.

## Effect on English Reading Report

This adds a concrete debugging story for the code reproduction section: the project did not merely run scripts, but traced a weak tracking result back through motion target initialization, endpoint-z termination, and IsaacLab step ordering, then validated the hypothesis with a live diagnostic. It is useful evidence for the report's reproducibility-boundary discussion.

## Next Step

Patch the local robot-order FK train/eval wrappers so that command targets are warmed immediately after reset. Then rerun the full robot-order FK tracking eval and compare done rate, endpoint-z error, body error, and reward before deciding whether to launch a stronger PPO run.

## Git Commit

Pending at the time this file was created.
