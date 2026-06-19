# Progress Update

## Goal

Bridge the official-csv-loop offline VAE/guidance action decode result into an IsaacLab tracking rollout probe without claiming paper-level closed-loop guidance.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/level_c/official_csv_loop_guidance_vae_action_decode_eval/level_c_official_csv_loop_guidance_vae_action_decode_eval.json`
- `res/level_c/official_csv_loop_state_latent_guidance_eval/level_c_official_csv_loop_state_latent_guidance_eval.json`
- `res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json`
- `reproduction/scripts/tracking_g1_official_csv_loop_policy_rollout_video_capture.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_guided_action_rollout_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/reproduction_report.md`
- `res/final_report/final_reproduction_report.json`
- `res/final_report/english_reading_report.md`
- `res/comparison/paper_vs_reproduction.csv`
- `res/comparison/paper_vs_reproduction.json`
- `res/comparison/paper_vs_reproduction.md`
- `res/artifact_manifest/artifact_manifest.json`
- `res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json`
- `res/docs/completion_matrix_status_audit/completion_matrix_status_audit.tsv`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`

## Commands Run

```bash
python3 reproduction/scripts/tracking_g1_official_csv_loop_guided_action_rollout_probe.py
python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_guided_action_rollout_probe.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

The new probe executes one 21-step decoded local VAE action sample for `base`, `guided`, and `teacher` variants inside the resource-adjusted official-csv-loop `Tracking-Flat-G1-v0` IsaacLab task.

New result paths:

- `res/level_c/official_csv_loop_guided_action_rollout_probe/tracking_g1_official_csv_loop_guided_action_rollout_probe.json`
- `res/level_c/official_csv_loop_guided_action_rollout_probe/official_csv_loop_guided_action_rollout_probe_assets.json`
- `res/level_c/official_csv_loop_guided_action_rollout_probe/official_csv_loop_guided_action_rollout_probe_timeseries.csv`
- `res/level_c/official_csv_loop_guided_action_rollout_probe/official_csv_loop_guided_action_rollout_probe_metrics.png`
- `res/runs/tracking_g1_official_csv_loop_guided_action_rollout_probe/guided_action_rollout_probe_20260619_055615_seed20260638/official_csv_loop_guided_action_rollout_probe_trace.npz`
- `logs/tracking_g1_official_csv_loop_guided_action_rollout_probe/tracking_g1_official_csv_loop_guided_action_rollout_probe.log`

Key metrics:

- rollout steps: `21`
- selected GPU: `4`
- base/guided max absolute action delta: `0.0`
- base/guided L2 mean: `0.0`
- base teacher MSE: `0.08418039977550507`
- guided teacher MSE: `0.08418039977550507`
- base target-body error mean: `0.07518018782138824`
- guided target-body error mean: `0.07855883985757828`
- teacher target-body error mean: `0.06338490545749664`

This is a useful action-to-sim bridge and a negative result for guided behavior change in this sampled window.

## Verification

All required verification commands passed after integration:

- `artifact_manifest.py`: `ok`, 390 artifacts
- `paper_vs_reproduction_comparison.py`: `ok`, 145 rows
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, 163 rows
- `verification_command_syntax_audit.py`: `ok`, 185 scripts
- `verification_command_script_manifest.py`: `ok`, 185 scripts
- `verification_command_coverage_audit.py`: `ok`, 193 commands
- `reproduction_master_audit.py`: `ok`, 249/249 artifacts passed

## Failed / Blocked Items

The first probe attempt reached the IsaacLab environment and completed the `base` 21-step variant, then failed before `guided` due to a local script bug: `torch.inference_mode()` caused an in-place update error during a later environment reset. The script was fixed by using `torch.no_grad()` for action stepping.

The successful result remains blocked from paper-level interpretation because it uses a resource-adjusted USD/motion path, one short decoded action sample, and no receding-horizon diffusion guidance loop. It is not Fig. 5/Fig. 6 reproduction and not real-robot evidence.

## Effect on English Reading Report

The English reading report now includes this probe as a concrete bridge from offline guidance to IsaacLab action execution. The report also records the negative result that base and guided decoded actions are identical in this sample, so the run supports engineering traceability rather than a claim of guided task success.

## Next Step

Use the same IsaacLab bridge to build a true receding-horizon guidance rollout: generate or select sequential state-latent windows, decode the current latent at each control step, feed actions into `Tracking-Flat-G1-v0`, and compare task costs over a longer rollout. This still must remain virtual/resource-adjusted unless official checkpoints and assets become available.

## Git Commit

Pending.
