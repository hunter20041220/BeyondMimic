# Progress Update

## Goal

Quantify the remaining robot-order FK tracking bottleneck after no-advance reset-target refresh, before starting another full PPO/downstream chain.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/docs/current_project_reproduction_state_20260622.md`
- `reproduction/docs/completion_matrix.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Baseline, seed-matched warmup, and no-advance target-refresh full eval JSON/CSV traces under `res/tracking/` and `res/runs/`.

## Files Modified

- `reproduction/scripts/robot_order_fk_reset_state_action_distribution_diagnostic.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/update_course_reports.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/current_project_reproduction_state_20260622.md`
- `reproduction/docs/progress/20260622_024557_reset_state_action_distribution.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/robot_order_fk_reset_state_action_distribution_diagnostic.py
python3 reproduction/scripts/robot_order_fk_reset_state_action_distribution_diagnostic.py
```

## Results

The diagnostic status is `ok_robot_order_fk_reset_state_action_distribution_diagnostic`.

It compares the same-seed baseline, reset-command warmup, and no-advance target-refresh full eval traces (`2048` envs x `299` steps each). The no-advance target refresh reduces step-0 body-position error by `-43.02953788638115` m, but step-0 joint-velocity error increases by `17.829124450683594`, first-five-step action mean increases by `0.07184725403785702`, post-step0 done-rate delta is `+0.047659854760906034`, and `ee_body_pos` termination-fraction delta is `+0.0478825904055184`.

## Verification

Full repository verification is run after report/audit refresh in the same round.

## Failed / Blocked Items

This is a diagnostic, not a teacher-quality fix. The current PPO checkpoint remains too weak for final DAgger/VAE/diffusion use. The next full PPO run should wait until reset-state, last-action observation, initial-velocity, endpoint-threshold, and `ee_body_pos` termination consistency are repaired.

## Effect on English Reading Report

The reading report can now explain a concrete reproduction difficulty rather than just saying tracking is weak: fixing stale reset targets improves the visible step-0 body target, but exposes a reset velocity/action distribution mismatch that worsens post-step0 termination. This supports an honest local-virtual-pipeline narrative and avoids claiming paper-level tracking reproduction.

## Next Step

Patch reset-state/action consistency in the tracking eval/train wrapper, then rerun the full robot-order FK task eval. If post-step0 done rate improves, proceed to full PPO on GPUs 4/7 and then redo teacher rollout, VAE, state-latent denoiser, and guidance rollout.

## Git Commit

Pending at the time this progress file is written.
