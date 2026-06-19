# Progress Update

## Goal

Extend the local full-bundle closed-loop guidance evidence from a single composed-cost receding-horizon rollout to four task-conditioned proxy rollouts, then integrate the new evidence into the comparison table, report assets, absence audit, final report, English reading report, artifact manifest, and master audit.

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
- `res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/official_csv_loop_task_conditioned_guidance_report_assets.py`
- `reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Generated audit/report outputs under `res/artifact_manifest`, `res/comparison`, `res/final_report`, `res/master_audit`, `res/report_assets`, `res/required_artifact_absence`, and verification-audit directories.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py reproduction/scripts/official_csv_loop_task_conditioned_guidance_report_assets.py reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.py
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_task_conditioned_guidance_report_assets.py
BM_TASK_CONDITIONED_REPORT_VARIANT=full_bundle /mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_task_conditioned_guidance_report_assets.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py
python3 reproduction/scripts/visual_evidence_index.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/final_deliverables_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Added four full-bundle local virtual task-conditioned receding-horizon latent-guidance rollouts: joystick, waypoint, obstacle_avoidance, and composed.
- Each task ran 299 IsaacLab control steps and produced JSON/TSV metrics, MP4 path evidence, keyframes, metrics plots, and metrics CSVs.
- Guided reward means: joystick `0.022738767414039195`, waypoint `0.021866608513861796`, obstacle_avoidance `0.02571137477160995`, composed `0.021387746856613304`.
- Guided target-body error means: joystick `0.08253771811723709`, waypoint `0.0809725970029831`, obstacle_avoidance `0.08061826974153519`, composed `0.08186342567205429`.
- `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_matrix.json` now records 23 rows, 8 aggregate rows, 4 full-bundle task-conditioned rows, and 23 video-linked rows.
- `res/comparison/paper_vs_reproduction.json` now records 166 rows; the new full-bundle task-conditioned entry is `qualitative_only`.
- `res/artifact_manifest/artifact_manifest.json` now records 691 artifacts.
- `res/master_audit/reproduction_master_audit.json` is `ok` with 280 checked artifacts.

## Verification

All required verification commands passed in `logs/full_bundle_task_conditioned_guidance_verification_20260620.log`.

Key statuses:

- `artifact_manifest.py`: `ok`, 691 artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`, 166 rows.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, 170 rows.
- `verification_command_syntax_audit.py`: `ok`, 186 scripts, 0 failed.
- `verification_command_script_manifest.py`: `ok`, 186 scripts.
- `verification_command_coverage_audit.py`: `ok`, 194 commands, 10/10 smoke pass.
- `required_artifact_absence_audit.py`: `ok`, 26 rows.
- `final_deliverables_audit.py`: `ok`, 38 rows.
- `reproduction_master_audit.py`: `ok`.

## Failed / Blocked Items

- Running the report-asset scripts with system `python3` failed because that interpreter lacks `matplotlib`. The correct environment is `envs/bm_analysis/bin/python`, and rerunning there passed.
- The new rollouts are local virtual/resource-adjusted evidence, not official BeyondMimic checkpoints, not unpatched official replay, not paper Fig. 5/Fig. 6 success-rate reproduction, not TensorRT deployment, and not real-robot validation.
- Current paper-level gaps remain: official VAE/diffusion checkpoints, true official DAgger rollout logs, official closed-loop Fig. 5/Fig. 6 task videos/metrics, TensorRT deployment, and real Unitree G1 results.

## Effect on English Reading Report

The English reading report now has a stronger code-reproduction section for simulation-side guided control. It can discuss task-conditioned full-bundle closed-loop evidence across four proxy tasks while explicitly preserving the boundary that this project does not fully reproduce BeyondMimic at paper level.

## Next Step

The next technical step is to push from local virtual guidance evidence toward either a stronger official replay/conversion gate or a more formal multi-seed/full-bundle PPO and closed-loop evaluation path, while keeping all claims separate from official paper-level results.

## Git Commit

Pending at the time this progress file was written.
