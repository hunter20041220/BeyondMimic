# Progress Update

## Goal

Add a stricter, auditable local success/fall/collision proxy over the existing scaled-PPO official-importer-export Fig. 5/Fig. 6-adjacent closed-loop traces, without claiming official paper-level success/fall/collision results.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/progress_report_audit.py
```

Full verification log:

```text
/mnt/infini-data/test/BeyondMimic/logs/verification_scaled_ppo_success_fall_collision_proxy_20260621_081629.log
```

## Results

New asset directory:

```text
/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/
```

Key metrics:

```text
row_count: 20
task_count: 4
seed_group_count: 5
trace_npz_count: 20
mp4_count: 20
overall_success_proxy_rate: 0.9
overall_fall_height_proxy_rate: 0.1
overall_body_error_spike_anomaly_proxy_rate: 0.05
overall_completed_299_rate: 1.0
overall_guidance_signal_positive_rate: 1.0
paper_level_success_rate_available: false
paper_level_fall_rate_available: false
paper_level_collision_rate_available: false
collision_contact_signal_available: false
```

Generated files:

```text
success_fall_collision_proxy.json
success_fall_collision_proxy_rows.csv
success_fall_collision_proxy_aggregate.csv
success_fall_collision_proxy.md
success_fall_collision_proxy_rates.png
README.md
```

## Verification

All required verification commands passed.

```text
artifact_manifest: ok, artifacts=1282
paper_vs_reproduction_comparison: ok
final_reproduction_report: ok
completion_matrix_status_audit: ok
verification_command_syntax_audit: ok
verification_command_script_manifest: ok
verification_command_coverage_audit: ok
required_artifact_absence_audit: ok
progress_report_audit: ok
reproduction_master_audit: ok
```

## Failed / Blocked Items

- No official BeyondMimic Fig. 5/Fig. 6 success/fall/collision thresholds are public in the available artifact set.
- The saved local traces do not contain contact/collision labels, so collision remains explicitly unavailable as a true contact metric.
- This proxy does not use official BeyondMimic VAE/diffusion checkpoints.
- This proxy is not TensorRT deployment evidence, mocap evidence, or real-robot evidence.
- Current project state must still remain `goal_complete=false`.

## Effect on English Reading Report

This adds a stronger and more honest evidence paragraph for the English reading report. The report can now discuss local success-like, fall-like, and anomaly-like behavior over 20 scaled-PPO closed-loop traces while explicitly stating that these are local proxies rather than official paper-level success/fall/collision results.

## Next Step

Continue toward a real virtual paper-facing gate by either improving the IsaacLab live headless gate or running a protocol-aligned official-importer-export tracking replay/eval when the simulator gate is stable.

## Git Commit

Pending at report creation time.
