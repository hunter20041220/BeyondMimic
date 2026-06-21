# Progress Update

## Goal

Convert the strongest current official-importer-export scaled PPO closed-loop guidance evidence into a stricter Fig. 5/Fig. 6 task-protocol proxy table and report-ready plots, while preserving the boundary that this is not official paper-level Fig. 5/Fig. 6 success/fall/collision evidence.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/fig5_fig6_task_protocol_proxy.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

## Results

New report assets:

```text
res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/
```

Key metrics:

- Status: `ok_official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy`
- Rows: `20`
- Seed groups: `5`
- Tasks: `4`
- Trace NPZ count: `20`
- Local MP4 path count: `20`
- Recorded 299-step completion rate: `1.0`
- Positive guidance signal rate: `1.0`
- Endpoint/root-reference proxy pass rate: `1.0`
- Target-body mean proxy pass rate: `1.0`
- Local task-protocol proxy pass rate: `0.8`
- Reward-improved-vs-denoised rate: `0.6`
- Tracking-error-not-worse-vs-denoised rate: `0.5`
- Mean final root XY error: `0.0061050763586953626` m
- Paper-level reproduced panel count: `0`

This is an analysis/report-assets step over existing closed-loop traces, not a new training run or formal GPU experiment.

## Verification

The quick regeneration chain passed:

- `official_importer_export_fig5_fig6_task_protocol_proxy.py`: `ok`
- `official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py`: `ok`
- `paper_vs_reproduction_comparison.py`: `ok`
- `final_reproduction_report.py`: `ok`
- `reproduction_master_audit.py`: `ok`

Full required verification is run after this progress note is written.

## Failed / Blocked Items

No command failed in this round. The paper-level gap remains: these are local proxy thresholds over local virtual tasks, not the official BeyondMimic Fig. 5/Fig. 6 task protocol, not official VAE/diffusion checkpoints, not TensorRT or CppAD deployment, not mocap/real-world context, and not real-robot evidence.

## Effect on English Reading Report

This adds a clear comparison point for the report: the scaled PPO downstream chain improves the local task-protocol proxy pass rate from the earlier `0.65` to `0.8`, while still showing mixed guided-vs-denoised behavior. It gives the report and PPT a stronger, visualizable simulation result without overclaiming paper-level reproduction.

## Next Step

Use these scaled-PPO Fig. 5/Fig. 6 proxy assets in the reading report, then continue toward more paper-facing closed-loop validation: either stricter task success/fall/collision proxy definitions or a live IsaacLab deployed-control gate that directly consumes the local VAE/denoiser/guidance loop.

## Git Commit

Pending at the time this progress note is written.
