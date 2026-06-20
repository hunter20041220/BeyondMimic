# Progress Update

## Goal

Add a paper-facing but conservative virtual diagnostic for the BeyondMimic Fig. 6A motion-inpainting/keyframe family on the recovered official-importer-export G1 IsaacLab chain. The goal was to create auditable closed-loop evidence without claiming paper-level Fig. 6A reproduction.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/completion_matrix.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/fig5_fig6_proxy_protocol_matrix.json`
- `reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.py`
- `reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Generated audit/report artifacts under `res/artifact_manifest/`, `res/comparison/`, `res/docs/completion_matrix_status_audit/`, `res/final_report/`, `res/master_audit/`, `res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/`, `res/required_artifact_absence/`, and `res/verification_command_*`.

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.py
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

Final verification log:

```text
logs/verification/20260621_inpainting_guidance_verification_rerun.log
```

## Results

- Added optional `inpainting` support to the local task-conditioned guidance runner via a synthetic future-keyframe/root-path cost.
- Added `tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.py`, which launches the existing importer-export task-conditioned runner in a fresh process with `BM_TASK_CONDITIONED_TASKS=inpainting`.
- Ran one 299-step local IsaacLab closed-loop inpainting diagnostic on physical GPU 4.
- Result status: `ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval`.
- Key metric: guided keyframe error mean `0.3349786878951531`; denoised keyframe error mean `0.24249927912314906`; guided minus denoised delta `0.09247940877200406`.
- Interpretation: this is a negative/diagnostic proxy. The rollout completes and records assets, but the guided variant is worse than the denoised baseline under the local keyframe proxy metric.
- Updated Fig. 5/Fig. 6 proxy matrix: `6` panels mapped, `4` panels with local importer-export closed-loop proxy support, `16` referenced rollout/video rows, and `0` paper-level reproduced panels.

## Verification

Final verification statuses:

- `required_artifact_absence_audit.py`: `ok`, 29 rows.
- `paper_vs_reproduction_comparison.py`: `ok`, 188 rows.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, 179 rows.
- `verification_command_syntax_audit.py`: `ok`, 188 scripts, 0 failed.
- `verification_command_script_manifest.py`: `ok`, 188 scripts.
- `verification_command_coverage_audit.py`: `ok`, 196 commands.
- `artifact_manifest.py`: `ok`, 974 artifacts.
- `reproduction_master_audit.py`: `ok`, 311/311 artifacts passed.

## Failed / Blocked Items

- First final verification attempt failed at `reproduction_master_audit.py` because the new local inpainting MP4 was not yet classified by `required_artifact_absence_audit.py`; it was treated as a potentially unclassified local paper-level video.
- Fixed by classifying `res/visualization/official_importer_export_full_bundle_inpainting_guidance_rollout` as local proxy/reference visualization evidence, not a paper-level video.
- No official BeyondMimic VAE/diffusion checkpoint was added.
- No paper Fig. 6A cartwheel/keyframe protocol was reproduced.
- No TensorRT deployment or real robot result was produced.

## Effect on English Reading Report

The English report can now cite a concrete Fig. 6A-adjacent virtual diagnostic instead of only offline/debug inpainting evidence. The report explicitly frames it as negative diagnostic evidence: the closed-loop path works, but the local guidance objective needs improvement because the guided keyframe proxy error is worse than the denoised baseline on this seed.

## Next Step

The best next technical step is a multi-seed paper-style keyframe/inpainting protocol: define explicit keyframes, fall/success thresholds, transition smoothness metrics, and guidance-scale selection before attempting to describe Fig. 6A as more than a local diagnostic proxy.

## Git Commit

Planned commit message: `feat: add importer inpainting guidance proxy`.

Commit hash: to be recorded by Git after this progress file is committed.
