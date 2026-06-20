# Progress Update

## Goal

Add report/PPT-ready local visual evidence for the official-importer-export scaled PPO checkpoint while preserving the boundary that it is not an official BeyondMimic teacher, not Fig. 5/Fig. 6 guided diffusion, not TensorRT deployment, and not real-robot evidence.

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
- Existing policy/video/report scripts under `reproduction/scripts/`

## Files Modified

- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/progress/20260620_194406_scaled_importer_policy_video.md`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py`
- `reproduction/scripts/visual_media_inventory_audit.py`
- `res/final_report/english_reading_report.md`

## Commands Run

```bash
BM_TERMINATE_WANGJC_GPU_GUARD=1 envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_IMPORTER_SCALED_PPO_POLICY_VIDEO_SEED=20260695 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py
```

Verification commands are recorded after the audit refresh below.

## Results

The rollout capture completed with status `ok_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture`.

New local visualization assets:

- `res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture.json`
- `res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json`
- `res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/official_importer_export_full_bundle_scaled_ppo_policy_rollout_vs_reference.mp4`
- `res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/official_importer_export_full_bundle_scaled_ppo_policy_rollout_keyframes.png`
- `res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/official_importer_export_full_bundle_scaled_ppo_policy_rollout_metrics.csv`

Key metrics from the asset JSON:

- frame count: `299`
- reward mean: `0.024723995476961136`
- done count total: `299`
- action absolute mean: `0.030157173052430153`
- target-body error mean/max: `0.344759464263916` / `0.3872191905975342`

## Verification

Passed after one classification fix in `visual_media_inventory_audit.py`.

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/completion_matrix_status_audit.py reproduction/scripts/verification_command_syntax_audit.py reproduction/scripts/verification_command_script_manifest.py reproduction/scripts/verification_command_coverage_audit.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/visual_media_inventory_audit.py reproduction/scripts/visual_evidence_index.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
envs/bm_analysis/bin/python reproduction/scripts/final_deliverables_audit.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

Final verification status:

- `artifact_manifest`: `ok`, `930` artifacts, missing `0`
- `paper_vs_reproduction`: `ok`, `180` rows
- `required_artifact_absence`: `ok`, `29` rows
- `visual_media_inventory`: `ok`, `307` media rows, `46` videos classified as local/debug/released evidence
- `visual_evidence_index`: `ok`, `21` report-ready local video rows indexed
- `completion_matrix_status_audit`: `ok`, `175` rows, status counts `complete=73`, `partial=98`, `blocked=3`, `out_of_scope=1`
- `reproduction_master_audit`: `ok`, `304/304` artifacts passed

## Failed / Blocked Items

No new failed run was produced by this video capture. The rollout remains a local virtual visualization from a weak local scaled PPO checkpoint, so the same paper-level blockers remain: no official teacher checkpoint, no official DAgger logs, no official VAE/diffusion checkpoints, no Fig. 5/Fig. 6 paper-level videos/metrics, no TensorRT/asynchronous deployment benchmark, and no real robot evidence.

## Effect on English Reading Report

The English report now has a concrete visual artifact for explaining the recovered tracking pipeline. The text explicitly labels it as qualitative engineering evidence rather than a paper-level result.

## Next Step

Refresh artifact manifest, paper-vs-reproduction comparison, final report, completion matrix audit, visual evidence audits, required-artifact absence audit, verification command audits, and master audit; then commit and push the small code/docs/audit artifacts while leaving MP4 and raw run outputs out of GitHub.

## Git Commit

Commit message: `feat: add scaled importer policy video evidence`.

The exact hash is recorded in the user-facing turn report after the final amend/push step.
