# Progress Update

## Goal

Create report/PPT-ready visual evidence for the five-seed full-bundle task-conditioned closed-loop guidance rollouts without committing large MP4 files or overclaiming paper-level reproduction.

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
- `res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json`
- `res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/local_proxy_success_boundary.json`
- `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_matrix.json`

## Files Modified

- `reproduction/scripts/official_csv_loop_full_bundle_guidance_video_contact_sheet.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/final_report/reproduction_report.md`
- `res/final_report/final_reproduction_report.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`
- `res/visual_media_inventory/visual_media_inventory_audit.json`
- `res/visual_media_inventory/visual_media_inventory_audit.tsv`
- `res/report_assets/visual_evidence_index/visual_evidence_index.json`
- `res/verification_command_script_manifest/verification_command_script_manifest.json`
- `res/verification_command_script_manifest/verification_command_script_manifest.tsv`
- `res/verification_command_coverage/verification_command_coverage_audit.json`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_full_bundle_guidance_video_contact_sheet.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

Verification logs:

```text
logs/verification/20260620_guidance_video_contact_sheet_verification.log
logs/verification/20260620_guidance_video_contact_sheet_final_verification.log
```

## Results

New report-facing assets:

```text
res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/full_bundle_guidance_video_index.json
res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/full_bundle_guidance_video_index.csv
res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/full_bundle_guidance_video_contact_sheet.png
res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/README.md
```

Key metrics:

```text
video rows: 20
seed groups: 5
tasks: 4
local MP4 files indexed: 20
total indexed MP4 size: 18,496,702 bytes
contact sheet size: 473,335 bytes
contact sheet resolution: 1224 x 1410
```

The script records each MP4 path, SHA256 hash, task, seed group, reward/error metrics, keyframe PNG, metrics PNG/CSV, GitHub policy, claim level, and limitation. MP4 files remain local and are intentionally not committed to GitHub.

## Verification

The final standard verification chain passed:

```text
artifact_manifest: ok, 779 artifacts
paper_vs_reproduction_comparison: ok, 168 rows
final_reproduction_report: ok
completion_matrix_status_audit: ok, 170 rows, 0 invalid statuses
verification_command_syntax_audit: ok, 186 scripts, 0 failed
verification_command_script_manifest: ok, 186 scripts
verification_command_coverage_audit: ok, 194 commands, 10 smoke commands passing
reproduction_master_audit: ok, 286/286 artifacts passing
visual_media_inventory: ok, 270 rows, 40 videos, 166 PNGs
```

## Failed / Blocked Items

No new verification failures were introduced.

Still not paper-level complete:

- No official BeyondMimic VAE/diffusion checkpoint.
- No official Fig. 5/Fig. 6 success/fall/collision evaluation protocol result.
- No true official DAgger rollout logs.
- No TensorRT/asynchronous deployment benchmark matching the paper hardware path.
- No real Unitree G1 robot result.

The new visual asset is local virtual/resource-adjusted evidence only. It must not be described as official BeyondMimic Fig. 5/Fig. 6 video reproduction.

## Effect on English Reading Report

The English reading report now has a concrete visual-evidence paragraph and a compact contact sheet path that can be used in the course report/PPT. This improves the reproduction section by showing robot-motion evidence while preserving the honest claim boundary.

## Next Step

Use the contact sheet in the English report/PPT visual appendix, then either add a concise task/result table around it or return to the official G1 conversion/replay path to reduce the remaining gap to official paper-level replay.

## Git Commit

Included in this round's Git commit; use the repository history for the final commit hash.
