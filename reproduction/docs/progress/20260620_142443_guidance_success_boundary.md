# Progress Update

## Goal

Add a report-facing, auditable local proxy success-boundary summary for the five-seed full-bundle task-conditioned closed-loop guidance rollouts. The goal is to make the English reading report/PPT evidence clearer without overclaiming it as official BeyondMimic Fig. 5/Fig. 6 or real-robot reproduction.

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
- `res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json`
- `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_matrix.json`

## Files Modified

- `reproduction/scripts/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/final_report/reproduction_report.md`
- `res/final_report/final_reproduction_report.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`
- `res/verification_command_coverage/verification_command_coverage_audit.json`
- `res/verification_command_script_manifest/verification_command_script_manifest.json`
- `res/verification_command_script_manifest/verification_command_script_manifest.tsv`
- `res/visual_media_inventory/visual_media_inventory_audit.json`
- `res/visual_media_inventory/visual_media_inventory_audit.tsv`
- `res/report_assets/visual_evidence_index/visual_evidence_index.json`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary.py
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

The final verification chain was captured in:

```text
logs/verification/20260620_guidance_success_boundary_final_verification.log
```

## Results

New local proxy success-boundary assets were written to:

```text
res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/
```

Key metrics:

```text
rows: 20
seed groups: 5
tasks: 4
overall completion rate at 299 steps: 1.0
overall positive guidance-signal rate: 1.0
overall action-changed rate: 1.0
overall local proxy pass rate: 0.9
overall reward improved vs. denoised rate: 0.5
overall tracking error not worse vs. denoised rate: 0.8
```

The new asset includes JSON, row CSV, aggregate CSV, README, and one compact PNG rate plot. Large MP4 rollout videos remain local and are not committed to GitHub.

## Verification

The standard verification chain passed:

```text
artifact_manifest: ok, 774 artifacts
paper_vs_reproduction_comparison: ok
final_reproduction_report: ok
completion_matrix_status_audit: ok, 170 rows, 0 invalid statuses
verification_command_syntax_audit: ok, 186 scripts, 0 failed
verification_command_script_manifest: ok, 186 scripts
verification_command_coverage_audit: ok, 194 commands, 10 smoke commands passing
reproduction_master_audit: ok, 285/285 artifacts passing
```

## Failed / Blocked Items

No new verification failures were introduced.

Still blocked or explicitly non-paper-level:

- No official BeyondMimic VAE/diffusion checkpoint.
- No official Fig. 5/Fig. 6 success/fall/collision rollout protocol result.
- No true official DAgger rollout logs.
- No TensorRT/asynchronous deployment benchmark matching the paper hardware path.
- No real Unitree G1 robot result.

The new success-boundary audit is a local virtual proxy over resource-adjusted IsaacLab rollouts and local checkpoints. It must not be described as official paper-level reproduction.

## Effect on English Reading Report

The English reading report now has a clearer code reproduction evidence block for the five-seed full-bundle task-conditioned guidance experiment. It can cite a compact success-boundary table and plot while explicitly stating that the result is local proxy evidence, not official BeyondMimic Fig. 5/Fig. 6 or real-robot evidence.

## Next Step

Use this report-facing evidence in the English reading report/PPT, then continue either:

1. strengthening the local virtual pipeline with a paper-facing task table and selected still frames, or
2. returning to the official-blocker side by trying a more faithful G1 asset/conversion path before claiming official replay.

## Git Commit

Included in this round's Git commit; use the repository history for the final commit hash.
