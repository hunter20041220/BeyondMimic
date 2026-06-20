# Progress Update

## Goal

Extend the strongest current official-importer-export closed-loop guidance bridge from a single task-conditioned seed group into a small multi-seed local virtual audit, then wire the result into the comparison table, report assets, final report, and master audit without overstating it as paper-level BeyondMimic reproduction.

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
- New result summaries under `res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/`
- New report assets under `res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/`

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py`
- `reproduction/scripts/official_importer_export_full_bundle_task_conditioned_guidance_multiseed_report_assets.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/reproduction_report.md`
- `res/final_report/english_reading_report.md`
- Refreshed audit outputs under `res/artifact_manifest/`, `res/comparison/`, `res/final_report/`, `res/master_audit/`, `res/required_artifact_absence/`, `res/verification_command_coverage/`, and `res/verification_command_script_manifest/`.

## Commands Run

```bash
CUDA_VISIBLE_DEVICES=4 envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_task_conditioned_guidance_multiseed_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

## Results

- Added a three-seed-group, four-task local virtual audit for official-importer-export task-conditioned latent guidance.
- Produced 12 task rows: joystick, waypoint, obstacle avoidance, and composed objectives across three seed groups.
- All rows completed 299 local IsaacLab rollout steps and all rows retained MP4 paths.
- Aggregate guided reward means: joystick `0.02282794576253505`, waypoint `0.022316898471585484`, obstacle avoidance `0.023011198332232145`, composed `0.02340046236257265`.
- Added report assets: aggregate CSV, bar chart, seed scatter chart, README, and JSON asset summary.

## Verification

After classifying the new local MP4s as local virtual reference/report evidence rather than required paper-level Fig. 5/Fig. 6 videos, the required verification chain passed:

- `artifact_manifest.py`: ok, 943 artifacts.
- `paper_vs_reproduction_comparison.py`: ok, 184 rows.
- `final_reproduction_report.py`: ok.
- `completion_matrix_status_audit.py`: ok, 175 rows, 0 invalid.
- `verification_command_syntax_audit.py`: ok, 186 scripts.
- `verification_command_script_manifest.py`: ok, 186 scripts.
- `verification_command_coverage_audit.py`: ok, 194 commands, 10 smoke-pass commands.
- `reproduction_master_audit.py`: ok, 306/306 master artifacts passed.

## Failed / Blocked Items

- The first master-audit rerun failed because the required-artifact absence audit discovered eight new local MP4 files under the official-importer-export multi-seed visualization directory and had not yet classified them. I fixed the audit classification so these videos are treated as local virtual/report artifacts, not paper-level required Fig. 5/Fig. 6 evidence.
- No official BeyondMimic VAE or diffusion checkpoint is available.
- No official Fig. 5/Fig. 6 success/fall/collision protocol or video has been reproduced.
- No TensorRT/asynchronous robot deployment evidence was produced in this round.
- No real Unitree G1 hardware was used.

## Effect on English Reading Report

The English reading report now has a stronger, less anecdotal code-reproduction section for the official-importer-export path: it can cite a three-seed-group local virtual guidance audit instead of only a single task-conditioned rollout. The text explicitly says this is local proxy evidence, not full paper-level reproduction.

## Next Step

The next useful paper-facing step is to convert these official-importer-export local guidance rollouts into a compact success-boundary/contact-sheet asset, then decide whether to spend GPU time on a longer official-importer-export teacher/VAE/diffusion run or return to the unresolved live IsaacLab/Kit gate.

## Git Commit

Planned commit message: `feat: add importer guidance multiseed evidence`. The final commit hash and push status are reported in the user-facing turn summary.
