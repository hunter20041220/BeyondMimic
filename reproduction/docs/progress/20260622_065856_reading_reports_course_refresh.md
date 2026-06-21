# Progress Update

## Goal

Refresh the course-facing English reading report, Chinese reading report, and Chinese project/defense report so they explain the BeyondMimic reproduction as a public-resource partial reproduction with a local virtual pipeline, not as a full paper-level result.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.json`

## Files Modified

- `reproduction/scripts/update_course_reports.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/final_report/final_reproduction_report.json`
- `res/final_report/reproduction_report.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`

## Commands Run

- `git status --short`
- `git log -1 --oneline`
- `find`, `sed`, `rg`, `jq`, and small Python JSON inspections over reports and audits
- `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader`
- `du -sh download other reproduction res logs envs cache tmp`
- `python3 -m py_compile reproduction/scripts/update_course_reports.py reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

- Added a module-to-evidence table to the English and Chinese reading reports, mapping tracking, DAgger, VAE, state-latent dataset, diffusion, guidance, and deployment to local implementation/audit evidence.
- Added an evidence ladder that separates exact public evidence, resource-adjusted local virtual evidence, qualitative proxy evidence, non-public missing artifacts, and hardware-only items.
- Strengthened the Chinese project/defense report with a clearer step-by-step narrative: paper reading, inventory, environment recovery, released-data/source audit, tracking data repair, PPO/eval, downstream VAE/diffusion/guidance, unified task protocol, and failure-boundary management.
- Replaced long storage-path rows in course reports with artifact-role labels, reducing path clutter while keeping the storage-management story.
- Registered this progress update in the artifact manifest and master audit.

## Verification

Verification is run after regenerating the reports and audits. Passing verification means the refreshed reports remain synchronized with the generated final report, artifact manifest, comparison table, completion matrix, verification-command audits, and master audit.

## Failed / Blocked Items

- No new PPO, teacher rollout, VAE, diffusion, or guidance training was launched in this report-focused round.
- The main simulation-side paper-level blockers remain: tracking teacher quality, wrist endpoint / `ee_body_pos` termination semantics, true DAgger rollout data, official VAE/diffusion checkpoints, strict Fig. 5/Fig. 6 metrics/videos, TensorRT/asynchronous deployment, and real robot validation.

## Effect on English Reading Report

The English report now reads more like a course reading report and less like a raw engineering log. It explains the paper idea, method decomposition, formula-to-code mapping, reproduction evidence levels, current partial results, and the scientific reason the project still cannot claim full BeyondMimic reproduction.

## Next Step

Return to the tracking mainline: run a wrist-endpoint target/body/FK/termination probe, and only after a gate improves done rate and joint/action transients should the project move to a full GPU 4/7 PPO rerun and downstream teacher rollout -> VAE -> state-latent -> denoiser -> guidance refresh.

## Git Commit

Pending.
