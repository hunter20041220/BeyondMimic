# Progress Update

## Goal

Refresh the course-facing English reading report, Chinese reading report, and Chinese project/defense report so they reflect the current BeyondMimic reproduction baseline without overclaiming paper-level success. Also reduce disk pressure by removing debug-only weights that are not scientific reproduction artifacts.

## Files Read

- `prompt06211658.txt`
- `goal.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `reproduction/scripts/update_course_reports.py`
- `reproduction/scripts/cleanup_failed_large_artifacts.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`
- `res/tracking/robot_order_fk_deterministic_reset_live_probe/robot_order_fk_deterministic_reset_live_probe.json`

## Files Modified

- `reproduction/scripts/update_course_reports.py`
- `reproduction/scripts/cleanup_failed_large_artifacts.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.tsv`

## Commands Run

```bash
find res logs tmp cache -xdev -type f -printf '%s\t%p\n' | sort -nr | head -80
du -sh res logs cache tmp envs
jq '{status, artifact_count, missing_count}' res/artifact_manifest/artifact_manifest.json
jq '{status, artifact_count, artifact_pass_count, artifact_fail_count, completion_matrix_counts}' res/master_audit/reproduction_master_audit.json
jq '{status,total_rows,comparison_type_counts}' res/comparison/paper_vs_reproduction.json
python3 -m py_compile reproduction/scripts/cleanup_failed_large_artifacts.py reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/update_course_reports.py
python3 reproduction/scripts/cleanup_failed_large_artifacts.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/update_course_reports.py
```

## Results

- The current audited baseline before regeneration was `artifact_manifest: 1506`, `master_audit: 377/377`, and `paper_vs_reproduction: 229 rows`.
- The English reading report now explicitly frames the project as a public-resource, auditable partial reproduction and local virtual BeyondMimic-like pipeline, not a full paper-level reproduction.
- The Chinese reading report mirrors the same claim boundary and includes the latest deterministic reset tracking gate.
- The Chinese project report now gives a defense-friendly narrative from paper reading, module decomposition, environment recovery, formulas-to-code, tracking quality debugging, downstream VAE/diffusion/guidance, storage policy, and next steps.
- Debug-only VAE/diffusion smoke weights were removed after their JSON/TSV summaries were preserved:
  - `res/level_c/vae_checkpoint_smoke/debug_conditional_vae_checkpoint_smoke.pt`
  - `res/level_c/diffusion_checkpoint_smoke/debug_diffusion_transformer_checkpoint_smoke.pt`
  - `res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/checkpoint/debug_bounded_diffusion_checkpoint.pt`
- The cleanup script recorded `679218570` bytes freed in this pass. It did not remove current tracking checkpoints, teacher rollout shards, state-latent datasets, videos, downloaded raw materials, or environments.

## Verification

The immediate post-cleanup verification passed:

```bash
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/update_course_reports.py
```

Full manifest, comparison, final-report, completion-matrix, verification-command, progress-report, final-deliverables, and master-audit refreshes should be run next in the same round before commit.

## Failed / Blocked Items

- GitHub push from the previous round failed with `invalid credentials`; the next push should retry with a valid GitHub token or existing credential.
- The cleanup is intentionally conservative. Large current teacher rollout shards, state-latent datasets, current robot-order PPO checkpoints, videos, and environments remain local because they still support current evidence or future downstream reruns.
- Tracking remains the main technical blocker: deterministic reset improves joint velocity but worsens done rate, so no full PPO rerun is justified from that patch alone.

## Effect on English Reading Report

This round directly improves the course deliverable. The English report now has a clearer thesis: the project does not fully reproduce BeyondMimic, but it reproduces and audits a large public subset, implements a local virtual pipeline, and identifies the missing paper-level artifacts. The report also now includes the latest tracking-quality conclusion and storage/artifact-management policy.

## Next Step

Refresh all machine-readable audits and commit the report/storage update. After that, return to the tracking mainline: inspect termination/body-target semantics around `ee_body_pos`, endpoint z, initial joint velocity, last-action observation, and reset target consistency before launching another full PPO run.

## Git Commit

Pending.
