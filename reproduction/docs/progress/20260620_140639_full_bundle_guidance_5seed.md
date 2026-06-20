# Progress Update

## Goal

Extend the local virtual official-CSV-loop full-bundle task-conditioned latent-guidance rollout evaluation from the previous three-seed evidence set to five seed groups, refresh report-facing assets, and keep the audit chain compatible with larger multiseed runs.

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

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/*`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/*`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/guided_vs_unguided_closed_loop_matrix/*`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/visual_evidence_index/visual_evidence_index.json`
- `/mnt/infini-data/test/BeyondMimic/res/visual_media_inventory/*`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/*`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/*`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/*`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_*/*`

## Commands Run

```bash
find res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rollout/seed_group_4 -maxdepth 2 -type f -name '*.mp4' -printf '%s %p\n' | sort -n
nvidia-smi -i 4,7 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
tail -60 logs/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/seed_group_4/composed/tracking_g1_official_csv_loop_receding_latent_guidance_rollout_eval.log
BM_FULL_BUNDLE_REUSE_EXISTING_SEED_GROUPS=1 BM_FULL_BUNDLE_EXTRA_SEED_GROUPS_JSON='{"seed_group_3":{"joystick":20260721,"waypoint":20260722,"obstacle_avoidance":20260723,"composed":20260724},"seed_group_4":{"joystick":20260731,"waypoint":20260732,"obstacle_avoidance":20260733,"composed":20260734}}' /mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_report_assets.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/guided_vs_unguided_closed_loop_matrix.py
```

Standard verification was run multiple times after the audit compatibility fix:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

## Results

- The full-bundle task-conditioned latent-guidance multiseed evaluation now records `20` rows: `5` seed groups x `4` tasks.
- All `20` rows completed `299` rollout steps for all four variants, for `23920` total rollout-variant steps.
- The aggregate checks record `all_rows_ok=true`, `all_rollouts_299_steps=true`, `all_rows_have_mp4_paths=true`, `uses_full_public_motion_bundle=true`, and `seed_group_count_at_least_3=true`.
- The guided-vs-unguided closed-loop report matrix now has `43` rows, including `20` full-bundle task-conditioned multiseed rows.
- The visual media inventory now records `268` media files: `40` videos, `164` PNG files, `30` PDF files, `30` SVG files, and `4` GIF files.
- Large MP4 rollout videos remain local evidence under `res/visualization/...` and are not intended for GitHub commit.

## Verification

Final verification passed with JSON `status=ok` for:

- `res/artifact_manifest/artifact_manifest.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/final_report/final_reproduction_report.json`
- `res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json`
- `res/verification_command_syntax/verification_command_syntax_audit.json`
- `res/verification_command_script_manifest/verification_command_script_manifest.json`
- `res/verification_command_coverage/verification_command_coverage_audit.json`
- `res/master_audit/reproduction_master_audit.json`

Final master audit: `284/284` artifacts passed, `0` failed.

## Failed / Blocked Items

- An intermediate `jq` inspection command failed because it assumed an older list-shaped comparison schema. This was a read-only inspection failure and did not modify results.
- The first standard verification run failed in `reproduction_master_audit.py` because the audit expected a fixed `three_seed_groups` key and exactly `12` full-bundle multiseed rows. The audit was corrected to accept `seed_group_count_at_least_3` and derive expected rollout steps from the actual row count.
- After refreshing the guided-vs-unguided matrix, master audit failed once more because it still expected exactly `12` full-bundle multiseed matrix rows. The audit was corrected to require at least the previous evidence volume while accepting larger seed counts.
- This is a local virtual closed-loop evaluation using the resource-adjusted official-CSV-loop chain. It is not official BeyondMimic Fig. 5/Fig. 6 reproduction, not an official VAE/diffusion checkpoint, and not real-robot evidence.

## Effect on English Reading Report

This update strengthens the code-reproduction section with a larger multiseed, task-conditioned, closed-loop virtual guidance evidence set. It supports statements about local reproduction effort, seed sensitivity, task-conditioned latent guidance behavior, and report/PPT visual material. It must still be described as local virtual evidence below paper-level official reproduction.

## Next Step

Use the five-seed matrix to write a concise English report subsection and table. The next technical step should be a paper-facing closed-loop diffusion guidance evaluation that compares task-conditioned guided variants against teacher/VAE/denoised baselines with clearer task metrics, while preserving the boundary that official Fig. 5/Fig. 6 and real-robot results are still not reproduced.

## Git Commit

Pending at the time this progress file was written.
