# Progress Update

## Goal

Strengthen the report/PPT-facing visual evidence trail after the IsaacLab headless gate and local closed-loop rollout work. The aim was not to start another training run, but to make the existing significant full-bundle videos, plots, and tables easier to cite honestly in the English reading report.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json`
- `res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json`
- `res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json`
- `res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json`
- `res/level_c/official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset/level_c_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset.json`
- `res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json`
- `res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json`
- `res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json`
- `reproduction/scripts/visual_evidence_index.py`
- `reproduction/scripts/visual_media_inventory_audit.py`

## Files Modified

- `reproduction/docs/visual_appendix_for_reading_report.md`
- `res/report_assets/visual_evidence_index/visual_evidence_index.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`
- `res/verification_command_coverage/verification_command_coverage_audit.json`
- `reproduction/docs/progress/20260620_125839_visual_report_appendix.md`

## Commands Run

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
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

- Added `reproduction/docs/visual_appendix_for_reading_report.md`, an English report-facing appendix that groups the most useful local videos and plots by reproduction stage.
- Refreshed the visual evidence index:
  - `29` asset JSON files indexed.
  - `15` report-ready MP4 paths indexed.
  - `62` PNG assets indexed.
  - `69` table/README assets indexed.
- Re-ran the visual media inventory:
  - `244` total visual/media rows.
  - `32` local video files.
  - `148` PNG files.
  - `30` SVG files.
  - `30` PDF files.
  - `4` GIF files.
- The recommended report assets now include reference replay, full-bundle policy rollout, VAE closed-loop rollout, receding-horizon latent guidance rollout, and full-bundle task-conditioned multiseed guidance plots.

## Verification

- `visual_evidence_index.py`: `ok`.
- `visual_media_inventory_audit.py`: `ok`.
- `artifact_manifest.py`: `ok`, `768` artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `170` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `186` scripts, `0` failures.
- `verification_command_script_manifest.py`: `ok`, `186` scripts.
- `verification_command_coverage_audit.py`: `ok`, `194` commands, `10` smoke-pass commands.
- `reproduction_master_audit.py`: `ok`.

## Failed / Blocked Items

- This round did not run a new PPO, VAE, diffusion, or IsaacLab rollout experiment.
- Large MP4 videos remain local artifacts and should not be committed to GitHub.
- The local videos are resource-adjusted virtual evidence, not official BeyondMimic Fig. 5/Fig. 6 reproduction.
- Official BeyondMimic VAE/diffusion checkpoints, true DAgger logs, TensorRT paper-hardware deployment, and real Unitree G1 evidence remain absent.

## Effect on English Reading Report

The new appendix gives the reading report a ready-made visual evidence section. It identifies which local assets are useful for explaining the reproduction pipeline, while preserving the exact limitation language needed to avoid overclaiming.

## Next Step

Move from evidence organization back to mainline reproduction: use GPU 4 and GPU 7 for a stronger full-bundle checkpoint rollout or task-conditioned guidance evaluation only if the current GPU guard confirms both devices are free, then update the visual appendix with any genuinely new video/plot assets.

## Git Commit

Pending at the time this progress file was written.
