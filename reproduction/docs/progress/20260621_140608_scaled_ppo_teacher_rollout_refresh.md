# Progress Update

## Goal

Advance from PPO checkpoint evaluation into the downstream teacher-rollout stage by refreshing the official-importer-export full-bundle scaled-PPO teacher rollout dataset. This is the bridge toward VAE training, state-latent trajectory datasets, and diffusion/guidance experiments.

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
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.py`
- `reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets.py`

## Files Modified

- `res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json`
- `res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/base_compatible_official_importer_export_scaled_training_run_for_teacher_rollout.json`
- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/`
- `res/visual_media_inventory/`
- `res/report_assets/visual_evidence_index/`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260621_140608_scaled_ppo_teacher_rollout_refresh.md`

Large rollout shards and GPU logs were regenerated under `res/runs/` and are intentionally not committed to GitHub.

## Commands Run

```bash
git status --short
git log -5 --oneline
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits -i 4,7
python3 reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

## Results

- Refreshed the scaled-PPO teacher rollout dataset from the local iteration-999 PPO checkpoint.
- Status: `ok_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_completed`.
- Selected GPUs: 4 and 7.
- `num_envs_per_rank=2048`, `total_envs=4096`.
- `shard_count=2`.
- `total_env_steps=1224704`.
- `rollout_steps=299`.
- `motion_count=40`.
- `total_motion_frames=11960`.
- `action_dim=29`.
- `done_count_total=1223466`.
- `timeout_count_total=0`.
- `reward_mean_over_steps=0.02392365585575037`.
- `dataset_npz_total_size_bytes=1919836221`.

Report assets refreshed:

- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/teacher_rollout_reward_done_timeseries.png`
- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/teacher_rollout_motion_step_coverage.png`
- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/teacher_rollout_action_distribution.png`
- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/teacher_rollout_shard_summary.csv`
- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/teacher_rollout_action_summary.csv`

## Verification

Pre-final verification in this round passed:

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

Results:

- teacher rollout report assets: `status=ok`.
- `visual_media_inventory_audit.py`: `status=ok`, `471` visual media rows, `85` local video rows.
- `visual_evidence_index.py`: `status=ok`, `31` report-ready MP4 rows indexed.

The full repository verification chain passed after this file was written:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

Key full-chain results:

- `artifact_manifest.py`: `status=ok`, `1365` artifacts.
- `paper_vs_reproduction_comparison.py`: `status=ok`.
- `final_reproduction_report.py`: `status=ok`.
- `completion_matrix_status_audit.py`: `status=ok`, `199` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `status=ok`, `199` scripts, `0` failures.
- `verification_command_script_manifest.py`: `status=ok`.
- `verification_command_coverage_audit.py`: `status=ok`, `207` commands categorized, `10` smoke commands passed.
- `reproduction_master_audit.py`: `status=ok`.

## Failed / Blocked Items

- No failure occurred in the teacher rollout refresh.
- This is still a local virtual teacher rollout dataset from a local scaled PPO checkpoint, not the official BeyondMimic DAgger rollout dataset.
- It does not claim official paper teacher policy quality, Fig. 5/Fig. 6 closed-loop guidance, TensorRT deployment, or real-robot validation.
- GPU memory per card was below the formal PPO-training 10GB threshold because this was rollout dataset collection, not formal PPO training.

## Effect on English Reading Report

This strengthens the reproduction narrative after PPO evaluation: the project now has current-server evidence that the scaled local PPO checkpoint can generate a two-shard teacher rollout dataset over the 40-motion public bundle, with action distribution, motion coverage, and reward/done plots for the report/PPT.

## Next Step

Use this refreshed teacher rollout dataset to refresh VAE training/evaluation, state-latent trajectory dataset generation, and downstream diffusion/guidance evaluation. Formal PPO training remains a separate step if more training compute is desired.

## Git Commit

Pending at file creation time; see Git history for the commit containing this progress update.
