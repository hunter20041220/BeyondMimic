# Progress Update

## Goal

Move the Level B tracking reproduction from official-importer-export task diagnostics to a real local PPO training and checkpoint-evaluation run using the same official-importer-export G1 USDA asset.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- Existing PPO harnesses and prior full-bundle tracking outputs under `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/` and `/mnt/infini-data/test/BeyondMimic/res/tracking/`.

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_training_run.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.py
BM_OFFICIAL_IMPORTER_EXPORT_PPO_MAX_ITERATIONS=1 BM_PPO_NUM_ENVS_PER_RANK=128 CUDA_VISIBLE_DEVICES=4,7 envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_training_run.py
BM_OFFICIAL_IMPORTER_EXPORT_PPO_MAX_ITERATIONS=300 BM_PPO_NUM_ENVS_PER_RANK=512 CUDA_VISIBLE_DEVICES=4,7 envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_training_run.py
CUDA_VISIBLE_DEVICES=4,7 envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/final_deliverables_audit.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

## Results

The 1-iteration gate passed, so the run was expanded directly to a full 300-iteration PPO training job.

Training result:

- Status: `ok_official_importer_export_full_bundle_ppo_training_completed`
- GPUs: physical GPU 4 and GPU 7
- World size: `2`
- Environments: `512` per rank, `1024` total
- PPO iterations: `300`
- Seed: `20260680`
- Duration: `519.372` seconds
- Checkpoints: `7`
- Rank-0 learning iteration: `299`
- Rank-0 timesteps: `7372800`
- Robot asset: `res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda`
- Motion input: `res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle.npz`

Checkpoint evaluation result:

- Status: `ok_official_importer_export_full_bundle_ppo_checkpoint_eval_completed`
- Eval environments: `512`
- Eval steps: `299`
- Total env steps: `153088`
- Loaded iteration: `299`
- Reward mean: `0.02351330920281418`
- Done count total: `152841`
- Timeout count total: `0`
- Anchor-position error mean: `0.05962799150608854`
- Body-position error mean: `0.6082496278262058`
- Joint-position error mean: `0.9147374291085081`

Report assets:

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/training_curve.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/tracking_error_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/reward_done_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/gpu_usage_eval.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/ppo_checkpoint_eval_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/ppo_checkpoint_eval_gpu_summary.csv`

## Verification

The first verification pass after integrating this result passed:

- `paper_vs_reproduction_comparison.py`: passed; 171 rows.
- `artifact_manifest.py`: passed; 813 artifacts before this progress file was added.
- `final_deliverables_audit.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed.
- `verification_command_syntax_audit.py`: passed.
- `verification_command_script_manifest.py`: passed.
- `verification_command_coverage_audit.py`: passed.
- `reproduction_master_audit.py`: passed; 293/293 master artifacts passed.

## Failed / Blocked Items

No new training or evaluation failure was introduced in this round.

Remaining limitations:

- The result is local virtual evidence, not the official BeyondMimic teacher checkpoint.
- The motion bundle has artificial clip boundaries.
- The training budget is 300 PPO iterations, not paper-scale teacher training.
- The high done count means the checkpoint should not be interpreted as a mature paper-level tracking teacher.
- DAgger rollout logs, official VAE/diffusion checkpoints, Fig.5/Fig.6 closed-loop videos/metrics, TensorRT deployment, and real robot results remain incomplete.
- Per-card memory stayed below 10GB because the official 512-env/rank H20 run fit in roughly 5GB/card; this was recorded rather than inflated.

## Effect on English Reading Report

This round materially strengthens the reproduction section: the project can now report an end-to-end local tracking PPO path using the official-importer-export G1 asset, not only an enriched scaffold. It provides concrete training and evaluation curves for the report/PPT while preserving the central conclusion that the project does not fully reproduce BeyondMimic at paper level.

## Next Step

The next mainline step is to use this more official asset-path checkpoint to collect a teacher rollout dataset, then repeat the VAE/state-latent/diffusion/guidance chain or generate a policy rollout video for presentation.

## Git Commit

Pending at the time this progress file was written.
