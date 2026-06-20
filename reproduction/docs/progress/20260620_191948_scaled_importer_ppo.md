# Progress Update

## Goal

Advance the tracking-side reproduction beyond the earlier 300-iteration official-importer-export PPO run by adding a larger local two-GPU PPO training/evaluation pass, while preserving the boundary between local virtual evidence and official BeyondMimic paper-level results.

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

- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`

## Commands Run

```bash
BM_TERMINATE_WANGJC_GPU_GUARD=1 envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MAX_ITERATIONS=1000 BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_NUM_ENVS_PER_RANK=2048 BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_SEED=20260693 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_NUM_ENVS=2048 BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_SEED=20260694 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_eval_report_assets.py
```

## Results

- GPU guard found no matching `wangjc` processes to terminate.
- Training completed on GPUs 4 and 7 with `4096` total environments, `1000` PPO iterations, and `21` local checkpoints.
- Rank0 reached iteration `999` and `98304000` timesteps.
- Checkpoint evaluation loaded `model_999.pt` and ran `2048` environments x `299` steps, totaling `612352` env steps.
- Eval metrics: reward mean `0.023619265105562864`, done count total `612030`, anchor/body/joint position error means `0.05958613292329686`, `0.7010389600310437`, and `0.904943812252287`.
- Report-ready plots and CSVs were generated under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/`.

## Verification

Passed after regenerating the audit chain:

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_eval_report_assets.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/reproduction_master_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

Results: artifact manifest `ok` with `920` artifacts; paper-vs-reproduction comparison `ok`; required-artifact absence audit `ok` with `29` rows; final report generation `ok`; completion matrix status audit `ok` with `174` rows and `0` invalid statuses; verification syntax audit `ok` with `186` scripts and `0` failures; verification script manifest `ok` with `186` scripts; verification command coverage `ok` with `194` commands and `10` smoke commands passing; master audit `ok`.

## Failed / Blocked Items

- Peak training memory was about `6203` MiB on GPU4 and `6199` MiB on GPU7, below the requested 10GB/card threshold. This is recorded honestly instead of inflating memory use.
- The local policy still has weak reward and very high termination counts, so it is not a mature tracking teacher.
- The generated checkpoints are local reproduction artifacts and are explicitly excluded from official paper-level checkpoint claims.
- Official unpatched `csv_to_npz.py`/`replay_npz.py`, true DAgger logs, official VAE/diffusion checkpoints, Fig. 5/Fig. 6 paper-level rollouts, TensorRT deployment, and real-robot validation remain incomplete.

## Effect on English Reading Report

The English report now has a stronger tracking reproduction subsection: it can state that the recovered official-importer-export path runs a larger PPO job end to end, while also explaining why this remains local virtual evidence rather than an official paper-level reproduction.

## Next Step

Run the required verification chain, update generated audit/report artifacts, then commit and push only scripts, docs, small JSON/CSV/PNG assets, and this progress update.

## Git Commit

Pending.
