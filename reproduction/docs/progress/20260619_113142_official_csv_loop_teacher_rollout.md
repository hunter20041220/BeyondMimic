# Progress Update

## Goal

Use the official csv-loop PPO checkpoint from the previous round to collect a stronger local teacher rollout dataset for downstream VAE/state-latent/diffusion reproduction.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_ppo_training_run/tracking_g1_official_csv_loop_ppo_training_run.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_csv_loop_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_teacher_rollout_dataset.py
BM_OFFICIAL_CSV_LOOP_TEACHER_ROLLOUT_SEED=20260631 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_csv_loop_teacher_rollout_dataset.py
```

## Results

- Status: `ok_official_csv_loop_teacher_rollout_dataset_completed`
- Checkpoint: local official-loop-motion PPO `model_299.pt`
- GPUs: physical `[4,7]`, `CUDA_VISIBLE_DEVICES=4,7`
- Rollout: `512` envs/rank x `2` ranks x `299` steps = `306176` env steps
- Shards: `2`
- Raw compressed dataset size: `514388245` bytes under ignored `res/runs`
- Reward means by rank: `[0.025898048654198647, 0.025928476825356483]`
- Done count total: `26331`
- Timeout count total: `0`
- GPU mean utilization: about `91.62%` on GPU4 and `91.24%` on GPU7
- Peak memory: about `6775` MiB on GPU4 and `6765` MiB on GPU7

## Verification

Passed:

```bash
envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/blocked_gate_audit.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/progress_report_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

The combined log is `/mnt/infini-data/test/BeyondMimic/logs/verification_20260619_113142_official_csv_loop_teacher_rollout.log`. Final key statuses: artifact manifest `ok` with `333` artifacts; paper-vs-reproduction `ok` with `139` rows; required-artifact absence `ok` with `22` rows; master audit `ok`; `goal_complete=false`.

## Failed / Blocked Items

- First wrapper attempt did not start because the inherited base preflight accepted only `ok_resource_adjusted_ppo_training_completed`; this was fixed to also accept `ok_official_csv_loop_ppo_training_completed`.
- Peak memory remains below the requested `10GB/card` formal high-memory threshold, although both GPUs were actively used.
- The dataset is not the official BeyondMimic DAgger rollout log because it depends on the enriched-USD runtime patch and a local 300-iteration PPO checkpoint.

## Effect on English Reading Report

This gives the report a stronger reproduction chain: official csv-loop motion -> local PPO checkpoint -> quantitative eval -> two-shard teacher rollout dataset. It should still be presented as local virtual evidence, not official paper-level DAgger.

## Next Step

Refresh audit artifacts, run the full verification chain, then commit and push. The next technical step should be to rebuild VAE/state-latent/diffusion data from this stronger official-loop teacher rollout dataset.

## Git Commit

Pending at file update time; see the Git commit for this progress update.
