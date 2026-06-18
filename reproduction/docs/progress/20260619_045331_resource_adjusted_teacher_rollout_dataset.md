# Progress Update

## Goal

Collect a clearly labeled resource-adjusted teacher rollout dataset from the evaluated local `model_99.pt` checkpoint, while preserving the official-vs-local boundary for BeyondMimic reproduction evidence.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/tracking_g1_resource_adjusted_ppo_checkpoint_eval.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260619_045331_resource_adjusted_teacher_rollout_dataset.md`

## Commands Run

- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `nvidia-smi -i 4,7`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `tail -80 logs/tracking_g1_resource_adjusted_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_dataset.log`
- `find res/runs/tracking_g1_resource_adjusted_teacher_rollout_dataset -maxdepth 5 -type f -printf '%p %s\n'`
- `envs/bm_analysis/bin/python - <<'PY' ... update GPU telemetry summary ... PY`

## Results

- Completed fixed-GPU rollout collection on physical GPUs `[4, 7]` with `CUDA_VISIBLE_DEVICES=4,7`.
- Collected two raw `.npz` shards under ignored `res/runs`.
- Total collected env steps: `306176`.
- Raw compressed dataset size: `514367800` bytes.
- Reward means by rank: `[0.02579653076827526, 0.025758858770132065]`.
- Done count total: `26258`; timeout count total: `0`.
- GPU telemetry: peak memory about `6775` MiB on GPU 4 and `6765` MiB on GPU 7; mean utilization about `91.85%` and `93.24%`.
- The pre-run GPU guard terminated one user-authorized `/mnt/infini-data/test/wangjc/` process on GPU 4 and saved the guard record.

## Verification

Full audit bundle is scheduled after integrating this new evidence into artifact manifest, comparison table, final report, completion matrix audits, and master audit.

## Failed / Blocked Items

- This rollout is not an official BeyondMimic DAgger dataset.
- It is not generated from an official public BeyondMimic teacher checkpoint.
- Peak GPU memory was below 10GB/card, so it is recorded as a rollout dataset gate rather than a formal high-memory training experiment.
- Official replay, official DAgger logs, Fig.5/Fig.6 paper-level closed-loop videos, TensorRT/asynchronous deployment evidence, and real robot evidence remain incomplete.

## Effect on English Reading Report

This adds a concrete, auditable code-reproduction milestone: the local pipeline now moves beyond static audits and checkpoint evaluation into state/action rollout data generation. The reading report can use this as evidence of independent reproduction work while explicitly stating that it remains resource-adjusted and not a paper-level official result.

## Next Step

Run the full verification bundle, refresh comparison/final/master audit artifacts, commit the scripts and small JSON/Markdown results, and push to GitHub. After that, the next technical step is a loader/contract audit for the rollout shards before using them for VAE/state-latent experiments.

## Git Commit

Pending at time of writing.
