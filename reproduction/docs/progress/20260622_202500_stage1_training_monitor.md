# Progress Update

## Goal

Add a read-only live monitor for the two active Stage 1 PPO teacher-policy training lines: the GPU 4+7 LAFAN1 paper-contract run and the GPU 5+6 multi-source public-plus-available run.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/paper_contract_tracking_parameters.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`
- active PPO logs under `/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_official_importer_export_paper_contract_ppo_training_run/` and `/mnt/infini-data/test/BeyondMimic/logs/tracking_stage1_multisource_paper_contract_ppo_training_run/`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/watch_stage1_tracking_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/watch_stage1_tracking_training.sh`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_202500_stage1_training_monitor.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/watch_stage1_tracking_training.py
python3 reproduction/scripts/watch_stage1_tracking_training.py --once --no-clear
chmod +x reproduction/scripts/watch_stage1_tracking_training.py reproduction/scripts/watch_stage1_tracking_training.sh
```

## Results

The monitor prints a single terminal dashboard with both active jobs:

- GPU 4+7 LAFAN1 paper-contract PPO teacher: process status, GPU memory/utilization, PPO parameters, latest learning iteration, ETA, reward/losses, motion errors, termination terms, and latest checkpoint.
- GPU 5+6 multi-source PPO teacher: same fields plus bundle metadata for 49 motions, 448358 frames, and 2.49 h public-plus-available training input.

The script is read-only and only parses logs, checkpoint filenames, process status, and `nvidia-smi`. It does not modify training runs or checkpoint files.

## Verification

Syntax and one-shot runtime parsing passed. The full project verification suite is run after this progress file is written.

## Failed / Blocked Items

No monitor-specific failure. The underlying Stage 1 teacher quality remains unresolved until the active training jobs finish and their checkpoints pass evaluation.

## Effect on English Reading Report

This improves experiment traceability for the reading report and defense: training progress, hardware usage, local-versus-official claim boundaries, and checkpoint provenance can now be monitored without interrupting active PPO jobs.

## Next Step

Keep both Stage 1 training lines running. After stronger checkpoints are available, evaluate teacher quality and only then collect closed-loop state-action trajectories for VAE and latent diffusion training.

## Git Commit

Pending.
