# Progress Update

## Goal

Advance the scaled official-importer-export PPO downstream chain from offline guidance into local closed-loop task-conditioned latent-guidance rollouts, then register the result in the audit/report pipeline without upgrading it to a paper-level BeyondMimic claim.

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
- Existing task-conditioned guidance, report-assets, manifest, comparison, final-report, completion-matrix, and master-audit scripts.

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/official_csv_loop_task_conditioned_guidance_report_assets.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/PROGRESS.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- `reproduction/docs/progress/20260621_060523_scaled_closed_loop_guidance.md`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py
CUDA_VISIBLE_DEVICES=4,7 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py
BM_METADATA_ONLY=1 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py
BM_TASK_CONDITIONED_REPORT_VARIANT=official_importer_export_scaled_ppo envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_task_conditioned_guidance_report_assets.py
```

## Results

The scaled closed-loop summary is:

```text
res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json
status: ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval
tasks: joystick, waypoint, obstacle_avoidance, composed
rollout steps per task: 299
```

Guided reward means:

```text
joystick: 0.022449076233313336
waypoint: 0.025156304768183858
obstacle_avoidance: 0.0229376406832458
composed: 0.025132083756082932
```

Guided target-body error means:

```text
joystick: 0.3439415395259857
waypoint: 0.3440071940422058
obstacle_avoidance: 0.34300488233566284
composed: 0.3445764183998108
```

Report assets were generated under:

```text
res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_summary/
```

The local MP4s remain in `res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout/` and are not intended for GitHub commit.

## Verification

Full verification commands are run after this progress file is written:

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

## Failed / Blocked Items

No task failed in this round. The result remains blocked from paper-level status because it uses local proxy costs and local scaled PPO/VAE/denoiser checkpoints. It is not official BeyondMimic Fig. 5/Fig. 6 success/failure evidence, not TensorRT/asynchronous deployment, and not real robot evidence.

## Effect on English Reading Report

The English reading report now has a stronger code reproduction section: the scaled PPO downstream chain is no longer only an offline guidance table; it has executed local closed-loop task-conditioned rollouts and report-ready plots. The report still explicitly states that this project does not fully reproduce BeyondMimic at paper level.

## Next Step

Use the scaled closed-loop result to decide whether to run a multi-seed scaled task-conditioned guidance set, a stricter Fig. 5/Fig. 6 proxy-protocol pass over scaled traces, or a deployment-path ONNX/TensorRT audit for the scaled VAE/denoiser.

## Git Commit

Pending at time of writing.
