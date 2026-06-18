# Progress Update

## Goal

Advance BeyondMimic reproduction beyond environment recovery without claiming official paper-level completion: re-test the official G1 URDF in-memory importer path on the current GPU4 headless setup, then use the existing full resource-adjusted teacher rollout dataset for a clearly labeled conditional action VAE training gate.

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
- `res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json`
- `res/tracking/g1_resource_adjusted_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_dataset.json`
- `res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json`

## Files Modified

- `reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py`
- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/known_limitations.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/progress/20260619_053608_resource_adjusted_teacher_rollout_vae.md`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py
```

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py
```

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py
```

```bash
envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py
```

Verification commands are recorded again after report/audit regeneration below.

## Results

The official G1 in-memory importer probe reached `AppLauncher` on physical GPU4 and entered the Isaac Sim URDF importer branch with `dest_path=""`, but it did not return from import or export a robot stage. The child process was killed after Vulkan `ERROR_DEVICE_LOST`; the summary JSON classifies this as `ok_with_vulkan_device_lost_blocker`. This preserves the official replay blocker as real evidence rather than a missing-script guess.

The resource-adjusted teacher rollout VAE training completed over all currently available teacher rollout shards: `306176` `(policy_obs, action)` samples, obs dim `160`, action dim `29`, train/validation/test split `244940/30618/30618`, latent dim `32`, hidden dim `512`, 40 epochs, `CUDA_VISIBLE_DEVICES=4,7`, two visible CUDA devices, and DataParallel enabled. Final metrics: validation action MSE `0.0029512199107557535`, test action MSE `0.002976319403387606`, test mean absolute action error `0.04116625525057316`.

## Verification

Passed after rerun. Commands completed successfully:

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py
```

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
git diff --check
```

Final audit results: artifact manifest `305` artifacts, paper-vs-reproduction comparison `124` rows, required artifact absence audit `19` rows, completion matrix `161` rows, verification command syntax `180` scripts with `0` failures, verification command coverage `188` commands, progress audit `38` rows, and master audit `ok`.

## Failed / Blocked Items

- Official G1 URDF file-output conversion remains blocked by `permissionToSave=False` / empty USD.
- Official G1 URDF in-memory conversion on current GPU4 is blocked by Vulkan `ERROR_DEVICE_LOST` before a valid stage export.
- Official `csv_to_npz.py`, unmodified official `replay_npz.py` replay loop, paper-level tracking evaluation, true DAgger logs, official VAE/diffusion checkpoints, closed-loop Fig. 5/Fig. 6 videos/metrics, TensorRT/asynchronous deployment, and real robot results remain incomplete.

## Effect on English Reading Report

This gives the report a stronger reproduction narrative: the environment can run substantial virtual experiments, the official G1 importer failure is localized and retained as failed-run evidence, and the local resource-adjusted rollout data now supports a full downstream conditional VAE training result. The report must still describe this as resource-adjusted local evidence, not as official BeyondMimic paper-level VAE reproduction.

## Next Step

Refresh all audit artifacts, commit/push this round, then choose between continuing the official USD/importer recovery path or using the trained resource-adjusted VAE to build a clearly labeled state-latent dataset / closed-loop surrogate evaluation.

## Git Commit

Pending before commit; final hash recorded in the assistant turn report after `git commit`.
