# Progress Update

## Goal

Push the BeyondMimic reproduction mainline forward after the resource-adjusted teacher rollout VAE by building a full state/action-latent dataset and training a full resource-adjusted denoiser over all generated windows, while keeping official-vs-local boundaries explicit.

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
- `res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json`
- `res/tracking/g1_resource_adjusted_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_dataset.json`

## Files Modified

- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/known_limitations.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/progress/20260619_055350_resource_adjusted_state_latent_diffusion.md`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py
```

```bash
envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py
```

```bash
envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py
```

Final audit commands are recorded after regeneration.

## Results

Built a full resource-adjusted state/action-latent dataset from the existing teacher rollout shards and local resource-adjusted VAE checkpoint: `306176` samples, `285696` 21-step windows, split counts `228558/28569/28569`, obs dim `160`, latent dim `32`, token dim `192`, weighted posterior reconstruction MSE `0.002923722844570875`.

Trained a resource-adjusted state-latent denoiser on all generated windows for 30 epochs with `CUDA_VISIBLE_DEVICES=4,7` and DataParallel enabled. Test noisy token MSE was `0.08264570789677757`, test predicted token MSE was `0.03726350223379476`, and test denoising improvement ratio was `0.5491175139992032`.

## Verification

Passed. Commands completed successfully:

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/required_artifact_absence_audit.py reproduction/scripts/reproduction_master_audit.py
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

Final audit results: required artifact absence `20` rows, artifact manifest `313` artifacts, verification syntax `182` scripts with `0` failures, verification coverage `190` commands, and master audit `ok`.

## Failed / Blocked Items

This is still resource-adjusted local evidence. It is not the official BeyondMimic DAgger rollout dataset, official VAE checkpoint, official diffusion checkpoint, TensorRT deployment, Fig. 5/Fig. 6 closed-loop guidance, or real robot result.

## Effect on English Reading Report

The reading report can now describe a coherent virtual reproduction chain: resource-adjusted PPO teacher candidate -> full teacher rollout dataset -> local conditional action VAE -> full state/action-latent dataset -> full denoiser training. This is valuable independent reproduction work, but the report must still label it as resource-adjusted and non-official.

## Next Step

Refresh all audit/report artifacts and then either evaluate the denoiser in an offline guidance surrogate or return to official G1 replay/closed-loop rollout recovery.

## Git Commit

Pending before commit; final hash recorded in the assistant turn report after `git commit`.
