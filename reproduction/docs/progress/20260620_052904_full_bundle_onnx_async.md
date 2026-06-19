# Progress Update

## Goal

Extend the deployment-path evidence from the single official-csv-loop VAE/denoiser audit to the broader 40-motion full-bundle local VAE and state-latent denoiser. This supports the reading-report deployment discussion while preserving the boundary that the result is ONNXRuntime CPU evidence, not TensorRT, not the paper Mini-PC, and not real robot deployment.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/scripts/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.py`
- `res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json`
- `res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json`
- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json`
- `res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json`

## Files Modified

- `reproduction/scripts/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`

## Commands Run

```bash
env -u BM_OFFICIAL_CSV_LOOP_ONNX_VARIANT \
  /mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python \
  reproduction/scripts/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.py

BM_OFFICIAL_CSV_LOOP_ONNX_VARIANT=full_bundle \
  /mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python \
  reproduction/scripts/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.py
```

## Results

The original single official-csv-loop ONNX async audit still passes after parameterization.

The new full-bundle audit was written to:

```text
res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/
```

Key full-bundle results:

- status: `ok_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit`
- ONNXRuntime providers: `AzureExecutionProvider`, `CPUExecutionProvider`
- CUDA/TensorRT ONNXRuntime providers: unavailable and recorded
- VAE encoder mu max abs ONNX-vs-PyTorch error: `4.76837158203125e-07`
- VAE encoder logvar max abs error: `7.152557373046875e-07`
- VAE decoder action max abs error: `1.7881393432617188e-07`
- denoiser token max abs error: `1.7881393432617188e-07`
- async thread-pool throughput speedup vs sequential mean: `2.7034503136567967`

Generated ONNX paths:

- `res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/official_csv_loop_full_bundle_vae_encoder_local.onnx`
- `res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/official_csv_loop_full_bundle_vae_decoder_local.onnx`
- `res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/official_csv_loop_full_bundle_state_latent_denoiser_local.onnx`

The ONNX files are recorded in manifests and absence audits, but they are not intended for GitHub commit by default.

## Verification

Pending after this progress note:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

- This is not a TensorRT run because the local ONNXRuntime build exposes CPU/Azure providers only.
- This is not a paper Mini-PC latency result.
- This is not CppAD guidance or a live asynchronous IsaacLab controller integration.
- This is not official BeyondMimic checkpoint evidence.
- Real robot deployment remains unavailable.

## Effect on English Reading Report

The English reading report can now say that deployment-path evidence was extended from the single official-loop local checkpoint pair to the 40-motion full-bundle local VAE/denoiser pair. The report must still describe it as local CPU ONNXRuntime evidence rather than TensorRT or paper hardware reproduction.

## Next Step

Run the verification suite, refresh manifests/reports, and commit/push the small code, audit, and report files. The next technical reproduction step should remain close to the paper-facing path: either strengthen task-conditioned closed-loop guidance/video evidence or attempt a real TensorRT/CUDA provider audit if the provider stack can be installed under `ROOT/envs`.

## Git Commit

Pending.
