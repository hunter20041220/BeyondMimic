# Progress Update

## Goal

Add an auditable local deployment-path gate for the official-csv-loop VAE and state-latent denoiser. The intent is to support the English reading report with honest ONNXRuntime/async evidence while keeping the boundary clear: this is not TensorRT, not the paper Mini-PC latency result, not an official BeyondMimic checkpoint, and not real-robot evidence.

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
- `reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.py`
- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `reproduction/scripts/level_c_official_csv_loop_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_official_csv_loop_state_latent_diffusion_training.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- `reproduction/docs/current_environment_and_reproduction_status.md`
- `reproduction/docs/completion_matrix.md`

## Files Modified

- `reproduction/scripts/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/current_environment_and_reproduction_status.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- `reproduction/docs/progress/20260619_170528_onnx_async_deployment_audit.md`

## Commands Run

```bash
envs/bm_diffusion/bin/python reproduction/scripts/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.py
```

Planned verification after this progress note:

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

## Results

The new audit wrote:

- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json`
- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.tsv`
- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/level_c_official_csv_loop_vae_denoiser_onnx_async_latency.csv`
- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/official_csv_loop_vae_encoder_local.onnx`
- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/official_csv_loop_vae_decoder_local.onnx`
- `res/level_c/official_csv_loop_vae_denoiser_onnx_async/official_csv_loop_state_latent_denoiser_local.onnx`

The ONNX files are local runtime outputs and remain ignored by Git. The audit JSON records their paths, sizes, and SHA256 hashes.

Key results:

- Status: `ok_official_csv_loop_vae_denoiser_onnx_async_audit`
- ONNXRuntime providers available: `AzureExecutionProvider`, `CPUExecutionProvider`
- VAE encoder mu max abs ONNX-vs-PyTorch error: `2.980232238769531e-07`
- VAE encoder logvar max abs ONNX-vs-PyTorch error: `5.960464477539062e-07`
- VAE decoder action max abs ONNX-vs-PyTorch error: `2.086162567138672e-07`
- Denoiser token max abs ONNX-vs-PyTorch error: `1.601874828338623e-07`
- Async proxy requests: `80`
- Async proxy workers: `4`
- Sequential mean per request: `0.8577487664297223` ms
- Async mean per request: `0.3490717732347548` ms
- Async throughput speedup vs sequential mean: `2.457227516510985`

## Verification

Verification is run after this note is committed into the working tree. Expected checks:

- The new script compiles through syntax audit.
- Manifest includes the new script and small JSON/TSV/CSV outputs.
- Paper-vs-reproduction gains one qualitative-only row for local CPU ONNXRuntime deployment-path evidence.
- Master audit remains `ok`.

## Failed / Blocked Items

- `CUDAExecutionProvider` is not available in this ONNXRuntime build.
- TensorRT provider is not available in this ONNXRuntime build.
- This is not the paper RTX 4060 Mobile Mini-PC deployment benchmark.
- This is not CppAD-guided real-time deployment.
- This is not live IsaacLab deployed control.
- This is not real Unitree G1 evidence.

## Effect on English Reading Report

The report can now say that the local VAE and denoiser are exportable to executable ONNX graphs and that a local CPU async runtime proxy was measured. It must also say that TensorRT/CUDA-provider/Mini-PC deployment remains unreproduced.

## Next Step

After verification and commit, the strongest next technical step is either:

1. install or build an ONNXRuntime/TensorRT runtime path and rerun the same audit with GPU/TensorRT providers, or
2. continue strengthening closed-loop task-conditioned guidance with multi-seed virtual rollouts and clearer task-success metrics.

## Git Commit

Pending at note creation time.
