# Progress Update

## Goal

Advance the BeyondMimic reproduction from offline VAE/action-decode evidence toward a closed-loop IsaacLab evaluation, while preserving the paper-level boundary. The concrete target for this round was a full 299-step two-GPU local VAE action-reconstruction rollout: PPO teacher action -> local conditional VAE posterior mean -> local VAE decoded action -> `Tracking-Flat-G1-v0` environment step.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/g1_official_csv_loop_teacher_rollout_dataset/tracking_g1_official_csv_loop_teacher_rollout_dataset.json`
- `res/level_c/official_csv_loop_teacher_rollout_vae_training/level_c_official_csv_loop_teacher_rollout_vae_training.json`
- `res/level_c/official_csv_loop_guided_action_rollout_probe/tracking_g1_official_csv_loop_guided_action_rollout_probe.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.py`
- `reproduction/scripts/official_csv_loop_vae_closed_loop_rollout_report_assets.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Regenerated audit/report outputs under `res/artifact_manifest/`, `res/comparison/`, `res/docs/completion_matrix_status_audit/`, `res/final_report/`, `res/master_audit/`, `res/verification_command_coverage/`, and `res/verification_command_script_manifest/`.

## Commands Run

```bash
python3 reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_vae_closed_loop_rollout_report_assets.py
python3 -m py_compile reproduction/scripts/official_csv_loop_vae_closed_loop_rollout_report_assets.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py
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

The new rollout succeeded:

```text
status: ok_official_csv_loop_vae_closed_loop_rollout_eval
physical GPUs: 4, 7
parallel envs: 2048
rollout steps: 299
total env steps: 612352
duration seconds: 1375.844
teacher/VAE action MSE mean: 0.004145793081027608
teacher/VAE action absolute-error mean: 0.04706366988752399
reward mean: 0.02731888730664516
done count total: 52621
timeout count total: 0
```

GPU telemetry was recorded honestly:

```text
GPU4 peak memory: 19229 MiB, mean utilization: 97.21818181818182%
GPU7 peak memory: 8129 MiB, mean utilization: 79.16363636363636%
peak_memory_each_gpu_at_least_10gb: false
```

This is a successful two-GPU virtual rollout, but it does not meet the stricter "each GPU at least 10GB" condition because GPU7 stayed below 10GB. The result was not inflated.

## Verification

All required verification commands passed after integration:

```text
artifact_manifest: ok, 404 artifacts, 0 missing
paper_vs_reproduction: ok, 146 rows
final_reproduction_report: ok
completion_matrix_status_audit: ok, 164 rows
verification_command_syntax_audit: ok, 185 scripts
verification_command_script_manifest: ok, 185 scripts
verification_command_coverage_audit: ok, 193 commands, 10 smoke checks passed
reproduction_master_audit: ok, 251/251 artifacts passed
required_artifact_absence_audit: ok, 25 rows
```

## Failed / Blocked Items

- The rollout log contains repeated PhysX patch-buffer overflow warnings during the high-parallelism run. The run still completed and wrote both shard metrics.
- GPU7 did not reach the 10GB peak-memory target. This is recorded as a limitation, not hidden.
- This run is not the official BeyondMimic VAE checkpoint, not an autonomous VAE policy, not receding-horizon diffusion guidance, not Fig. 5/Fig. 6 reproduction, and not real-robot evidence.

## Effect on English Reading Report

This gives the reading report a stronger engineering narrative: released-data/static audits -> IsaacLab official-loop PPO and teacher rollout -> local conditional VAE training -> offline guided action decode -> short decoded-action bridge -> full local VAE action-reconstruction closed-loop rollout. The report now has a concrete closed-loop simulation result while still clearly stating that paper-level guided diffusion and official checkpoints remain unreproduced.

## Next Step

The next useful step is to move from VAE action reconstruction to receding-horizon closed-loop guidance: repeatedly select a current observation, generate/guidance-update a state-latent/action plan, decode the current action through the local VAE, step IsaacLab, and log success/failure metrics. This must still be labeled as local virtual evidence unless official checkpoints and paper task definitions become available.

## Git Commit

Pending at the time this progress file was written.
