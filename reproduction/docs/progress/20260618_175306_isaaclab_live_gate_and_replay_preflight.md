# Progress Update

## Goal

Refine the IsaacLab live headless gate after the Vulkan EGL ICD repair, distinguish Isaac Sim fast-shutdown sentinel semantics from real runtime warnings, and prepare the official `whole_body_tracking` replay path without starting training.

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
- Isaac Sim `SimulationApp` implementation under `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/lib/python3.10/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py`
- Official `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py`
- Official `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_live_gate_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/blocked_gate_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_preflight.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- Refreshed small audit/report artifacts under `/mnt/infini-data/test/BeyondMimic/res`

## Commands Run

```bash
env CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/isaaclab_live_gate_probe.py
python3 reproduction/scripts/tracking_official_replay_preflight.py
python3 reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py
python3 reproduction/scripts/blocked_gate_audit.py
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

- IsaacLab live headless gate is now `ok_with_runtime_warning`, not fully blocked.
- `AppLauncher(headless=True)` reaches `after_app`, emits payload, and the `fast_shutdown=False` candidate reaches `BM_SENTINEL:after_close`.
- The previous missing `after_close` was partly explained by Isaac Sim default `fast_shutdown=True`, which exits the process immediately.
- CUDA P2P/IOMMU warning is still retained in logs and is not erased: `cuda_p2p_iommu_runtime_warning_retained=true`.
- Added official replay preflight:
  - Official `csv_to_npz.py` exists.
  - Official `replay_npz.py` exists.
  - Local replay patch exists for avoiding WandB registry download.
  - Selected G1 LAFAN1 CSV `walk1_subject1.csv` has 36 columns, finite values, and enough frames.
  - Planned commands are recorded but not executed.
- Artifact manifest increased to 231 artifacts.
- Master audit increased to 193/193 passing artifacts.
- Blocked gate counts now separate `clear_with_runtime_warning` from truly blocked gates.

## Verification

All required verification commands passed. Latest verification log:

`/mnt/infini-data/test/BeyondMimic/logs/setup/verification_20260618_175230_isaaclab_replay_preflight.log`

Key refreshed outputs:

- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_gpu_foundation_settings_audit/isaaclab_gpu_foundation_settings_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_preflight/tracking_official_replay_preflight.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`

## Failed / Blocked Items

- No official `csv_to_npz.py` conversion was executed in this round.
- No rendered `replay_npz.py` video or tracking task smoke was executed in this round.
- CUDA P2P/IOMMU warning remains a runtime risk for Isaac Sim/Kit.
- PPO tracking training, official teacher rollouts, true DAgger rollouts, VAE closed-loop evaluation, state-latent rollout dataset, full diffusion closed-loop evaluation, Fig. 5/Fig. 6 paper-level videos/metrics, TensorRT deployment, and real robot validation remain unfinished.

## Effect on English Reading Report

This round strengthens the reproduction section by showing a concrete environment recovery step: the project moved from a live IsaacLab gate that appeared blocked to a bounded `ok_with_runtime_warning` state with explicit evidence. It also creates a clear next experimental gate for official tracking replay while preserving the distinction between environment readiness, replay smoke, and paper-level reproduction.

## Next Step

Run the planned official `csv_to_npz.py` conversion on the selected short G1 LAFAN1 clip under the live-gate environment, validate the generated `motion.npz`, then attempt bounded local/official replay only if conversion succeeds. Do not start PPO or closed-loop diffusion yet.

## Git Commit

Pending at time of writing.
