# Progress Update

## Goal

Run a bounded diagnostic of the official `whole_body_tracking/scripts/replay_npz.py` entrypoint without modifying the
official worktree, so the current official replay blocker is localized more precisely than "replay not done."

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_full_replay/tracking_g1_resource_adjusted_csv_full_replay_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/blocked_gate_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py
python3 -m py_compile reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/blocked_gate_audit.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/blocked_gate_audit.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/reproduction_master_audit.py
```

The full verification bundle is rerun after this progress file is written.

## Results

- New audit status: `ok_with_official_replay_npz_entry_blocker`.
- Latest blocker: `official_urdf_converter_layer_save_blocked`.
- The unmodified official `replay_npz.py` entry reaches AppLauncher.
- It fails before fake-WandB artifact download and before replay-loop execution.
- The log records multiple `Cannot save layer ... saving not allowed` errors under `/tmp/IsaacLab/...`, followed by an empty robot prim with no contact sensors.
- The failed log is retained under `res/failed_runs/tracking_official_replay_npz_entry_diagnostic/`.

## Verification

Pending in this file at creation time. The expected full bundle is:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/blocked_gate_audit.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/progress_report_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

- Official G1 USD/URDF converter output remains blocked.
- Official `csv_to_npz.py` and official `replay_npz.py` have not produced a valid paper-level replay artifact.
- Formal PPO tracking training/evaluation, teacher rollout data, DAgger logs, VAE/diffusion closed-loop evaluation, Fig. 5/Fig. 6 videos, TensorRT deployment, and real robot validation remain incomplete.

## Effect on English Reading Report

This gives the reproduction section a concrete failure analysis: the official replay entry can be launched, but the
public environment still fails inside the official URDF-to-USD conversion/write path before motion artifact handling.
That is stronger and more honest evidence than simply saying "official replay could not be run."

## Next Step

Continue official URDF/USD converter recovery, or run a controlled short PPO diagnostic only with explicit
resource-adjusted labeling and GPU telemetry. The paper-level claim remains blocked until an official replay/conversion
artifact passes.

## Git Commit

Pending at creation time; the final commit hash is reported in the user-facing turn summary.
