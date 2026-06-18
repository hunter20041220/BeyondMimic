# Progress Update

## Goal

Probe whether the official G1 URDF converter blocker can be repaired through the Isaac Sim 4.5 Python `ImportConfig`
surface, then fold the result into the auditable reproduction reports without claiming official replay success.

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
- IsaacLab converter sources under `download/IsaacLab-v2.1.0/source/isaaclab/isaaclab/sim/converters/`

## Files Modified

- `reproduction/scripts/tracking_g1_urdf_import_config_variant_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/blocked_gate_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/progress/20260619_020706_g1_import_config_probe.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_import_config_variant_probe.py
```

Full verification commands were run after integration; see the final turn report for pass/fail status.

## Results

The probe reached `AppLauncher(headless=True)` and enumerated `URDFCreateImportConfig` in Isaac Sim 4.5. The surface
does not expose `set_make_instanceable` or an instanceable USD path setter. The baseline official G1 URDF converter
attempt produced an openable USD file, but it contained zero prims, zero joints, and zero rigid-body-like prims.

This closes the Python-level instanceable-patch route for the active official replay blocker. It does not produce
official `motion.npz`, replay, PPO, trained policy, video, or paper-level metric evidence.

## Verification

The result is now wired into artifact manifest, paper-vs-reproduction comparison, blocked-gate audit, final report, and
master audit. The master audit checks that no valid robotish USD is claimed and that `goal_complete=false` remains true.

## Failed / Blocked Items

- Official G1 USD converter output remains blocked.
- Official `csv_to_npz.py` and `replay_npz.py` still do not reach paper-level replay.
- Formal PPO tracking training/evaluation is not yet run.
- DAgger/teacher rollout data, VAE/diffusion closed-loop evaluation, Fig. 5/Fig. 6 videos, TensorRT deployment, and real robot remain incomplete or unavailable.

## Effect on English Reading Report

This adds a precise reproducibility note for the reading report: the environment can enter IsaacLab/Kit, but the
official tracking replay path is blocked by asset conversion rather than by a missing registry artifact alone. It also
supports a sober distinction between official-code diagnostics and resource-adjusted virtual validation.

## Next Step

Return to the main reproduction line: run the strongest available virtual task/evaluation or a controlled PPO diagnostic
on the resource-adjusted path, while only revisiting official converter work if a new lower-level importer route is
identified.

## Git Commit

Pending at file creation time.
