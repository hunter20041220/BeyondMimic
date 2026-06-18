# Progress Update

## Goal

Refine the IsaacLab/Isaac Sim tracking gate by testing whether the official Unitree G1 URDF can be imported into an
in-memory stage under raw `SimulationApp` with the IsaacLab headless experience. This probes whether the previous
AppLauncher in-memory crash is caused by the AppLauncher wrapper or by the lower Kit/GPU runtime path.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/scripts/tracking_g1_urdf_in_memory_import_probe.py`
- `reproduction/scripts/tracking_simulationapp_save_policy_probe.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`

## Files Modified

- Added `reproduction/scripts/tracking_g1_urdf_simulationapp_in_memory_import_probe.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Updated `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- Updated `reproduction/scripts/reproduction_master_audit.py`
- Updated `reproduction/scripts/final_reproduction_report.py`
- Updated `reproduction/docs/known_limitations.md`
- Updated `reproduction/docs/experiment_protocol.md`
- Regenerated `reproduction/docs/final_reproduction_report.md`
- Regenerated small JSON/TSV audit outputs under `res/artifact_manifest`, `res/final_report`, `res/master_audit`,
  `res/final_deliverables_audit`, `res/verification_command_script_manifest`,
  `res/verification_command_coverage`, and `res/tracking/official_replay_conversion`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_simulationapp_in_memory_import_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
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

- New probe JSON:
  `res/tracking/g1_urdf_simulationapp_in_memory_import/tracking_g1_urdf_simulationapp_in_memory_import_probe.json`
- Status: `ok_with_vulkan_device_lost_before_payload`
- Current blocker: `simulationapp_in_memory_import_vulkan_device_lost_before_payload`
- The raw `SimulationApp` run reached the IsaacLab headless app sentinel and the URDF importer entered the
  `dest_path=""` in-memory branch.
- The process then crashed before payload with Vulkan/GPU device loss and return code `-11`.
- No G1 USD, official `motion.npz`, replay video, PPO training, policy evaluation, DAgger rollout, VAE/diffusion
  closed-loop run, checkpoint, or robot result is claimed.
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json` now records this as the
  latest official replay conversion blocker.

## Verification

- `artifact_manifest.py`: passed, 242 artifacts, missing count 0
- `paper_vs_reproduction_comparison.py`: passed, 122 comparison rows retained
- `final_reproduction_report.py`: passed
- `completion_matrix_status_audit.py`: passed, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: passed, 161 scripts, 0 failures
- `verification_command_script_manifest.py`: passed, 161 scripts
- `verification_command_coverage_audit.py`: passed, 169 commands, 10 smoke-pass commands
- `reproduction_master_audit.py`: passed, 204 audited artifacts

## Failed / Blocked Items

- The official tracking replay gate remains blocked.
- The current evidence shows that both AppLauncher and raw `SimulationApp` can reach the G1 URDF in-memory import
  branch, but the current Kit/Vulkan runtime crashes before an exported robot stage can be captured.
- The next technical target is a non-RTX/no-render importer setting, another Isaac/Kit conversion mode, or a valid
  preconverted G1 USD before retrying official `csv_to_npz.py` / `replay_npz.py`.

## Effect on English Reading Report

This provides a clearer reproduction boundary for the reading report: the project has restored IsaacLab/Isaac Sim
package and live-app startup layers, but official closed-loop tracking remains blocked at the G1 URDF import/conversion
gate. It supports an honest statement that the reproduction work is substantial and auditable, while not yet a
paper-level live tracking or Fig. 5/Fig. 6 reproduction.

## Next Step

Try a rendering-minimized or preconverted-asset path for the official G1 USD gate, then rerun the official
`csv_to_npz.py` / `replay_npz.py` conversion and replay checks. Do not start long PPO tracking training until a valid
official G1 USD, `motion.npz`, and replay gate are available.

## Git Commit

Pending at the time this progress note is written.
