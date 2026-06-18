# Progress Update

## Goal

Localize the current official `whole_body_tracking` replay blocker after the IsaacLab live gate reached the headless AppLauncher sentinel but official G1 URDF conversion still failed to produce a valid USD or `motion.npz`.

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
- `reproduction/scripts/tracking_urdf_conversion_probe.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- IsaacLab `urdf_converter.py` and `urdf_converter_cfg.py`
- Isaac Sim URDF importer command docs and `commands.py`

## Files Modified

- `reproduction/scripts/tracking_urdf_path_tiny_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- `res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json`
- `res/final_report/final_reproduction_report.json`
- `res/final_report/reproduction_report.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`
- Verification/audit refresh JSON/TSV files under `res/verification_command_*` and `res/final_deliverables_audit/`

## Commands Run

```bash
git status --short
git log --oneline --decorate -5
envs/bm_analysis/bin/python reproduction/scripts/tracking_urdf_path_tiny_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
git diff --stat
```

## Results

- Added a contrast probe that checks official G1 `package://unitree_description/...` mesh references and attempts Isaac Sim URDF conversion for a tiny local URDF, the official G1 URDF, and a generated absolute-mesh-path G1 URDF.
- Static mesh audit passes: all official G1 package mesh references resolve locally, so the current blocker is not a missing unpacked mesh file.
- Runtime probe status: `ok_with_blocker_classified`.
- Current blocker: `usd_layer_save_forbidden_and_vulkan_device_lost_before_payload`.
- The log records AppLauncher reaching `BM_SENTINEL:after_app`, repeated USD layer errors of `saving not allowed`, then Vulkan `ERROR_DEVICE_LOST` / GPU crash before a valid URDF conversion payload.
- Official replay conversion audit now reports latest blocker `usd_layer_save_forbidden_and_vulkan_device_lost_before_payload`.
- No `motion.npz`, replay video, tracking smoke metric, PPO checkpoint, teacher rollout dataset, VAE/diffusion closed-loop result, or robot result is claimed.

## Verification

All required verification commands passed.

- `artifact_manifest.py`: `ok`, `234` artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`, comparison JSON/CSV refreshed.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `161` rows.
- `verification_command_syntax_audit.py`: `ok`, `161` scripts, `0` failed.
- `verification_command_script_manifest.py`: `ok`, `161` scripts.
- `verification_command_coverage_audit.py`: `ok`, `169` commands, `10` smoke pass.
- `reproduction_master_audit.py`: `ok`, `196/196` master artifacts passed.

## Failed / Blocked Items

- Official replay remains blocked.
- Isaac Sim/Kit can start headless, but URDF conversion hits project-local USD layer save-forbidden errors and Vulkan device loss.
- No valid official G1 USD from the URDF importer was generated in this round.
- No official motion replay, tracking task smoke, PPO training/evaluation, DAgger rollout, VAE closed-loop rollout, diffusion closed-loop rollout, Fig.5/Fig.6 paper-level video, TensorRT deployment, or real robot validation is complete.

## Effect on English Reading Report

This provides a stronger and more honest reproduction narrative for the reading report: the environment restoration is no longer simply described as "IsaacLab broken"; it can be stated that IsaacLab/Isaac Sim package imports and headless AppLauncher startup are available, while the next reproducibility blocker is specifically the Isaac Sim URDF-to-USD write/runtime path for official replay. This also supports a reproducibility reflection about simulator asset conversion being a hidden dependency in robotics papers.

## Next Step

Try to bypass the URDF importer save-policy path by testing an official/IsaacLab-supported preconverted G1 USD path, a MJCF-based G1 asset path, or a minimal stage-save permission probe under an Isaac Sim writable Omniverse/user cache. Only after a valid G1 USD exists should official `csv_to_npz.py` / `replay_npz.py` be retried.

## Git Commit

Pending at the time this progress note is written.
