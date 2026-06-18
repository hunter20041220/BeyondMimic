# Progress Update

## Goal

Refresh the official `whole_body_tracking/scripts/replay_npz.py` entry diagnostic after the current IsaacLab
`AppLauncher(headless=True)` gate passed, using the current physical GPU4 setup and preserving a machine-readable
blocker if official replay still fails.

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
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_probe.py`
- `/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic.log`

## Commands Run

```bash
git status --short
git diff --stat
sed -n '1,280p' reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py
sed -n '280,560p' reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py
nvidia-smi -i 4,7 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py
jq '{status, generated_at, latest_blocker, checks, run: {returncode: .run.returncode, duration_seconds: .run.duration_seconds, stalled: .run.stalled, markers: .run.markers}}' \
  res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json
```

## Results

The official replay entry diagnostic now runs in a more robust wrapper:

- The Kit child process is started in a new session so its shutdown signal does not kill the audit parent.
- The wrapper targets physical `cuda:4` without `CUDA_VISIBLE_DEVICES` masking, matching the current headless gate.
- Kit multi-GPU renderer/physics auto-selection is disabled through Kit args.
- A new log marker catches the driver shader-cache shutdown error after AppLauncher construction.

The current run wrote:

```text
/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json
```

Current status:

```text
status: ok_with_official_replay_npz_entry_blocker
latest_blocker: official_urdf_converter_layer_save_blocked
returncode: 0
duration_seconds: 30.029
```

Important markers:

```text
before_runpy=true
add_app_launcher_args=true
after_real_app_launcher=true
fake_wandb_download=false
bounded_loop_complete=false
permission_to_save_false=true
failed_to_save_layer=true
empty_robot_after_converter=true
driver_shader_cache_shutdown_error=true
```

## Verification

The script compiled successfully and produced a refreshed audit JSON plus retained failed-run log. Full project audit
refresh is run after this progress note is added.

## Failed / Blocked Items

Official replay is still blocked before motion loading. Because `fake_wandb_download=false`, this is not a missing
registry-only issue. The blocker is still localized to the official G1 URDF/USD converter layer-save path, where the
robot prim remains empty. The existing 299-step resource-adjusted replay remains useful engineering evidence, but it is
not official `csv_to_npz.py` output and not paper-level replay evidence.

## Effect on English Reading Report

This result improves the report's reproduction section by showing a clear boundary: the environment can launch
IsaacLab/Kit and enter the official replay entrypoint, but official asset conversion still prevents paper-level replay.
It supports an honest statement that the project progressed from package/import recovery to a concrete official replay
blocker, without overstating the resource-adjusted workaround.

## Next Step

Continue official converter recovery at the USD layer-save/empty-robot boundary, or proceed with explicitly labeled
resource-adjusted downstream experiments while preserving official replay as blocked.

## Git Commit

Pending at the time this file is written.
