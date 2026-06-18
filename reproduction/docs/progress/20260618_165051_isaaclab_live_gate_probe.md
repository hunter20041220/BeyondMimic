# Progress Update

## Goal

Diagnose the current IsaacLab live headless gate on the migrated server before attempting official replay or PPO training.

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
- `res/blocked_gates/blocked_gate_audit.json`
- `logs/setup/isaaclab_headless_app_gate.log`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/app/app_launcher.py`

## Files Modified

- `.gitignore`
- `.vscode/settings.json`
- `reproduction/scripts/isaaclab_live_gate_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/blocked_gate_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/final_reproduction_report.md`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/blocked_gates/blocked_gate_audit.json`
- `res/blocked_gates/blocked_gate_audit.tsv`
- `res/final_report/final_reproduction_report.json`
- `res/final_report/reproduction_report.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`
- refreshed lightweight setup and verification audit JSON/TSV files under `res/setup/` and `res/verification_command_*`

## Commands Run

- `rg` and `sed` inspections over IsaacLab `app_launcher.py`
- `tail -n 120 logs/setup/isaaclab_headless_app_gate.log`
- `nvidia-smi -q -i 6`
- `python3 reproduction/scripts/kit_inotify_budget_audit.py`
- `python3 reproduction/scripts/inotify_live_usage_audit.py`
- `python3 reproduction/scripts/isaaclab_live_gate_probe.py`
- `python3 reproduction/scripts/blocked_gate_audit.py`
- `python3 reproduction/scripts/vscode_watcher_exclude_audit.py`
- full required verification command bundle listed below

## Results

- `bm_tracking` package/import layer remains usable: Isaac Sim and IsaacLab import with EULA accepted.
- Current inotify limits are now `max_user_watches=1048576` and `max_user_instances=10240`; the old `8192/128` blocker is historical and now downgraded to `needs_review`.
- Live inotify usage is high because a Cursor/VS Code file watcher holds about 1.03M watches, but it still has positive headroom after the sysctl increase.
- Added `.vscode/settings.json` watcher/search excludes for large project directories.
- IsaacLab AppLauncher probes reached `BM_SENTINEL:after_app` and reported `is_running=true`, but both failed to reach `BM_SENTINEL:after_close` and logged `ERROR_INCOMPATIBLE_DRIVER` / `carb::graphics::createInstance failed`.
- New current blocker is recorded as `vulkan_incompatible_driver`.
- `artifact_manifest` increased to 227 artifacts.
- `master_audit` now reports 189/189 passed artifacts.
- `blocked_gate_audit` now records 7 gates: 4 blocked, 1 needs_review, 1 clear, 1 out_of_scope.

## Verification

The required verification bundle passed:

- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

Verification log: `logs/setup/verification_20260618_isaaclab_live_gate_probe.log` (kept local because `logs/` is ignored).

## Failed / Blocked Items

- `isaaclab_live_headless_gate_ok=false`.
- Current technical blocker: `vulkan_incompatible_driver` in Kit/Isaac Sim graphics foundation.
- Official `whole_body_tracking` replay, tracking task smoke/eval, PPO training, teacher rollout collection, VAE/diffusion closed-loop evaluation, and paper-level Fig. 5/Fig. 6 videos/metrics remain blocked behind the live Kit gate.
- No official BeyondMimic VAE/diffusion checkpoint, no true DAgger rollout logs, no true Fig. 5/Fig. 6 rollout videos, and no real robot results are available.

## Effect on English Reading Report

This round strengthens the reproduction setup and limitations sections: it provides auditable evidence that the conda/package layer is restored, the old inotify issue is no longer the primary blocker on this host, and the remaining simulation blocker is a Vulkan/driver/Kit runtime issue. It also clarifies that current Level B/C simulation work cannot yet be presented as paper-level closed-loop reproduction.

## Next Step

Repair the Isaac Sim Vulkan runtime on the host, then rerun `python3 reproduction/scripts/isaaclab_live_gate_probe.py`. If it reaches the success sentinel without graphics foundation errors, proceed to official `whole_body_tracking` motion replay before PPO.

## Git Commit

Pending at the time this progress file was written.
