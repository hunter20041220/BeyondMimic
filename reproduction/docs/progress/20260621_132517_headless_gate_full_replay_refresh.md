# Progress Update

## Goal

Re-run the current IsaacLab `AppLauncher(headless=True)` gate on the live server and, once it passed, run the full 40-motion official `replay_npz.py` loop through the official-importer-export G1 USDA path. Refresh report-ready replay plots and the representative reference replay MP4 for the English reading report/PPT.

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
- `reproduction/scripts/isaaclab_current_headless_gate.py`
- `reproduction/scripts/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.py`
- `reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py`
- `reproduction/scripts/official_importer_export_replay_full_dataset_report_assets.py`

## Files Modified

- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json`
- `res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.csv`
- `res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.tsv`
- `res/report_assets/official_importer_export_replay_full_dataset/`
- `res/visualization/official_importer_export_full_dataset_reference_replay/`
- `res/visual_media_inventory/`
- `res/report_assets/visual_evidence_index/`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260621_132517_headless_gate_full_replay_refresh.md`

## Commands Run

```bash
git status --short
git log -3 --oneline
python3 reproduction/scripts/isaaclab_current_headless_gate.py
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7
python3 reproduction/scripts/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.py
python3 reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_replay_full_dataset_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

The first `python3 reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py` attempt failed because the system interpreter lacked `matplotlib`; it was immediately rerun successfully with `envs/bm_analysis/bin/python`.

## Results

- Current IsaacLab headless gate passed on physical GPU 4:
  - `status=ok`
  - `returncode=0`
  - `duration_seconds=17.579`
  - `app_launcher_headless_success_sentinel=true`
  - `payload_is_running=true`
  - no inotify, Vulkan incompatible-driver, traceback, timeout, or GPU-foundation fatal marker
- Full official replay loop refresh passed:
  - `status=ok_official_replay_npz_loop_full_dataset_with_official_importer_export`
  - `row_count=40`
  - `ok_count=40`
  - `failed_count=0`
  - `total_replayed_steps=11960`
  - `shutdown_warning_count=0`
  - `total_duration_seconds=2494.959`
- Refreshed report assets:
  - completion-by-family plot
  - duration-by-motion plot
  - rows/family/summary CSVs
  - representative `walk1_subject1` reference replay MP4 and keyframes

## Verification

The replay/report asset refresh passed before the full repository verification chain:

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_replay_full_dataset_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

Full verification passed after this file was written:

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

Key full-chain results:

- `artifact_manifest.py`: `status=ok`, `1363` artifacts.
- `paper_vs_reproduction_comparison.py`: `status=ok`.
- `final_reproduction_report.py`: `status=ok`.
- `completion_matrix_status_audit.py`: `status=ok`, `199` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `status=ok`, `199` scripts, `0` failures.
- `verification_command_script_manifest.py`: `status=ok`.
- `verification_command_coverage_audit.py`: `status=ok`, `207` commands categorized, `10` smoke commands passed.
- `reproduction_master_audit.py`: `status=ok`.

## Failed / Blocked Items

- The system `python3` interpreter failed to run the reference replay video asset script due to missing `matplotlib`. The correct project environment `envs/bm_analysis/bin/python` succeeded.
- This full replay result is still local virtual reference replay through the captured official-importer-export G1 USDA path. It is not unmodified live converter-entry success, trained policy evaluation, PPO training, DAgger data, Fig. 5/Fig. 6 closed-loop guidance, TensorRT deployment, or real-robot evidence.
- Official BeyondMimic trained checkpoints and real robot artifacts remain absent.

## Effect on English Reading Report

The report can now state that the current server revalidated the IsaacLab headless gate and refreshed a full 40-motion official replay-loop result. The English reading report/PPT can use:

- `res/visualization/official_importer_export_full_dataset_reference_replay/official_importer_export_full_dataset_reference_replay_kinematic.mp4`
- `res/report_assets/official_importer_export_replay_full_dataset/official_importer_export_replay_completion_by_family.png`
- `res/report_assets/official_importer_export_replay_full_dataset/official_importer_export_replay_duration_by_motion.png`
- `res/report_assets/official_importer_export_replay_full_dataset/official_importer_export_replay_full_dataset_summary.csv`

## Next Step

Proceed from replay to the official-importer-export tracking task smoke/eval and then PPO tracking training/evaluation. Formal PPO GPU experiments should use GPU 4 and GPU 7, record GPU memory/runtime/seed/config, and only kill external `wangjc` GPU processes after command-line verification.

## Git Commit

Pending at file creation time; see Git history for the commit containing this progress update.
