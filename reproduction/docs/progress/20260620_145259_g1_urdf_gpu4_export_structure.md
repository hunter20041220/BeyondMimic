# Progress Update

## Goal

Narrow the current official G1 URDF/USD conversion blocker after the GPU4 in-memory importer probe produced a large local USDA export, without claiming official replay or paper-level tracking success.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_export_structure_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/english_reading_report.md`
- Generated audit/report outputs under `/mnt/infini-data/test/BeyondMimic/res/`.

## Commands Run

- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_export_structure_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_export_structure_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_deliverables_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

- The source GPU4 probe records `after_parse_import_in_memory=true`, `vulkan_device_lost=true`, `timed_out=true`, and a local USDA export at `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda`.
- The large export is `311027678` bytes and remains ignored by Git.
- The new structure audit status is `ok_with_physics_usd_export_but_vulkan_device_lost`.
- Structure counts: `40` rigid-body API rows, `1` articulation root, `29` revolute joints, `29` joint-state rows, `29` drive rows, `105` mesh defs, `56` capsule defs.
- All `29/29` audited action joints and `18/18` checked target bodies are present in the exported text.
- Current latest blocker: `official_g1_importer_exports_physics_stage_but_vulkan_device_lost_before_payload_or_replay`.
- Artifact manifest now records `783` artifacts.
- Master audit now records `288/288` passing artifacts.

## Verification

All required verification commands passed before this progress file was added:

- `artifact_manifest.py`: `ok`, `783` artifacts, `0` missing
- `paper_vs_reproduction_comparison.py`: `ok`
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, `170` rows
- `verification_command_syntax_audit.py`: `ok`, `186` scripts, `0` failed
- `verification_command_script_manifest.py`: `ok`, `186` scripts
- `verification_command_coverage_audit.py`: `ok`, `194` commands
- `reproduction_master_audit.py`: `ok`, `288/288` artifacts passed

## Failed / Blocked Items

- The official in-memory importer still hits Vulkan `ERROR_DEVICE_LOST` before payload/clean close.
- The new USDA export has not been wired into unpatched official `csv_to_npz.py` or `replay_npz.py`.
- This is not official replay, not official motion preprocessing success, not PPO tracking training/evaluation, not DAgger, not official VAE/diffusion, not TensorRT deployment, not Fig. 5/Fig. 6 reproduction, and not real-robot evidence.
- Paper-level remaining gaps are unchanged: official G1 replay path, paper-scale PPO teacher evaluation, true DAgger logs, official checkpoints, closed-loop guided diffusion paper videos/metrics, TensorRT/asynchronous deployment, and real robot validation.

## Effect on English Reading Report

The reading report now has a sharper IsaacLab/tracking discussion: the official importer can produce a nonempty G1 physics/articulation stage on GPU4, but the runtime still fails before official replay can consume it. This supports an honest reproduction narrative that distinguishes asset/importer progress from paper-level closed-loop results.

## Next Step

Try to consume the exported official-importer USDA in a bounded project-local replay preflight, or build a non-mutating compatibility audit that compares this export against the enriched USD and official `Tracking-Flat-G1-v0` requirements before attempting another Kit launch.

## Git Commit

Pending at the time this progress file is written; the final commit hash is reported in the user-facing turn summary.
