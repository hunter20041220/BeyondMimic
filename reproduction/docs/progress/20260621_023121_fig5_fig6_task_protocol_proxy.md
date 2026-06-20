# Progress Update

## Goal

Turn the existing official-importer-export 5-seed closed-loop guidance traces into a stricter Fig.5/Fig.6-adjacent local task-protocol proxy table for the English reading report, without claiming paper-level reproduction.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json`
- Per-task rollout summary JSON and trace NPZ paths referenced from the multiseed summary.

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py 2>&1 | tee logs/official_importer_export_fig5_fig6_task_protocol_proxy.log
```

The final successful run wrote:

```text
res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/fig5_fig6_task_protocol_proxy.json
```

## Results

- Status: `ok_official_importer_export_fig5_fig6_task_protocol_proxy`.
- Rows: `20`.
- Seed groups: `5`.
- Tasks: `joystick`, `waypoint`, `obstacle_avoidance`, `composed`.
- Trace NPZ files found: `20`.
- MP4 paths found: `20`.
- Recorded 299-step completion rate: `1.0`.
- Endpoint/root-reference proxy pass rate: `1.0`.
- Target-body mean proxy pass rate: `1.0`.
- Local task-protocol proxy pass rate: `0.65`.
- Reward improved vs denoised rate: `0.45`.
- Tracking error not worse vs denoised rate: `0.5`.
- Mean final root XY error: `0.005920683296880743` m.
- Paper-level reproduced panel count: `0`.

## Verification

Full verification passed after the new asset was wired into the audit chain:

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

Observed results:

- artifact manifest: `ok`, `1145` artifacts;
- paper-vs-reproduction comparison: `ok`, `191` rows;
- final reproduction report: `ok`;
- completion matrix status audit: `ok`, `182` rows, `0` invalid statuses;
- verification command syntax audit: `ok`, `189` scripts, `0` failed;
- verification command script manifest: `ok`, `189` scripts;
- verification command coverage audit: `ok`, `197` commands;
- reproduction master audit: `ok`, `314` key artifacts.

## Failed / Blocked Items

Two initial script-compatibility failures occurred before the final successful run:

- legacy `seed_group_0_existing` rows had top-level `seed: null`; fixed by falling back to `summary.config.seed`;
- NumPy scalar values were not JSON serializable; fixed by adding a recursive `jsonable()` sanitizer.

Failure note:

```text
res/failed_runs/official_importer_export_fig5_fig6_task_protocol_proxy_initial_failures.md
```

Paper-level gaps remain: official BeyondMimic VAE/diffusion checkpoints, exact Fig.5/Fig.6 protocols, true success/fall/collision metrics, TensorRT/asynchronous deployment traces, mocap/real-world context, and real robot evidence.

## Effect on English Reading Report

This round gives the English report a concrete table for discussing how far the local virtual guided-control evidence reaches toward the paper's Fig.5/Fig.6 claims. It also makes the limitations sharper: the local controller completes the traces and tracks local endpoint/body proxies, but reward/error improvement over the denoised baseline is mixed. The report can therefore present both engineering progress and honest non-reproduction boundaries.

## Next Step

Commit the round, attempt GitHub push, then continue toward protocol-aligned simulated joystick/obstacle/inpainting gates.

## Git Commit

This progress note is intended to be committed with the round using message `report: add fig5 fig6 task protocol proxy`.
