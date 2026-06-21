# Storage Cleanup Note

## Scope

This note records conservative storage handling during the robot-order FK PPO evidence round.

## Deleted

| Path | Approx Size | Reason |
| --- | ---: | --- |
| `/mnt/infini-data/test/BeyondMimic/tmp/g1_urdf_simulationapp_in_memory_import/g1_current_stage_export.qXOXHD` | 40 MB | Temporary USD stage export from an old probe; the probe script recreates `ROOT/tmp/g1_urdf_simulationapp_in_memory_import` and the file is not a checkpoint, report asset, manifest, or current best evidence. |

## Not Deleted

- Current robot-order FK PPO checkpoints under `res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training/` were retained because they are the current strongest local virtual tracking baseline.
- Successful teacher-rollout, state-latent, VAE, diffusion, and video artifacts under `res/runs/` were retained. Some are large or duplicated historical runs, but they are referenced by audits or required-artifact absence scans and should only be removed after a dedicated retention manifest is written.
- `res/failed_runs/` was retained because the largest files are only tens of KB and the failure evidence is useful for audit traceability.

## Current Storage Observation

`res/runs/` is still the dominant cleanup candidate at roughly 9.9 GB. The safest next cleanup pass should target duplicated historical successful raw runs after preserving summary JSON/CSV/README evidence, not current robot-order FK PPO artifacts.
