# Visual Appendix for the BeyondMimic Reading Report

This appendix summarizes local visual evidence that can be used in the English reading report or presentation. It is intentionally claim-bounded: these assets demonstrate local virtual reproduction progress under the documented resource-adjusted IsaacLab pipeline, not official paper-level BeyondMimic results.

## Recommended Assets

| Section | Primary asset | Supporting plot/table | Suggested use |
|---|---|---|---|
| Reference motion replay | `res/visualization/official_csv_loop_reference_replay/official_csv_loop_reference_replay_kinematic.mp4` | `res/visualization/official_csv_loop_reference_replay/official_csv_loop_reference_replay_keyframes.png` | Show that public CSV motion data can be converted into a local visual replay artifact. |
| Local PPO policy rollout | `res/visualization/official_csv_loop_full_bundle_policy_rollout/official_csv_loop_policy_rollout_vs_reference.mp4` | `res/report_assets/official_csv_loop_full_bundle_ppo_checkpoint_eval/reward_done_timeseries.png` | Show local virtual tracking-policy behavior after the full public-motion bundle PPO run. |
| Local VAE closed-loop reconstruction | `res/visualization/official_csv_loop_vae_closed_loop_rollout/official_csv_loop_vae_closed_loop_rollout_vs_reference.mp4` | `res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/vae_closed_loop_shard_summary.csv` | Explain how the conditional action VAE reconstructs teacher actions in closed-loop IsaacLab rollout. |
| Receding-horizon latent guidance | `res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/official_csv_loop_receding_latent_guidance_rollout_vs_reference.mp4` | `res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/official_csv_loop_receding_latent_guidance_rollout_metrics.png` | Present the strongest current local bridge toward the paper's guided latent diffusion controller. |
| Task-conditioned local guidance | `res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/full_bundle_task_conditioned_guidance_multiseed_bars.png` | `res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/full_bundle_task_conditioned_guidance_multiseed_aggregate.csv` | Summarize joystick, waypoint, obstacle-avoidance, and composed local proxy tasks across seed groups. |
| Official-importer-export tracking task diagnostic | `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_tracking_errors.png` | `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_completion_table.csv` | Report the current 40/40 official-importer-export G1 task-construction and zero-action diagnostic, including reward, reset, termination, and tracking-error contracts. |
| Official-importer-export task-conditioned guidance videos | `res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/importer_export_guidance_video_contact_sheet.png` | `res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/importer_export_guidance_video_index.csv` | Show the report-ready contact sheet and index for 20 local MP4 rollouts across 4 tasks and 5 seed groups; the MP4s remain local and are not GitHub artifacts. |
| Visual evidence inventory | `res/report_assets/visual_evidence_index/visual_evidence_index.md` | `res/visual_media_inventory/visual_media_inventory_audit.json` | Audit trail for all indexed videos, plots, tables, paths, sizes, and claim boundaries. |

## What These Assets Support

- The local IsaacLab/Isaac Sim pipeline can produce visual robot rollout artifacts after the headless gate.
- The project has progressed beyond static audits into replay, task-contract evaluation, local PPO policy rollout, teacher rollout data, VAE reconstruction, and local receding-horizon/task-conditioned latent guidance.
- The report can show both quantitative plots and robot-motion videos rather than only JSON logs.
- Each asset is tied to a machine-readable audit or asset JSON so the report can cite reproducible evidence paths.

## What These Assets Do Not Support

- They do not prove official BeyondMimic paper-level Fig. 5 or Fig. 6 reproduction.
- They do not use official unpublished BeyondMimic VAE/diffusion checkpoints.
- They do not prove official DAgger rollout data or paper-scale teacher-policy quality.
- They do not prove TensorRT/asynchronous deployment on the paper's hardware.
- They do not contain real Unitree G1 robot execution.

## Suggested Reading Report Wording

> I generated local visualization evidence for reference motion replay, a resource-adjusted PPO tracking-policy rollout, VAE action reconstruction, and receding-horizon latent-guidance rollouts. These videos are useful for understanding the BeyondMimic pipeline and for demonstrating that the reproduction reached live IsaacLab execution. However, they should be interpreted as local virtual evidence under a resource-adjusted asset path, not as official reproduction of the paper's closed-loop Fig. 5/Fig. 6 results.

## Audit Links

- `res/report_assets/visual_evidence_index/visual_evidence_index.json`
- `res/report_assets/visual_evidence_index/visual_evidence_index.csv`
- `res/report_assets/visual_evidence_index/visual_evidence_index.md`
- `res/visual_media_inventory/visual_media_inventory_audit.json`
- `res/visual_media_inventory/visual_media_inventory_audit.tsv`
- `res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/importer_export_guidance_video_index.json`
- `res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/importer_export_guidance_video_contact_sheet.png`

Current status: this project still does not fully reproduce BeyondMimic at paper level.
