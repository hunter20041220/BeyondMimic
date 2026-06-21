# Progress Update

## Goal

Refresh the scaled-PPO task-conditioned latent-guidance multi-seed closed-loop evidence after the latest scaled PPO, VAE, state-latent diffusion, offline guidance, and single-seed video refresh chain. Keep the claim level local-virtual and audit-facing only.

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
- `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py`
- `reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py`
- `reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py`
- `reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py`

## Files Modified

- `res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json`
- `res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json`
- `res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/seed_group_{1,2,3,4}/*/*_task_conditioned_latent_guidance_rollout_eval.json`
- `res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/seed_group_{1,2,3,4}/seed_group_*_importer_export_task_conditioned_latent_guidance_rollout_eval.json`
- `res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/*`
- `res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/*`
- `res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/*`
- `res/visual_media_inventory/visual_media_inventory_audit.json`
- `res/visual_media_inventory/visual_media_inventory_audit.tsv`
- `res/report_assets/visual_evidence_index/visual_evidence_index.json`
- `reproduction/docs/progress/20260621_153941_scaled_ppo_multiseed_guidance_refresh.md`
- `reproduction/scripts/artifact_manifest.py`

## Commands Run

```bash
python3 reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py
python3 reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

## Results

- Multi-seed rollout status: `ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval`.
- Rows: 20 local virtual closed-loop rollouts.
- Seed groups: 5, including the previously refreshed single-seed baseline plus 4 re-run seed groups.
- Tasks: joystick, waypoint, obstacle avoidance, composed.
- MP4 evidence paths: 20/20 present, minimum observed MP4 size 2,621,247 bytes.
- Total rollout variant steps: 23,920.
- Task-protocol proxy status: `ok_official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy`.
- Task-protocol proxy rates: 299-step completion 1.0, guidance-signal-positive 1.0, endpoint proxy pass 1.0, local task protocol proxy pass 0.8.
- Success/fall/collision proxy status: `ok_official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy`.
- Success/fall/collision proxy rates: success proxy 0.9, fall-height proxy 0.1, body-error-spike anomaly proxy 0.05, completed-299 proxy 1.0.
- Visual media inventory after refresh: 471 media files, including 85 videos.
- Visual evidence index after refresh: 31 indexed videos.

These are local virtual closed-loop evidence artifacts using local scaled PPO/VAE/denoiser/guidance checkpoints and proxy task costs. They are not official BeyondMimic checkpoints, not paper Fig. 5/Fig. 6 reproduction, not TensorRT deployment evidence, and not real-robot evidence.

## Verification

The full verification chain passed after this progress file was registered in the artifact manifest:

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

- `artifact_manifest.py`: ok, 1369 artifacts.
- `paper_vs_reproduction_comparison.py`: ok.
- `final_reproduction_report.py`: ok.
- `completion_matrix_status_audit.py`: ok, 199 rows, 0 invalid statuses.
- `verification_command_syntax_audit.py`: ok, 199 scripts, 0 failed syntax checks.
- `verification_command_script_manifest.py`: ok, 199 scripts.
- `verification_command_coverage_audit.py`: ok, 207 commands, 10 smoke-pass commands.
- `reproduction_master_audit.py`: ok.

## Failed / Blocked Items

- No command failed in the scaled-PPO multi-seed guidance refresh itself.
- GPU use was light and selected GPU 4 for rollout/render workers; this was not a formal two-GPU training experiment.
- Paper-level blockers remain: no official BeyondMimic VAE/diffusion checkpoint, no true paper Fig. 5/Fig. 6 rollout videos, no true DAgger rollout logs, no TensorRT/asynchronous real deployment timing, and no real Unitree G1 hardware validation.

## Effect on English Reading Report

This round strengthens the reproducibility evidence for the report's code reproduction section: the project now has refreshed 5-seed, 4-task local virtual guidance evidence linked to the latest scaled-PPO chain, plus explicit proxy metrics that help explain what can be validated in simulation and what remains outside public reproduction.

## Next Step

Refresh the full audit chain, commit only code/docs/small JSON/CSV/PNG evidence, push to GitHub, then return to the remaining IsaacLab live gate and official task evaluation blockers.

## Git Commit

Committed with this progress update; see Git history for the exact hash.
