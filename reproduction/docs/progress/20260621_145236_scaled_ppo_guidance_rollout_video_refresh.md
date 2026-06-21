# Progress Update

## Goal

Refresh the scaled-PPO task-conditioned closed-loop guidance rollout videos after the refreshed scaled-PPO VAE/state-latent/diffusion/guidance chain. The goal is to produce report- and PPT-facing local virtual MP4 evidence for joystick, waypoint, obstacle avoidance, and composed proxy tasks, then refresh Fig. 5/Fig. 6 proxy tables and visual-evidence audits without claiming official paper-level BeyondMimic reproduction.

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
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/visual_media_inventory_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/visual_evidence_index.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260621_145236_scaled_ppo_guidance_rollout_video_refresh.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/joystick/joystick_task_conditioned_latent_guidance_rollout_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/waypoint/waypoint_task_conditioned_latent_guidance_rollout_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/obstacle_avoidance/obstacle_avoidance_task_conditioned_latent_guidance_rollout_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/composed/composed_task_conditioned_latent_guidance_rollout_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/fig5_fig6_task_protocol_proxy.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/fig5_fig6_task_protocol_proxy_rows.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/success_fall_collision_proxy.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/success_fall_collision_proxy_rows.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/visual_evidence_index/visual_evidence_index.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.tsv`

Large MP4 videos, trace NPZ files, generated worker scripts, and raw logs remain local under `res/visualization/`, `res/runs/`, and `logs/`; they are intentionally not staged for GitHub. Their paths, metrics, and claim levels are recorded in JSON/CSV/Markdown artifacts.

## Commands Run

- `git status --short && git log -3 --oneline`
- `wc -l goal.md README.md reproduction/PROGRESS.md reproduction/RUNBOOK.md reproduction/docs/final_reproduction_report.md reproduction/docs/known_limitations.md reproduction/docs/experiment_protocol.md res/comparison/paper_vs_reproduction.json res/artifact_manifest/artifact_manifest.json res/master_audit/reproduction_master_audit.json res/required_artifact_absence/required_artifact_absence_audit.json`
- `find reproduction/scripts -maxdepth 1 -type f | rg 'scaled_ppo.*(task_conditioned|guidance|rollout|video|fig5|fig6|success|visual|report_assets)' | sort`
- `jq '{status, generated_at, input_statuses, metrics, checks, aggregate, outputs, report_assets}' res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json`
- `find res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout -maxdepth 3 -type f -printf '%s %p\n' | sort -nr | head -40`
- `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7`
- `python3 reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.py`
- `ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 <mp4>`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_report_assets.py`
- `python3 reproduction/scripts/visual_media_inventory_audit.py`
- `python3 reproduction/scripts/visual_evidence_index.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- Single-seed scaled-PPO task-conditioned rollout status: `ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval`.
- Input chain: scaled PPO training run `ok`, checkpoint eval `ok`, scaled VAE `ok`, scaled diffusion denoiser `ok`, scaled offline guidance `ok`.
- Tasks refreshed: joystick, waypoint, obstacle_avoidance, composed.
- Rollout length: `299` steps for each task.
- MP4s refreshed and validated by ffprobe:
  - `/mnt/infini-data/test/BeyondMimic/res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout/joystick/official_csv_loop_task_conditioned_latent_guidance_rollout_vs_reference.mp4`, `2729052` bytes, `9.967` seconds.
  - `/mnt/infini-data/test/BeyondMimic/res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout/waypoint/official_csv_loop_task_conditioned_latent_guidance_rollout_vs_reference.mp4`, `2761667` bytes, `9.967` seconds.
  - `/mnt/infini-data/test/BeyondMimic/res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout/obstacle_avoidance/official_csv_loop_task_conditioned_latent_guidance_rollout_vs_reference.mp4`, `2703250` bytes, `9.967` seconds.
  - `/mnt/infini-data/test/BeyondMimic/res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout/composed/official_csv_loop_task_conditioned_latent_guidance_rollout_vs_reference.mp4`, `2646066` bytes, `9.967` seconds.
- Task metrics:
  - Joystick: guided reward mean `0.022449076233313336`, target-body error mean `0.3439415395259857`, guidance cost delta mean `6.848652177431113e-05`.
  - Waypoint: guided reward mean `0.025156304768183858`, target-body error mean `0.3440071940422058`, guidance cost delta mean `8.282827892431049e-06`.
  - Obstacle avoidance: guided reward mean `0.0229376406832458`, target-body error mean `0.34300488233566284`, guidance cost delta mean `8.742658290575978e-07`.
  - Composed: guided reward mean `0.025132083756082932`, target-body error mean `0.3445764183998108`, guidance cost delta mean `6.488199816100972e-05`.
- Scaled-PPO Fig. 5/Fig. 6 local task-protocol proxy refreshed: `20` rows, `5` seed groups, `20` MP4 paths, local task-protocol proxy pass rate `0.8`, paper-level reproduced panel count `0`.
- Scaled-PPO success/fall/collision proxy refreshed: local success proxy rate `0.9`, fall-height proxy rate `0.1`, body-error spike anomaly proxy rate `0.05`; true paper collision/contact signal remains unavailable.
- Visual media inventory refreshed: `471` media files, `85` videos.
- Visual evidence index refreshed: `31` core videos.

## Verification

- `python3 reproduction/scripts/artifact_manifest.py`: passed, `artifact_count=1367` before adding this progress artifact.
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`: passed.
- `python3 reproduction/scripts/final_reproduction_report.py`: passed.
- `python3 reproduction/scripts/completion_matrix_status_audit.py`: passed, `rows=199`, `invalid_status_count=0`.
- `python3 reproduction/scripts/verification_command_syntax_audit.py`: passed, `scripts=199`, `failed=0`.
- `python3 reproduction/scripts/verification_command_script_manifest.py`: passed, `scripts=199`.
- `python3 reproduction/scripts/verification_command_coverage_audit.py`: passed, `commands=207`, `smoke_pass=10`.
- `python3 reproduction/scripts/reproduction_master_audit.py`: passed, `status=ok`.

After this progress file is added to the manifest list, `artifact_manifest.py` and `reproduction_master_audit.py` must be rerun once more before commit.

## Failed / Blocked Items

- No command failed in this round.
- MP4s are local visual evidence and intentionally not committed to GitHub because they are large binary media. Their paths, durations, sizes, metrics, claim levels, and limitations are recorded in small JSON/CSV/Markdown artifacts.
- The refreshed closed-loop rollouts are local virtual official-importer-export scaled-PPO guidance rollouts, not official BeyondMimic Fig. 5/Fig. 6 results, not TensorRT/asynchronous deployment, and not real-robot evidence.
- Still missing official DAgger rollout logs, official BeyondMimic VAE/diffusion checkpoints, paper-level success/fall/collision protocol metrics, true collision/contact labels, TensorRT deployment evidence, and real-robot results.

## Effect on English Reading Report

This round materially improves the English reading report and PPT evidence. It gives four refreshed robot-motion MP4s for task-conditioned guidance plus updated proxy tables for Fig. 5/Fig. 6-style discussion. The report can now show concrete local virtual guided rollouts while clearly stating that the project still does not reproduce the official paper-level BeyondMimic closed-loop or real-robot results.

## Next Step

The next best paper-facing step is to either rerun the 5-seed scaled-PPO task-conditioned guidance multiseed rollout with the refreshed denoiser, or create a compact contact sheet/index for the four refreshed single-seed MP4s so they can be inserted directly into the English report/PPT. If runtime permits, the multiseed refresh is stronger evidence; if not, the contact-sheet/report packaging is the fastest reporting upgrade.

## Git Commit

Pending at time of writing. Commit message target: `feat: refresh scaled ppo guidance rollout videos`.
