# Progress Update

## Goal

Generate a LAFAN1 paper-contract video folder after the completed 4/7 Stage 1 PPO run, using the selected best local teacher checkpoint plus the downstream local VAE, diffusion, and guidance artifacts. The goal was to produce MuJoCo control-form videos rather than matplotlib skeleton plots.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_teacher_rollout_vae_training/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json`
- `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- `/mnt/infini-data/test/BeyondMimic/download/official/motion_tracking_controller/config/g1/controllers.yaml`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_trace_mesh_video.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_065329_lafan1_paper_contract_mujoco_videos.md`

## Commands Run

```bash
mujoco_mp4/.venv/bin/python -m py_compile reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= BM_LAFAN1_VIDEO_FRAMES=30 BM_LAFAN1_VIDEO_FPS=30 mujoco_mp4/.venv/bin/python reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= BM_LAFAN1_VIDEO_FRAMES=450 BM_LAFAN1_VIDEO_FPS=30 mujoco_mp4/.venv/bin/python reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py
```

The 450-frame command was logged under `/mnt/infini-data/test/BeyondMimic/logs/lafan1_paper_contract_mujoco_action_control_videos_*.log`.

## Results

New output folder:

`/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/`

Generated local MuJoCo MP4s:

- `reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

Each primary video has a keyframe PNG, three-frame keyframe strip, per-frame metrics CSV, and summary JSON. The top-level summary is:

`/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/lafan1_paper_contract_video_suite_summary.json`

The suite summary reports `all_mp4_exist=true` and `all_metrics_csv_exist_for_primary_videos=true`.

## Verification

Visual sanity checks were performed on keyframe strips for reference, teacher policy, and guided latent videos. The robot is rendered as the MuJoCo G1 mesh on a floor and remains near the fixed camera center due to root XY recentering plus root assist.

The 5/6 multi-source PPO training was checked before and after video generation and remained alive. GPU 5/6 were still occupied by the training workers, with utilization around 75 percent after video generation.

## Failed / Blocked Items

The first smoke run failed because the patched MuJoCo XML was written outside the asset directory, breaking relative mesh paths such as `meshes/left_wrist_pitch_link.STL`. The script was fixed to write the patched XML beside the original G1 XML, preserving mesh-relative paths.

The selected teacher rollout is weak: all sampled envs in the teacher rollout shard had `first_done=0`. The resulting teacher/VAE/diffusion/guided videos are valid MuJoCo action-to-PD visualization evidence, but the motion quality is not paper-level. Fall proxy counts were:

- reference: `0`
- teacher policy: `38`
- VAE reconstructed: `38`
- denoised latent: `59`
- guided latent: `59`

## Effect on English Reading Report

This adds a concrete media asset section for the report: "Local MuJoCo action-control visualization from the LAFAN1 paper-contract chain." It helps explain the full intended pipeline from Stage 1 teacher to VAE/diffusion/guidance and demonstrates control-form MuJoCo rendering, while also honestly showing that the current teacher is weak and does not yet reproduce BeyondMimic at paper level.

Recommended claim level:

`Local MuJoCo action-to-PD visualization evidence from the current weak LAFAN1 paper-contract chain; not official IsaacLab rendered MP4, not official BeyondMimic checkpoint, not paper-level closed-loop Fig.5/Fig.6, and not real robot.`

## Next Step

Let the 5/6 multi-source PPO continue. Once it finishes, run checkpoint sweep on that run and compare whether the multi-source teacher improves over the weak 4/7 LAFAN1 teacher before regenerating videos from a better checkpoint.

## Git Commit

Pending. Large MP4/PNG artifacts are local and should not be committed; only scripts, progress markdown, and small audit/report text should be considered for Git.
