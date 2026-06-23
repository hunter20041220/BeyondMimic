# Progress Update

## Goal

Regenerate the six LAFAN1 MuJoCo videos from a verified-continuous segment and mark the old reset-spliced action-control videos as failed diagnostics.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/lafan1_paper_contract_video_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260622_222256_seed20260804/rank_0/teacher_rollout_shard.npz`
- `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260622_222256_seed20260804/rank_1/teacher_rollout_shard.npz`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`

## Commands Run

```bash
mujoco_mp4/.venv/bin/python -m py_compile reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES= BM_LAFAN1_VIDEO_FPS=30 BM_LAFAN1_MIN_CONTINUOUS_FRAMES=20 mujoco_mp4/.venv/bin/python -u reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,duration,r_frame_rate -of default=noprint_wrappers=1 <each_new_mp4>
```

## Results

The old `res/visualization/lafan1_paper_contract_videos` action-control MP4s were marked as failed diagnostics because the selected rollout context has reset-spliced motion-time-step jumps.

The corrected suite found the longest clean segment:

```text
rank/env: 1/419
source frames: 79:156
motion time steps: 2314..2390
frames: 77
duration: 2.566666666666667 s
done_count: 0
non_plus_one_count: 0
```

New outputs:

- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/reference_action_control/reference_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/guided_latent_action_control/guided_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/lafan1_continuous_video_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/failed_discontinuous_action_control_audit.json`

## Verification

All six new MP4s have `77` frames, `30 FPS`, and duration `2.566667` seconds. The top-level summary reports:

```text
all_mp4_exist=true
all_primary_metrics_csv_exist=true
all_continuous_primary_time_steps=true
no_temporal_stretching=true
old_discontinuous_suite_marked_failed=true
```

## Failed / Blocked Items

- The new videos are short because the current local teacher only provides a short stable continuous segment.
- The current teacher remains weak; these videos fix temporal discontinuity, but they do not prove paper-level motion quality.
- Denoised/guided videos use local VAE/denoiser proxy actions over the continuous segment, not official BeyondMimic Fig. 5/Fig. 6 guidance.
- True Isaac rendered MP4 remains blocked on the H20 Isaac Sim rendering stack.

## Effect on English Reading Report

The report can now separate failed reset-spliced diagnostics from corrected continuous MuJoCo video evidence. This improves the honesty of the reproduction section while still keeping the claim level local and non-paper-level.

## Next Step

After the 5/6 multi-source teacher training finishes, repeat the same continuous-segment gate using the stronger checkpoint and regenerate longer continuous videos only if the teacher can provide longer `done=false`, `+1` motion-time-step segments.

## Git Commit

Pending.
