# Progress Update

## Goal

Capture a real local virtual policy rollout visualization from the official csv-loop PPO checkpoint, rather than relying only on reference-motion or offline metric plots.

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
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `reproduction/third_party/official/whole_body_tracking/.../mdp/observations.py`
- `res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_policy_rollout_video_capture.py`
- `reproduction/scripts/visual_media_inventory_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Generated/updated report and audit outputs under `res/`.

## Commands Run

- `python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_policy_rollout_video_capture.py`
- `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7`
- `python3 reproduction/scripts/tracking_g1_official_csv_loop_policy_rollout_video_capture.py`
- `envs/bm_analysis/bin/python res/visualization/official_csv_loop_policy_rollout/tracking_g1_official_csv_loop_policy_rollout_render.py`
- `python3 reproduction/scripts/visual_media_inventory_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`
- Full verification sequence after this progress note.

## Results

- Captured a 299-step, one-environment local virtual policy rollout from the official csv-loop PPO checkpoint.
- Recorded robot/reference target-body positions from `MotionCommand` tensors during Tracking-Flat-G1-v0 execution.
- Generated policy-vs-reference MP4, keyframes, metrics CSV, README, capture JSON, and asset JSON under `res/visualization/official_csv_loop_policy_rollout/`.
- The first capture succeeded but render failed because the trace stores 14 target bodies rather than 40 URDF bodies. I fixed the renderer to support both 14-target-body and 40-body traces and reran the render successfully.
- Final policy rollout video metrics:
  - frames: `299`
  - bodies: `14`
  - target bodies: `14`
  - reward mean: `0.026445439085364342`
  - done count: `25`
  - action abs mean: `0.49431127309799194`
  - target-body error mean: `0.08224783092737198`
  - target-body error max: `0.24832472205162048`
- Visual media inventory now records `2` local video files.
- Artifact manifest increased to `384` artifacts.
- Master audit passed after adding policy rollout capture/video checks.

## Verification

The policy rollout capture JSON verifies GPU selection, 299 rollout steps, capture success, render success, and no paper-level/Fig.5/Fig.6/robot overclaiming. The video asset JSON records MP4/keyframe hashes, sizes, metrics, and claim level.

## Failed / Blocked Items

- This is still a local virtual resource-adjusted policy rollout video, not unpatched official replay.
- It is not BeyondMimic Fig. 5/Fig. 6 guided diffusion, not TensorRT/asynchronous deployment, and not real robot evidence.
- The official paper-level DAgger dataset/checkpoints and closed-loop guided diffusion metrics remain missing.

## Effect on English Reading Report

This is the strongest visual artifact so far for the reproduction section. The report can now show an actual robot policy rollout versus reference motion, not only static audits or reference-only motion replay.

## Next Step

Use this policy rollout video in the report/PPT and continue toward closed-loop VAE/diffusion guidance: denoiser sample -> VAE decoder -> action interface -> Tracking-Flat-G1-v0 rollout -> policy-vs-guided metrics/video.

## Git Commit

Pending at the time of writing this progress file.
