# Failure Analysis

Current videos cannot be described as successful BeyondMimic reproduction. The latest six MuJoCo videos are continuous and generated through simulation stepping, but they are diagnostics. They still show poor motion quality, high fall proxies, and MuJoCo instability warnings such as QACC warnings.

## Evidence

- Video suite: `res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_continuous_video_suite_summary.json`
- Checks: `{"all_continuous_primary_time_steps": true, "all_mp4_exist": true, "all_primary_metrics_csv_exist": true, "does_not_claim_complete_beyondmimic_reproduction": true, "does_not_claim_real_robot": true, "selected_segment_single_source_motion": true}`
- Selected segment: `{"continuity": {"all_motion_time_step_deltas_plus_one": true, "done_count": 0, "env_index": 606, "frames": 298, "motion_time_step_end": 418474, "motion_time_step_start": 418177, "no_temporal_stretching": true, "non_plus_one_count": 0, "rank": 1, "selection_rule": "longest segment with done_count == 0 and motion_time_steps strictly consecutive +1", "source_end_exclusive": 299, "source_shard": "/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260623_055602_seed20260854/rank_1/teacher_rollout_shard.npz", "source_start": 1, "timeout_count": 0}, "done_count": 0, "duration_seconds": 9.933333333333334, "end_exclusive": 299, "env_index": 606, "length": 298, "motion_time_step_end": 418474, "motion_time_step_start": 418177, "rank": 1, "reward_mean": -0.08196809452227098, "shard": "/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260623_055602_seed20260854/rank_1/teacher_rollout_shard.npz", "single_source_motion_boundary_ok": true, "source_motion": {"duration_seconds": 246.6, "end_frame_exclusive": 420595, "frame_count": 12330, "motion": "`
- MuJoCo warning log: `logs/mujoco/MUJOCO_LOG_stage1_multisource_continuous_rerun_20260623_143500.txt`

## Data Layer

Evidence: the current bundle includes `49` motions and explicitly records skipped sources. It does not silently pad PBHC/ASAP 23-DoF sources into 29-DoF G1 actions.

Possible causes:

- Some motions may be kinematically valid but dynamically difficult.
- Root frame / ground height may still differ from the training model.
- The exact paper curated 2.5h dataset is not fully reconstructed.

How to verify:

- Replay each source motion with FK and ground-contact checks.
- Plot ankle/wrist/root heights for each motion.
- Reject impossible clips before PPO training.

## Teacher Policy Layer

Evidence: best 5/6 checkpoint reward mean is `0.024131401152315747`, body-position error mean is `1.0095036663737982`, and non-timeout done rate is `0.19413670568561872`.

Likely impact: downstream VAE and diffusion imitate weak behavior. Better denoising cannot rescue a failed teacher distribution.

How to verify:

- Inspect reward component breakdown and termination reason distribution.
- Compare action scale / PD targets / reset pose against official whole_body_tracking config.
- Run smaller single-motion training until the teacher visibly tracks before multi-source training.

## VAE Layer

Evidence: VAE test action MSE is `0.003289680986199528`. This is good offline reconstruction but not a proof of closed-loop VAE stability.

Risks:

- Offline reconstruction may match bad teacher actions.
- KL collapse / latent degeneracy may still occur.
- Full DAgger is not reproduced from official logs.

## Diffusion Layer

Evidence: test pred token MSE improves from `0.07281625297452722` to `0.04322136765612023`.

Risks:

- Token MSE does not enforce physically executable trajectory.
- Inverse transform / scaling / horizon alignment may be wrong.
- The denoiser is trained from weak-teacher trajectories.

## Guidance Layer

Evidence: offline task-cost gradients are nonzero and best costs improve for proxy tasks, but no current paper-level receding-horizon closed-loop task success is available for this chain.

Risks:

- Guidance may act on token variables that do not map cleanly to stable actions.
- Guidance scale can break the learned distribution.
- Current implementation is not the paper TensorRT/asynchronous deployment path.

## MuJoCo / Isaac Deployment Layer

Risks to audit:

| risk | current evidence | how to test | priority |
|---|---|---|---|
| joint order mismatch | action-scale and G1 mapping audits exist but video instability remains | one-joint impulse test and compare visual joint | high |
| action scale mismatch | theta_sp formula is recorded in video summaries | sweep action scale while holding reference action | high |
| default pose mismatch | controller default position is imported from motion_tracking_controller config | compare default pose in Isaac/MuJoCo | high |
| PD gain mismatch | local MuJoCo uses position actuators/root assist | audit gain/armature against official controller | high |
| root frame mismatch | reference/video recenters root XY for display | compare root transform convention with training obs | high |
| observation normalization mismatch | PPO obs adapter is local | log obs mean/std and compare training normalization | high |
| diffusion inverse transform mismatch | local token windows are custom | decode known training windows and compare action reconstruction | medium |

## Priority Fix

Do not optimize the report videos first. The first fix is to make a single-motion Stage 1 teacher reliably track in physics, then recollect rollout data and retrain VAE/diffusion. Longer videos will only make weak-control failure more visible.
