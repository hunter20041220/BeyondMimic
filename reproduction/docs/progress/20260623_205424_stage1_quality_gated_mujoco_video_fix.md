# Progress Update

## Goal

修复 Stage-1 multi-source MuJoCo 视频自动选中 near-floor root target 的问题：加入 root height、reward、stability 门槛，先生成 root 高度正常的短 reference replay，再测试 teacher action-control 是否仍然不稳，并把结果写入审计。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/diagnose_stage1_mujoco_video_failure.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_best_teacher_rollout_dataset/tracking_stage1_multisource_best_teacher_rollout_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_video_stability_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_205424_stage1_quality_gated_mujoco_video_fix.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/stage1_multisource_quality_gated_mujoco_action_control_videos.py
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_quality_gated_mujoco_action_control_videos.py
python3 -m py_compile reproduction/scripts/stage1_multisource_quality_gated_video_stability_audit.py
python3 reproduction/scripts/stage1_multisource_quality_gated_video_stability_audit.py
```

## Results

新增 quality-gated 输出目录：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/`

选中片段：

- source motion：`lafan1_sprint1_subject4`
- motion steps：`286550..286579`
- frames：`30`
- reward mean：`0.05464879038433234`
- root z min/mean/max：`0.7880 / 0.7894 / 0.7905 m`

生成视频：

- `reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

关键指标：

- reference root z mean：`0.7894 m`
- teacher `fall_proxy_count=0`，root height min/max：`0.644 / 0.746 m`
- VAE `fall_proxy_count=0`，root height min/max：`0.542 / 0.733 m`
- diffusion `fall_proxy_count=0`，root height min/max：`0.526 / 0.734 m`
- guided `fall_proxy_count=0`，root height min/max：`0.532 / 0.735 m`

## Verification

新增审计：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_selector_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/stage1_multisource_quality_gated_video_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_stability_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_stability_audit.md`

视觉检查：

- reference keyframes 正常站立且居中；
- teacher/guided keyframes 不再贴地崩坏，但末帧出现明显前倾和 root height 下滑。

## Failed / Blocked Items

- 当前 teacher rollout 没有 `>=60` 帧且 root height 正常的稳定连续片段，因此不能生成诚实的长视频。
- action-control 仍是 MuJoCo position actuators + root assist，不是 native MuJoCo PPO obs/action adapter。
- VAE/diffusion/guidance 视频只能作为短时 pipeline diagnostic，不能声称 paper-level closed-loop guidance。

## Effect on English Reading Report

报告中可以写：旧视频失败是 selector/root target 问题；修复后 reference 与短时 control variants 已不再因 near-floor target 立即失败。但也必须写清楚：当前只是 short-horizon local MuJoCo diagnostic，长时运动控制和 paper-level task guidance 仍未完成。

## Next Step

优先处理 Stage-1 teacher 长时稳定性和 MuJoCo obs/action adapter：要么采集更长正常 root-height teacher rollout，要么实现可信的 MuJoCo closed-loop observation adapter，而不是继续硬拉短视频长度。

## Git Commit

本轮尚未提交，等待完整验证链完成后提交。
