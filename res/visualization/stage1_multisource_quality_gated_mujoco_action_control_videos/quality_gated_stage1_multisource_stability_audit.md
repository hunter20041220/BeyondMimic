# Quality-Gated Stage-1 MuJoCo Stability Audit

## 结论

本轮修复已经解决了旧视频的 near-floor root target 问题：新选段 root z 正常，reference replay 可以正常站立显示。teacher/VAE/diffusion/guided action-control 在 30 帧短视频里 `fall_proxy_count=0`，但仍存在 root height 下滑，因此只能视为短时诊断通过，不能视为长时稳定控制。

## 选段

- Motion: `lafan1_sprint1_subject4`
- Motion steps: `286550..286579`
- Frames: `30`
- Reward mean: `0.05464879038433234`
- Root z mean: `0.7893778244654338`

## 指标

- Reference root z: min `0.7880`, mean `0.7894`, max `0.7905` m
- Teacher fall proxy count: `0`
- Teacher root height min/max: `0.643982828232582` / `0.7457355254466121` m
- Teacher root position error mean/max: `0.1438565948581903` / `0.22297892348363033` m

## 未解决

- 当前 teacher rollout 没有 `>=60` 帧的正常 root-height 稳定片段；视频只有 30 帧，不应硬拉长。
- action-control 仍使用 MuJoCo position actuators + root assist，不是 native MuJoCo PPO closed-loop obs/action adapter。
- 当前结果不是 BeyondMimic paper-level Fig.5/Fig.6，也不是真实机器人结果。

JSON: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_stability_audit.json`
