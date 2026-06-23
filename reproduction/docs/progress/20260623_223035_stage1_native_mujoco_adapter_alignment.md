# Progress Update

## Goal

继续解决 Stage-1 multi-source MuJoCo 视频站不稳的问题。在 root target selector 已修复后，检查并修复 MuJoCo 侧 PPO observation/action adapter，避免继续 open-loop replay IsaacLab rollout actions。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_action_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/motion_tracking_controller/src/MotionCommand.cpp`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_223035_stage1_native_mujoco_adapter_alignment.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py
```

## Results

新增 approximate native MuJoCo PPO adapter probe：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/native_ppo_obs_adapter_probe/native_ppo_obs_adapter_probe.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/native_ppo_obs_adapter_probe/native_ppo_obs_adapter_probe_metrics.csv`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/native_ppo_obs_adapter_probe/native_ppo_obs_adapter_probe_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_native_adapter_comparison.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_native_adapter_comparison.md`

关键修复：reference frame 对齐。

- 初版 adapter 直接使用 motion bundle global reference，`motion_anchor_pos_b_norm` 首帧约 `8.98 m`。
- 修复后按官方 `MotionCommand.reset()` 语义做 yaw + translation 对齐，首帧 `motion_anchor_pos_b_norm` 降到约 `0.0163 m`。

修复后指标：

- `fall_proxy_count=0`
- root height min/max：`0.6802 / 0.7830 m`
- root position error mean/max：`0.0317 / 0.1062 m`
- 对比旧 open-loop teacher-action replay：root height min 提升 `+0.0362 m`，root position error mean 降低 `-0.1122 m`

## Verification

本文件写入后继续运行完整 artifact/master audit 链。

## Failed / Blocked Items

- 该 probe 仍是 approximate native adapter，不是官方 `motion_tracking_controller` ONNX deployment。
- 仍使用 root assist。
- 只验证 30 帧短片段，没有证明 15 秒长时稳定。
- VAE/diffusion/guidance 尚未迁移到 aligned native adapter。
- 当前不得声称完整复现 BeyondMimic。

## Effect on English Reading Report

报告可以写明一个实质工程修复：MuJoCo adapter 原先没有把 reference motion frame 对齐到当前 robot anchor，导致 policy observation 里出现约 9 米的虚假 anchor error。按官方 MotionCommand 语义修复后，短时 native obs->PPO->action control 不再摔倒，证明复现问题的一部分来自 adapter，而不是 BeyondMimic 方法本身。

## Next Step

把 aligned reference-frame adapter 迁移为统一 MuJoCo rollout runner，然后让 teacher、VAE decoder、diffusion/guidance 都使用这个 runner 做闭环控制，而不是 open-loop action replay。

## Git Commit

本轮尚未提交，等待完整验证链完成后提交。
