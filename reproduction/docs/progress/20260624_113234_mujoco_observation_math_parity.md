# Progress Update

## Goal

本轮目标是继续执行训练前 hard gate：审计 MuJoCo native PPO/VAE/diffusion 闭环所依赖的 160-D observation adapter 是否和 BeyondMimic/whole_body_tracking 官方 IsaacLab 观测公式一致。当前不启动新训练、不生成新成功视频、不把已有失败视频改口成成功。

## Files Read

- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/utils/math.py`
- `res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`

## Files Modified

- `reproduction/scripts/mujoco_observation_math_parity_audit.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_113234_mujoco_observation_math_parity.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_observation_math_parity_audit.py
python3 reproduction/scripts/mujoco_observation_math_parity_audit.py
python3 -m py_compile reproduction/scripts/mujoco_native_observation_adapter_contract.py reproduction/scripts/mujoco_observation_math_parity_audit.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
```

## Results

- 新增 `mujoco_observation_math_parity_audit.py`，用 `envs/bm_tracking/bin/python` 调用 IsaacLab 官方 `isaaclab.utils.math`，对本地 NumPy quaternion/frame helper 做公式级数值对齐。
- 已验证的公式包括：
  - `matrix_from_quat`
  - yaw-only quaternion
  - `subtract_frame_transforms`
  - Rot6D flatten order
  - body-frame linear/angular velocity conversion
  - `MotionCommand` delta-yaw alignment 的 quaternion 公式
- 最大绝对误差均为浮点舍入量级：
  - `matrix_from_quat`: `4.44e-16`
  - `yaw_quat_sign_invariant`: `2.50e-16`
  - `subtract_frame_transforms_position`: `9.99e-16`
  - `rot6_flatten_order`: `6.66e-16`
  - `base_velocity_body_frame`: `8.88e-16`
- 这说明当前本地基础数学 helper 没有明显 quaternion 顺序、frame subtraction 或 Rot6D 列顺序写反的问题。

## Verification

- `mujoco_observation_math_parity_audit.py` 输出：
  - `blocked_observation_runtime_parity_missing_but_math_fixtures_pass`
- `mujoco_native_observation_adapter_contract.py` 输出：
  - `blocked_native_mujoco_observation_adapter_not_validated`
- 已把 action adapter 的旧 blocker 修正：上一轮 no-action-clipping XML 已经解除 MuJoCo ctrlrange clipping 风险，因此当前主要 blocker 收敛到 observation runtime parity。

## Failed / Blocked Items

- 仍没有捕获同一状态下的 official IsaacLab `observation_manager` 160-D policy observation sample。
- 仍没有逐 slice 对比以下 8 个 policy obs term：
  - `command`
  - `motion_anchor_pos_b`
  - `motion_anchor_ori_b`
  - `base_lin_vel`
  - `base_ang_vel`
  - `joint_pos`
  - `joint_vel`
  - `actions`
- 当前 MuJoCo native probe 仍带有 `world_to_init`/centered visualization 对齐逻辑，尚未证明等价于官方训练时 `MotionCommand` 的 runtime observation_manager 输出。
- 因此当前不能继续把 PPO/VAE/diffusion MuJoCo 视频声明成可信 motion-control 复现证据。

## Effect on English Reading Report

本轮给报告提供了一条更诚实的失败原因链：失败视频不是简单“模型弱”或“MuJoCo 不行”，而是 native deployment adapter 还没有完成训练分布一致性验证。可以在 reproduction limitation 中写明：action formula 和基础 observation math 已经过公式级审计，但 runtime observation_manager parity 仍未通过，所以 current MuJoCo PPO/VAE/diffusion closed-loop videos 仍只能算 diagnostic evidence。

## Next Step

下一步应写 IsaacLab observation sample capture gate：在 `Tracking-Flat-G1-v0` 或等价官方环境中，为同一 motion phase、robot state、last_action 捕获 official observation_manager 输出，然后和 MuJoCo native observation builder 逐 slice 比较。只有这个 gate 通过后，才继续跑新的 teacher/VAE/diffusion 闭环视频。

## Git Commit

待标准验证通过后提交。当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
