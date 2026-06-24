# IsaacLab Observation Manager Sample Gate

- Status: `ok_isaaclab_observation_manager_sample_captured_but_mujoco_parity_pending`
- Generated: `2026-06-24T04:14:17.162444+00:00`
- Scope: official IsaacLab observation sample only; no MuJoCo parity, no training, no video.
- 当前不得声称完整复现 BeyondMimic；本 gate 只是为下一步 MuJoCo obs adapter parity 提供官方样本。

## Sample

- JSON: `/mnt/infini-data/test/BeyondMimic/res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json`
- NPZ: `/mnt/infini-data/test/BeyondMimic/res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.npz`
- Policy obs dim: `160`
- Policy term names: `['command', 'motion_anchor_pos_b', 'motion_anchor_ori_b', 'base_lin_vel', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']`
- Motion time steps: `[232]`

## Failed / Blocking Checks

- `mujoco_native_parity_ready`

## Interpretation

- 如果本 gate 成功，说明官方 IsaacLab observation_manager 样本已经可捕获。
- 这仍不代表 MuJoCo 160-D observation adapter 正确。
- 下一步必须在同一 reset/state/last_action 条件下对 MuJoCo builder 的 8 个 policy slices 做数值对比。
