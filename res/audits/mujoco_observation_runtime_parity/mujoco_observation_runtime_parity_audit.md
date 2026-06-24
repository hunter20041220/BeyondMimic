# MuJoCo Observation Runtime Parity Audit

- Status: `blocked_mujoco_injected_state_observation_runtime_parity_mismatch`
- Generated: `2026-06-24T04:46:17.532848+00:00`
- Scope: load MuJoCo G1, inject captured IsaacLab state, call `mj_forward`, compare 160-D observation slices.
- 当前不得声称完整复现 BeyondMimic；本审计不是训练、不是 rollout、不是视频成功。

## Term Errors

- `command` dim=58 max_abs_error=0.000000e+00 passed=`True` source=`motion_file_or_last_action`
- `motion_anchor_pos_b` dim=3 max_abs_error=5.219149e-03 passed=`False` source=`mujoco_runtime_injected_state`
- `motion_anchor_ori_b` dim=6 max_abs_error=3.175157e-01 passed=`False` source=`mujoco_runtime_injected_state`
- `base_lin_vel` dim=3 max_abs_error=3.092577e-08 passed=`True` source=`mujoco_runtime_injected_state`
- `base_ang_vel` dim=3 max_abs_error=8.795058e-08 passed=`True` source=`mujoco_runtime_injected_state`
- `joint_pos` dim=29 max_abs_error=2.700835e-08 passed=`True` source=`mujoco_runtime_injected_state`
- `joint_vel` dim=29 max_abs_error=0.000000e+00 passed=`True` source=`mujoco_runtime_injected_state`
- `actions` dim=29 max_abs_error=0.000000e+00 passed=`True` source=`motion_file_or_last_action`

## Failed / Blocking Checks

- `mujoco_anchor_pose_matches_isaaclab_sample`
- `candidate_mujoco_models_any_anchor_orientation_matches_isaaclab`
- `all_runtime_observation_slices_pass`

## Interpretation

- 这个 gate 比 pure formula parity 更强，因为它真的加载 MuJoCo model 并读取 `data.xpos/xquat/qpos/qvel`。
- 但它仍然只是 injected-state adapter parity，没有执行 policy closed-loop、没有 `mj_step` 物理稳定性结论。
- 如果该 gate 失败，当前前倾/不抬腿视频很可能仍由 MuJoCo body frame/default pose/joint order/velocity frame mismatch 导致。
