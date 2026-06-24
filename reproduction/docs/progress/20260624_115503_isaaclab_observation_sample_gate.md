# Progress Update

## Goal

本轮继续推进训练前 hard gate：捕获官方 IsaacLab `Tracking-Flat-G1-v0` 的 `observation_manager` policy observation 样本，为后续 MuJoCo native observation adapter 的 same-state 数值对齐提供官方基准。当前不启动 PPO/VAE/diffusion 训练，不生成新视频，不声称已有 MuJoCo 视频成功。

## Files Read

- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/mujoco_observation_math_parity_audit.py`
- `reproduction/scripts/tracking_g1_current_task_env_construction_gate.py`
- `reproduction/scripts/isaaclab_current_headless_gate.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/managers/observation_manager.py`
- `res/tracking/g1_current_task_env_construction_gate/tracking_g1_current_task_env_construction_gate.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

## Files Modified

- `reproduction/scripts/isaaclab_observation_manager_sample_gate.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_115503_isaaclab_observation_sample_gate.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/isaaclab_observation_manager_sample_gate.py
python3 reproduction/scripts/isaaclab_observation_manager_sample_gate.py
python3 -m py_compile reproduction/scripts/mujoco_native_observation_adapter_contract.py reproduction/scripts/isaaclab_observation_manager_sample_gate.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
```

## Results

- 成功启动 IsaacLab `AppLauncher(headless=True, enable_cameras=False)`。
- 成功创建官方 `Tracking-Flat-G1-v0` 环境。
- 使用当前 official importer USD：
  - `res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda`
- 使用 robot-order FK-repaired motion：
  - `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/dance1_subject1/motion.npz`
- 成功捕获 official `observation_manager` policy observation：
  - policy obs shape: `[1, 160]`
  - critic obs shape: `[1, 286]`
  - policy terms: `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b`, `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`, `actions`
  - policy term dims: `58 + 3 + 6 + 3 + 3 + 29 + 29 + 29 = 160`
  - motion time steps: `[232]`
  - robot anchor body index: `11`
  - motion anchor body index: `7`
  - official runtime body indexes: `[0, 6, 12, 24, 7, 13, 25, 11, 22, 30, 36, 23, 31, 37]`

## Verification

- `isaaclab_observation_manager_sample_gate.py` 输出：
  - `ok_isaaclab_observation_manager_sample_captured_but_mujoco_parity_pending`
- `mujoco_native_observation_adapter_contract.py` 现在记录：
  - `isaaclab_observation_sample_available=true`
  - `native_obs_adapter_ready=false`
  - `observation_runtime_parity_ready=false`

## Failed / Blocked Items

- MuJoCo native adapter 仍未通过。
- 当前只捕获了 official IsaacLab 样本，还没有在同一 reset/state/last_action 条件下运行 MuJoCo builder 并逐 slice 对比：
  - `command`
  - `motion_anchor_pos_b`
  - `motion_anchor_ori_b`
  - `base_lin_vel`
  - `base_ang_vel`
  - `joint_pos`
  - `joint_vel`
  - `actions`
- 因此当前 PPO/VAE/diffusion MuJoCo 视频仍不能声明为可信 motion-control 复现证据。

## Effect on English Reading Report

本轮为报告提供了更强的工程证据：我们不是凭空猜测 160-D observation，而是已经从官方 IsaacLab `observation_manager` 捕获了 policy observation 的真实 term 顺序、维度、motion time step 和 body index 映射。报告中可以把它写作 deployment adapter audit 的关键进展：official observation sample captured, but MuJoCo numerical parity is still pending。

## Next Step

下一步应写 `mujoco_observation_same_state_parity_audit.py`：读取本轮 official sample，按 motion time step `[232]`、body indexes、last action 和同一 reference motion 构造 MuJoCo-side observation，然后对 8 个 policy slices 做数值误差表。只有这个 gate 通过后，才继续 teacher/VAE/diffusion 的 MuJoCo closed-loop 视频。

## Git Commit

待标准验证通过后提交。当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
