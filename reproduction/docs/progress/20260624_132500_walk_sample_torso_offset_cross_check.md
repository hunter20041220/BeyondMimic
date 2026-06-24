# Progress Update

## Goal

继续执行“先审公式、参数、obs/action adapter，再训练/视频”的主线。本轮目标是用一个低动态、non-terminated 的 walk 样本复验上一轮从 terminated dance 样本反推出的 MuJoCo->IsaacLab `torso_link` offset 是否稳定。

## Files Read

- `reproduction/scripts/isaaclab_observation_manager_sample_gate.py`
- `reproduction/scripts/mujoco_observation_runtime_parity_audit.py`
- `reproduction/scripts/mujoco_torso_frame_offset_audit.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml`
- `res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda`

## Files Modified

- Updated `reproduction/scripts/isaaclab_observation_manager_sample_gate.py` so the motion/output directory and capture mode can be configured without overwriting the existing dance sample.
- Updated `reproduction/scripts/mujoco_observation_runtime_parity_audit.py` to accept an alternate sample/output directory.
- Updated `reproduction/scripts/mujoco_torso_frame_offset_audit.py` to accept an alternate sample/output directory and to distinguish terminated vs non-terminated single-sample status.
- Added `reproduction/scripts/mujoco_torso_frame_offset_cross_sample_audit.py`.
- Updated `reproduction/scripts/mujoco_native_observation_adapter_contract.py`.
- Updated `reproduction/scripts/artifact_manifest.py`.

## Commands Run

```bash
BM_ISAACLAB_OBS_SAMPLE_GPU=4 \
BM_ISAACLAB_OBS_SAMPLE_AFTER_ZERO_STEP=0 \
BM_ISAACLAB_OBS_SAMPLE_OUT=/mnt/infini-data/test/BeyondMimic/res/audits/isaaclab_observation_manager_walk_sample_gate \
BM_ISAACLAB_OBS_SAMPLE_MOTION=/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz \
python3 reproduction/scripts/isaaclab_observation_manager_sample_gate.py

BM_MUJOCO_OBS_RUNTIME_PARITY_SAMPLE=/mnt/infini-data/test/BeyondMimic/res/audits/isaaclab_observation_manager_walk_sample_gate/isaaclab_policy_obs_sample.json \
BM_MUJOCO_OBS_RUNTIME_PARITY_OUT=/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_observation_runtime_parity_walk_sample \
python3 reproduction/scripts/mujoco_observation_runtime_parity_audit.py

BM_MUJOCO_TORSO_FRAME_OFFSET_SAMPLE=/mnt/infini-data/test/BeyondMimic/res/audits/isaaclab_observation_manager_walk_sample_gate/isaaclab_policy_obs_sample.json \
BM_MUJOCO_TORSO_FRAME_OFFSET_OUT=/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_torso_frame_offset_walk_sample \
python3 reproduction/scripts/mujoco_torso_frame_offset_audit.py

python3 reproduction/scripts/mujoco_torso_frame_offset_cross_sample_audit.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
```

## Results

新增 walk 样本和 cross-sample 审计：

- `res/audits/isaaclab_observation_manager_walk_sample_gate/isaaclab_policy_obs_sample.json`
- `res/audits/mujoco_observation_runtime_parity_walk_sample/mujoco_observation_runtime_parity_audit.json`
- `res/audits/mujoco_torso_frame_offset_walk_sample/mujoco_torso_frame_offset_audit.json`
- `res/audits/mujoco_torso_frame_offset_cross_sample/mujoco_torso_frame_offset_cross_sample_audit.json`

walk 样本质量：

- motion: `walk1_subject1`
- capture mode: `after_reset_no_step`
- motion_time_steps: `[157]`
- terminated: `false`
- command metrics: anchor/body/joint errors all `0.0`

walk MuJoCo runtime parity:

- command, base velocity, joint position, joint velocity, actions pass.
- `motion_anchor_pos_b` error remains `0.005084645669435443`.
- `motion_anchor_ori_b` error remains `0.14584921251276103`.

single-sample walk offset can restore the walk sample:

- raw pos/orient error: `0.005084645669435443` / `0.14584921251276103`
- corrected pos/orient error: `7.555605446851743e-10` / `4.4325126062616516e-08`
- walk q offset: `[0.9956580662748654, -0.05612639773281594, -0.01978585107039479, 0.07157766856187599]`

但是 cross-sample 结论更重要：

- dance q offset: `[0.981741168614011, 0.10665968775146022, -0.1357829358628406, -0.07981843888240901]`
- walk q offset: `[0.9956580662748654, -0.05612639773281594, -0.01978585107039479, 0.07157766856187599]`
- q-offset sign-invariant error: about `0.162786`
- p-offset L2 difference: about `0.001093 m`

结论：固定 torso frame offset 不跨样本稳定，不能作为最终 MuJoCo observation adapter patch。

## Verification

- `isaaclab_observation_manager_sample_gate.py`: walk sample captured successfully.
- `mujoco_observation_runtime_parity_audit.py`: expected blocked due anchor terms.
- `mujoco_torso_frame_offset_audit.py`: walk single sample blocked pending cross-sample validation.
- `mujoco_torso_frame_offset_cross_sample_audit.py`: `blocked_fixed_torso_offset_not_stable_across_walk_and_dance_samples`.
- `mujoco_native_observation_adapter_contract.py`: still blocked, as intended.

## Failed / Blocked Items

- 当前不能使用单个 fixed torso offset 修复 native MuJoCo observation adapter。
- 当前不能把 MuJoCo PPO/VAE/diffusion 视频差归因成纯训练质量问题；adapter/frame parity 仍未过。
- 下一步要定位 IsaacLab/PhysX articulation body frame 与 MuJoCo MJCF body frame 在腰部链条上的状态相关差异，而不是直接重训。

## Effect on English Reading Report

该结果能支撑报告中的失败分析：本项目不是简单“模型没学到 walk/single-leg”，而是 native MuJoCo deployment path 还没有通过官方 IsaacLab observation semantics parity。尤其是 `torso_link`/motion anchor term 是 policy 输入中的核心 tracking error，错误会直接导致 actor 保守前倾或不抬腿。

## Next Step

1. 审计 IsaacLab/PhysX body frame 与 MuJoCo `data.xquat` 的 waist chain 语义差异。
2. 对比 USD `PhysicsRevoluteJoint` localRot/localPos 与 MJCF body/joint frame。
3. 建立一个可跨样本通过的 torso/anchor extraction adapter。
4. 只有 adapter 通过 walk+dance/single-leg parity 后，才继续 PPO/VAE/diffusion/guidance 视频。

## Git Commit

Pending.
