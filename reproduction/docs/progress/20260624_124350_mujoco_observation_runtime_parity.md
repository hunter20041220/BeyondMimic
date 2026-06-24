# Progress Update

## Goal

本轮继续执行“先审公式/参数再训练”的 hard gate：把 observation 检查从 pure formula / same-state raw tensor parity 推进到 MuJoCo runtime injected-state parity。目标是验证：把官方 IsaacLab captured root pose、joint state、root velocity、joint velocity 写入 MuJoCo G1 model 后，由 MuJoCo `data.xpos/xquat/qpos/qvel` 构造的 160-D observation 是否能和 IsaacLab noise-free critic shared terms 对齐。

本轮不训练、不生成视频、不启动 PPO/VAE/diffusion rollout。

## Files Read

- `reproduction/scripts/mujoco_observation_same_state_parity_audit.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `reproduction/scripts/mujoco_native_action_adapter_contract.py`
- `mujoco_mp4/configs/g1_joint_mapping.yaml`
- `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json`
- `res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`

## Files Modified

- `reproduction/scripts/mujoco_observation_runtime_parity_audit.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_124350_mujoco_observation_runtime_parity.md`
- `res/audits/mujoco_observation_runtime_parity/mujoco_observation_runtime_parity_audit.json`
- `res/audits/mujoco_observation_runtime_parity/mujoco_observation_runtime_parity_audit.tsv`
- `res/audits/mujoco_observation_runtime_parity/mujoco_observation_runtime_parity_audit.md`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.md`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.tsv`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_observation_runtime_parity_audit.py
python3 reproduction/scripts/mujoco_observation_runtime_parity_audit.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
```

## Results

- MuJoCo runtime audit 成功执行，使用 `mujoco_mp4/.venv/bin/python` 中的 `mujoco==3.9.0`。
- 成功加载 G1 MuJoCo model，注入 IsaacLab captured root/joint/qvel state，并执行 `mj_forward`。
- 逐项 observation 结果：
  - `command`: pass, max abs error `0.000e+00`
  - `base_lin_vel`: pass, max abs error `3.093e-08`
  - `base_ang_vel`: pass, max abs error `8.795e-08`
  - `joint_pos`: pass, max abs error `2.701e-08`
  - `joint_vel`: pass, max abs error `0.000e+00`
  - `actions`: pass, max abs error `0.000e+00`
  - `motion_anchor_pos_b`: fail, max abs error `5.219e-03`
  - `motion_anchor_ori_b`: fail, max abs error `3.175e-01`
- root frame 本身对齐：
  - root position error `0.0`
  - root quaternion sign-invariant error `~1.1e-16`
- blocker 集中在 `torso_link` / motion anchor frame：
  - anchor position error `5.856e-04 m`
  - anchor quaternion sign-invariant error `0.133646`
- 多个候选 G1 MJCF/XML 均出现同一 torso orientation mismatch：
  - local GMR `g1_mocap_29dof.xml`
  - official `unitree_description/mjcf/g1.xml`
  - PBHC G1 XML
  - mjlab / unitree_rl_mjlab G1 XML
  - ASAP old G1 XML

## Verification

- `mujoco_observation_runtime_parity_audit.py` 输出：
  - `blocked_mujoco_injected_state_observation_runtime_parity_mismatch`
- `mujoco_native_observation_adapter_contract.py` 现在记录：
  - `runtime_observation_builder_executed=true`
  - `runtime_observation_all_slices_pass=false`
  - `runtime_anchor_frame_mismatch_detected=true`
  - `success_video_claim_allowed=false`

## Failed / Blocked Items

- MuJoCo runtime observation adapter 仍 blocked。
- 失败项明确定位为 anchor/torso frame mismatch，而不是 command、joint order、joint velocity、base velocity 或 last_action。
- 当前不能把 IsaacLab PPO checkpoint 直接放进 MuJoCo native observation loop 后声称 policy rollout 成功。
- 当前不能启动新的长训练来“赌效果变好”；必须先解决 MuJoCo MJCF frame 与 IsaacLab USD/URDF importer frame 的语义差异，或改用官方 deployment controller/ONNX metadata 中定义的 frame 语义。

## Effect on English Reading Report

这能解释为什么之前的 MuJoCo action-control 视频会前倾、不抬腿：不是所有公式都错，也不是 joint order 全错，而是关键的 `motion_anchor_ori_b` 由 MuJoCo `torso_link` frame 产生时和 IsaacLab official sample 有明显差异。报告里可以写：the injected-state MuJoCo runtime audit isolates the remaining observation adapter blocker to the torso/anchor frame mismatch.

## Next Step

下一步应审 `motion_tracking_controller` 的 Pinocchio/local-frame implementation 和 exported ONNX metadata，确认 deployment 侧是否使用与 IsaacLab USD importer 一致的 torso frame，或者需要在 MuJoCo observation builder 中加入固定 frame correction。只有这个 runtime anchor frame gate 过了，才能继续 no-root-assist PPO/VAE/diffusion videos。

## Git Commit

待标准验证通过后提交。当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
