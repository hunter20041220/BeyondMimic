# Progress Update

## Goal

继续执行“先审公式、参数、obs/action adapter，再训练/视频”的主线。本轮不启动训练、不生成新视频，只针对当前 MuJoCo 运动控制视频前倾、走路/单脚站不稳的问题，审计 MuJoCo `torso_link` frame 是否与 IsaacLab importer/exported frame 不一致。

## Files Read

- `reproduction/scripts/mujoco_observation_runtime_parity_audit.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/isaaclab_observation_manager_sample_gate.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json`
- `res/audits/mujoco_observation_runtime_parity/mujoco_observation_runtime_parity_audit.json`

## Files Modified

- Added `reproduction/scripts/mujoco_torso_frame_offset_audit.py`
- Updated `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Added this progress report.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_torso_frame_offset_audit.py reproduction/scripts/mujoco_native_observation_adapter_contract.py reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/mujoco_torso_frame_offset_audit.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
```

## Results

新增 torso frame offset 审计产物：

- `res/audits/mujoco_torso_frame_offset/mujoco_torso_frame_offset_audit.json`
- `res/audits/mujoco_torso_frame_offset/mujoco_torso_frame_offset_audit.tsv`
- `res/audits/mujoco_torso_frame_offset/mujoco_torso_frame_offset_audit.md`

关键数值：

- 原始 MuJoCo anchor `motion_anchor_pos_b` error: `0.005219148958698112`
- 原始 MuJoCo anchor `motion_anchor_ori_b` error: `0.3175156624836241`
- 右乘候选 MuJoCo->IsaacLab torso quaternion offset 后，`motion_anchor_pos_b` error: `1.4629287503620247e-09`
- 右乘候选 MuJoCo->IsaacLab torso quaternion offset 后，`motion_anchor_ori_b` error: `1.1436359326211232e-07`
- 候选四元数 offset: `[0.981741168614011, 0.10665968775146022, -0.1357829358628406, -0.07981843888240901]`
- 候选 world position offset: `[-7.13339350e-05, 5.85632682e-04, -1.69969719e-05]`

解释：当前失败的 MuJoCo action-control 视频很可能在 policy 学习质量之前，就已经受到 native MuJoCo observation adapter 的 `torso_link`/anchor frame mismatch 影响。一个维度正确的 160-D obs 仍然可以在语义上错误，从而让 actor 看到错误 anchor 误差并输出前倾/保守站姿。

## Verification

本轮审计当前状态：

- `mujoco_torso_frame_offset_audit.py`: `blocked_torso_frame_offset_hypothesis_single_terminated_sample_requires_walk_validation`
- `mujoco_native_observation_adapter_contract.py`: `blocked_native_mujoco_observation_adapter_not_validated`

该结果是预期的：offset 假设被单样本支持，但不能直接作为最终 adapter 修复。

## Failed / Blocked Items

- 当前 IsaacLab observation sample 来自 `dance1_subject1`，并且 `terminated_after_zero_step=true`。
- 因此该 offset 只能作为 frame-mismatch hypothesis，不能直接放入 MuJoCo PPO/VAE/diffusion rollout。
- 必须再抓取一个 non-terminated、低动态 walk/single-leg official IsaacLab observation sample，验证同一个 offset 是否成立。
- 当前仍不得声称 MuJoCo PPO/VAE/diffusion control videos 成功。

## Effect on English Reading Report

这一轮为报告提供了一个更诚实的失败分析：视频差不应简单写成“模型没学会”，而应写成“native MuJoCo deployment adapter 尚未通过 IsaacLab observation-manager parity，尤其是 torso/anchor frame mismatch”。这能解释为什么 reference/video 链条看起来像前倾站姿，而不是正常抬腿或行走。

## Next Step

1. 抓取 non-terminated walk observation sample，优先 `walk1_subject1` 或 `walk3_subject1`。
2. 用同一 MuJoCo injected-state runtime parity 和 torso offset audit 复验。
3. 如果 offset 跨样本稳定，再实现正式 adapter correction，并重新跑 no-root-assist native MuJoCo PPO rollout gate。
4. 只有 adapter gate 通过后，才继续 teacher/VAE/diffusion/guidance 视频。

## Git Commit

Pending.
