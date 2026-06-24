# Progress Update

## Goal

本轮继续执行“训练前先审公式/参数”的 hard gate：在已经捕获官方 IsaacLab `Tracking-Flat-G1-v0` observation sample 的基础上，补齐 raw tensors，并新增同状态 observation 公式对齐审计。目标是确认本地 NumPy/未来 MuJoCo adapter 所需的 8 段 160-D policy observation 公式、slice 顺序、Rot6D flatten、body-frame velocity、default joint offset 和 last_action 语义没有在同一状态样本上出错。

本轮不启动 PPO/VAE/diffusion 训练，不生成新视频，不声称 MuJoCo runtime rollout 成功。

## Files Read

- `reproduction/scripts/isaaclab_observation_manager_sample_gate.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/mujoco_observation_math_parity_audit.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/mdp/observations.py`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

## Files Modified

- `reproduction/scripts/isaaclab_observation_manager_sample_gate.py`
- `reproduction/scripts/mujoco_observation_same_state_parity_audit.py`
- `reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_121631_mujoco_observation_same_state_parity.md`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.npz`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_observation_manager_sample_gate.json`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_observation_manager_sample_gate.tsv`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_observation_manager_sample_gate.md`
- `res/audits/isaaclab_observation_manager_sample_gate/isaaclab_observation_manager_sample_gate_worker.py`
- `res/audits/mujoco_observation_same_state_parity/mujoco_observation_same_state_parity_audit.json`
- `res/audits/mujoco_observation_same_state_parity/mujoco_observation_same_state_parity_audit.tsv`
- `res/audits/mujoco_observation_same_state_parity/mujoco_observation_same_state_parity_audit.md`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.tsv`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.md`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
python3 reproduction/scripts/isaaclab_observation_manager_sample_gate.py
python3 reproduction/scripts/mujoco_observation_same_state_parity_audit.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
```

## Results

- 成功重新捕获 IsaacLab official observation sample。
- 新 sample 现在包含：
  - noisy `policy_terms`
  - noise-free `critic_terms`
  - 32 组 `raw_state` tensors
  - `critic_shared_terms_available=true`
  - `raw_state_available_for_same_state_parity=true`
- 新增 `mujoco_observation_same_state_parity_audit.py` 并通过。
- 同状态公式/slice 对齐结果：
  - `command`: max abs error `0.000e+00`
  - `motion_anchor_pos_b`: max abs error `1.463e-09`
  - `motion_anchor_ori_b`: max abs error `1.144e-07`
  - `base_lin_vel`: max abs error `0.000e+00`
  - `base_ang_vel`: max abs error `0.000e+00`
  - `joint_pos`: max abs error `2.701e-08`
  - `joint_vel`: max abs error `0.000e+00`
  - `actions`: max abs error `0.000e+00`
- policy-vs-critic 在带噪声项上有明显差异，这是官方 `PolicyCfg.enable_corruption=True` 的预期现象，不是公式错。

## Verification

- `isaaclab_observation_manager_sample_gate.py`:
  - `ok_isaaclab_observation_manager_sample_captured_but_mujoco_parity_pending`
- `mujoco_observation_same_state_parity_audit.py`:
  - `ok_same_state_observation_formula_slices_match_official_sample_but_mujoco_runtime_pending`
- `mujoco_native_observation_adapter_contract.py`:
  - `same_state_observation_formula_parity_ready=true`
  - `observation_runtime_parity_ready=false`
  - `native_obs_adapter_ready=false`
  - `success_video_claim_allowed=false`

## Failed / Blocked Items

- MuJoCo runtime observation builder 仍未被证明和 IsaacLab runtime 等价。
- 当前 `native_adapter_validated_against_isaaclab_observation_manager=false`。
- 当前 `native_adapter_validated_against_deployment_controller=false`。
- 当前 `native_rollout_preconditions_ready=false`。
- 因此不能启动新的长训练，也不能把现有 PPO/VAE/diffusion MuJoCo 视频声明为可信 closed-loop motion-control 结果。

## Effect on English Reading Report

本轮为报告中的 reproduction audit 提供了更细的证据链：项目不是只拼一个 160-D 向量，而是已经把官方 IsaacLab observation 的 noisy policy terms、noise-free critic shared terms 和 raw tensors 捕获下来，并在同一状态上逐 slice 验证公式。报告里可以诚实写：same-state formula parity passed, but native MuJoCo runtime parity and deployment-frame parity are still pending.

## Next Step

下一步应该把 same-state formula parity 推进到真正的 MuJoCo runtime builder parity：在 MuJoCo 中构造/记录同类 state、reference phase、last_action、normalizer 输入，逐 slice 比较 `command`、anchor error、base velocity、joint offset 和 last action。只有 runtime parity 过了，才能继续无 root assist 的 PPO/VAE/diffusion action-control 视频。

## Git Commit

待标准验证通过后提交。当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
