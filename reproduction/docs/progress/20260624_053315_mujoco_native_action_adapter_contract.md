# Progress Update

## Goal

在继续训练和重新生成单脚站立/走路视频之前，补齐 BeyondMimic 论文 action 公式到 MuJoCo PD setpoint 的 native adapter 前置审计，避免把 reference target、IK target、root-assist 视频误当成真正 policy action control。

## Files Read

- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
- `reproduction/third_party/official/motion_tracking_controller/config/g1/controllers.yaml`
- `res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- `mujoco_mp4/configs/g1_joint_mapping.yaml`
- `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml`
- `res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

## Files Modified

- `reproduction/scripts/mujoco_native_action_adapter_contract.py`
- `reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_053315_mujoco_native_action_adapter_contract.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_native_action_adapter_contract.py
python3 reproduction/scripts/mujoco_native_action_adapter_contract.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
```

标准验证命令也在本轮后续执行：

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- 新增 `mujoco_native_action_adapter_contract`，验证论文/IsaacLab 的 action 语义：
  - `theta_sp = theta_default + action_scale * normalized_action`
  - IsaacLab `JointPositionAction` 执行 affine transform：raw action * scale + offset。
  - RSL-RL wrapper 可将 normalized action clip 到指定范围。
  - 29 个 joint order 与 `g1_action_scale_audit`、MuJoCo mapping、MuJoCo PD actuator 顺序一致。
  - 零 action 会回到官方 deployment standby default pose，而不是 all-zero fallback。
  - `+1/-1` action 在 raw formula 层精确对应 `default ± action_scale`。

## Verification

新增公式 gate 通过：

- `mujoco_native_action_adapter_contract`: `ok_native_action_adapter_formula_contract_ready_not_rollout`

同时保留 rollout readiness warning：

- MuJoCo actuator `ctrlrange` 会截断 ankle-roll 的 unit action setpoint。
- IsaacLab InitialStateCfg 和 motion_tracking_controller standby default_position 在 ankle pitch 上相差约 0.033 rad。
- 这不是 physics rollout，也不是 teacher/VAE/diffusion 成功声明。

## Failed / Blocked Items

- Native observation adapter 尚未完成。
- No-root-assist MuJoCo physics rollout 尚未完成。
- 当前视频链仍不能作为最终成功视频。
- 当前不得声称完整复现 BeyondMimic，也不得把 formula fixture 当成 policy rollout 成功。

## Effect on English Reading Report

报告中可以更准确地说明：本地复现已经把论文 action 公式、官方 G1 action scale、部署默认姿态和 MuJoCo actuator 顺序对齐到一个可审计 fixture，但最终效果差的关键仍在 teacher quality、native observation adapter、MuJoCo/IsaacLab 动力学差异和 no-root-assist rollout gate。

## Next Step

下一步应该实现 native observation adapter smoke：用同一段 reference motion 和当前 MuJoCo state 构造 policy obs，调用 teacher/VAE/diffusion 输出 normalized action，再使用本轮 verified adapter 进入 MuJoCo PD 和 `mj_step`。成功 gate 必须禁用 root assist，并记录 raw setpoint、ctrlrange-clipped setpoint、fall/done、root height、joint error、target body error。

## Git Commit

本文件创建时尚未提交；验证通过后随审计脚本和结果一起提交。
