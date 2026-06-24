# Progress Update

## Goal

本轮目标是继续按“先对照论文公式和附录参数，再训练/出视频”的主线推进，修复当前 MuJoCo 执行层里最直接的 action adapter blocker：论文公式要求

```text
theta_sp = theta_default + alpha * normalized_action
```

而旧 MuJoCo position actuator `ctrlrange` 对左右 ankle roll 的合法 `default +/- action_scale` setpoint 进行了预物理裁剪。这会让 policy/VAE/diffusion 即使输出了合法动作，也在进入 MuJoCo physics 前被改坏，因此会导致小步、前倾、抬腿幅度不足等失败现象。

## Files Read

- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `reproduction/third_party/official/motion_tracking_controller/config/g1/controllers.yaml`
- `res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml`
- `res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `res/audits/appendix_parameter_matrix/beyondmimic_appendix_parameter_matrix_audit.json`
- `res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`

## Files Modified

- `reproduction/scripts/mujoco_native_action_adapter_contract.py`
  - 新增 no-action-clipping MuJoCo PD XML 生成逻辑。
  - 保留原始 XML 的裁剪风险记录，同时用 patched XML 重新验证 `unit_targets_inside_mujoco_ctrlrange`。
  - 报告中区分 original ctrlrange violation 和 patched ctrlrange result。
- `mujoco_mp4/scripts/mujoco_pd_control_video.py`
  - 生成 MuJoCo PD actuator 时，不再只用 mechanical joint range 作为 position actuator ctrlrange。
  - 改为使用 joint range 与 `deployment_default +/- action_scale` 的并集，避免合法论文 action 被预物理裁剪。
- `reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py`
  - 修正文案：action ctrlrange patch 已覆盖 unit setpoints，但这不是 rollout 成功。
- `reproduction/scripts/beyondmimic_appendix_parameter_matrix_audit.py`
  - 更新 allowed next work，把 action ctrlrange 修复项改成后续使用 patched no-clipping range 做 no-root-assist adapter probes。
- `reproduction/scripts/artifact_manifest.py`
  - 登记 patched XML、MuJoCo PD video script 和本进度文件。

## Commands Run

```bash
python3 reproduction/scripts/mujoco_native_action_adapter_contract.py
python3 -m py_compile reproduction/scripts/mujoco_native_action_adapter_contract.py mujoco_mp4/scripts/mujoco_pd_control_video.py
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
python3 reproduction/scripts/beyondmimic_appendix_parameter_matrix_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
```

## Results

- `res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
  - status: `ok_native_action_adapter_formula_and_no_clip_ctrlrange_patch_ready`
  - original violating joints: 2
    - `left_ankle_roll_joint`
    - `right_ankle_roll_joint`
  - original max excess: about `0.176777 rad`
  - patched violating joints: 0
  - `unit_targets_inside_mujoco_ctrlrange=true`
- 新增 patched XML：
  - `res/audits/mujoco_native_action_adapter_contract/g1_clean_walk_control_suite_pd_no_action_clip.xml`
  - SHA256: `c72f82a55a4a52f8b67555cecc2e39b1ac9cc4e4865268544c52c5a0cf42aeea`

## Verification

本轮局部验证通过：

- Python compile 通过。
- Action adapter audit 通过。
- Appendix parameter matrix 仍为 blocked，这是预期状态。
- Pretraining hard gate 仍为 blocked，这是预期状态。
- Code/formula appendix contract 仍为 blocked，这是预期状态。

还需要继续运行标准八项全量验证：

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

## Failed / Blocked Items

本轮没有生成新视频，也没有开始新训练。仍然 blocked 的关键项：

- `mujoco_native_observation_adapter_contract` 仍为 `blocked_native_mujoco_observation_adapter_not_validated`。
- teacher quality gate 仍未通过，当前 teacher/VAE/diffusion 不能证明学到 walk 或 single-leg reference 的完整姿态。
- root-assist/material/no-root-assist physics video gate 仍未通过。
- VAE/diffusion/guidance 的长训练和最终成功视频文件夹仍不允许开始。

## Effect on English Reading Report

这轮可用于报告中的“失败分析与工程修正”部分：

- 说明当前 MuJoCo 视频失败并不只可能是训练不够，也包括执行层 action 被 MuJoCo actuator range 裁剪的问题。
- 可以展示论文公式、官方 action scale、MuJoCo actuator ctrlrange 三者的对照。
- 可以诚实说明：action adapter 的公式和 ctrlrange 前置闸门已修复，但 native observation 和 teacher quality 仍未通过，所以不能声称控制复现成功。

## Next Step

下一步应该做 numeric IsaacLab-vs-MuJoCo observation parity probe：

1. 对同一连续 motion、同一 robot state，导出 IsaacLab observation manager 的 160 维 policy obs。
2. 在 MuJoCo adapter 中重建同一 160 维 obs。
3. 按 term 比较 command、anchor pos/orientation、base velocities、joint offsets、joint velocity、last action。
4. 只有 observation/action 都通过后，才能重新跑 no-root-assist teacher/VAE/diffusion videos。

## Git Commit

待本轮标准验证通过后提交。

