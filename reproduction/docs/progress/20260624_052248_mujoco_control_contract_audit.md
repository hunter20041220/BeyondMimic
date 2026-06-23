# Progress Update

## Goal

在继续 teacher/VAE/diffusion 训练和最终单脚站立/走路视频之前，补充一个 MuJoCo 控制合同审计，明确当前 MuJoCo 视频是否真正满足 BeyondMimic 论文里的 action/PD/control 语义。

## Files Read

- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- `mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `mujoco_mp4/scripts/mujoco_trace_mesh_video.py`
- `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml`
- `mujoco_mp4/res/control_videos/reference_control/reference_control_summary.json`
- `res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/reference_action_control/reference_action_control_summary.json`

## Files Modified

- `reproduction/scripts/mujoco_control_contract_audit.py`
- `reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260624_052248_mujoco_control_contract_audit.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_control_contract_audit.py reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/mujoco_control_contract_audit.py
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

- 官方 G1 PD/action scale/armature 数值已经有可读审计，且 MuJoCo PD XML 中 29 个 position actuator 的数值补丁与官方行表一致。
- 当前 MuJoCo 视频链仍不满足论文控制语义：
  - 视频脚本直接写 absolute joint targets 或 IK-fitted targets。
  - 当前路径没有证明 native policy observation adapter，也没有证明 `obs -> policy/VAE/diffusion -> normalized action -> theta0 + alpha * action -> PD -> mj_step`。
  - root assist 默认开启，用于稳定/居中展示，但阻断无辅助控制成功声明。
  - MuJoCo floor/material/friction 与官方 IsaacLab material randomization 不等价。

## Verification

本轮新增审计的预期状态是 blocked，而不是通过：

- `mujoco_control_contract_audit`: `blocked_mujoco_control_semantics_not_native_policy_control`
- `beyondmimic_model_chain_paper_contract_audit`: 继续 blocked，并新增 `mujoco_control_contract_native_ready=false`

## Failed / Blocked Items

- 当前 MuJoCo 视频只能作为 diagnostic/report visualization，不能作为 teacher/VAE/diffusion 成功控制视频。
- 单脚站立和走路视频要成为最终成功结果之前，必须先实现或验证 native action adapter，并禁用 root assist 的成功 gate。
- 当前不得声称完整复现 BeyondMimic，也不得把 root-assist/IK/absolute-target MuJoCo 视频作为 paper-level closed-loop result。

## Effect on English Reading Report

这次审计为报告提供一个更诚实的技术结论：失败视频不一定说明论文公式本身错误，而是当前本地 MuJoCo 可视化/控制适配器尚未等价于论文 native action control path。报告中应把这批视频列为诊断可视化，不列为复现成功证据。

## Next Step

下一步应先实现 native MuJoCo/Isaac action adapter gate：重建观测，加载 teacher/VAE/diffusion 输出 normalized action，应用 `theta0 + alpha * action`，禁用 root assist，记录 fall/done、root height、action norm、target error。只有 gate 通过后再生成新的成功视频。

## Git Commit

本文件创建时尚未提交；本轮验证通过后再随审计脚本和结果一起提交。
