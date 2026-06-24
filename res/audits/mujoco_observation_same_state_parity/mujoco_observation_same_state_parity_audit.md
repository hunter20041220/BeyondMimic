# MuJoCo Observation Same-State Parity Audit

- Status: `ok_same_state_observation_formula_slices_match_official_sample_but_mujoco_runtime_pending`
- Generated: `2026-06-24T04:14:34.765394+00:00`
- Scope: one captured IsaacLab state, local NumPy formula recomputation, no MuJoCo physics rollout.
- 当前不得声称完整复现 BeyondMimic；本审计只验证 observation 公式/slice 的同状态样本对齐。

## Term Errors

- `command` dim=58 max_abs_error=0.000000e+00 passed=`True` policy_vs_critic=0.000000e+00
- `motion_anchor_pos_b` dim=3 max_abs_error=1.462929e-09 passed=`True` policy_vs_critic=1.391676e-01
- `motion_anchor_ori_b` dim=6 max_abs_error=1.143636e-07 passed=`True` policy_vs_critic=4.139441e-02
- `base_lin_vel` dim=3 max_abs_error=0.000000e+00 passed=`True` policy_vs_critic=3.182523e-01
- `base_ang_vel` dim=3 max_abs_error=0.000000e+00 passed=`True` policy_vs_critic=8.767864e-02
- `joint_pos` dim=29 max_abs_error=2.700835e-08 passed=`True` policy_vs_critic=9.655893e-03
- `joint_vel` dim=29 max_abs_error=0.000000e+00 passed=`True` policy_vs_critic=4.997391e-01
- `actions` dim=29 max_abs_error=0.000000e+00 passed=`True` policy_vs_critic=0.000000e+00

## Failed Checks

- `mujoco_runtime_builder_executed`

## Interpretation

- policy observation 在官方配置里带 corruption/noise，因此不能作为精确公式对齐参考。
- 本审计使用 critic 中同名的无噪声 shared terms 作为 deterministic reference。
- 这一步通过也只说明同状态公式和 slice 顺序正确；还没有证明 MuJoCo runtime 的 state、frame alignment、normalizer 和 last_action 全部正确。
