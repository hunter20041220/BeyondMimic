# Quality-Gated MuJoCo Adapter Diagnostic

## 结论

Primary blocker inference: `teacher_action_targets_or_isaac_to_mujoco_action_mapping`.

在同一个 normal-root quality-gated 片段上，新增 `reference_joint_pd_control` 作为 MuJoCo PD/root-assist baseline。它使用 reference joint qpos 作为 PD target，不是 policy 输出。

## 指标对比

- Reference-PD fall proxy count: `0`
- Reference-PD root height min/max: `0.756292804305334` / `0.772163249466415` m
- Reference-PD root position error mean/max: `0.05327024329438863` / `0.07349643487315234` m
- Teacher-action fall proxy count: `0`
- Teacher-action root height min/max: `0.643982828232582` / `0.7457355254466121` m
- Teacher-action root position error mean/max: `0.1438565948581903` / `0.22297892348363033` m
- Teacher target vs reference target per-frame mean abs gap: mean `0.5034109839455593`, max `0.5896512676785843` rad

## 解释

- 如果 Reference-PD 明显更稳，下一步优先查 teacher action、action scale、obs/action adapter。
- 如果 Reference-PD 也明显下滑，下一步优先查 MuJoCo PD/root-assist/asset dynamics 适配。
- 该诊断仍是短时 root-assist MuJoCo local diagnostic，不是 paper-level BeyondMimic 控制结果。

JSON: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_adapter_diagnostic.json`
