# Quality-Gated Action Contract Audit

## 结论

同一 quality-gated normal-root 片段上，teacher action-derived PD targets 与 reference joint qpos 差距较大。这支持下一步优先检查 teacher action/action-scale/default-pose/joint-order/obs-action adapter，而不是继续怀疑 root target。

## 总体差距

- per-frame mean abs gap: mean `0.5034109839455593`, median `0.5028769604484502`, max `0.5896512676785843` rad
- high-gap joints (>0.5 rad mean): `13`
- low-correlation joints (<0.2): `15`
- sign flip improves joints: `16`

## Top Gap Joints

- `left_knee_joint`: mean gap `1.2293`, max gap `1.4789`, corr `0.5814848881792867`
- `right_hip_yaw_joint`: mean gap `1.1055`, max gap `1.6232`, corr `-0.9357409225590029`
- `right_shoulder_pitch_joint`: mean gap `1.0774`, max gap `1.1669`, corr `-0.26958846195893144`
- `right_shoulder_yaw_joint`: mean gap `0.8909`, max gap `1.2342`, corr `-0.20787702924940085`
- `left_hip_pitch_joint`: mean gap `0.8900`, max gap `1.5778`, corr `0.6873296929694028`
- `left_shoulder_yaw_joint`: mean gap `0.8531`, max gap `1.0968`, corr `-0.8401763302524032`
- `left_wrist_roll_joint`: mean gap `0.7578`, max gap `0.9497`, corr `-0.36984818175863077`
- `left_hip_roll_joint`: mean gap `0.7180`, max gap `1.1058`, corr `-0.81797828054012`

## Claim Boundary

该结果是本地 MuJoCo bridge/static target audit，不是官方 BeyondMimic teacher 结论，也不是 paper-level control result。

JSON: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.json`
TSV: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.tsv`
