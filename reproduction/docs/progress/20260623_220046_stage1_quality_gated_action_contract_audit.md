# Progress Update

## Goal

定位 quality-gated MuJoCo 视频仍然站不稳的下一层原因：在 reference target 已修复、reference-joint PD baseline 正常后，检查 teacher action-derived targets 是否与同片段 reference joint qpos 满足可信 action contract。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/stage1_multisource_quality_gated_video_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_stability_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_adapter_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_action_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_220046_stage1_quality_gated_action_contract_audit.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/stage1_multisource_quality_gated_action_contract_audit.py
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_quality_gated_action_contract_audit.py
```

## Results

新增 action contract audit：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.tsv`

关键结果：

- Selected segment：`lafan1_sprint1_subject4`, motion steps `286550..286579`, frames `30`, reward mean `0.05465`
- Teacher action-derived target vs reference joint qpos per-frame mean abs gap：mean `0.5034 rad`, median `0.5029 rad`, max `0.5897 rad`
- High-gap joints mean > `0.5 rad`：`13`
- Low-correlation joints corr < `0.2`：`15`
- Sign-flip improves joints：`16`，说明不是一个全局符号翻转能解释
- Top gap joints 包括 `left_knee_joint`、`right_hip_yaw_joint`、`right_shoulder_pitch_joint`、`right_shoulder_yaw_joint`、`left_hip_pitch_joint`、`left_shoulder_yaw_joint`、`left_wrist_roll_joint`、`left_hip_roll_joint`

## Verification

本轮新增文件已纳入 `artifact_manifest.py`。完整验证链在本 progress 写入后运行并刷新 manifest/master audit。

## Failed / Blocked Items

- Teacher action-derived targets 与 reference joints 差距过大，当前 MuJoCo teacher/VAE/diffusion/guidance 视频不能作为可信 paper-level closed-loop result。
- 下一层 blocker 是 native MuJoCo obs/action adapter 或 IsaacLab action scale/default pose/joint order/normalization contract。
- 长时稳定控制仍未证明；当前 result 只支持 short-horizon diagnostic conclusion。

## Effect on English Reading Report

报告中可以明确写出：视频差不是因为 reference 数据全坏，也不是因为 MuJoCo 不能渲染；reference joint PD baseline 在同片段能短时稳定，而 teacher action bridge 明显偏离 reference contract。这是复现实验失败分析的重要证据，也能说明为什么当前不能把 VAE/diffusion/guidance 视频当作 paper-level result。

## Next Step

核对官方 `whole_body_tracking` 与 `motion_tracking_controller` 的 action scale、default joint pose、joint order、PD gain、observation history/normalization，并实现 native MuJoCo obs/action adapter 或优先在 IsaacLab 内评估 teacher checkpoint。

## Git Commit

本轮尚未提交，等待完整验证链完成后提交。
