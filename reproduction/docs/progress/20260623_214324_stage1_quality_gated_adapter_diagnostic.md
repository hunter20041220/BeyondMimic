# Progress Update

## Goal

在 quality-gated normal-root 片段上继续定位下一层 blocker：判断 MuJoCo PD/root-assist adapter 是否能稳定跟踪 reference joint qpos，以及 teacher action-derived targets 是否明显偏离 reference joints。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/stage1_multisource_quality_gated_video_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_selector_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_paper_contract_mujoco_action_control_videos.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_adapter_diagnostic.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_214324_stage1_quality_gated_adapter_diagnostic.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/stage1_multisource_quality_gated_adapter_diagnostic.py
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_quality_gated_adapter_diagnostic.py
```

## Results

新增 reference-joint PD baseline：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/reference_joint_pd_control/reference_joint_pd_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/reference_joint_pd_control/reference_joint_pd_control_metrics.csv`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/reference_joint_pd_control/reference_joint_pd_control_summary.json`

对比结果：

- Reference-PD `fall_proxy_count=0`
- Reference-PD root height min/max：`0.7563 / 0.7722 m`
- Reference-PD root position error mean/max：`0.0533 / 0.0735 m`
- Teacher-action root height min/max：`0.6440 / 0.7457 m`
- Teacher-action root position error mean/max：`0.1439 / 0.2230 m`
- Teacher action-derived target 与 reference joint target per-frame mean abs gap：mean `0.5034 rad`，max `0.5897 rad`

## Verification

新增审计：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_adapter_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_adapter_diagnostic.md`

视觉检查：

- reference-joint PD keyframes 正常站立并居中；
- 该 baseline 比 teacher-action control 更接近 target root height。

## Failed / Blocked Items

- 当前 baseline 仍使用 root assist，不是 unassisted humanoid control。
- teacher action-derived targets 和 reference joint qpos 差距较大，下一步需要做 action contract audit，检查 action scale/default pose/joint order/sign/normalization。
- 这仍不是 paper-level BeyondMimic Fig.5/Fig.6。

## Effect on English Reading Report

报告可以写明：selector/root target 问题已修；MuJoCo PD/root-assist 在同一短片段上能跟 reference joint qpos；后续失败更可能来自 teacher action/action-scale/obs-action adapter，而不是视频渲染或 root target。

## Next Step

新增 action contract audit，比较 teacher action-derived targets 与 reference joints 的 per-joint error、correlation、sign flip 和 scale/default pose contract。

## Git Commit

本轮尚未提交，等待完整验证链完成后提交。
