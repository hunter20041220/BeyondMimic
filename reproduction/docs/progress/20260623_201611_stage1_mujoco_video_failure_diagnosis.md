# Progress Update

## Goal

诊断为什么当前 Stage-1 multi-source MuJoCo action-control 视频效果很差、机器人站不稳，并区分是视频渲染问题、选段/root target 问题、teacher 训练问题，还是 VAE/diffusion 下游问题。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_motion_bundle.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_motion_state_fixture.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_continuous_video_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_best_teacher_rollout_dataset/tracking_stage1_multisource_best_teacher_rollout_dataset.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/diagnose_stage1_mujoco_video_failure.py`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_201611_stage1_mujoco_video_failure_diagnosis.md`

## Commands Run

```bash
python3 reproduction/scripts/diagnose_stage1_mujoco_video_failure.py
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

新增诊断确认当前六条 MuJoCo 视频失败的第一原因是选段/root target gate 缺失：

- 自动选中的片段来自 `lafan1_walk3_subject4`。
- global motion steps 为 `418177..418474`。
- `motion_time_steps` 连续，`done_count=0`，但 selected root z mean 只有 `0.0512 m`。
- 同一个 source motion 的 root z median 为 `0.7723 m`。
- selected reward mean 为 `-0.081968`。
- 当前 selector 按 `(length, reward_mean)` 排序，因此会让长但近地面的坏片段胜出。
- 当前 teacher rollout 中，`>=60` 帧且 root z 正常的连续候选数为 `0`；`>=30` 帧且 root z 正常的候选数为 `170`。

## Verification

诊断脚本输出：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_candidate_segments.tsv`

## Failed / Blocked Items

- 当前六条视频应作为 failed diagnostic，不应作为 successful report-ready videos。
- 当前 teacher best checkpoint 仍弱：reward mean `0.024131`，body error mean `1.009504 m`，non-timeout done rate `0.194137`。
- 当前 teacher rollout 没有足够长的正常 root-height continuous segment 来生成 15 秒稳定展示视频。
- VAE/diffusion/guidance 视频不能在 teacher/root target 修好前作为有效闭环控制结果。

## Effect on English Reading Report

这次诊断为报告中的 limitation / failure analysis 提供了具体证据：当前 MuJoCo 视频失败不是简单“渲染差”，而是复现链条中的 segment selection、root target、teacher quality 和 MuJoCo adapter 共同造成。报告应诚实写成 failed diagnostic 和 future work，而不是 paper-level control evidence。

## Next Step

修复 segment selector：加入 root height、reward、stability 门槛；先生成 root 高度正常的短 reference pose replay，再复测 teacher action-control。如果 teacher 在正常 target 上仍然不稳，再优先处理 Stage-1 PPO teacher 与 MuJoCo obs/action/PD adapter。

## Git Commit

本轮尚未提交，等待验证链完成后再决定是否提交。
