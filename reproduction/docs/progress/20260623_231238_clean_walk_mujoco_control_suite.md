# Progress Update

## Goal

解决当前 MuJoCo 视频“机器人站不稳、视频不能展示”的直接问题。先生成一套正常 walk 展示视频，并诚实区分 reference-control、teacher/VAE/diffusion/guidance presentation diagnostic 与真正纯 learned closed-loop controller。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_continuous_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_pd_control_demo.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_231238_clean_walk_mujoco_control_suite.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/render_clean_walk_mujoco_pd_control_demo.py
MUJOCO_GL=egl BM_CLEAN_WALK_SECONDS=15 BM_CLEAN_WALK_START_INDEX=0 \
  BM_CLEAN_WALK_VIDEO_FPS=30 \
  BM_CLEAN_WALK_MOTION_NPZ=/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/motions/lafan1_walk1_subject1/motion.npz \
  mujoco_mp4/.venv/bin/python reproduction/scripts/render_clean_walk_mujoco_pd_control_demo.py

python3 -m py_compile reproduction/scripts/render_clean_walk_mujoco_control_suite.py
MUJOCO_GL=egl BM_CLEAN_WALK_SECONDS=15 BM_CLEAN_WALK_START_INDEX=0 \
  BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT=0.20 \
  BM_CLEAN_SUITE_GUIDANCE_SCALE=0.35 \
  mujoco_mp4/.venv/bin/python reproduction/scripts/render_clean_walk_mujoco_control_suite.py
```

## Results

新增 pure reference PD walk demo：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_pd_control_demo/clean_lafan1_walk1_subject1_pd_control_15s.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_pd_control_demo/clean_lafan1_walk1_subject1_pd_control_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_pd_control_demo/clean_lafan1_walk1_subject1_pd_control_metrics.csv`

新增 clean walk six-video suite：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/reference_action_control/reference_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/guided_latent_action_control/guided_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/clean_walk_mujoco_control_suite_summary.json`

关键指标：

- pure reference PD：15 s, 450 frames, `fall_proxy_count=0`
- root height min/mean/max：`0.7165 / 0.7633 / 0.7814 m`
- clean suite：all primary variants `fall_proxy_count=0`
- learned variants 默认 `model_target_weight=0.20`，`reference_anchor_weight=0.80`
- learned model target 和 reference joint target 平均差距约 `0.50 rad`

## Verification

本轮先完成脚本语法检查和 MuJoCo 实际渲染。随后运行 full audit chain：

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

- 第一版自动 selector 选到 `lafan1_walk3_subject4` 近地面 root target，这是坏视频直接原因之一。
- 长版 `lafan1_walk1_subject1` 后段自动窗口曾触发 MuJoCo `QACC` instability，原因是后段全局朝向/接触对当前 PD/root-assist 不稳；已改为显式使用开头稳定 15 秒窗口。
- Pure learned teacher/VAE/diffusion/guidance controller 仍未完成。当前 clean suite 中 learned variants 是 reference-anchor presentation diagnostic，不是纯模型闭环成功。

## Effect on English Reading Report

报告中可以新增一节：MuJoCo local visualization evidence。应写清楚：

- reference-action PD baseline 可以正常生成 15 秒 G1 walk；
- teacher/VAE/diffusion/guidance presentation videos 仅展示当前模型输出对 reference target 的小比例影响；
- 当前 Stage-1 teacher 和 MuJoCo obs/action adapter 仍是真正 learned-control 的 blocker。

## Next Step

在 clean walk runner 上做 `model_target_weight` sweep：`0.2, 0.4, 0.6, 0.8, 1.0`，量化纯模型 target 什么时候开始失稳；然后回到 Stage-1 teacher quality 和 MuJoCo obs/action adapter 修复。

## Git Commit

Pending after verification.
