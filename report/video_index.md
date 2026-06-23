# 视频索引说明

本项目本地索引到 `363` 个视频。注意：视频数量多不代表 paper-level 复现完成。

## 最新六条 MuJoCo action-control 诊断视频

路径根目录：

```text
res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/
```

六条视频：

```text
reference_action_control.mp4
teacher_policy_action_control.mp4
vae_reconstructed_action_control.mp4
diffusion_denoised_latent_action_control.mp4
guided_latent_action_control.mp4
guided_vs_unguided_action_control.mp4
```

当前检查：

```json
{
  "all_continuous_primary_time_steps": true,
  "all_mp4_exist": true,
  "all_primary_metrics_csv_exist": true,
  "does_not_claim_complete_beyondmimic_reproduction": true,
  "does_not_claim_real_robot": true,
  "selected_segment_single_source_motion": true
}
```

这些视频是连续片段，但控制质量差。报告中应写为“MuJoCo local diagnostic videos”，不要写成 BeyondMimic paper-level Fig.5/Fig.6。
