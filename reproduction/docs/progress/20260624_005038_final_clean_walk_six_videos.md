# Progress Update

## Goal

把前面诊断出的 MuJoCo walk 视频问题落成一套可用的、统一路径下的正常 walk 六视频：reference、teacher policy、VAE reconstructed、diffusion denoised latent、guided latent、guided-vs-unguided。目标是报告/PPT 中能展示“本地 MuJoCo 运动控制形式的 normal walk”，而不是继续使用坏 selector 或短 smoke 视频。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_clean_walk_candidate_chain_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_candidate_chain_sweep/clean_walk_candidate_chain_sweep_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_005038_final_clean_walk_six_videos.md`

## Commands Run

```bash
MUJOCO_GL=egl \
BM_CLEAN_SUITE_OUT_ROOT=/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure \
BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT=1.0 \
BM_CLEAN_SUITE_BEST_TEACHER_JSON=/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json \
BM_CLEAN_SUITE_VAE_CKPT=/mnt/infini-data/test/BeyondMimic/res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training/resource_adjusted_teacher_rollout_vae_20260621_141139_seed20260701/resource_adjusted_teacher_rollout_action_vae.pt \
BM_CLEAN_SUITE_DENOISER_CKPT=/mnt/infini-data/test/BeyondMimic/res/runs/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training/resource_adjusted_state_latent_diffusion_20260621_142629_seed20260703/resource_adjusted_state_latent_denoiser.pt \
mujoco_mp4/.venv/bin/python reproduction/scripts/render_clean_walk_mujoco_control_suite.py

mujoco_mp4/.venv/bin/python - <<'PY'
from pathlib import Path
import imageio.v2 as imageio
base=Path('/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure')
for mp4 in sorted(base.glob('*/*.mp4')):
    r=imageio.get_reader(mp4)
    try:
        n=r.count_frames()
        meta=r.get_meta_data()
        fps=meta.get('fps') or 30
        print(mp4.parent.name, n, fps, n/fps)
    finally:
        r.close()
PY
```

## Results

新增最终输出目录：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/`

六条 MP4：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/reference_action_control/reference_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/guided_latent_action_control/guided_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

核心指标：

- 每条视频：`450` frames, `30 FPS`, `15.0 s`
- suite status：`ok`
- `model_target_weight=1.0`
- `reference_anchor_weight=0.0`
- primary variants 全部 `fall_proxy_count=0`
- teacher root height min/mean/max：`0.7001 / 0.7374 / 0.7801 m`
- VAE root height min/mean/max：`0.7326 / 0.7437 / 0.7799 m`
- diffusion root height min/mean/max：`0.7334 / 0.7437 / 0.7799 m`
- guided root height min/mean/max：`0.7324 / 0.7433 / 0.7799 m`

## Verification

已验证：

- MP4 文件存在且非空；
- 每条视频帧数为 `450`；
- FPS 为 `30`；
- 视频时长为 `15.0 s`；
- summary JSON 为 `ok`；
- primary variants 的 `fall_proxy_count=0`；
- keyframe strip 中机器人位于画面中央，没有贴地摔倒或飞出画面。

## Failed / Blocked Items

- 动作仍偏前倾、偏僵，视觉质量低于 BeyondMimic 论文视频。
- 该结果仍启用 MuJoCo root assist。
- 该结果使用本地 scaled-PPO / local VAE / local denoiser，不是官方 BeyondMimic VAE/diffusion checkpoint。
- 该结果不是 IsaacLab rendered MP4，不是真实机器人结果，不是 paper-level Fig.5/Fig.6 closed-loop task evaluation。

## Effect on English Reading Report

报告中可以把这套视频作为“local MuJoCo normal-walk control evidence”，用来说明复现工程已经从 matplotlib skeleton 和坏 selector 进展到可展示的本地物理仿真控制视频。但必须同步写明：该证据只能支持 local virtual reproduction，不支持完整 paper-level claim。

## Next Step

继续提高 Stage-1 teacher 质量，并尽量走官方 IsaacLab/ONNX metadata/motion_tracking_controller 路线验证 teacher policy；随后再用更强 teacher 重新采集 rollout dataset 训练 VAE/diffusion。

## Git Commit

待提交。
