# Progress Update

## Goal

对 `hub_singleleg_video_single_leg_stand_1` 这条动作跑一遍当前 MuJoCo 可视化完整链，输出并检查：

- reference action-control
- teacher policy action-control
- VAE reconstructed action-control
- diffusion denoised latent action-control
- guided latent action-control
- guided-vs-unguided comparison

本轮目标是看这条 Hub 单腿站立动作在当前复现链路下的真实效果，而不是把它包装成 paper-level 成功。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_clean_walk_candidate_chain_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_mujoco_action_control_videos.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/motions/hub_singleleg_video_single_leg_stand_1/motion.npz`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/clean_walk_mujoco_control_suite_summary.json`

## Files Modified

- 新增本进度文件：`/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_030017_hub_singleleg_full_chain.md`

视频、PNG、metrics CSV 和 summary JSON 生成在 `res/visualization/` 下，属于本地大/派生产物，不提交 GitHub。

## Commands Run

环境与 motion 检查：

```bash
python3 - <<'PY'
import pathlib, numpy as np
p=pathlib.Path('/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/motions/hub_singleleg_video_single_leg_stand_1/motion.npz')
data=np.load(p, allow_pickle=True)
print(data.files)
for k in data.files:
    a=data[k]
    print(k, a.shape, a.dtype)
PY
```

完整链视频生成：

```bash
BM_CLEAN_SUITE_OUT_ROOT=/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure \
BM_CLEAN_WALK_MOTION_NPZ=/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/motions/hub_singleleg_video_single_leg_stand_1/motion.npz \
BM_CLEAN_WALK_SECONDS=15.0 \
BM_CLEAN_WALK_START_INDEX=0 \
BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT=1.0 \
BM_CLEAN_SUITE_BEST_TEACHER_JSON=/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json \
BM_CLEAN_SUITE_VAE_CKPT=/mnt/infini-data/test/BeyondMimic/res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training/resource_adjusted_teacher_rollout_vae_20260621_141139_seed20260701/resource_adjusted_teacher_rollout_action_vae.pt \
BM_CLEAN_SUITE_DENOISER_CKPT=/mnt/infini-data/test/BeyondMimic/res/runs/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training/resource_adjusted_state_latent_diffusion_20260621_142629_seed20260703/resource_adjusted_state_latent_denoiser.pt \
MUJOCO_GL=egl /mnt/infini-data/test/BeyondMimic/mujoco_mp4/.venv/bin/python reproduction/scripts/render_clean_walk_mujoco_control_suite.py
```

视频元信息与 metrics 检查：

```bash
/mnt/infini-data/test/BeyondMimic/mujoco_mp4/.venv/bin/python - <<'PY'
import imageio.v3 as iio, pathlib
root=pathlib.Path('/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure')
for mp4 in sorted(root.glob('*/*.mp4')):
    print(mp4.parent.name, iio.immeta(mp4).get('fps'), iio.immeta(mp4).get('duration'))
PY
```

## Results

Motion 输入：

- 路径：`/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/motions/hub_singleleg_video_single_leg_stand_1/motion.npz`
- 帧数：`799`
- FPS：`50`
- 可用时长：约 `15.98 s`
- 本次渲染：`450` frames, `30 FPS`, `15.0 s`
- 选中 root z：min `0.7983`, mean `0.8071`, max `0.8163`

输出目录：

`/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/`

输出视频：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/reference_action_control/reference_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/guided_latent_action_control/guided_latent_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

关键指标：

| Variant | root z min | root z mean | root z max | fall proxy | root error mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 0.7532 | 0.7671 | 0.7889 | 0 | 0.0507 |
| teacher | 0.7336 | 0.7382 | 0.7885 | 0 | 0.0741 |
| VAE | 0.7340 | 0.7371 | 0.7887 | 0 | 0.0747 |
| diffusion | 0.7330 | 0.7368 | 0.7887 | 0 | 0.0745 |
| guided | 0.7334 | 0.7369 | 0.7887 | 0 | 0.0746 |

## Verification

- 所有六个 MP4 均存在。
- 所有六个 MP4 均为 `15.0 s`, `30 FPS`。
- 五个单视频 variant 均有 metrics CSV。
- `fall_proxy_count=0`。
- 关键帧已人工查看。

视觉判断：

- `reference_action_control` 能看出单腿/抬腿站立姿态变化，动作本身可作为 reference visualization。
- `teacher_policy_action_control` 没有摔倒，但没有很好复现 reference 的单腿抬腿关键姿态，更像保守前倾站立。
- `VAE`、`diffusion`、`guided` 与 teacher 视觉上接近，同样没有体现出明显单腿站立动作细节。

## Failed / Blocked Items

- 当前 learned variants 不应声称成功复现 Hub 单腿站立控制，只能说是 local MuJoCo diagnostic/control-chain visualization。
- 这不是官方 IsaacLab rendered MP4。
- 这不是真实机器人结果。
- 这不是 BeyondMimic paper-level Fig.5/Fig.6。
- 当前链路仍使用 root assist；teacher/VAE/diffusion/guidance 对该 balance motion 的动作语义保真度不足。

## Effect on English Reading Report

这条结果适合写进报告的 “negative/limited reproduction evidence”：

- 说明 reference motion 已经能被 MuJoCo 可视化，且 Hub balance 数据可用于动作展示。
- 说明当前 teacher -> VAE -> diffusion -> guidance 链路还没有学到高保真 balance skill。
- 可以作为诚实复现审计中的例子：`fall_proxy=0` 不等于动作语义复现成功。

建议报告表述：

> The Hub single-leg balance motion can be rendered as a plausible MuJoCo reference visualization. However, the current teacher/VAE/diffusion/guided variants remain conservative and do not reproduce the characteristic lifted-leg balance pose. Therefore, this result is useful as a diagnostic chain check, but not as paper-level successful skill reproduction.

## Next Step

如果要继续把这条动作做好，优先方向不是调视频，而是：

1. 为 balance motion 单独训练/筛选更强 teacher policy。
2. 用该 teacher 重新采集 balance rollout dataset。
3. 重新训练或 fine-tune VAE/diffusion。
4. 再生成同一套六视频对照。

## Git Commit

Pending.
