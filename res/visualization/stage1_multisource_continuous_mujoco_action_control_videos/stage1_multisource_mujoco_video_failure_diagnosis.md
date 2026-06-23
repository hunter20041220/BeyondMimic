# Stage-1 MuJoCo 视频失败诊断

## 结论

当前视频差的首要原因不是视频编码，也不是单纯 VAE/diffusion 公式写错，而是前端控制证据链已经坏了：自动选中的连续片段虽然 `motion_time_steps` 连续、`done_count=0`，但 root/pelvis target z 只有约 `0.0512 m`，接近地面；同一个 source motion 的中位 root z 为 `0.7723 m`。后续 MuJoCo root assist 和 PD action-control 都围绕这个近地面目标运行，所以机器人站不稳是预期结果。

## 关键证据

- 选中 motion: `lafan1_walk3_subject4`
- 选中全局 motion steps: `418177..418474`
- 选中片段 root z: min `0.0459`, mean `0.0512`, max `0.0544` m
- 整个 source root z: min `0.0457`, median `0.7723`, max `0.8066` m
- 选中片段 reward mean: `-0.081968`
- 当前选择规则: `length` 优先，再看 `reward_mean`；因此会选择长但不可展示的坏片段。
- teacher best reward mean: `0.024131`
- teacher body error mean: `1.009504` m
- teacher non-timeout done rate: `0.194137`
- 60 帧及以上、root z 正常的连续候选数: `0`
- 30 帧及以上、root z 正常的连续候选数: `170`

## 为什么六个视频都会差

这六个视频共享同一段 reference/root target。`reference_action_control` 直接做 pose replay，因此它会首先暴露低 root 高度问题；`teacher_policy_action_control`、`vae_reconstructed_action_control`、`diffusion_denoised_latent_action_control` 和 `guided_latent_action_control` 又共享这个 root target，同时 teacher 本身 reward 很低、done/fall 很多，所以它们不是独立失败，而是同一条坏数据/弱 teacher 链路的下游表现。

## 下一步修复顺序

1. 旧六个视频标记为失败诊断，不作为展示结果。
2. 修改 segment selector：要求 root z 不低于站立阈值，优先 reward/stability，再考虑 length。
3. 先重新生成一条 root 高度正常的 reference pose replay。
4. 再在同一连续片段上跑 teacher action-control。
5. teacher 稳定后才继续 VAE/diffusion/guidance 视频；否则下游视频只会学习弱 teacher 和坏 target。

## Claim Boundary

这份诊断只说明本地 MuJoCo 视频链路为什么失败；它不是 BeyondMimic paper-level 复现结果，也不是真实机器人结果。

JSON: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.json`
候选片段 TSV: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_candidate_segments.tsv`
