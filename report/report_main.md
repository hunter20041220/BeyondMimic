# BeyondMimic 复现工程中文技术报告

中文化时间：`2026-06-23T08:20:26.138903+00:00`

## 0. 一句话总览

这个工程已经搭起了一条接近 BeyondMimic 思路的本地复现链路：

```text
多来源 G1 reference motions
  -> whole_body_tracking / IsaacLab PPO motion tracking teacher
  -> teacher rollout 状态-动作轨迹
  -> conditional VAE 压缩 action 到 latent action
  -> state-latent diffusion denoiser
  -> classifier / task-cost guidance
  -> MuJoCo 中 action-to-PD 闭环控制和视频诊断
```

但是当前还不能说完成论文级复现。最主要原因是 Stage 1 teacher policy 质量仍然弱，MuJoCo action-control 视频不能稳定完成正常动作。也就是说，VAE 和 diffusion 的离线指标有进步，但底层 teacher/control 分布质量不够好，导致闭环视频效果差。

## 1. 论文 BeyondMimic 的核心思路

BeyondMimic 不是简单播放动作动画，而是把 humanoid control 拆成三层：

1. **Motion tracking teacher**：先用强化学习让 G1 机器人跟踪大量人类/动画 reference motion，得到能在物理环境中输出 action 的 teacher policy。
2. **Latent action policy / VAE**：用 teacher rollout 得到状态-动作轨迹，再训练 conditional VAE，把高维 action 压缩为 latent action。
3. **State-latent diffusion + guidance**：在 state + latent 序列上训练 diffusion model，推理时用 joystick、waypoint、inpainting、obstacle cost 等任务代价做 guidance。

论文里真正关键的不是“画骨架”或“直接设置 qpos”，而是：

```text
当前机器人状态 -> policy / VAE / diffusion 输出 action -> PD controller -> 物理仿真 step -> 新状态反馈
```

本项目现在正在沿着这条主线复现，但 teacher 质量还没有达到论文级。

## 2. 本项目当前做到了哪一步

### 2.1 数据和 motion bundle

当前 Stage 1 multi-source bundle：

- motion 数量：`49`
- 总时长：`2.491` 小时
- 总帧数：`448358`
- 来源统计：

```json
{
  "BeyondMimic Zenodo ablation reference CSV": 1,
  "HuB supplemental 29-DoF pkl": 8,
  "Unitree-retargeted LAFAN1": 40
}
```

这个时长接近论文提到的约 2.5 小时 motion pool，但不能说它就是作者未公开的 exact curated 2.5h 数据集。当前包含的是本机能审计、能转换、能进入训练合同的公开/本地可用动作。

### 2.2 Stage 1 motion tracking teacher

使用 `HybridRobotics/whole_body_tracking` 相关任务、G1 obs/action 合同和 PPO 训练流程，5/6 卡 multi-source 训练完成后做了 checkpoint sweep。

当前 best checkpoint：

- best iteration：`29999`
- reward mean：`0.0241314`
- local non-timeout done rate：`0.194137`
- body-position error mean：`1.0095`
- joint-position error mean：`1.67395`

结论：这条 teacher 可以用于打通后续流程，但质量明显不够，不能当作稳定 teacher policy，更不能当作 BeyondMimic 官方 teacher。

### 2.3 Teacher rollout 数据

用 best teacher 采集了 rollout 数据：

- rollout env steps：`612352`
- shard 数：`2`
- done count：`118220`

这些数据可以用于本地 VAE/diffusion 训练，但它们继承了弱 teacher 的问题，因此是 partial evidence。

### 2.4 Conditional VAE

VAE 的作用是学习：

```text
obs_t, action_t -> encoder -> latent z_t
obs_t, z_t -> decoder -> reconstructed action_t
```

当前结果：

- test action MSE：`0.00328968`
- test action absolute error mean：`0.0425109`

这个指标说明 action reconstruction 离线可用，但不能证明 VAE decoder 放回物理环境后就能稳定控制。

### 2.5 State-latent diffusion

当前 state-latent dataset：

- window count：`571392`
- token dim：`192`

Diffusion denoiser 结果：

- noisy token MSE：`0.0728163`
- pred token MSE：`0.0432214`
- 相对 denoising improvement：`40.64%`

这是本轮最清楚的正向结果：模型确实学会了一定 token-level denoising。但它还是离线 token 预测，不能直接等同于 humanoid closed-loop control 成功。

### 2.6 Guidance

当前 guidance 是 offline proxy：

- selected windows：`8192`
- rows/tasks：`48`

它说明 guidance cost 可以对 diffusion sample 产生非零梯度和局部改进，但还不是论文 Fig.5/Fig.6 那种真实闭环任务验证。

### 2.7 MuJoCo 视频

最新六条视频已经改成连续片段，不再是 reset 拼接或硬拉长 offline sample：

```text
reference_action_control.mp4
teacher_policy_action_control.mp4
vae_reconstructed_action_control.mp4
diffusion_denoised_latent_action_control.mp4
guided_latent_action_control.mp4
guided_vs_unguided_action_control.mp4
```

视频检查：

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

但视频效果仍差，fall proxy 很高，所以它们只能写作“MuJoCo local action-control diagnostic videos”，不能写成 paper-level simulation result。

## 3. 为什么现在效果不好

当前失败不是单纯“视频脚本错了”，而是链路上游质量不足：

1. **teacher policy 弱**：reward 很低，done/fall 高，body/joint error 高。
2. **rollout 数据质量弱**：VAE 和 diffusion 学到的是弱 teacher 的 action distribution。
3. **MuJoCo adapter 仍有 gap**：IsaacLab 训练出的 obs/action/PD/action scale/termination 与 MuJoCo 控制闭环并非天然完全一致。
4. **offline 指标不能保证 closed-loop 成功**：MSE 下降只说明 token denoising 学到了统计结构，不说明机器人不会摔。

所以后续不要盲目继续堆 VAE/diffusion，而要优先把 Stage 1 teacher 修到稳定跟踪。

## 4. 当前结果可以怎么写进报告

可以写：

- 已经完成公开资料、官方 Stage 1 代码、数据来源和复现边界审计；
- 已经构建约 2.49h 的本地 multi-source motion bundle；
- 已经跑通 PPO teacher -> teacher rollout -> VAE -> state-latent dataset -> diffusion -> offline guidance -> MuJoCo diagnostic video 的完整本地链路；
- diffusion denoising 指标从 `0.0728163` 降到 `0.0432214`，约 `40.64%` improvement；
- 当前闭环控制视频效果差，说明 teacher/control 质量仍是主要 blocker。

不能写：

- 不能说已经复现 BeyondMimic 完整 paper-level 结果；
- 不能说当前 MuJoCo 视频等于 Fig.5/Fig.6；
- 不能说 VAE/diffusion checkpoint 是官方 checkpoint；
- 不能说仿真结果是真实机器人结果。

## 5. 后续最重要的工作

1. 先选一个 clean single motion，把 Stage 1 teacher 练到稳定；
2. 严查 reward、termination、reset、action scale、PD gain、joint order、obs normalization；
3. teacher 稳定后重新采集 teacher rollout；
4. 用高质量 rollout 重训 VAE；
5. 再构造 state-latent dataset 和 diffusion；
6. 最后再做 receding-horizon closed-loop guidance 和 MuJoCo/Isaac 视频。

## 6. 文件导航

先看 `REPORT_FILE_MAP.md`。它告诉你每个文件是什么、哪些适合写报告、哪些只是自动清单。

## 7. Claim Boundary

当前不得声称完整复现 BeyondMimic。当前 MP4 是本地 MuJoCo 虚拟诊断视频，不是真实机器人结果，不是官方 Isaac rendered paper video，也不是 paper-level Fig.5/Fig.6 结果。
