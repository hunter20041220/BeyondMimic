# 失败分析：为什么当前运动控制视频效果差

## 1. 现象

最新六条 MuJoCo action-control 视频已经保证连续 motion segment，不再是跳变拼接；但机器人仍然不能稳定完成动作，teacher/VAE/diffusion/guided variants 都表现出明显失稳。

## 2. 主要原因排序

### 原因一：当前视频选段/root target 本身不适合作为站立运动展示

本轮新增诊断脚本确认，最新六条视频虽然已经满足 `motion_time_steps` 连续、`done_count=0`，但自动选中的片段来自：

- source motion：`lafan1_walk3_subject4`
- global motion steps：`418177..418474`
- selected root z：min `0.0459` m，mean `0.0512` m，max `0.0544` m
- whole source root z median：`0.7723` m
- selected reward mean：`-0.081968`

也就是说，视频 pipeline 选中了一个接近地面的片段。`reference_action_control` 会直接把这段低 root target 作为 pose replay；后面的 teacher/VAE/diffusion/guided action-control 又共享同一段 root target，并启用了 root assist，因此控制器会围绕一个近地面目标运行。机器人站不稳首先是这个 root target/segment selection 问题，不应该简单归因为“视频渲染不好”。

更重要的是，按旧规则 `length` 优先、再看 `reward_mean`，前 10 个长连续候选大多是近地面或 get-up/fall 片段；按 `>=60` 帧且 root z 正常的门槛，目前候选数为 `0`。这说明当前 teacher rollout 里还没有足够长、稳定、站立高度正常的连续片段可用于 15 秒展示视频。

新增证据：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_candidate_segments.tsv`

### 原因二：Stage 1 teacher 本身很弱

best teacher 指标仍然不好：

- reward mean：`0.0241314`
- body error mean：`1.0095`
- joint error mean：`1.67395`
- non-timeout done rate：`0.194137`

这说明 teacher 还没有学到高质量 motion tracking。后续 VAE/diffusion 学到的是这个弱 teacher 的 action distribution。

### 原因三：rollout 数据带有大量 done/fall 信号

teacher rollout done count：`118220`。这会污染 VAE 和 state-latent diffusion 的训练分布。

### 原因四：离线 MSE 和闭环控制不是一回事

VAE MSE、diffusion MSE 都是离线指标。机器人在 MuJoCo 里会受到接触、动力学、PD 控制、root height、joint limit、action scale 等因素影响。

### 原因五：IsaacLab -> MuJoCo adapter 仍可能有 contract gap

需要继续查：

- joint order
- action scale
- PD gain
- default joint pose
- obs normalization
- last action
- root twist / IMU-like state
- termination semantics
- reference phase 和 motion_time_step

## 3. 当前视频应该怎么表述

正确说法：

```text
These videos are continuous MuJoCo local action-control diagnostics.
They reveal that the current teacher/control chain is still unstable.
```

错误说法：

```text
These videos reproduce BeyondMimic Fig.5/Fig.6.
```

## 4. 结论

下一步不要优先美化视频，也不要继续把当前六条视频当成功结果。正确顺序是：

1. 把旧六条视频标记为 failed diagnostic；
2. 修改 segment selector，加入 root height、reward、stability 门槛；
3. 先生成 root 高度正常的 reference pose replay；
4. 再生成同一片段的 teacher action-control；
5. 如果 root target 正常后 teacher 仍然不稳，再回到 Stage 1 PPO teacher 训练和 MuJoCo adapter。

teacher 稳定之前，VAE/diffusion/guidance 的闭环视频很难好看，也不能作为 paper-level result。
