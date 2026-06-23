# 失败分析：为什么当前运动控制视频效果差

## 1. 现象

最新六条 MuJoCo action-control 视频已经保证连续 motion segment，不再是跳变拼接；但机器人仍然不能稳定完成动作，teacher/VAE/diffusion/guided variants 都表现出明显失稳。

## 2. 主要原因排序

### 原因一：Stage 1 teacher 本身很弱

best teacher 指标：

- reward mean：`0.0241314`
- body error mean：`1.0095`
- joint error mean：`1.67395`
- non-timeout done rate：`0.194137`

这说明 teacher 还没有学到高质量 motion tracking。后续 VAE/diffusion 学到的是这个弱 teacher 的 action distribution。

### 原因二：rollout 数据带有大量 done/fall 信号

teacher rollout done count：`118220`。这会污染 VAE 和 state-latent diffusion 的训练分布。

### 原因三：离线 MSE 和闭环控制不是一回事

VAE MSE、diffusion MSE 都是离线指标。机器人在 MuJoCo 里会受到接触、动力学、PD 控制、root height、joint limit、action scale 等因素影响。

### 原因四：IsaacLab -> MuJoCo adapter 仍可能有 contract gap

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

下一步不要优先美化视频，而是先把 Stage 1 teacher 修好。teacher 稳定之前，VAE/diffusion/guidance 的闭环视频很难好看。
