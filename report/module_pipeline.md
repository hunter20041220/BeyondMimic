# BeyondMimic 复现流程图解

## 主线

```text
Stage 1: reference motion -> PPO motion tracking teacher
Stage 2: teacher rollout -> conditional VAE
Stage 3: state + latent trajectory -> diffusion denoiser
Stage 4: task cost / classifier guidance -> guided latent trajectory
Stage 5: VAE decoder -> action -> MuJoCo PD control -> video and metrics
```

## 每一阶段输入输出

### Stage 1：Motion Tracking Teacher

- 输入：G1 reference motion bundle、IsaacLab task、reward/termination/PPO 配置。
- 输出：PPO checkpoint / policy action。
- 当前状态：best iteration `29999`，但 reward 低、error 高。

### Stage 2：Teacher Rollout

- 输入：Stage 1 best teacher。
- 输出：obs/action rollout dataset。
- 当前状态：`612352` env steps，done count `118220`。

### Stage 3：Conditional VAE

- 输入：teacher rollout 的 obs/action。
- 输出：encoder、decoder、latent action。
- 当前状态：test action MSE `0.00328968`。

### Stage 4：State-Latent Diffusion

- 输入：obs + latent token windows。
- 输出：denoised state-latent sequence。
- 当前状态：MSE `0.0728163` -> `0.0432214`。

### Stage 5：Guidance + MuJoCo

- 输入：当前状态、task cost、diffusion future plan、VAE decoder。
- 输出：action-to-PD closed-loop control。
- 当前状态：视频连续但动作差，仍是 failure/diagnostic。

## 关键控制公式

```text
theta_sp = theta_0 + alpha * action
tau ~= Kp * (theta_sp - theta) - Kd * theta_dot
```

注意：真正的运动控制必须输出 action 再进入物理仿真，不能直接把 reference qpos 写进机器人状态冒充 policy。
