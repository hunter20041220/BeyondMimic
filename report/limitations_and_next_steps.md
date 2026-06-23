# 下一步建议

## 第一优先级：修 Stage 1 teacher

1. 选一个最干净、最短、最容易跟踪的单一 motion。
2. 确认 reference qpos/qvel/body pose 连续且物理合理。
3. 检查 reward 和 termination，不要让 wrist/endpoint 过早终止支配训练。
4. 检查 action scale、PD gain、joint order、default pose。
5. 跑 single-motion PPO，先追求稳定动作，而不是一开始追求 2.5h 全量。

## 第二优先级：重新采集高质量 teacher rollout

teacher 能稳定后再采集 state-action trajectory。否则 VAE/diffusion 会继续学习失败动作。

## 第三优先级：重训 VAE 和 diffusion

用高质量 rollout 重新训练：

```text
teacher rollout -> VAE -> state-latent dataset -> diffusion denoiser
```

## 第四优先级：重做 closed-loop guidance 视频

先做：

1. reference replay
2. teacher policy rollout
3. VAE decoder rollout
4. diffusion unguided rollout
5. guided rollout
6. guided-vs-unguided comparison

每条视频都必须保证：

- 单一连续 motion 或连续 receding-horizon context；
- action -> PD target -> MuJoCo step；
- 不直接写 qpos 冒充控制；
- 不把 21-step offline sample 硬拉成十几秒视频。
