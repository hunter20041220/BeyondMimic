# 中文伪代码：BeyondMimic 本地复现主线

## Algorithm 1：Stage 1 Motion Tracking Teacher

```text
输入：G1 reference motions、IsaacLab tracking task、PPO config
输出：motion tracking teacher policy

1. 读取并验证 reference motion bundle
2. 创建 Tracking-Flat-G1-v0 环境
3. 初始化 PPO actor-critic
4. 对每个训练 iteration:
      obs <- env.reset/step 返回的机器人状态和 reference phase
      action <- policy(obs)
      theta_sp <- theta_0 + alpha * action
      env.step(theta_sp)
      reward <- DeepMimic-style tracking reward + smoothness terms
      done <- termination / timeout / fall
      PPO update
5. 每 500 iteration 保存 checkpoint
6. checkpoint sweep 选择 best teacher
```

## Algorithm 2：Teacher Rollout Collection

```text
输入：best teacher checkpoint
输出：state-action rollout dataset

for each environment shard:
    reset env
    for t in rollout horizon:
        obs_t = current observation
        action_t = teacher(obs_t)
        next_obs, reward, done = env.step(action_t)
        save(obs_t, action_t, reward, done, motion_id, motion_time_step)
```

## Algorithm 3：Conditional VAE

```text
输入：teacher rollout obs/action
输出：encoder E、decoder D

for batch in rollout_dataset:
    mu, logvar = E(obs, action)
    z = mu + eps * exp(0.5 * logvar)
    action_hat = D(obs, z)
    loss = MSE(action_hat, action) + beta * KL(q(z|obs,action) || N(0,I))
    update(E, D)
```

## Algorithm 4：State-Latent Diffusion

```text
输入：obs 序列、VAE latent z 序列
输出：denoiser

token_t = concat(obs_t, z_t)
window = [token_t, ..., token_{t+H}]

for batch in windows:
    sigma = sample_noise_level()
    noisy = window + sigma * noise
    pred = denoiser(noisy, sigma)
    loss = MSE(pred, window)
    update(denoiser)
```

## Algorithm 5：Guided Receding-Horizon Control

```text
输入：当前 MuJoCo state、历史 state-latent、task cost
输出：物理仿真中的连续 action control

while episode not done:
    current_state = read_mujoco_state()
    sample future state-latent trajectory using diffusion
    apply guidance: trajectory <- trajectory - lambda * grad(task_cost)
    z_t = first latent token from guided trajectory
    action_t = VAE_decoder(obs_t, z_t)
    theta_sp = theta_0 + alpha * action_t
    mujoco.step(theta_sp)
    render frame and log metrics
```

当前项目只部分做到 Algorithm 5；视频仍然是诊断失败结果，不是论文级成功控制。
