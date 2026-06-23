# 关键代码复现说明

这里不是把所有源码完整贴一遍，而是告诉你报告里应该引用哪些核心代码、每个脚本在论文流程里对应哪一步。

## 1. Stage 1 teacher checkpoint sweep

路径：

```text
reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py
reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py
```

作用：遍历 5/6 卡 multi-source PPO 训练保存的 checkpoint，用同一套 eval 合同筛选 best teacher。它对应论文第一阶段 motion tracking policy 的本地复现入口。

伪代码：

```python
for checkpoint in saved_checkpoints:
    env = make_tracking_env(task="Tracking-Flat-G1-v0")
    policy = load_checkpoint(checkpoint)
    metrics = rollout_eval(env, policy)
    save_metrics(checkpoint, metrics)
select_best_checkpoint(metrics_table)
```

## 2. Teacher rollout dataset

路径：

```text
reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py
```

作用：用 best teacher 在环境里 rollout，保存 obs/action/reward/done/motion_time_steps。这一步对应 VAE 和 diffusion 的数据来源。

## 3. Conditional VAE

路径：

```text
reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py
```

核心逻辑：

```python
mu, logvar = encoder(obs, action)
z = reparameterize(mu, logvar)
action_hat = decoder(obs, z)
loss = mse(action_hat, action) + beta * kl(mu, logvar)
```

这对应论文里把 teacher 高维 action 压缩为 latent action 的模块。

## 4. State-latent dataset

路径：

```text
reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py
```

作用：把 obs 和 VAE latent 拼成 token window：

```python
token_t = concat(obs_t, z_t)
trajectory = [token_t, token_{t+1}, ...]
```

## 5. Diffusion denoiser

路径：

```text
reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py
```

核心逻辑：

```python
clean_tokens = state_latent_window
noise = sample_gaussian()
noisy_tokens = clean_tokens + sigma * noise
pred_tokens = denoiser(noisy_tokens, timestep)
loss = mse(pred_tokens, clean_tokens)
```

## 6. Guidance

路径：

```text
reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py
```

核心逻辑：

```python
for guidance_scale in scales:
    guided_sample = diffusion_sample - scale * grad(task_cost)
    score = evaluate_task_cost(guided_sample)
```

当前这一步仍是 offline proxy，不是 paper Fig.5/Fig.6 闭环控制。

## 7. MuJoCo 连续视频

路径：

```text
reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py
```

核心要求：

```text
必须选连续 motion_time_steps
必须用 action -> PD target -> mujoco step
不能直接设置 reference qpos 冒充控制
不能把 offline 21-step sample 硬拉成 15 秒视频
```

## 8. 报告生成和中文化

路径：

```text
reproduction/scripts/generate_report_package.py
reproduction/scripts/localize_report_to_chinese.py
```

`generate_report_package.py` 生成基础报告包；`localize_report_to_chinese.py` 把报告重写成中文并生成文件地图。
