# 公式和符号说明

## PD target action

```text
theta_sp = theta_0 + alpha * action
tau ~= Kp * (theta_sp - theta) - Kd * theta_dot
```

## VAE loss

```text
L = MSE(action_hat, action) + beta * KL(q(z|obs, action) || N(0, I))
```

## Diffusion denoising

```text
clean_token = concat(obs, latent)
noisy_token = clean_token + sigma * noise
loss = MSE(denoiser(noisy_token, sigma), clean_token)
```

## Guidance

```text
guided_sample = sample - lambda * grad(task_cost(sample))
```
