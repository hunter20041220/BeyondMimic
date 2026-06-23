# Stage 3: State-latent diffusion

1. **Input**: State-latent token windows from teacher rollout and VAE.
2. **Noising**: x_t = sqrt(a_bar)x_0 + sqrt(1-a_bar)eps.
3. **Denoising**: MLP denoiser predicts clean token from noisy token + timestep.
4. **Current result**: Noisy MSE 0.0728 -> pred MSE 0.04322, about 40.6% improvement.
