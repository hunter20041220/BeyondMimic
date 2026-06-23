# Stage 2: Conditional action VAE

1. **Input**: Teacher rollout obs/action pairs.
2. **Encoder**: q(z|obs, action) produces 32-D latent posterior.
3. **Decoder**: D(obs,z) reconstructs 29-D action.
4. **Current result**: Low offline action MSE, but closed-loop stability not proven.
