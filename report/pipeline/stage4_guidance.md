# Stage 4: Test-time guidance

1. **Input**: Unconditional denoiser output and task cost.
2. **Gradient**: Compute task-cost gradient w.r.t. trajectory tokens.
3. **Receding action**: Decode current latent to action and step simulation.
4. **Current status**: Offline proxy guidance only for this chain; closed-loop task success absent.
