# Official-CSV-Loop Local Receding-Horizon Latent Guidance Rollout

This directory contains one local virtual closed-loop task-conditioned rollout comparing teacher, VAE base, denoised-latent, and guided-latent variants.

The guided-latent variant recomputes a short state-latent horizon at each control step, applies the local denoiser and one composed-cost guidance update, then decodes the current latent through the local VAE.

## Claim Level

local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout. This is not official BeyondMimic latent diffusion guidance, not Fig. 5/Fig. 6 paper-level evidence, and not real-robot validation.
