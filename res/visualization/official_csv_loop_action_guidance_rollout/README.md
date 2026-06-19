# Official-CSV-Loop Local Action Guidance Rollout

This directory contains a local virtual closed-loop rollout comparing teacher, VAE base, and action-guided variants.

The guided action is `a_guided = a_vae + alpha * (a_teacher - a_vae)` with alpha recorded in the asset JSON.

## Claim Level

local_virtual_teacher_consistency_action_guidance_rollout. This is not official BeyondMimic latent diffusion guidance, not Fig. 5/Fig. 6 paper-level evidence, and not real-robot validation.
