# Official-importer-export local proxy success boundary

This folder summarizes 20 local virtual task-conditioned latent-guidance rollouts across 5 seed groups
on the official-importer-export G1 USDA path over the 40-motion public bundle.

The rates are local proxy diagnostics only: 299-step completion, positive guidance signal,
action change, guided reward improvement over the denoised baseline, and guided tracking error
not worsening relative to the denoised baseline.

Claim boundary: these are not official BeyondMimic Fig. 5/Fig. 6 success rates, not official
checkpoints, not TensorRT deployment metrics, and not real-robot validation.
