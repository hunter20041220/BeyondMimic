# Scaled PPO Reward / Termination Diagnostic

The ee_body_pos termination component dominates both evaluated local checkpoints, explaining the near-total non-timeout done counts. This points to tracking/termination configuration or teacher quality as the next mainline diagnosis target.

Dominant final termination: `ee_body_pos` at `0.9988405361622074` of envs per step.
Dominant best-checkpoint termination: `ee_body_pos` at `0.9985874137750836` of envs per step.

Claim level: local virtual diagnostic only. This is not a paper-level BeyondMimic result.
