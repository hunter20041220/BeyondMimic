# Scaled PPO Checkpoint Completion Proxy

This report asset summarizes local completion/termination proxies for the iteration-999
official-importer-export scaled PPO checkpoint evaluation.

## Metrics

- num envs: `2048`
- eval steps: `299`
- attempted env-steps: `612352`
- done count total: `611642`
- local completion proxy rate: `0.0011594638377926403`
- local non-timeout done rate: `0.9988405361622074`
- timeout rate: `0.0`

## Claim Boundary

This is a local virtual proxy over an existing checkpoint eval. It is not the paper's official
success/fall/collision protocol, not an official BeyondMimic teacher checkpoint evaluation,
not DAgger, not Fig.5/Fig.6 guided diffusion, and not real robot evidence.
