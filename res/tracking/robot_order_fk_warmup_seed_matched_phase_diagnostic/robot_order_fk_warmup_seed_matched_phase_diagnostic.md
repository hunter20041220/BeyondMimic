# Robot-Order FK Warmup Seed-Matched Phase Diagnostic

This is a post-hoc analysis over completed full checkpoint-eval traces. It does not claim paper-level tracking.

## Key Findings

- Non-warmup done rate: `0.1782798129180602`.
- Seed-matched warmup done rate: `0.22153761235367894`.
- Same-seed done-rate delta: `0.04325779943561875`.
- Step-0 done count delta: `-1452.0`.
- Step-0 body-error delta: `-43.02924793958664`.
- Post-step0 done-rate delta: `0.04578210203439598`.
- ee_body_pos termination fraction delta: `0.04554896530100333`.
- Sampling top-bin post-step0 delta: `0.0`.

## Interpretation

Seed matching confirms that reset-command warmup is not a teacher-quality fix. It removes the stale step-0 body target spike, but post-step0 done rate and ee_body_pos termination increase while the adaptive-sampling top bin stays unchanged. The likely next target is command/observation phase consistency: refresh motion targets after reset without introducing a one-command-step mismatch, or apply the same reset warmup consistently during training and evaluation.

## Next Experiment

Run a targeted reset-target refresh variant that recomputes body_pos_relative_w at reset without advancing MotionCommand.time_steps, then only run full PPO after this termination gate improves.
