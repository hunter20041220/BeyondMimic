# Robot-Order FK Reset / Termination Alignment Audit

This audit does not launch simulation or claim paper-level tracking. It joins current robot-order FK data,
existing eval traces, and official source order to define the next live tracking fix.

## Findings

- Motion bundle status: `ok_fk_repaired_robot_order_motion_npz`; motions `40`, frames `11960`.
- Split zero-action done rate: `0.1811036789297659` from `2166/11960` done signals.
- Multi-seed step0 done-rate mean: `1.0`.
- Multi-seed step0 body-error mean: `43.29219436645508`.
- Multi-seed step0 anchor-error mean: `0.039055753499269485`.
- Multi-seed step1 body-error mean: `0.2652200361092885`.

## Source-Linked Interpretation

- `MotionCommand` zero-initializes `body_pos_relative_w` and later populates it in `_update_command()`.
- `ee_body_pos` uses z-only ankle/wrist body-position termination with a 0.25 m threshold.
- `ManagerBasedRLEnv.step()` computes termination before `command_manager.compute()`.
- Therefore the next live probe should test command warmup immediately after reset before doing more PPO.

## Next Mainline Probe

Run a small GPU-4 IsaacLab probe that records endpoint z-errors before and after an explicit command update
after reset. If the first-step all-done spike disappears, patch the local train/eval wrappers to warm up
commands after reset, then rerun full tracking eval before rebuilding downstream VAE/diffusion artifacts.

## Claim Boundary

This is tracking-quality diagnosis only. It is not paper-level tracking, not DAgger, not Fig. 5/Fig. 6,
not TensorRT deployment, and not real-robot evidence.
