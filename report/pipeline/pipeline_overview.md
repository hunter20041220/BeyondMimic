# BeyondMimic reproduction pipeline

1. **Data sources**: LAFAN1 G1 CSV, Zenodo result data, HuB candidates; exact paper curated set not fully public.
2. **Motion preprocessing**: Validate G1 generalized coordinates, FK/body tensors, metadata, duration statistics.
3. **PPO teacher**: Official whole_body_tracking / IsaacLab task; latest 5/6 run completes but teacher remains weak.
4. **Teacher rollout**: Collect obs/action/reward/done/motion_time_steps from selected local teacher.
5. **Conditional VAE**: Encode teacher action distribution into 32-D latent and decode action from obs+z.
6. **State-latent diffusion**: Train denoiser over 21-token state+latent windows; token MSE improves.
7. **Classifier guidance**: Offline joystick/waypoint/smoothness proxy costs; not paper-level closed loop.
8. **MuJoCo video**: Render continuous action-to-PD diagnostics; videos reveal instability/fall proxies.
