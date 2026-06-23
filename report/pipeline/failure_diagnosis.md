# Failure diagnosis map

1. **Data**: Check retargeting, joint order, root frame, FPS, impossible segments.
2. **Teacher**: Low reward/high done means downstream models imitate weak behavior.
3. **VAE/diffusion**: Offline MSE can improve while physical rollout remains invalid.
4. **Deployment**: Check action scale, PD gain, default pose, obs normalization, joint map.
