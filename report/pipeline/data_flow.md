# Data flow

1. **Raw / released sources**: BVH, 36-col CSV, MCAP/rosbag-derived released evidence.
2. **Train-ready references**: G1 generalized-coordinate CSV/NPZ with joint/body tensors.
3. **Stage1 bundle**: 49 motions / 2.49 h local public+available bundle.
4. **Rollout shards**: 612,352 local weak-teacher state-action samples.
5. **State-latent windows**: 571,392 windows with 160-D obs + 32-D latent.
