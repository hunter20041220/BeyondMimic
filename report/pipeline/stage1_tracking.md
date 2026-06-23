# Stage 1: PPO motion tracking

1. **Input**: Reference motion NPZ, Unitree G1 asset, tracking reward, PPO config.
2. **Policy loop**: obs -> PPO actor -> 29-D action -> PD target in physics.
3. **Reward/termination**: Tracking terms and reset gates decide learning signal.
4. **Current result**: Best 5/6 checkpoint reward mean ~0.024 and high error: weak teacher.
