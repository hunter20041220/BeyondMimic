# LAFAN1 jumps1_subject1 MuJoCo Clean Baseline

本目录展示原始 Unitree-retargeted LAFAN1 `jumps1_subject1.csv` 的 MuJoCo baseline。

Claim boundary: source/reference baseline only; not teacher/RL, not VAE, not diffusion, not guidance, not real robot.

- Window: `stable_dynamic_164s_179s` frames `4920:5370` (164.00-179.00s at 30 FPS)
- Original CSV replay MP4: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/original_csv_reference_replay/original_csv_reference_replay.mp4`
- Reference action control MP4: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/reference_action_control/reference_action_control.mp4`
- Reference action control fall_proxy_count: `0`
- Reference action control mean joint error: `0.151691`

这一步只证明原始动作和 reference PD baseline 可在 MuJoCo 中展示；不能解锁 teacher/VAE/diffusion/guidance 长训练。
