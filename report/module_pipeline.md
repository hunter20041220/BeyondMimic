# Module Pipeline

## Module 0: Project code base

- Official Stage 1 code: `download/official/whole_body_tracking`
- Official deployment/controller reference: `download/official/motion_tracking_controller`
- IsaacLab: `download/dependencies/IsaacLab-v2.1.0`
- Local reproduction scripts: `reproduction/scripts`
- MuJoCo package: `mujoco_mp4`
- Report-ready official released-data replays: `official_mp4`

## Module 1: Data

G1-retargeted LAFAN1, one train-ready Zenodo tkd CSV, and HuB candidates were converted into the current 49-motion / 2.49h local bundle.

## Module 2: Stage1 teacher

PPO training completed on GPUs 5/6, but current reward/error metrics show a weak teacher.

## Module 3: Teacher rollout

The selected weak teacher generated local rollout shards used by VAE/diffusion.

## Module 4: Conditional VAE

Offline action reconstruction works, but true DAgger/closed-loop VAE success is not proven.

## Module 5: State-latent diffusion

Token denoising improves by about 40.6%, but physical control remains unstable.

## Module 6: Guidance

Offline guidance proxy works over selected windows. Paper-level closed-loop joystick/waypoint/inpainting/obstacle success is not reproduced.

## Module 7: MuJoCo/Isaac rendering

MuJoCo videos are generated; true Isaac rendered MP4 on H20 is blocked by rendering stack.
