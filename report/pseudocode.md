# Pseudocode for All Stages

## Algorithm 1: Motion Preprocessing

```text
Input: retargeted G1 CSV / BVH-like source, expected joint order, FPS
Output: reference motion NPZ with q_ref, v_ref, body poses, body velocities, metadata
1. Load source motion table.
2. Validate root pose and 29 actuated joint columns.
3. Convert or verify generalized coordinates in Unitree G1 order.
4. Run FK to recover body positions/orientations/velocities.
5. Reject sources whose DoF mapping is not audited.
6. Save per-motion NPZ and a bundle manifest with duration/source provenance.
```

## Algorithm 2: PPO Motion Tracking Teacher Training

```text
Input: processed reference bundle, G1 asset, tracking task config, PPO config
Output: teacher policy checkpoint
for each environment reset:
    sample a motion and phase
    initialize robot near the reference state
for each control step:
    build observation from proprioception and reference tracking cues
    action = PPO_actor(observation)
    theta_sp = theta_default + action_scale * action
    physics.step(PD(theta_sp))
    reward = tracking_reward + regularization
    done = termination_checks(robot_state, reference_error)
    store transition
after horizon:
    update PPO actor/critic with RSL-RL
    save checkpoint every configured interval
```

## Algorithm 3: Teacher Rollout Collection

```text
Input: trained teacher checkpoint, reference bundle, simulation environment
Output: state-action rollout shards
load teacher policy
for each rank/env:
    reset to sampled motion phase
    for T steps:
        obs_t = env.get_obs()
        a_t = teacher(obs_t)
        obs_{t+1}, reward_t, done_t = env.step(a_t)
        record obs_t, a_t, reward_t, done_t, motion_time_step_t
        if done: record reset boundary; do not stitch across resets for video evidence
save NPZ shards and metrics
```

## Algorithm 4: Conditional VAE Training

```text
Input: teacher rollout obs/action pairs
Output: VAE checkpoint and latent action representation
for minibatch (obs, action):
    mu, logvar = Encoder([obs, action])
    z = mu + exp(0.5 * logvar) * epsilon
    action_hat = Decoder([obs, z])
    loss = MSE(action_hat, action) + beta * KL(q(z|obs,action) || N(0,I))
    update encoder and decoder
```

## Algorithm 5: DAgger Loop (paper target; current project is partial)

```text
Input: student/VAE policy, teacher policy, simulation environment
Output: aggregated on-policy state-action dataset
dataset = initial teacher rollouts
repeat:
    rollout student policy in simulation
    at visited states, query teacher action
    append (student_state, teacher_action) to dataset
    retrain or fine-tune VAE/student
Current project: offline teacher-rollout VAE exists; full official DAgger logs are not available.
```

## Algorithm 6: State-Latent Trajectory Dataset

```text
Input: teacher rollout observations, actions, trained VAE
Output: windows of tau = [state, latent] tokens
for each rollout shard:
    infer z_t from VAE posterior
    concatenate token_t = [obs_t, z_t]
    create fixed-length windows of 21 tokens
    split windows into train/validation/test
save state-latent dataset and index metadata
```

## Algorithm 7: Diffusion Denoiser Training

```text
Input: clean state-latent token windows x0
Output: denoiser checkpoint
for minibatch x0:
    sample timestep k
    epsilon ~ Normal(0, I)
    x_k = sqrt(alpha_bar[k]) * x0 + sqrt(1 - alpha_bar[k]) * epsilon
    x0_hat = Denoiser(x_k, k)
    loss = MSE(x0_hat, x0)
    update denoiser
```

## Algorithm 8: Guided Diffusion Inference

```text
Input: current state, denoiser, task cost C(tau), VAE decoder
Output: current action for receding-horizon control
initialize noisy future trajectory tau_K
for k from K to 1:
    tau_hat = denoiser(tau_k, k)
    cost = C(tau_hat)
    grad = d cost / d tau_hat
    tau_{k-1} = reverse_step(tau_k, tau_hat) - guidance_scale * grad
take current latent z_t from tau_0
action_t = VAE_decoder(current_proprioception, z_t)
execute action_t in physics
```

## Algorithm 9: MuJoCo / Isaac Video Rendering

```text
Input: robot model, action or qpos sequence, camera config, metrics config
Output: MP4, keyframes, metrics CSV
load G1 MJCF/USD/URDF-derived model
map actions to robot joint order
if action-control:
    theta_sp = theta_default + action_scale * clip(action)
    for each frame: set actuator ctrl, step MuJoCo, render RGB
if reference replay:
    write qpos for visualization and call mj_forward
save frames to MP4 and compute root/action/fall/error metrics
```

## Algorithm 10: Failure Diagnosis Checklist

```text
Check data: joint order, FPS, root frame, ground contact, impossible segments.
Check teacher: reward, done rate, body/joint error, termination breakdown.
Check VAE: reconstruction MSE, KL, closed-loop rollout, DAgger coverage.
Check diffusion: token scaling, inverse transform, physically valid trajectory.
Check guidance: cost frame, gradient target, guidance scale.
Check deployment: action scale, PD gains, default pose, control frequency, observation normalization.
```
