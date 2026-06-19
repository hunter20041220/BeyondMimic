# BeyondMimic Reading Report

## Abstract

BeyondMimic studies a difficult problem in humanoid robotics: how to move from motion tracking to versatile, task-directed control. The paper is interesting because it does not treat motion imitation as the final goal. Instead, it uses motion tracking as a source of behavioral competence, distills that competence into latent action representations, and then uses diffusion plus guidance to compose new behaviors.

This report combines paper reading with a reproduction-oriented audit. The local project does not fully reproduce BeyondMimic at paper level. However, it does reconstruct a substantial part of the public and virtual pipeline: released-data tables and figures, official tracking code contracts, IsaacLab task gates, official-loop motion replay/training/evaluation evidence, local teacher rollouts, conditional VAE training, state-latent denoising, and full-split offline guidance. The strongest current virtual chain is:

```text
official-loop tracking/PPO eval
-> local teacher rollout dataset
-> local conditional action VAE
-> local state-latent trajectory windows
-> local state-latent denoiser
-> full validation/test offline guidance
```

The important boundary is that this is still not closed-loop guided diffusion in IsaacLab, not the official BeyondMimic DAgger dataset, not an official VAE/diffusion checkpoint, not TensorRT deployment, and not a real Unitree G1 result.

## 1. Introduction

Humanoid control is hard because the robot must satisfy many constraints at once. It must remain balanced, track contacts, coordinate high-dimensional joints, react to changing commands, and avoid physically implausible motions. Traditional motion tracking can produce impressive imitation, but it often remains tied to reference clips. A controller that only tracks motion may struggle when asked to solve new tasks such as reaching a target, avoiding an obstacle, or composing motion with a novel objective.

BeyondMimic is interesting because it treats tracking as a foundation rather than as the endpoint. The paper asks whether a humanoid can first learn a strong tracking teacher, then use the teacher's behavior to build a latent representation and a generative planner. The resulting system is meant to combine the physical reliability of reinforcement-learned tracking with the flexibility of diffusion-based trajectory generation.

## 2. Background and Related Work

Motion imitation and tracking are central to humanoid control. A tracking policy can learn reusable locomotion and whole-body coordination from motion capture data, especially when trained in simulation with domain randomization and careful reward design. However, tracking alone is usually reference-conditioned. It answers the question "how do I follow this motion?" more naturally than "how do I create a new motion that satisfies this task?"

Reinforcement learning gives the tracking teacher physical grounding. PPO-style training in IsaacLab or related simulation environments can optimize stability, contact behavior, and reward terms over many parallel environments. This part of the pipeline is engineering-heavy: robot assets, observation schemas, action dimensions, motion preprocessing, reset logic, termination logic, and reward terms all matter.

DAgger and policy distillation connect the teacher to a dataset. Instead of relying only on offline clips, the system can collect trajectories from the teacher under the states it actually visits. This helps reduce distribution shift. In BeyondMimic, this teacher trajectory data is the bridge from tracking to generative control.

The conditional VAE is used as a latent action representation. A high-dimensional humanoid action can be compressed into a lower-dimensional latent variable conditioned on state. This is useful because diffusion over raw action or state-action trajectories would be more difficult and less structured.

Diffusion models provide a way to generate trajectories by denoising. In the BeyondMimic pipeline, diffusion operates over state-latent trajectories. This is a clever design choice: the model is not simply copying clips, and it is not planning directly in raw joint torques. It generates a representation that remains connected to the learned low-level action model.

Classifier guidance or task-cost guidance allows test-time optimization. A diffusion sample can be nudged toward a task objective such as velocity tracking, reaching, smoothness, or obstacle avoidance. This is the mechanism that turns a motion prior into a versatile controller.

## 3. Paper Summary

The paper's main claim is that a humanoid controller can go beyond motion tracking by using guided diffusion over learned latent trajectories. The system can be understood as four stages:

1. Train a motion tracking teacher in simulation.
2. Collect teacher or DAgger-style trajectories.
3. Train a conditional VAE and a state-latent diffusion model.
4. Use guidance at test time to solve new tasks and deploy the resulting behavior.

The tracking teacher gives the system physically plausible skills. The VAE provides a compact action space. The state-latent diffusion model learns the distribution of feasible behavior. Guidance then biases generation toward task objectives without retraining a new policy for every task.

The real robot component is also important. It shows that the method is intended not only as an offline generation method, but as a deployment-oriented humanoid control system. In this local reproduction project, real robot validation is explicitly out of scope unless Unitree G1 hardware is confirmed.

## 4. Method Understanding

The elegant part of BeyondMimic is the division of labor. Reinforcement learning handles physical competence. The VAE handles action abstraction. Diffusion handles sequence-level generation. Guidance handles task specificity.

This decomposition matters because humanoid control is too difficult to solve with a single generic model. A pure diffusion model might generate smooth trajectories that are not physically executable. A pure tracking policy might be stable but not flexible. A pure task policy might overfit to one objective. BeyondMimic combines these pieces into a pipeline where each component solves a narrower problem.

A key assumption is that the teacher rollout distribution is good enough to support downstream generative modeling. If the teacher is weak, narrow, or trained on imperfect assets, the VAE and diffusion model inherit those limitations. Another assumption is that the latent representation is controllable: the decoder must turn generated latents into actions that remain meaningful in closed loop.

The most important unresolved question for reproduction is closed-loop validation. Offline denoising and guidance can show that the math and data path work, but the real test is whether generated latents produce stable humanoid behavior when rolled out in IsaacLab or on hardware.

## 5. Reproduction Setup

The local project root is:

```text
/mnt/infini-data/test/BeyondMimic
```

The reproduction uses separate environments for analysis, diffusion, and tracking. The current audit state records:

- `bm_analysis`: restored for Python analysis, plotting, pandas, ONNX, and report scripts.
- `bm_diffusion`: restored with PyTorch CUDA support for VAE/diffusion/guidance experiments.
- `bm_tracking`: restored enough to import Isaac Sim, IsaacLab packages, and the official `whole_body_tracking` code.

The project also uses strict artifact boundaries:

- raw downloads remain under `download/` and are treated as read-only.
- checkpoints, large runs, videos, and latent shards stay under ignored result directories.
- small audit JSON/TSV/CSV/Markdown artifacts are committed to GitHub.

The GitHub workflow records each meaningful round with a progress Markdown file under `reproduction/docs/progress/`.

## 6. Reproduction Results and Evidence

### 6.1 Current Audit Totals

The current machine-readable comparison table has 143 rows:

```text
exactly_comparable: 58
approximately_comparable: 19
qualitative_only: 53
not_publicly_reproducible: 10
requires_real_robot: 3
```

The latest artifact manifest records 345 artifacts. The master audit passes with 237 out of 237 artifacts. The required artifact absence audit has 24 rows and explicitly records that local checkpoints are not official BeyondMimic checkpoints.

### 6.2 Released-Data and Static Audits

The project has reproduced and audited many released-data artifacts: paper table values, released-data figures, source coverage, paper panel mapping, formula/code trace, and required artifact absence. These results are valuable because they make the paper claims traceable, but they should not be confused with retraining the full system.

The tracking side includes official-code and contract audits for observation/action schema, reward terms, termination logic, motion preprocessing, ONNX contracts, MuJoCo/ROS launch surfaces, and official entry-point diagnostics.

### 6.3 IsaacLab and Tracking Evidence

The project recovered the IsaacLab package layer and the official `whole_body_tracking` code path. It also advanced from basic import checks to live task gates, resource-adjusted replay, official-loop replay, PPO training, checkpoint evaluation, and teacher rollout collection.

The strongest current tracking evidence is the official-csv-loop PPO checkpoint evaluation:

```text
status: ok_official_csv_loop_ppo_checkpoint_eval_completed
env steps: 512 x 299 = 153088
done_count_total: 13127
anchor position error mean: 0.10621154815407102
body position error mean: 0.18640418467812714
joint position error mean: 1.218346951597909
```

This is not paper-scale PPO training. It uses an enriched-USD runtime patch and does not produce official paper tracking metrics. Still, it is important because it connects the official-loop motion path to a trained local checkpoint and evaluation.

### 6.4 Teacher Rollout Dataset

The project collected a local teacher rollout dataset from the official-loop PPO checkpoint:

```text
sample_count: 306176
shards: 2
policy_obs_dim: 160
action_dim: 29
```

This dataset is useful for downstream local modeling. It is not the official BeyondMimic DAgger dataset.

### 6.5 Conditional Action VAE

The local official-loop conditional action VAE was trained on the teacher rollout dataset:

```text
samples: 306176
train/validation/test: 244940 / 30618 / 30618
latent_dim: 32
hidden_dim: 512
epochs: 40
test action MSE: 0.0033218273892998695
test action absolute error mean: 0.04307248070836067
```

This result supports the paper's VAE stage conceptually and technically. It does not reproduce the official BeyondMimic VAE checkpoint because that checkpoint and official DAgger data are not available.

### 6.6 State-Latent Diffusion

The local official-loop state-latent dataset was built from the teacher rollout and VAE:

```text
samples: 306176
windows: 285696
train/validation/test windows: 228556 / 28570 / 28570
sequence_length: 21
obs_dim: 160
latent_dim: 32
token_dim: 192
weighted posterior reconstruction MSE: 0.0032909737434238195
```

The local state-latent denoiser was trained over all windows:

```text
epochs: 30
batch_windows: 2048
validation pred token MSE: 0.03781931714287826
test pred token MSE: 0.037761972951037545
test noisy token MSE: 0.08398369699716568
test denoising improvement ratio: 0.5503654363737768
```

This is one of the strongest local reproduction results because it recreates the shape of the paper's downstream diffusion pipeline over a locally collected virtual dataset.

### 6.7 Offline Guidance

The latest result evaluates offline guidance over all validation/test windows from the official-loop denoiser:

```text
total selected windows: 57140
validation windows: 28570
test windows: 28570
tasks: velocity_command, latent_smoothness, latent_magnitude, composed
guidance scales: 0, 0.0005, 0.001, 0.002, 0.005, 0.01
aggregate rows: 48
```

All four tasks have positive best-scale cost deltas:

```text
velocity_command: 1.5221161936487983e-07
latent_smoothness: 1.0589483862054907e-06
latent_magnitude: 2.0676824151193147e-06
composed: 1.7070461498461627e-07
```

This demonstrates that the local denoiser output can be connected to task-cost gradients. It is still offline guidance, not closed-loop guided humanoid control.

The newest action-decode gate also maps the guided latents back through the local VAE decoder:

```text
total windows: 57140
decoded action dimension: 29
decoded action steps per task: 1199940
tasks with finite decoded actions: 4 / 4
```

The project now saves report assets for this stage under:

```text
res/report_assets/official_csv_loop_guidance_vae_action_decode/
```

These plots compare guided and unguided decoded actions and summarize teacher-action MSE. They are useful for a presentation, but they still do not show a robot rollout video.

## 7. What Is Not Yet Reproduced

This project does not fully reproduce BeyondMimic at paper-level.

The following paper-level components remain missing or blocked:

- the official BeyondMimic DAgger rollout logs;
- the official conditional VAE checkpoint;
- the official state-latent diffusion checkpoint;
- closed-loop guided diffusion rollout in IsaacLab;
- Fig. 5 and Fig. 6 paper-level videos and metrics;
- TensorRT/asynchronous deployment evidence;
- real Unitree G1 robot results.

The local official-loop pipeline is meaningful, but it remains a virtual surrogate. It should be described as qualitative and engineering evidence, not as a complete reproduction of the paper's final claims.

## 8. Personal Reflections

The main lesson from this reproduction is that robotics papers are pipelines, not just algorithms. The diffusion model is only one layer. The robot asset, simulator startup, motion preprocessing, reset semantics, reward terms, policy checkpoint, rollout collection, latent model, and deployment interface all affect whether the method can be reproduced.

I also learned that "public code available" does not always mean "paper result reproducible." The tracking code path can be audited and partially executed, but the full BeyondMimic diffusion side depends on data and checkpoints that are not fully public. This forces a careful distinction between exact reproduction, approximate reproduction, local surrogate experiments, and non-reproducible paper-level claims.

The most intellectually interesting part of the paper is the use of latent diffusion as a bridge between imitation and task-directed behavior. Motion tracking alone gives skill, but not enough flexibility. Guidance alone gives objectives, but not physical competence. BeyondMimic combines them in a way that feels natural after seeing the pipeline, but is not obvious beforehand.

The most difficult engineering part was not training a small model. It was making the environment auditable: IsaacLab startup, GPU isolation, official entry points, artifact manifests, and honest result labeling. For robotics reproduction, these details are not secondary. They are part of the scientific result.

## 9. Future Work

The next technical step should be a closed-loop guidance gate in IsaacLab. The current offline guidance result proves that task costs can steer denoiser outputs, but it does not prove that decoded latents stabilize a humanoid in simulation. A useful next milestone would be:

```text
official-loop denoiser sample
-> VAE decoder
-> action sequence or receding-horizon action interface
-> IsaacLab Tracking-Flat-G1-v0 rollout
-> task metrics and failure analysis
```

After that, the project should attempt TensorRT or ONNX deployment audits only for models that actually participate in the local pipeline. Real robot experiments should remain out of scope unless Unitree G1 hardware is explicitly available and safety procedures are documented.

## 10. Conclusion

BeyondMimic is a strong example of modern robot learning as system composition. Its contribution is not only "use diffusion" or "track motions", but the way it combines tracking, latent action modeling, trajectory generation, and guidance.

The local reproduction does not prove the full paper. It does, however, provide an auditable and progressively stronger reconstruction of the method's public and virtual components. The current evidence is enough to support a serious reading report: it shows what the paper is trying to do, why the method is technically interesting, which components can be reproduced locally, and which claims remain unavailable without official checkpoints, rollout logs, closed-loop evaluation, or real hardware.
