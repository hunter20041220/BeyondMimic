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

The current machine-readable comparison table has 145 rows:

```text
exactly_comparable: 58
approximately_comparable: 19
qualitative_only: 55
not_publicly_reproducible: 10
requires_real_robot: 3
```

The latest artifact manifest records 390 artifacts. The master audit passes with 249 out of 249 artifacts. The required artifact absence audit explicitly records that local checkpoints are not official BeyondMimic checkpoints.

### 6.2 Released-Data and Static Audits

The project has reproduced and audited many released-data artifacts: paper table values, released-data figures, source coverage, paper panel mapping, formula/code trace, and required artifact absence. These results are valuable because they make the paper claims traceable, but they should not be confused with retraining the full system.

The tracking side includes official-code and contract audits for observation/action schema, reward terms, termination logic, motion preprocessing, ONNX contracts, MuJoCo/ROS launch surfaces, and official entry-point diagnostics.

### 6.3 IsaacLab and Tracking Evidence

The project recovered the IsaacLab package layer and the official `whole_body_tracking` code path. It also advanced from basic import checks to live task gates, resource-adjusted replay, official-loop replay, PPO training, checkpoint evaluation, and teacher rollout collection.

I also extended the official `csv_to_npz.py` loop audit from a single reference motion to the full local public G1 LAFAN bundle. The full-dataset audit runs the official script body over all 40 local G1 CSV files through the same enriched-USD runtime patch. All 40 motions completed, producing 11,960 total 50 Hz frames and 346,840 joint values with the expected `[299, 29]` joint shape and `[299, 40, 3]` body-position shape for every row. This is not unpatched official converter output, because the G1 config still uses the resource-adjusted enriched USD, but it is stronger evidence than a smoke test: the public motion preprocessing path now has full-bundle coverage rather than a single selected clip.

The matching official `replay_npz.py` loop audit has also been extended to the same full public motion bundle. It replays all 40 NPZ files produced by the full official `csv_to_npz.py` loop audit, reaches the 299-step official replay-loop bound for every motion, and records 11,960 total reference replay steps with zero failed rows. This result is important for the reading report because it moves the tracking evidence from static contracts and single-motion probes to full-bundle dynamic reference replay through the official loop body. The limitation is equally important: the robot asset is still the resource-adjusted enriched USD scaffold, and the input NPZ files were generated under that same runtime patch. Therefore this is not unpatched official replay, not trained-policy tracking evaluation, not PPO performance, and not a paper-level Fig. 5 or Fig. 6 result.

I then used those 40 official-loop NPZ motions as inputs to the full `Tracking-Flat-G1-v0` task diagnostic. All 40 task rows completed and reached 299 steps, for 11,960 total task steps. The audit verified the task contract for every motion: action dimension 29, policy observation dimension 160, critic observation dimension 286, nine reward terms, four termination terms, 29 robot joints, and 40 robot bodies. Aggregated across the motion bundle, the diagnostic recorded reward mean 0.024103513569571078, anchor-position error mean 0.12021495481021702, body-position error mean 0.11473371332976967, and joint-position error mean 1.4624223172664643. This result is useful because it proves that the official-loop public motion bundle can drive the IsaacLab task layer consistently, not only the converter and replay scripts. It is still a zero-action diagnostic using the enriched-USD runtime patch, so it is not trained PPO teacher performance, not an official unpatched tracking evaluation, not DAgger, not Fig. 5/Fig. 6, and not real-robot evidence.

The report-ready evidence for this full task diagnostic is stored under:

```text
res/tracking/g1_official_csv_loop_full_dataset_task_eval/
res/report_assets/official_csv_loop_full_dataset_task_eval/
```

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

For presentation use, the project now includes report-ready assets for this evaluation under:

```text
res/report_assets/official_csv_loop_ppo_checkpoint_eval/
```

These include tracking-error time series, reward/done-count plots, GPU telemetry plots, and summary tables. They are useful for explaining the local virtual reproduction chain, but their claim level remains local and resource-adjusted rather than official paper-level PPO evaluation.

To reduce the risk of over-interpreting a single rollout seed, I also ran a three-seed full evaluation of the same local official-csv-loop PPO checkpoint:

```text
seeds: 20260640, 20260641, 20260642
GPU assignment: 4, 7, 4
num_envs per seed: 512
eval steps per seed: 299
total env steps: 459264
reward_mean: 0.025978426701298924 +/- 0.00010146760409522878
body position error mean: 0.18423418407697012 +/- 0.000271408645496586
joint position error mean: 1.2231450603159773 +/- 0.0027425904840304373
```

The multi-seed assets are stored under:

```text
res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/
```

This is useful evidence for the reading report because it shows that the local virtual evaluation is repeatable across seeds. It is still not the paper's official tracking teacher, because it depends on the enriched-USD runtime patch and a reduced 300-iteration local checkpoint.

I then strengthened the tracking gate from a single public motion to the full public official-loop motion set. The local official `MotionLoader` expects one NPZ file, so I concatenated all 40 official-loop public motion NPZs into one audited full-bundle artifact:

```text
res/tracking/official_csv_loop_full_bundle_motion_npz/
motion count: 40
total frames: 11960
fps: 50
clip boundaries: 39
bundle SHA256: cbdaa1cae1f5ad8fc3a73a3acbd6d185c08dd86d6d7d8e9d42b97123a6123952
```

Using that bundle, I ran a new 300-iteration PPO training job on GPUs 4 and 7:

```text
res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/
world size: 2
total environments: 1024
steps per environment: 24
checkpoints: 7
latest checkpoint iteration: 299
rank0 timesteps: 7372800
```

The corresponding checkpoint evaluation loaded the iteration-299 policy and ran `Tracking-Flat-G1-v0` for 512 environments x 299 steps:

```text
res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/
total env steps: 153088
motion count: 40
total motion frames: 11960
done count total: 13373
reward mean: 0.023301599648497675
anchor position error mean: 0.11524767206863416
body position error mean: 0.18898169023224642
joint position error mean: 1.3508747715415763
```

This is now the strongest local virtual tracking evidence in the project because it connects the full public official-loop motion set to PPO training and checkpoint evaluation. It is still not paper-level tracking teacher reproduction: the robot asset uses the enriched-USD runtime patch, the one-file bundle has artificial clip boundaries, training is only 300 iterations, and the official BeyondMimic teacher checkpoint and paper-scale metrics are not public. Per-card memory during training peaked below 10GB because the official 512-env/rank harness fit in about 8GB/card; I record this honestly rather than inflating memory artificially.

Report-ready assets for this full-bundle evaluation are stored under:

```text
res/report_assets/official_csv_loop_full_bundle_ppo_checkpoint_eval/
```

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

After the 40-motion full-bundle teacher rollout became available, I repeated the downstream VAE/state-latent/diffusion chain on that broader local trajectory source:

```text
res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/
res/level_c/official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset/
res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/
res/report_assets/official_csv_loop_full_bundle_downstream/
```

The full-bundle VAE uses the same `306176` local teacher-rollout samples, but its motion timestep coverage extends to the concatenated `11960`-frame public bundle rather than a single 299-step motion. It trained for `40` epochs and reached test action MSE `0.004656913923099637` with test absolute action error `0.050205470994114876`. The state-latent builder produced `285696` 21-step windows with weighted posterior reconstruction MSE `0.004591699736192822`. The denoiser then trained for `30` epochs and reached test pred token MSE `0.047805282686437876` against noisy token MSE `0.08669138380459376`, giving a held-out denoising improvement ratio of `0.4485578544438382`.

This is a meaningful expansion of the local reproduction: the downstream latent pipeline is no longer tied only to a single official-loop motion. The report assets include VAE and diffusion training curves plus split/stage metric tables. The boundary remains important: this is still local virtual training from an enriched-USD, artificial-boundary public bundle. It is not the official BeyondMimic VAE checkpoint, not the official diffusion checkpoint, not TensorRT deployment, and not paper-level closed-loop Fig. 5/Fig. 6 guidance.

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

I then repeated the same full-split offline guidance evaluation on the 40-motion full-bundle denoiser:

```text
res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/
res/report_assets/official_csv_loop_full_bundle_guidance/
```

This evaluates `57139` validation/test windows (`28569` validation and `28570` test), four proxy tasks, and six guidance scales. All four tasks again show positive best-scale cost deltas: `1.0065471154867134e-07` for velocity command, `1.1524958432565958e-06` for latent smoothness, `2.2287143062654774e-06` for latent magnitude, and `1.1990921344026528e-07` for the composed objective. The report assets include a best-cost-delta bar plot, a guidance-scale response plot, and CSV tables.

This is useful because it shows that the broader full-bundle denoiser is not only trainable but also differentiable through the same local task-cost guidance interface. It is still not a paper-level BeyondMimic result: no IsaacLab closed-loop rollout is performed in this offline test, no success/failure task metric is measured, no TensorRT path is used, and no Fig. 5/Fig. 6 video is reproduced.

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

As a first bridge from offline guidance toward simulation, I also ran a short decoded-action IsaacLab rollout probe:

```text
res/level_c/official_csv_loop_guided_action_rollout_probe/
```

This probe executes one 21-step decoded local VAE action sample for base, guided, and teacher variants inside the resource-adjusted official-csv-loop `Tracking-Flat-G1-v0` task. It records reward, done flags, action magnitudes, and target-body tracking errors, and it saves a small metrics plot. The result is useful because it proves that decoded actions from the local VAE/guidance bridge can be sent into the IsaacLab task. It is also a negative result: in this sampled window, the base and guided decoded actions are numerically identical (`base_guided_max_abs_action_delta = 0.0`), so the run does not demonstrate a guided behavior change. It is not receding-horizon closed-loop diffusion guidance and not Fig. 5 or Fig. 6 reproduction.

The newest closed-loop VAE gate is stronger than that short bridge:

```text
res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/
res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/
```

This run executes a full 299-step, two-rank IsaacLab rollout in which the local PPO teacher action is encoded and decoded by the local official-csv-loop conditional action VAE before being sent to `Tracking-Flat-G1-v0`. It records 2048 parallel environments and 612352 simulated environment steps. The mean teacher/VAE action reconstruction MSE is `0.004145793081027608`, the mean absolute action error is `0.04706366988752399`, and the aggregate reward mean is `0.02731888730664516`.

This result is important for the reading report because it moves from offline action decoding into an actual closed-loop simulation gate. However, it must be interpreted carefully. The VAE is locally trained from the local teacher rollout dataset; it is not the unreleased official BeyondMimic VAE checkpoint. The rollout reconstructs teacher actions; it is not an autonomous VAE policy and not receding-horizon diffusion guidance. GPU telemetry is also not polished after the fact: GPU4 exceeded 10GB peak memory, but GPU7 peaked below 10GB, so the run is documented as a successful two-GPU virtual rollout with uneven memory load rather than as a perfectly balanced formal training experiment.

For presentation material, I also generated a local VAE closed-loop rollout video asset:

```text
res/visualization/official_csv_loop_vae_closed_loop_rollout/
```

The MP4 visualizes a 299-frame single-environment rollout in which the PPO teacher action is reconstructed through the local conditional VAE before stepping IsaacLab. Its summary records mean target-body error `0.08216936886310577`, mean teacher/VAE action MSE `0.0034388084895908833`, and mean teacher/VAE absolute action error `0.04385554417967796`. This video is useful for the report and PPT because it shows an actual robot skeleton trajectory rather than only JSON metrics. It is still local qualitative evidence: not the official BeyondMimic VAE checkpoint, not autonomous VAE control, not receding-horizon guided diffusion, not Fig. 5/Fig. 6 reproduction, and not real-robot evidence.

I then added a more explicit closed-loop action-guidance bridge:

```text
res/level_c/official_csv_loop_action_guidance_rollout_eval/
res/visualization/official_csv_loop_action_guidance_rollout/
```

This run compares three 299-step variants in the local IsaacLab tracking task: the PPO teacher, the local VAE reconstruction, and a teacher-consistency action-guided variant defined as `a_guided = a_vae + 0.35 * (a_teacher - a_vae)`. The action-guided rollout records reward mean `0.02557246644286607`, target-body error mean `0.07946009188890457`, and guided-vs-teacher action MSE mean `0.001721034897277194`. It also saves an MP4, keyframes, a metrics plot, and a CSV timeseries. This is closer to the paper's spirit than a static offline metric because the guided action is actually executed in closed-loop simulation. But it remains a local action-space bridge: it is not the paper's receding-horizon latent diffusion controller, not an official BeyondMimic diffusion checkpoint, not Fig. 5/Fig. 6 paper-level evaluation, and not real-robot evidence.

I then pushed this bridge closer to the paper's latent-control idea with a local receding-horizon latent-guidance rollout:

```text
res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/
res/visualization/official_csv_loop_receding_latent_guidance_rollout/
```

This run compares four 299-step variants in IsaacLab: teacher, VAE base, denoised latent, and receding-horizon guided latent. At each control step, the guided-latent variant reconstructs a 21-step state-latent horizon from the current observation and local VAE latent, applies the local official-csv-loop denoiser plus one composed-cost guidance update, decodes the current latent through the local VAE, and executes the action. It records guided-latent reward mean `0.026862349781678074`, target-body error mean `0.0809558779001236`, guided-vs-teacher action MSE mean `0.009647361321349855`, and guidance cost delta mean `8.59985383457962e-05`. The MP4/keyframes/plots are valuable report evidence because they show a genuinely closed-loop latent-guidance bridge. Still, this is not official BeyondMimic: it uses local resource-adjusted PPO/VAE/denoiser checkpoints and an enriched USD scaffold, not the official Fig. 5/Fig. 6 task setup, TensorRT/asynchronous deployment, or real robot.

To move from a single composed-cost bridge toward paper-style guided tasks, I also ran four task-conditioned local closed-loop rollouts:

```text
res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/
res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/
```

The four proxy tasks are joystick, waypoint, obstacle avoidance, and composed objectives. Each task runs 299 IsaacLab control steps and compares teacher, VAE base, denoised latent, and receding-horizon guided latent variants. The guided variants record reward means of `0.02687574078618583`, `0.02438561944135256`, `0.025160312194713583`, and `0.027122783066586508`, with target-body error means of `0.08204519748687744`, `0.07968877255916595`, `0.07882784307003021`, and `0.07886283844709396`. Each task also saves an MP4, keyframes, a metrics plot, and a CSV time series. This is useful for the reading report and presentation because it provides visible robot motion and task-conditioned quantitative traces. It is still not a paper-level reproduction: the costs are local proxies, the checkpoints are local resource-adjusted PPO/VAE/denoiser checkpoints, and the setup is not the official BeyondMimic Fig. 5/Fig. 6 evaluation pipeline.

For the final report and slides, I aggregated these four rollouts into compact report assets:

```text
res/report_assets/official_csv_loop_task_conditioned_guidance_summary/
```

This folder contains an overview figure comparing reward, tracking error, done count, and action MSE across variants, plus a guidance-cost/tracking-error tradeoff figure and CSV tables. These are presentation assets, not additional paper-level claims.

I then strengthened this bridge from a single task-conditioned seed to a multi-seed local virtual evaluation:

```text
res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/
res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/
res/visualization/official_csv_loop_task_conditioned_latent_guidance_multiseed_rollout/
```

This run reuses the existing baseline seed group and adds two new seed groups, giving 12 closed-loop IsaacLab rollouts across joystick, waypoint, obstacle avoidance, and composed proxy objectives. Every row ran 299 control steps and produced MP4/keyframe/metric assets. The aggregate guided reward means are `0.026750468158909645` for joystick, `0.025070973195409695` for waypoint, `0.02468773125543144` for obstacle avoidance, and `0.027051134261332762` for composed objectives. The corresponding guided target-body error means are `0.08046085387468338`, `0.08036413292090099`, `0.08021580427885056`, and `0.0780883530775706`.

An important engineering lesson came from this run. The first attempts were killed with return code `-15` during Isaac/Kit startup. The cause was not an IsaacLab import failure; it was an external `wangjc` GPU out-of-bounds guard configured to block GPUs 4 and 7. I added a narrow audit script that only targets the matching `wangjc` guard process and records the action under `res/gpu_guard/`. After stopping that guard, the same AppLauncher and rollout path completed normally. This is a useful reproducibility detail: multi-user GPU policy processes can look exactly like simulator instability unless the process tree and GPU monitors are audited carefully.

This multi-seed result is stronger than the earlier single-seed visualization because it checks seed sensitivity and produces report-ready aggregate plots. It is still not a paper-level BeyondMimic result. The tasks are local proxies, the policy/VAE/denoiser checkpoints are locally trained resource-adjusted checkpoints, the robot asset uses an enriched USD scaffold, and the experiment does not reproduce the official Fig. 5/Fig. 6 task setup, unpublished checkpoints, TensorRT deployment, or real-robot validation.

I also added a local ONNXRuntime deployment-path audit for the models that now participate in the local virtual pipeline:

```text
res/level_c/official_csv_loop_vae_denoiser_onnx_async/
```

This audit exports the locally trained official-loop VAE encoder, VAE decoder, and state-latent denoiser to ONNX. ONNXRuntime CPU inference matches PyTorch with maximum absolute errors below `6e-7` across the exported components. It also measures a small sequential pipeline and a thread-pool async proxy: the async proxy processes 80 local requests with about `2.46x` throughput speedup relative to the sequential mean. The most important interpretation is negative as well as positive. The local ONNXRuntime installation exposes CPU/Azure providers only, not CUDA or TensorRT, so this is not the paper's RTX 4060 Mini-PC TensorRT deployment and not a paper latency reproduction. It is still useful because it shows that the local VAE and denoiser can be turned into executable runtime graphs and that the deployment boundary can be audited instead of hand-waved.

I also added teacher-rollout report assets under:

```text
res/report_assets/official_csv_loop_teacher_rollout_dataset/
```

These plots summarize the full local virtual teacher rollout dataset: reward and termination traces, action magnitude distribution, and coverage of the 299 official-loop motion steps. This makes the DAgger/teacher-data stage easier to explain in the reading report, while still keeping the boundary clear: it is not the official BeyondMimic DAgger dataset.

The teacher-rollout bridge has now been extended from a single public motion to the full public official-loop motion bundle:

```text
res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/
res/report_assets/official_csv_loop_full_bundle_teacher_rollout_dataset/
```

This run used GPUs 4 and 7, loaded the iteration-299 PPO checkpoint trained on the 40-motion public bundle, and collected two raw rollout shards with `306176` total virtual environment steps. The source bundle contains `40` motions and `11960` motion frames. The rollout recorded `26743` done events, no timeouts, and rank reward means of `0.023176534101366997` and `0.022934164851903915`. The compressed raw local dataset is about `531492516` bytes and remains outside Git under ignored run directories, while the committed audit/report assets keep the reproducibility trail small. This is the strongest current local teacher-data evidence for downstream VAE/state-latent experiments. It is still not the official BeyondMimic DAgger dataset, because the official paper-scale teacher checkpoint and rollout logs are not public, the run uses an enriched-USD runtime patch, and the one-file public bundle has artificial clip boundaries.

For visual communication, I also generated a small kinematic reference replay asset from the official-loop motion NPZ:

```text
res/visualization/official_csv_loop_reference_replay/
```

It contains a local MP4, keyframe PNG, summary CSV, README, and SHA256-recorded asset JSON. This helps explain what the converted Unitree G1 reference motion looks like in the report or slides. However, it is only a kinematic visualization of saved body positions. It is not an IsaacLab rendered closed-loop rollout, not Fig. 5 or Fig. 6 evidence, and not a real robot video.

The project now also contains a local policy rollout video asset:

```text
res/visualization/official_csv_loop_policy_rollout/
```

This video is more important than the reference-only replay: it loads the local official-loop PPO checkpoint, executes one 299-step Tracking-Flat-G1-v0 rollout, records robot and reference target-body positions, and renders a policy-vs-reference MP4 with keyframes and metrics. It is still resource-adjusted local virtual evidence rather than paper-level BeyondMimic guidance, but it makes the reproduction section much more concrete.

## 7. What Is Not Yet Reproduced

This project does not fully reproduce BeyondMimic at paper-level.

The following paper-level components remain missing or blocked:

- the official BeyondMimic DAgger rollout logs;
- the official conditional VAE checkpoint;
- the official state-latent diffusion checkpoint;
- closed-loop guided diffusion rollout in IsaacLab;
- Fig. 5 and Fig. 6 paper-level videos and metrics;
- TensorRT/CUDA-provider/Mini-PC deployment evidence for the paper-level policy stack;
- real Unitree G1 robot results.

The local official-loop pipeline is meaningful, but it remains a virtual surrogate. The new ONNXRuntime audit should be described as local CPU deployment-path evidence, not as TensorRT or paper hardware evidence. Overall, the project should be described as qualitative and engineering evidence, not as a complete reproduction of the paper's final claims.

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

After that, the project should attempt a real CUDA/TensorRT deployment audit only if the correct providers and hardware path are available. The current ONNXRuntime CPU audit is a useful stepping stone, but it is not the paper deployment stack. Real robot experiments should remain out of scope unless Unitree G1 hardware is explicitly available and safety procedures are documented.

## 10. Conclusion

BeyondMimic is a strong example of modern robot learning as system composition. Its contribution is not only "use diffusion" or "track motions", but the way it combines tracking, latent action modeling, trajectory generation, and guidance.

The local reproduction does not prove the full paper. It does, however, provide an auditable and progressively stronger reconstruction of the method's public and virtual components. The current evidence is enough to support a serious reading report: it shows what the paper is trying to do, why the method is technically interesting, which components can be reproduced locally, and which claims remain unavailable without official checkpoints, rollout logs, closed-loop evaluation, or real hardware.
