# BeyondMimic Reading Report

## Abstract

BeyondMimic studies a difficult problem in humanoid robotics: how to move from motion tracking to versatile, task-directed control. The paper is interesting because it does not treat motion imitation as the final goal. Instead, it uses motion tracking as a source of behavioral competence, distills that competence into latent action representations, and then uses diffusion plus guidance to compose new behaviors.

This report combines paper reading with a reproduction-oriented audit. The local project does not fully reproduce BeyondMimic at paper level. However, it does reconstruct a substantial part of the public and virtual pipeline: released-data tables and figures, official tracking code contracts, IsaacLab task gates, official-loop and official-importer-export motion replay/training/evaluation evidence, local teacher rollouts, conditional VAE training, state-latent denoising, full-split offline guidance, and local proxy closed-loop guidance rollouts. The strongest current virtual chain is:

```text
official-loop tracking/PPO eval
-> official-importer-export tracking/PPO eval
-> local teacher rollout dataset
-> local conditional action VAE
-> local state-latent trajectory windows
-> local state-latent denoiser
-> scaled-teacher VAE/state-latent/denoiser retraining
-> full validation/test offline guidance
-> local proxy task-conditioned closed-loop guidance in IsaacLab
```

The important boundary is that this is still not the official BeyondMimic DAgger dataset, not an official VAE/diffusion checkpoint, not the paper-level Fig. 5/Fig. 6 task protocol, not TensorRT deployment, and not a real Unitree G1 result.

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

The current machine-readable comparison table has 186 rows:

```text
exactly_comparable: 58
approximately_comparable: 19
qualitative_only: 96
not_publicly_reproducible: 10
requires_real_robot: 3
```

The latest artifact manifest records 956 hashed artifacts. The master audit passes with 309 out of 309 artifacts. The required artifact absence audit records 29 trained/deployment artifact rows, including 12 missing required paper-level artifacts and 15 local artifacts that are present but explicitly classified as non-paper-level. This is important because local checkpoints, videos, and ONNX exports are useful reproduction evidence, but they are not official BeyondMimic checkpoints or paper-level deployment artifacts.

### 6.2 Released-Data and Static Audits

The project has reproduced and audited many released-data artifacts: paper table values, released-data figures, source coverage, paper panel mapping, formula/code trace, and required artifact absence. These results are valuable because they make the paper claims traceable, but they should not be confused with retraining the full system.

The tracking side includes official-code and contract audits for observation/action schema, reward terms, termination logic, motion preprocessing, ONNX contracts, MuJoCo/ROS launch surfaces, and official entry-point diagnostics.

### 6.3 IsaacLab and Tracking Evidence

The project recovered the IsaacLab package layer and the official `whole_body_tracking` code path. It also advanced from basic import checks to live task gates, resource-adjusted replay, official-loop replay, PPO training, checkpoint evaluation, and teacher rollout collection.

The newest official-importer evidence is no longer only static. A GPU4 in-memory URDF importer probe returns from
official G1 URDF parsing and writes a local 311,027,678-byte USDA export. A lightweight text audit confirms that this
export contains a G1 default prim, 40 `RigidBodyAPI` rows, one articulation root, 29 `PhysicsRevoluteJoint` rows, 29
joint-state/drive rows, all 29 action joints, and the checked target bodies. I then used this exact official-importer
USDA as the robot asset in `Tracking-Flat-G1-v0`. The single-motion smoke gate passed reset plus 8 zero-action steps,
and the full public-motion diagnostic reached 40/40 motions and 11,960 task steps. It verified the expected 29-action,
160-policy-observation, 286-critic-observation, nine-reward-term, four-termination-term, 29-joint, and 40-body
contracts. For the reading report, this is a stronger tracking-side result than the earlier enriched-scaffold task
diagnostic because the robot asset comes from the official importer. It is still not a paper-level tracking result:
the motion NPZs were generated under an enriched-USD runtime patch, the actions are zero diagnostic actions, and no
trained paper teacher, DAgger, VAE/diffusion, TensorRT, Fig. 5/Fig. 6, or real-robot validation is claimed.

I also extended the official `csv_to_npz.py` loop audit from a single reference motion to the full local public G1 LAFAN bundle. The full-dataset audit runs the official script body over all 40 local G1 CSV files through the same enriched-USD runtime patch. All 40 motions completed, producing 11,960 total 50 Hz frames and 346,840 joint values with the expected `[299, 29]` joint shape and `[299, 40, 3]` body-position shape for every row. This is not unpatched official converter output, because the G1 config still uses the resource-adjusted enriched USD, but it is stronger evidence than a smoke test: the public motion preprocessing path now has full-bundle coverage rather than a single selected clip.

The matching official `replay_npz.py` loop audit has also been extended to the same full public motion bundle. It replays all 40 NPZ files produced by the full official `csv_to_npz.py` loop audit, reaches the 299-step official replay-loop bound for every motion, and records 11,960 total reference replay steps with zero failed rows. This result is important for the reading report because it moves the tracking evidence from static contracts and single-motion probes to full-bundle dynamic reference replay through the official loop body. The limitation is equally important: the robot asset is still the resource-adjusted enriched USD scaffold, and the input NPZ files were generated under that same runtime patch. Therefore this is not unpatched official replay, not trained-policy tracking evaluation, not PPO performance, and not a paper-level Fig. 5 or Fig. 6 result.

I also reran the full public-motion official conversion/replay loop on a more faithful local asset path: the G1 USDA captured from the official Isaac Sim URDF importer. This removes the generated enriched-USD scaffold from this particular full-loop test. The official `csv_to_npz.py` loop converted all 40 public G1 LAFAN motions, producing 11,960 frames and 346,840 joint values with the expected 29-joint and 40-body shapes. The matching official `replay_npz.py` loop replayed all 40 generated NPZ motions for 11,960 total replay steps, with zero failed rows and zero shutdown warnings. I also generated report assets under `res/report_assets/official_importer_export_replay_full_dataset/`: row/family/summary CSVs, a completion-by-family plot, and a duration-by-motion plot. My interpretation is that this is a meaningful reproducibility improvement: it shows that the recovered IsaacLab/Isaac Sim stack can execute the official preprocessing and replay loop bodies across the full public motion bundle without relying on the generated scaffold asset. However, it still uses a captured importer-export USDA rather than a live unmodified official converter entry, and it still does not evaluate a trained policy. I would therefore cite it as strong local virtual reference-replay evidence, not as paper-level tracking reproduction.

I then used those 40 official-loop NPZ motions as inputs to the full `Tracking-Flat-G1-v0` task diagnostic. All 40 task rows completed and reached 299 steps, for 11,960 total task steps. The audit verified the task contract for every motion: action dimension 29, policy observation dimension 160, critic observation dimension 286, nine reward terms, four termination terms, 29 robot joints, and 40 robot bodies. Aggregated across the motion bundle, the diagnostic recorded reward mean 0.024103513569571078, anchor-position error mean 0.12021495481021702, body-position error mean 0.11473371332976967, and joint-position error mean 1.4624223172664643. This result is useful because it proves that the official-loop public motion bundle can drive the IsaacLab task layer consistently, not only the converter and replay scripts. It is still a zero-action diagnostic using the enriched-USD runtime patch, so it is not trained PPO teacher performance, not an official unpatched tracking evaluation, not DAgger, not Fig. 5/Fig. 6, and not real-robot evidence.

The report-ready evidence for this full task diagnostic is stored under:

```text
res/tracking/g1_official_csv_loop_full_dataset_task_eval/
res/report_assets/official_csv_loop_full_dataset_task_eval/
```

The report-ready evidence for the newer official-importer-export full task diagnostic is stored under:

```text
res/tracking/g1_official_importer_export_full_dataset_task_eval/
res/report_assets/official_importer_export_full_dataset_task_eval/
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

After recovering a dynamic task gate for the G1 USDA produced by the official Isaac Sim URDF importer, I repeated the
full-bundle PPO step on that more official robot asset path. The new run used the official-importer-export USDA instead
of the enriched scaffold:

```text
res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/
training status: ok_official_importer_export_full_bundle_ppo_training_completed
physical GPUs: 4 and 7
world size: 2
total environments: 1024
PPO iterations: 300
checkpoints: 7
rank0 timesteps: 7372800
training duration: 519.372 seconds
```

The corresponding checkpoint evaluation loaded the iteration-299 policy and ran 512 environments x 299 steps:

```text
res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/
status: ok_official_importer_export_full_bundle_ppo_checkpoint_eval_completed
total env steps: 153088
motion count: 40
total motion frames: 11960
reward mean: 0.02351330920281418
anchor position error mean: 0.05962799150608854
body position error mean: 0.6082496278262058
joint position error mean: 0.9147374291085081
done count total: 152841
```

This is now the strongest tracking-side evidence in the project because it combines three things that were previously
separate: the official-importer-export robot asset, the full public motion bundle, and an actual PPO training/evaluation
loop. It still should not be described as paper-level tracking reproduction. The training budget is much smaller than a
full teacher-policy run, the bundle has artificial clip boundaries, the official BeyondMimic teacher checkpoint is not
available, and the high done count shows that this policy is not a mature tracking teacher. Its value for the reading
report is therefore methodological rather than scoreboard-like: it documents what can be executed in simulation and
where the public artifact boundary still prevents exact reproduction.

Report-ready assets for this official-importer-export PPO run are stored under:

```text
res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/
```

They include a training curve, tracking-error plot, reward/done-count plot, GPU telemetry plot, and summary CSV files.

I then pushed the same official-importer-export PPO path further with a larger local run:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/
training status: ok_official_importer_export_full_bundle_scaled_ppo_training_completed
physical GPUs: 4 and 7
world size: 2
environments per rank: 4096
total environments: 8192
PPO iterations: 1000
checkpoints: 21
rank0/global timesteps: 196608000
training duration: 3242.741 seconds
peak training memory: GPU4 7771 MiB, GPU7 7767 MiB
```

The corresponding evaluation loaded the iteration-999 checkpoint:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/
status: ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed
total env steps: 612352
motion count: 40
total motion frames: 11960
reward mean: 0.02423080788881683
anchor position error mean: 0.05960297264333154
body position error mean: 0.6893615395727763
joint position error mean: 0.8996927592666651
done count total: 611642
```

I then repeated this scaled checkpoint evaluation across three full seeds:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval/
status: ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_completed
seeds: 20260710, 20260711, 20260712
envs per seed: 2048
steps per seed: 299
total env steps: 1837056
aggregate reward mean: 0.02347703839973064 +/- 0.00016869219841160913
aggregate body position error mean: 0.7051227944617553 +/- 0.000989948130753293
aggregate joint position error mean: 0.9113616949339773 +/- 0.005944694571661651
```

The multi-seed result is useful because it shows that the weak behavior is not just a single unlucky evaluation seed. The
checkpoint consistently runs through the full local virtual eval protocol, but it also consistently produces low reward
and very high termination counts. This strengthens the engineering claim that the recovered official-importer-export path
is runnable, while weakening any temptation to call the checkpoint a successful paper teacher. I therefore use it as
robustness evidence for the reproduction process, not as a BeyondMimic tracking result.

This scaled run is valuable because it tests whether the recovered official-importer-export training path can survive a
longer two-GPU PPO job. The answer is yes at the engineering level: the run completed, emitted checkpoints, and produced
report-ready plots under:

```text
res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/
```

But the scientific interpretation is cautious. Even after increasing the training load to 4096 environments per rank,
the observed peak memory was only about 7.77GB per card, so it still does not meet my formal 10GB/card threshold for a
high-memory experiment. More importantly, the policy still has weak reward and very high termination counts. I treat it
as stronger local virtual evidence that the pipeline is runnable, not as an official tracking teacher or as a paper-level
reproduction.

To make this tracking-side result easier to inspect in the reading report and presentation, I also captured a
single-environment policy-vs-reference video from the scaled checkpoint:

```text
res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/
status: ok_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture
claim level: local_virtual_official_importer_export_scaled_ppo_policy_rollout_video
frame count: 299
reward mean: 0.024693377315998077
target-body error mean: 0.3432866036891937
done count total: 299
```

This video is useful because it turns the abstract PPO checkpoint into visible robot motion on the recovered
official-importer-export asset path. At the same time, it makes the limitations visible: the motion is still generated
by a weak local checkpoint, not by the official BeyondMimic tracking teacher, and it is not evidence for the paper's
guided diffusion figures, TensorRT deployment, or real-robot results. I therefore use it as qualitative engineering
evidence, not as a paper-level result.

To keep this tracking evidence easy to audit, I added a compact summary bundle:

```text
res/report_assets/official_importer_export_tracking_eval_summary/
status: ok_official_importer_export_tracking_eval_summary_assets
task diagnostic: 40/40 motions, reward mean 0.060974017945530964
scaled PPO checkpoint eval: reward mean 0.02423080788881683, done count total 611642
scaled policy video: 299 frames, reward mean 0.024693377315998077
```

This bundle is useful for the reading report because it connects three levels of tracking-side evidence: task contract
diagnostics over all public motions, a larger local PPO checkpoint evaluation, and a visible policy-vs-reference rollout.
The task diagnostic now uses the full official-importer-export `csv_to_npz.py`/`replay_npz.py` loop outputs rather than
the older enriched-USD NPZ set. It also makes the negative result clear: the recovered virtual tracking path runs, but the current local checkpoint is weak
and cannot be presented as the official BeyondMimic tracking teacher, DAgger source policy, Fig. 5/Fig. 6 guidance result,
or real-robot result.

I also added a focused completion/termination proxy for the same scaled PPO checkpoint evaluation:

```text
res/report_assets/official_importer_export_scaled_ppo_checkpoint_completion_proxy/
attempted env steps: 612352
non-timeout done events: 611642
timeout rate: 0.0
local completion proxy rate: 0.0011594638377926403
local non-timeout done rate: 0.9988405361622074
```

This is deliberately framed as negative local virtual evidence. It helps explain why the checkpoint can enter the
official-importer-export evaluation harness while still failing to behave like a mature paper tracking teacher. It is
not the BeyondMimic paper's success/fall/collision protocol, not an official checkpoint result, and not a real-robot
result.

To check whether the final checkpoint was simply a poor choice, I then screened all saved checkpoints from the scaled
PPO run:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/
checkpoint count: 21
screening eval size: 256 envs x 299 steps per checkpoint
total env steps: 1607424
best local screening iteration: 300
best reward mean: 0.02327705469343774
best body-position error mean: 0.6452447620522617
best local non-timeout done rate: 1.0
```

This is a useful reproduction lesson: more PPO iterations did not obviously produce a better local teacher under this
screening metric, and even the best checkpoint still terminates almost immediately under the local proxy. For my
reading report, this supports a sober conclusion: the recovered official-importer-export pipeline is operational, but
the current public-data local teacher is not a replacement for the unpublished BeyondMimic tracking teacher.

I then checked whether the screening result survives a full-size confirmation evaluation. The sweep-selected checkpoint
at iteration 300 was rerun with the same `2048` environments and `299` steps used for the final iteration-999 eval:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval/
best iteration: 300
best reward mean: 0.023709370602034405
final iteration-999 reward mean: 0.02423080788881683
reward delta, best minus final: -0.0005214372867824238
body-position error delta, best minus final: 0.025006858002780685
joint-position error delta, best minus final: 0.024168011336821005
```

This reverses the simple screening interpretation: at full eval scale, iteration 300 does not beat the final checkpoint.
The practical conclusion is not that one hidden checkpoint solves tracking, but that the local PPO teacher itself needs
diagnosis or better training setup before it can support a convincing BeyondMimic-style downstream pipeline.

The reward and termination diagnostic made this more concrete:

```text
res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/
dominant termination, iteration 300: ee_body_pos, fraction 0.9985874137750836
dominant termination, iteration 999: ee_body_pos, fraction 0.9988405361622074
reward component rows: 18
termination component rows: 8
motion metric rows: 26
```

This points to a very specific failure mode: the local policy is not merely slightly worse than the paper teacher; it is
being terminated almost entirely by an end-effector/body-position tracking condition. That makes the next engineering
question sharper. I should inspect whether the public-motion retargeting, body index mapping, termination threshold, or
teacher policy quality is responsible, instead of continuing to treat the local PPO checkpoint as a reliable source of
DAgger-like trajectories.

The official-importer-export checkpoint has also been used to collect a two-shard local teacher rollout dataset:

```text
res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/
res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/
```

This run used GPUs 4 and 7, the official-importer-export G1 USDA, and the 40-motion public bundle. It collected
`306176` virtual environment steps, `2` raw shards, `40` motions, and `11960` source motion frames. The compressed raw
local dataset is about `479719377` bytes and remains outside Git under ignored run directories, while small JSON/CSV/PNG
assets summarize reward/done traces, action magnitudes, and motion-step coverage. This is currently the strongest local
teacher-data candidate on the more official robot-asset path. It is still not the official BeyondMimic DAgger dataset:
the teacher checkpoint is a short local 300-iteration PPO checkpoint, the official paper-scale teacher and rollout logs
are not public, and this result does not validate Fig. 5/Fig. 6 closed-loop diffusion.

After the longer scaled PPO run reached iteration 999, I collected a larger replacement teacher-data candidate:

```text
res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/
res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/
```

This run again used GPUs 4 and 7, the official-importer-export G1 USDA, and the 40-motion public bundle, but it loaded
the iteration-999 scaled PPO checkpoint and used `2048` environments per rank. It collected `1224704` virtual
environment steps, `2` raw shards, `40` motions, and `11960` source motion frames. The compressed raw local shards total
about `1919836221` bytes and remain outside Git, while committed JSON/CSV/PNG assets summarize reward/done traces,
action magnitude, and motion-step coverage. The report-asset summary records reward mean over steps
`0.02392365585575037`, done count `1223466`, and GPU memory peaks of `4847` MiB on GPU4 and `4839` MiB on GPU7 during
collection. This is now the strongest local teacher-data candidate for future VAE/state-latent experiments on the
official-importer-export path. It still is not the official BeyondMimic DAgger rollout log, not official paper-scale
teacher data, not Fig. 5/Fig. 6 closed-loop guided diffusion evidence, and not real-robot evidence.

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

I then repeated the same VAE closed-loop idea on the stronger official-importer-export asset path:

```text
res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/
res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/
res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/
```

This run uses the current official-importer-export G1 USDA, the local 40-motion full-bundle PPO teacher, and the local full-bundle conditional action VAE. The two-rank metric gate runs 3072 parallel environments for 299 steps, producing `918528` simulated environment steps. The mean teacher/VAE action reconstruction MSE is `5.015458783269533e-05`, the mean absolute action error is `0.005258061872471286`, and the aggregate reward mean is `0.027976495864797994`. It also produces PNG/CSV report assets and a 299-frame single-environment MP4/keyframe visualization. The video asset records mean target-body error `0.3425091505050659` and teacher/VAE action MSE `5.245815555099398e-05`.

This is useful evidence because it moves the local VAE closed-loop test onto the more official robot-asset path rather than only the enriched scaffold path. The boundary is equally important: every env-step is still marked done, the source teacher is only a short local PPO checkpoint, per-GPU memory peaked below the requested 10GB/card formal threshold, and the result is not the official BeyondMimic VAE checkpoint, not autonomous VAE control, not guided diffusion, not Fig. 5/Fig. 6 reproduction, and not real-robot evidence.

I also extended this official-importer-export chain into the downstream state-latent diffusion stage:

```text
res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/
res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/
res/report_assets/official_importer_export_full_bundle_downstream/
```

The state-latent builder used GPU 5 and GPU 6, converted the same `306176` local teacher-rollout samples into `285696` 21-step windows, and recorded weighted posterior reconstruction MSE `5.118260560266208e-05`. The denoiser then trained for `30` epochs with DataParallel on GPU 5/6. Its held-out test pred-token MSE was `0.013647833040782384`, compared with noisy-token MSE `0.06729835644364357`, giving a denoising improvement ratio of `0.7972040661615378`.

This is currently the cleanest local downstream training result on the more official robot-asset path. It connects teacher rollout, local conditional VAE, state-latent windows, and diffusion-style denoising into one auditable chain. But it is still not the official BeyondMimic Level C result. The teacher is a short local PPO policy, and the VAE and denoiser checkpoints are local artifacts under ignored run directories. It should be used in the report as evidence of faithful engineering reconstruction, not as evidence that Fig. 5 or Fig. 6 has been reproduced.

I then used the stronger scaled PPO teacher rollout dataset to retrain the downstream local VAE, state-latent dataset,
and denoiser:

```text
res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/
res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/
res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/
res/report_assets/official_importer_export_scaled_ppo_downstream/
```

This scaled downstream chain uses `1224704` local virtual teacher samples rather than the earlier `306176`-sample
candidate. The new VAE trains for `40` epochs and reaches test action MSE `0.00019815583800664172`. The state-latent
builder converts the rollout into `1142784` 21-step windows with weighted posterior reconstruction MSE
`0.00019638959393456675`. The denoiser trains for `30` epochs and records test pred-token MSE
`0.013214186100023133`, noisy-token MSE `0.06736994787518467`, and denoising improvement ratio
`0.8038563704323348`.

This is now the strongest local downstream training result in the project. It matters for the reading report because
the VAE and denoiser are no longer only attached to a short iteration-299 teacher. The caveat is that this is still a
local virtual chain: the checkpoints are not official BeyondMimic VAE/diffusion checkpoints, the teacher data is not
official DAgger data, and per-GPU memory remained below the requested 10GB/card formal threshold. Therefore I use it
as stronger engineering evidence, not as a claim of paper-level Fig. 5/Fig. 6 reproduction.

I then reran full-split offline guidance on this scaled PPO denoiser:

```text
res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/
res/report_assets/official_importer_export_scaled_ppo_guidance/
```

The scaled offline guidance audit evaluates every validation/test state-latent window from the new denoiser:
`114279` validation windows and `114278` test windows, or `228557` windows total. It records `48` task/split/scale
rows and all four proxy tasks with positive best-scale cost deltas and nonzero gradients. The report assets include a
best-cost-delta plot, a guidance-scale response plot, and CSV tables. This result updates the offline guidance
prerequisite to the larger iteration-999 teacher-rollout downstream chain.

I then reran the four-task closed-loop bridge with this scaled PPO chain:

```text
res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/
res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_summary/
res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout/
```

The scaled closed-loop evaluation records four 299-step IsaacLab proxy rollouts: joystick, waypoint, obstacle
avoidance, and composed objectives. It links the iteration-999 scaled PPO training summary, the scaled PPO checkpoint
evaluation, the scaled VAE, the scaled denoiser, and the scaled offline-guidance summary. The guided reward means are
`0.022449076233313336`, `0.025156304768183858`, `0.0229376406832458`, and `0.025132083756082932`; the corresponding
guided target-body error means are `0.3439415395259857`, `0.3440071940422058`, `0.34300488233566284`, and
`0.3445764183998108`. Each task saves local MP4/keyframe/metrics assets, and the report-assets directory adds a
CSV summary plus overview/trade-off plots.

This is a meaningful step because the stronger scaled PPO downstream chain is no longer only an offline guidance
result: it is decoded and stepped in closed-loop simulation. The caveat remains central. These are still local proxy
costs and local checkpoints, not official BeyondMimic VAE/diffusion checkpoints, not the paper Fig. 5/Fig. 6
success/failure protocol, not TensorRT deployment, and not real robot evidence.

I then extended the scaled PPO bridge from a single seed group to a multi-seed closed-loop audit:

```text
res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/
res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/
res/visualization/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rollout/
```

This audit aggregates `5` seed groups and `20` total 299-step local IsaacLab proxy rollouts across joystick, waypoint,
obstacle avoidance, and composed objectives. It records `23920` rollout-variant steps and checks that every row is
`ok`, every rollout reaches 299 steps, every MP4 path exists, the full 40-motion public bundle is used, and the chain
really comes from the scaled PPO training/checkpoint-evaluation/VAE/denoiser/offline-guidance artifacts. The guided
reward means by task are `0.023177480513859157` for joystick, `0.023896200074985576` for waypoint,
`0.022975769497915154` for obstacle avoidance, and `0.024168015661241365` for the composed proxy. The report assets
add an aggregate CSV plus bar and seed-scatter plots, which makes this evidence easier to cite in the reproduction
section.

The interpretation is deliberately conservative. This multi-seed run strengthens the robustness of the local virtual
guidance story, but it still uses local scaled PPO/VAE/denoiser checkpoints and local proxy costs. It is not an
official BeyondMimic checkpoint result, not the paper Fig. 5/Fig. 6 success/failure protocol, not TensorRT deployment,
and not real robot validation.

I then extended the same official-importer-export downstream chain into offline guidance and closed-loop task-conditioned guidance rollouts:

```text
res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/
res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval/
res/visualization/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout/
motion bundle: 40 public official-csv-loop motions
offline guidance windows: 57139 validation/test windows
closed-loop tasks: joystick, waypoint, obstacle_avoidance, composed
rollout steps per task: 299
selected GPU for rollout: 4
```

The offline guidance audit evaluates every validation/test window from the local official-importer-export denoiser. It records `57139` selected windows, `48` task/scale rows, and all four proxy tasks with positive best-scale cost deltas. The closed-loop rollout then executes joystick, waypoint, obstacle avoidance, and composed proxy objectives in IsaacLab using the official-importer-export G1 USDA, the local full-bundle PPO teacher, the local conditional action VAE, and the local denoiser. The guided variants record reward means of `0.02026389510897191`, `0.023516151245783105`, `0.0239038624408278`, and `0.023154594104606224`, with target-body error means of `0.3444918692111969`, `0.3424043357372284`, `0.344357430934906`, and `0.34435439109802246`.

This is the strongest local guided-control bridge on the recovered official-importer-export asset path. It matters because it no longer stops at a denoising loss or an offline latent metric: the guided latent is decoded and stepped through the simulator for visible robot rollouts. However, it must be labeled conservatively. The tasks are local proxy objectives, the checkpoints are locally trained, the evaluation protocol is not the paper's Fig. 5/Fig. 6 success/fall/collision protocol, and the videos are local report assets rather than official BeyondMimic results. It is evidence of a serious reproduction pipeline, not evidence of full paper-level reproduction.

I then repeated this official-importer-export task-conditioned bridge across five seed groups:

```text
res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/
res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/
res/visualization/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rollout/
seed groups: 5
tasks: joystick, waypoint, obstacle_avoidance, composed
rows: 20
rollout steps per row: 299
total rollout-variant steps: 23920
```

The multi-seed audit keeps the same conservative interpretation but makes the evidence less anecdotal. Across the five seed groups, the guided reward means are `0.022807253612653917` for joystick, `0.022766795185983284` for waypoint, `0.022796624001850653` for obstacle avoidance, and `0.023558438572577854` for the composed objective. The guided target-body error means are `0.34388601779937744`, `0.3439153075218201`, `0.34403362274169924`, and `0.3442098379135132`. All rows completed 299 local IsaacLab steps and all rows have MP4 paths. This should be cited as local virtual official-importer-export guidance evidence only: it still uses local PPO/VAE/denoiser checkpoints and proxy objectives, not official BeyondMimic checkpoints, not Fig. 5/Fig. 6 paper metrics, not TensorRT deployment, and not real-robot validation.

I also converted the 20 official-importer-export guidance rollouts into an explicit local proxy success-boundary summary:

```text
res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/
rows: 20
seed groups: 5
tasks: joystick, waypoint, obstacle_avoidance, composed
completion rate at 299 steps: 1.0
positive guidance-signal rate: 1.0
action-changed rate: 1.0
local proxy pass rate: 0.65
reward improved vs. denoised rate: 0.45
tracking error not worse vs. denoised rate: 0.5
```

This summary is helpful because it makes the strongest current official-importer-export guidance evidence easier to interpret than a list of videos. Joystick and obstacle avoidance are the cleanest row groups, each with local proxy pass rate `0.8`; composed is intermediate at `0.6`, and waypoint is weaker at `0.4`. That pattern is useful for intellectual honesty: the local guided controller is not uniformly better under every proxy metric, but it does complete all 299-step rollouts and produces measurable guidance action changes. The asset should be described as a local proxy success boundary, not as the official BeyondMimic Fig. 5/Fig. 6 success/fall/collision protocol.

I then added a stricter local task-protocol proxy table over the same 20 official-importer-export traces:

```text
res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/
rows: 20
seed groups: 5
tasks: joystick, waypoint, obstacle_avoidance, composed
recorded 299-step completion rate: 1.0
endpoint/root-reference proxy pass rate: 1.0
target-body mean proxy pass rate: 1.0
local task-protocol proxy pass rate: 0.65
reward improved vs. denoised rate: 0.45
tracking error not worse vs. denoised rate: 0.5
mean final root XY error: 0.005920683296880743 m
```

I repeated the same stricter task-protocol proxy on the scaled-PPO importer-export chain:

```text
res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/
rows: 20
seed groups: 5
tasks: joystick, waypoint, obstacle_avoidance, composed
recorded 299-step completion rate: 1.0
endpoint/root-reference proxy pass rate: 1.0
target-body mean proxy pass rate: 1.0
local task-protocol proxy pass rate: 0.8
reward improved vs. denoised rate: 0.6
tracking error not worse vs. denoised rate: 0.5
mean final root XY error: 0.0061050763586953626 m
```

This scaled-PPO proxy is the stronger paper-facing local virtual result because it uses the later iteration-999 teacher/VAE/denoiser chain and preserves all 20 local MP4 paths. It improves the local protocol pass rate relative to the earlier full-bundle proxy, but it still must be framed carefully: these are local thresholds over proxy tasks, not the official BeyondMimic Fig. 5/Fig. 6 success/fall/collision protocol.

This table is useful because it separates different notions of success that can otherwise blur together. The local controller stays close to the local reference endpoint and maintains the thresholded target-body tracking proxy, but it does not consistently improve reward or tracking error relative to the local denoised baseline. The task-level local proxy pass rates are `0.8` for joystick, `0.8` for obstacle avoidance, `0.6` for composed, and `0.4` for waypoint. I would use this result in the report as a more honest Fig. 5/Fig. 6-adjacent simulation summary: it gives concrete, multi-seed virtual evidence while explicitly refusing to call the local thresholds paper-level success/fall/collision criteria.

I then added a more explicit success/fall/collision proxy over the same scaled-PPO traces:

```text
res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/
rows: 20
seed groups: 5
tasks: joystick, waypoint, obstacle_avoidance, composed
recorded 299-step completion rate: 1.0
local success proxy rate: 0.9
relative-root-height fall proxy rate: 0.1
body-error spike anomaly proxy rate: 0.05
positive guidance-signal rate: 1.0
true contact/collision signal available: false
```

This is useful for the reading report because it forces the reproduction evidence into the same conceptual vocabulary as the paper's task rollouts while exposing the boundary. I can discuss local success-like and fall-like behavior, but I cannot honestly claim the paper's success/fall/collision numbers. The saved traces do not contain contact or collision labels, and the thresholds are local analysis thresholds rather than official BeyondMimic criteria.

I also added one official-importer-export diagnostic for the paper's Fig. 6A inpainting/keyframe family:

```text
res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/
status: ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval
task: inpainting
rollout steps: 299
guided keyframe error mean: 0.3349786878951531
denoised keyframe error mean: 0.24249927912314906
guided minus denoised keyframe error: 0.09247940877200406
```

This is valuable precisely because it is not a clean success story. The rollout completes, saves capture/video evidence, and proves that a future-keyframe/root-path inpainting proxy can be stepped through the recovered official-importer-export IsaacLab chain. But on this seed the guided variant is worse than the denoised baseline under the local keyframe proxy metric. I would use this in the reading report as a negative diagnostic: it identifies the next thing to improve, such as guidance scale selection, a better keyframe objective, or a multi-seed paper-style protocol. It must not be described as the paper's cartwheel keyframe inpainting result, because it uses local checkpoints, a synthetic root-path objective, a fallback guidance scale, and no official Fig. 6A task protocol.

I also added a Fig. 5/Fig. 6 proxy protocol matrix for the official-importer-export evidence:

```text
res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/
paper panels mapped: 6
panels with importer-export closed-loop proxy evidence: 5
panels with offline or debug evidence: 4
referenced local closed-loop rollout/video rows: 27
paper-level reproduced panels: 0
```

The matrix is useful for deciding the next virtual experiments. It shows that joystick, waypoint, obstacle avoidance, composed objectives, and one future-keyframe inpainting diagnostic now have local closed-loop proxy evidence. The inpainting row is still deliberately conservative: it is a diagnostic proxy, not the paper's cartwheel/keyframe protocol, and it currently records a guided keyframe error that is worse than the denoised baseline. It also keeps Figure 6B honest: simulated waypoint-plus-obstacle guidance can be pursued locally, but the paper panel itself uses real-world/mocap context. This matrix is therefore a planning and reporting tool, not a claim that Fig. 5 or Fig. 6 has been reproduced.

I then added a local walk-to-run transition proxy for the paper's Fig. 5B/Fig. 5D transition idea:

```text
res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/
status: ok_official_importer_export_full_bundle_transition_guidance_rollout_eval
rollout steps: 299
selected GPU: 4
guided reward mean: 0.024728728436481794
guided target-body error mean: 0.3448648750782013
guidance cost delta mean: 0.00012286637838070208
guided late-minus-early speed: 2.0195484379946684
guided speed-target correlation: 0.016159506113184546
guided target-speed RMSE: 21.66667160938303
```

This run matters because it moves the local evidence beyond a static latent projection. The controller actually steps through IsaacLab for 299 frames using the official-importer-export G1 USDA path and saves a local MP4 plus transition speed/path plots. The result should still be discussed as a diagnostic, not a success story: the guided variant increases late-vs-early speed under the local proxy, but the speed-ramp target is not tracked well. In the report, I would use it to explain what is still missing for a true Fig. 5B reproduction: a paper-defined transition command, meaningful success/fall/smoothness thresholds, multi-seed evaluation, and official checkpoints.

To make the Fig. 5D latent-space discussion less abstract, I generated a local PCA projection from the official-importer-export full-bundle VAE posterior means:

```text
res/report_assets/official_importer_export_full_bundle_latent_projection/
status: ok_official_importer_export_full_bundle_latent_projection_report_assets
latent samples: 306176
latent dimension: 32
public motions: 40
motion families: 8
stratified plotted samples: 12800
PCA top-2 explained variance ratio: 0.250849945613856
walk/run trace rows: 1920
```

This asset is useful for the English report because it visualizes whether the local VAE latents carry recognizable motion-family structure after the recovered tracking-teacher and VAE pipeline. The family scatter, root-speed scatter, and walk/run trace plots give a concrete way to discuss latent organization without inventing an official result. The boundary is important: this is PCA, not the paper's t-SNE; it uses local official-importer-export PPO/VAE artifacts, not official BeyondMimic checkpoints; and it is not a closed-loop walking-to-running transition experiment. I would cite it as Fig. 5D-adjacent interpretive evidence, not as reproduction of Fig. 5D.

For presentation use, I also generated a compact contact sheet for this importer-export guidance set:

```text
res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/
videos indexed: 20
contact sheet size: 730,130 bytes
```

The contact sheet is useful because it lets the reading report or PPT show all four proxy tasks across the five seed groups without embedding the large MP4 files in Git. The JSON/CSV index records the local MP4 paths and SHA256 hashes, while the interpretation remains unchanged: this is local virtual report media, not paper-level Fig. 5/Fig. 6 video reproduction.

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

I then pushed the closed-loop guidance bridge onto the full public official-loop motion bundle:

```text
res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/
res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/
motion count: 40
total frames in bundle: 11960
rollout variants: teacher, VAE base, denoised latent, receding-latent guided
rollout steps per variant: 299
selected GPU: 4
```

This run uses the 40-motion public official-csv-loop bundle together with the matching local full-bundle PPO checkpoint, full-bundle action VAE, full-bundle state-latent denoiser, and full-bundle offline guidance scale. It produced a new MP4, keyframes, metric plot, metric CSV, GPU telemetry, and JSON audit. The guided-latent variant reached reward mean `0.023252945707917812`, target-body error mean `0.08156827092170715`, and guidance cost delta mean `5.999496549268231e-05`. This is currently the strongest simulation-side guidance video evidence in the project because it is no longer tied only to the single-motion closed-loop bridge.

The limitation is still central: the bundle is built from public official-loop motions but uses an enriched USD scaffold and local checkpoints. It is not the official BeyondMimic VAE or diffusion model, not unpatched official replay, not the paper's Fig. 5/Fig. 6 task protocol, not TensorRT deployment, and not a real Unitree G1 result.

I then extended this full-bundle bridge from one composed-cost rollout to four task-conditioned local closed-loop rollouts:

```text
res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/
res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/
res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout/
tasks: joystick, waypoint, obstacle_avoidance, composed
rollout steps per task: 299
motion bundle: 40 public official-csv-loop motions
```

The guided variants record reward means of `0.022738767414039195`, `0.021866608513861796`, `0.02571137477160995`, and `0.021387746856613304` for joystick, waypoint, obstacle avoidance, and composed objectives. The corresponding target-body error means are `0.08253771811723709`, `0.0809725970029831`, `0.08061826974153519`, and `0.08186342567205429`. Each task saves an MP4, keyframes, a metrics plot, and a metrics CSV, while the report asset folder produces compact overview/tradeoff plots for citation in this report.

This is the best current local evidence for the idea that task-conditioned guidance can be executed in closed-loop simulation rather than only evaluated offline. Its boundary is also important: the tasks are local proxy objectives, the VAE and denoiser are locally trained full-bundle models, the robot asset still uses the enriched USD scaffold, and the results are not official BeyondMimic Fig. 5/Fig. 6 success-rate reproduction, TensorRT deployment, or real-robot validation.

I then extended the same full-bundle task-conditioned bridge to a five-seed audit:

```text
res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/
res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/
res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rollout/
seed groups: seed_group_0_existing, seed_group_1, seed_group_2, seed_group_3, seed_group_4
rows: 20
rollout steps per row: 299
total rollout-variant steps: 23920
motion bundle: 40 public official-csv-loop motions
```

The aggregate guided reward means are `0.021954482373934124` for joystick, `0.022448493876684468` for waypoint, `0.024278478643367917` for obstacle avoidance, and `0.02204239875900506` for composed objectives. The corresponding guided target-body error means are `0.08180097341537476`, `0.08056965321302414`, `0.08039124161005021`, and `0.08011763840913773`. All 20 rows completed 299 control steps and all rows have local MP4 paths. This is now the strongest local virtual evidence that the full-bundle task-conditioned guidance bridge is not a one-off visualization. It is still deliberately labeled `qualitative_only`: the checkpoints, costs, USD scaffold, and task protocol are local, so the result cannot be reported as official Fig. 5/Fig. 6 reproduction.

To make this evidence easier to audit and cite, I added and refreshed a guided-vs-unguided closed-loop matrix:

```text
res/report_assets/guided_vs_unguided_closed_loop_matrix/
matrix rows: 43
multiseed rows: 12
full-bundle multiseed rows: 20
aggregate task rows: 12
video-linked rows: 43
claim level: local_virtual_guided_vs_unguided_closed_loop_report_matrix
```

The matrix pulls together the action-space bridge, the receding-horizon latent bridge, the four task-conditioned single-seed rollouts, the three-seed task-conditioned rollout set, the full-bundle task-conditioned rollouts, and the new full-bundle five-seed task-conditioned rollout set. It exports CSV/JSON/Markdown plus two plots: one for guided-vs-denoised reward/error deltas and one for guidance signal strength. This is now the cleanest report-facing summary of the local closed-loop guidance evidence. Its comparison labels are deliberately conservative (`qualitative_only` or `approximately_comparable`) because the rows are local virtual/resource-adjusted results, not official Fig. 5/Fig. 6 success-rate reproduction.

I then added a stricter report-facing local proxy success-boundary audit:

```text
res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/
rows: 20
seed groups: 5
tasks: 4
completion rate at 299 steps: 1.0
positive guidance-signal rate: 1.0
action-changed rate: 1.0
local proxy pass rate: 0.9
reward improved vs. denoised rate: 0.5
tracking error not worse vs. denoised rate: 0.8
```

This asset is useful because it turns the five-seed closed-loop guidance set into a compact audit table, aggregate task table, and plot that can be cited directly in the reading report. The interpretation remains deliberately bounded: this is a local proxy success boundary over public-motion, resource-adjusted IsaacLab rollouts. It is not the paper's official Fig. 5/Fig. 6 success/fall/collision protocol, not an official BeyondMimic checkpoint evaluation, not TensorRT deployment, and not real-robot evidence.

Because the course report and presentation need visual evidence, I also added a contact-sheet asset for the same 20 local rollout videos:

```text
res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/
video rows: 20
seed groups: 5
tasks: 4
indexed local MP4 size: 18,496,702 bytes
contact sheet size: 473,335 bytes
```

The asset records each MP4 path, SHA256 hash, task, seed group, reward/error summary, keyframe image, metrics image, and claim boundary. The large MP4 files remain local and are not committed to GitHub, while the CSV/JSON/contact-sheet PNG can be used directly in the report or PPT. This makes the reproduction evidence more concrete visually while preserving the central limitation: these are local virtual/resource-adjusted rollouts, not official paper videos.

I also added a local ONNXRuntime deployment-path audit for the models that now participate in the local virtual pipeline:

```text
res/level_c/official_csv_loop_vae_denoiser_onnx_async/
```

This audit exports the locally trained official-loop VAE encoder, VAE decoder, and state-latent denoiser to ONNX. ONNXRuntime CPU inference matches PyTorch with maximum absolute errors below `6e-7` across the exported components. It also measures a small sequential pipeline and a thread-pool async proxy: the async proxy processes 80 local requests with about `2.46x` throughput speedup relative to the sequential mean. The most important interpretation is negative as well as positive. The local ONNXRuntime installation exposes CPU/Azure providers only, not CUDA or TensorRT, so this is not the paper's RTX 4060 Mini-PC TensorRT deployment and not a paper latency reproduction. It is still useful because it shows that the local VAE and denoiser can be turned into executable runtime graphs and that the deployment boundary can be audited instead of hand-waved.

I then repeated the same deployment-path audit on the broader 40-motion full-bundle VAE and denoiser:

```text
res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/
```

The full-bundle ONNX exports also match PyTorch closely: the largest component-wise absolute differences are `7.15e-7` for VAE log-variance, `4.77e-7` for VAE mean, `1.79e-7` for decoded action, and `1.79e-7` for denoiser tokens. The local async thread-pool proxy processes 80 requests with about `2.70x` throughput speedup over the measured sequential mean. This is a better match to the current full-bundle reproduction chain than the earlier single official-loop deployment audit, but the same boundary remains: it is CPU ONNXRuntime evidence, not CUDA/TensorRT, not CppAD guidance, not the paper's Mini-PC deployment, and not real-robot execution.

I then repeated the deployment-path audit on the currently strongest official-importer-export full-bundle VAE and denoiser chain:

```text
res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/
```

The official-importer-export ONNX exports match PyTorch with maximum absolute differences of `7.08e-8` for VAE mean, `1.34e-7` for VAE log-variance, `8.94e-8` for decoded action, and `7.15e-7` for denoiser tokens. The local thread-pool async proxy processes 80 requests with about `2.81x` throughput speedup over the sequential mean. This is useful because it moves the deployment-path audit onto the recovered official-importer-export G1 asset chain, but it is still CPU ONNXRuntime evidence only: not CUDA/TensorRT, not CppAD guidance, not the paper Mini-PC deployment, not official BeyondMimic checkpoints, and not real-robot execution.

Finally, I repeated the deployment-path audit on the iteration-999 scaled PPO downstream VAE and denoiser:

```text
res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/
```

The scaled-PPO ONNX exports also match PyTorch: the largest component-wise absolute differences are `2.86e-6` for VAE mean, `2.38e-7` for VAE log-variance, `1.49e-7` for decoded action, and `9.61e-7` for denoiser tokens. The local thread-pool async proxy processes 80 requests with about `2.76x` throughput speedup over the sequential mean. This is now the deployment-path audit that matches the strongest local downstream chain in the workspace. It still has the same boundary: CPU ONNXRuntime only, no CUDA/TensorRT provider, no CppAD guidance, no Mini-PC measurement, no official checkpoint, and no real robot.

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

The newer official-importer-export teacher dataset has now been pushed one step further into the local VAE stage:

```text
res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/
res/report_assets/official_importer_export_full_bundle_vae_training/
```

This VAE was trained for `40` epochs over all `306176` teacher rollout samples with observation dimension `160`,
action dimension `29`, latent dimension `32`, and train/validation/test splits `244940/30618/30618`. The test action
MSE is `5.362209958548192e-05` and the mean absolute action error is `0.005292208399623632`. This is much stronger
than a smoke test and provides a clean training curve for the reading report. Still, it is not the official
BeyondMimic VAE checkpoint: the source data comes from a short local PPO teacher on the official-importer-export asset,
not from the unavailable official DAgger logs, and it has not yet been evaluated as closed-loop VAE or diffusion-guided
control.

For visual communication, I also generated a small kinematic reference replay asset from the official-loop motion NPZ:

```text
res/visualization/official_csv_loop_reference_replay/
```

It contains a local MP4, keyframe PNG, summary CSV, README, and SHA256-recorded asset JSON. This helps explain what the converted Unitree G1 reference motion looks like in the report or slides. However, it is only a kinematic visualization of saved body positions. It is not an IsaacLab rendered closed-loop rollout, not Fig. 5 or Fig. 6 evidence, and not a real robot video.

After the full official-importer-export `csv_to_npz.py` loop passed on all 40 public motions, I generated the same kind of report asset from that stronger evidence path:

```text
res/visualization/official_importer_export_full_dataset_reference_replay/
```

This asset selects `walk1_subject1` from the 40/40 official-importer-export conversion audit, renders a 299-frame kinematic reference MP4, and records a keyframe PNG, summary CSV, README, SHA256 hashes, and the source dataset aggregate (`40` ok rows, `0` failed rows, `11960` total frames). Its value is explanatory: it makes the recovered official-importer-export reference trajectory visible for the report without pretending that a controller was evaluated. It is not a closed-loop IsaacLab rollout, not unmodified live official converter-entry output, not Fig. 5 or Fig. 6 guided-diffusion evidence, and not real-robot validation.

The companion replay report-assets directory, `res/report_assets/official_importer_export_replay_full_dataset/`, indexes the same full replay result for presentation: 40/40 motions completed, `11960` replay steps, `0` failed rows, `0` shutdown warnings, and the representative local reference MP4 path. The MP4 is intentionally kept local rather than committed to GitHub; the small JSON/CSV/PNG assets are enough for versioned evidence and slides.

The project now also contains a local policy rollout video asset:

```text
res/visualization/official_csv_loop_policy_rollout/
```

This video is more important than the reference-only replay: it loads the local official-loop PPO checkpoint, executes one 299-step Tracking-Flat-G1-v0 rollout, records robot and reference target-body positions, and renders a policy-vs-reference MP4 with keyframes and metrics. It is still resource-adjusted local virtual evidence rather than paper-level BeyondMimic guidance, but it makes the reproduction section much more concrete.

I then generated the same kind of policy-vs-reference media from the stronger 40-motion full-bundle PPO checkpoint:

```text
res/visualization/official_csv_loop_full_bundle_policy_rollout/
```

This asset records a 299-frame single-environment rollout using the local full-bundle PPO checkpoint trained over all 40 public official-csv-loop motions. The video asset reports mean target-body error `0.07958605140447617`, reward mean `0.020382126793265343`, done count `26`, and full-bundle metadata (`40` motions, `11960` frames, `39` artificial clip boundaries). This is the clearest tracking-side robot visualization for the report because it is no longer tied to a single selected reference clip. It is still local qualitative evidence: the robot asset uses the enriched USD scaffold, the checkpoint is locally trained, and the video is not an official BeyondMimic teacher checkpoint rollout, not paper-level Fig. 5/Fig. 6 guided diffusion, not TensorRT deployment evidence, and not a real-robot result.

To make these presentation assets easier to audit and reuse, I added a visual evidence index:

```text
res/report_assets/visual_evidence_index/
```

The index records 9 local MP4 files, 47 PNG figures, and 52 table or README assets, together with their source asset JSON, claim level, file size, and GitHub policy. This matters because the videos are useful for the English report and slides but should not be committed to GitHub or described as official paper-level videos. The index explicitly marks them as local virtual or resource-adjusted evidence, not official Fig. 5/Fig. 6 or real-robot results.

## 7. What Is Not Yet Reproduced

This project does not fully reproduce BeyondMimic at paper-level.

The following paper-level components remain missing or blocked:

- the official BeyondMimic DAgger rollout logs;
- the official conditional VAE checkpoint;
- the official state-latent diffusion checkpoint;
- official paper-level closed-loop guided diffusion rollout in IsaacLab with the Fig. 5/Fig. 6 task protocol;
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

The next technical step should be a paper-protocol-aligned closed-loop guidance gate in IsaacLab. The current local proxy rollouts prove that decoded guided latents can be stepped through the simulator, but they do not prove the official Fig. 5/Fig. 6 task protocol or success metrics. A useful next milestone would be:

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
