# Current Environment And Reproduction Status

Generated: 2026-06-19 18:10 Asia/Shanghai

This report answers three operational questions for the BeyondMimic reproduction workspace:

1. How complete is the current environment?
2. How much of the paper has been reproduced with auditable evidence?
3. What can still be verified in simulation before any real-robot work?

The short answer is: the project is now a substantial, auditable reproduction workspace, but it is not a complete paper-level reproduction of BeyondMimic. The most valuable next work is to push the local virtual chain from tracking and VAE reconstruction toward closed-loop latent-diffusion guidance, while preserving the strict boundary between official paper results and resource-adjusted local evidence.

## Environment Status

Root directory:

```text
/mnt/infini-data/test/BeyondMimic
```

The expected project-local directories exist:

```text
download/
other/
reproduction/
res/
logs/
envs/
cache/
tmp/
```

`download/` remains the raw read-only source area. `other/` remains the preserved old-server snapshot. Large runtime outputs, environments, caches, logs, runs, checkpoints, videos, USD assets, and raw datasets are ignored by Git.

### Host And GPU

Current host probe:

- Python: `/usr/bin/python3`, Python `3.10.12`
- User: `root`
- Project filesystem: `/mnt/infini-data`
- Disk: about `244T` total, about `293G` available, mounted at 100 percent usage
- Inodes: healthy, about 1 percent used
- NVIDIA driver: `570.124.06`
- CUDA reported by `nvidia-smi`: `12.8`
- GPUs: 8 x NVIDIA H20, about `95.09 GiB` each

At probe time, GPU 4, 6, and 7 were effectively free, while GPUs 0, 1, 2, 3, and 5 had existing load. For formal new GPU experiments in this project, the current policy is to use physical GPUs 4 and 7 and record memory/utilization.

One small probe failed because `nvidia-smi --query-gpu` does not accept `cuda_version` as a CSV query field. The value was recovered from the standard `nvidia-smi` header instead.

### `bm_analysis`

Status: usable.

Validated imports:

```text
numpy       2.2.6
pandas      2.3.3
matplotlib  3.10.9
onnx        1.22.0
onnxruntime 1.23.2
```

This environment is suitable for audits, table generation, plotting, manifest refresh, report generation, and ONNX CPU/runtime checks.

### `bm_diffusion`

Status: usable for PyTorch VAE/diffusion/guidance work.

Validated under `CUDA_VISIBLE_DEVICES=5,6`:

```text
torch              2.5.1+cu121
cuda_available     true
logical devices    2
device names       NVIDIA H20, NVIDIA H20
visible memory     about 95.09 GiB per logical GPU
```

The existing audit `res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json` is `ok`. The environment has already supported local VAE training, state-latent dataset generation, denoiser training, offline guidance evaluation, and action-decoding evaluation. These are local/resource-adjusted experiments, not official BeyondMimic checkpoints.

### `bm_tracking` / Isaac Sim / Isaac Lab

Status: package layer and AppLauncher gate are substantially restored, but official paper-level tracking remains blocked by the G1 conversion/replay chain.

Machine-readable evidence:

- `res/setup/env_probe/env_import_probe.json`: `ok_with_runtime_warning`
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`: `ok_with_runtime_warning`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`: `ok`

Important current checks:

- `isaaclab_import_ok=true`
- `isaacsim_import_ok=true` in the audited AppLauncher/runtime path
- `isaaclab_live_headless_gate_ok=true`
- `app_launcher_headless_success_sentinel=true`
- Current AppLauncher blocker: `none`
- CUDA P2P/IOMMU runtime warning is retained as a runtime risk

Plain non-Kit Python imports are not equivalent to a live Isaac Sim session. A direct import probe showed:

```text
isaaclab        ok
isaaclab_rl     ok
isaaclab_mimic  ok
onnx            1.16.1
onnxscript      0.1.0
isaacsim        failed in non-interactive EULA/stdin path
isaaclab_assets failed without carb runtime
isaaclab_tasks  failed without omni.kit runtime
```

This does not contradict the AppLauncher gate. It means the tracking stack should be exercised through the configured Isaac/Kit entrypoints, not by assuming all deep Kit modules import in a plain interpreter.

### Environment Completeness Judgment

Environment completeness is currently:

- `bm_analysis`: complete for analysis/reporting.
- `bm_diffusion`: complete for local PyTorch VAE/diffusion/guidance experiments.
- `bm_tracking`: complete enough for current AppLauncher and resource-adjusted IsaacLab task/rollout experiments, but not complete for unpatched official paper-level G1 replay/training because official G1 URDF/USD conversion remains blocked.

Therefore, the environment is not "fully complete" for the original BeyondMimic paper-level reproduction, but it is no longer stuck at package installation. The current blocker is a paper-facing robotics/runtime asset gate, not a missing Python package alone.

## Reproduction Progress

Current audit summary:

```text
master_audit: ok
master artifacts: 257/257 passed
artifact_manifest: 418 artifacts
paper_vs_reproduction: 147 rows
completion matrix: complete 73, partial 88, blocked 3, out_of_scope 1
goal_complete: false
```

Paper-vs-reproduction comparison:

```text
exactly_comparable:          58
approximately_comparable:    19
qualitative_only:            57
not_publicly_reproducible:   10
requires_real_robot:          3
```

These counts show meaningful progress, but they do not imply full reproduction. Many rows are audits, released-data reproductions, paper-faithful debug reimplementations, or local virtual experiments rather than official paper-level closed-loop results.

### What Is Solidly Completed

The following parts are currently strong and reportable with proper boundaries:

- Local inventory, source ledger, paper/source mapping, and reproducibility documentation.
- Released-data figure/table reproduction for the public dataset scope.
- Paper table value audit and released-data statistical summaries.
- Paper/source coverage, panel mapping, formula/code trace, and experiment protocol.
- Official tracking code/config static audits.
- Observation/action schema, reward, randomization, termination, ONNX, motion preprocessing, MuJoCo/ROS launch, and deployment contract audits.
- IsaacLab/AppLauncher headless gate in the current runtime path.
- Resource-adjusted G1 enriched USD scaffold and task-contract gates.
- Official `csv_to_npz.py` and `replay_npz.py` loop-body evidence under an enriched-USD runtime patch.
- Local official-csv-loop PPO training and checkpoint evaluation at reduced scale.
- Local teacher rollout dataset from that reduced-scale checkpoint.
- Local conditional action VAE training on the collected rollout dataset.
- Local state/action-latent dataset construction.
- Local state-latent denoiser training.
- Full validation/test offline state-latent guidance surrogate.
- Offline guided latent to VAE action decoding.
- Local closed-loop VAE action-reconstruction rollout evaluation.
- Local teacher-consistency action-guidance rollout with video/metrics assets.
- Local receding-horizon latent-guidance rollout with video/metrics assets.
- English reading report draft and final-report copy.

### Most Important Recent Local Results

Official-csv-loop PPO chain:

- `res/tracking/g1_official_csv_loop_ppo_training_run/tracking_g1_official_csv_loop_ppo_training_run.json`
- `res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json`
- `res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json`
- `res/tracking/g1_official_csv_loop_teacher_rollout_dataset/tracking_g1_official_csv_loop_teacher_rollout_dataset.json`

These runs use GPUs 4 and 7 and are valuable local virtual evidence. They are still not paper-level official PPO because the motion/asset path uses the enriched-USD runtime patch and training is far below paper scale.

The latest checkpoint evaluation has now been repeated across three seeds:

```text
seeds: 20260640, 20260641, 20260642
GPU assignment: 4, 7, 4
num_envs per seed: 512
eval_steps per seed: 299
total_env_steps: 459264
reward_mean: 0.025978426701298924 +/- 0.00010146760409522878
body_pos_error_mean: 0.18423418407697012 +/- 0.000271408645496586
joint_pos_error_mean: 1.2231450603159773 +/- 0.0027425904840304373
```

Report assets are saved under `res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/`. This improves stability evidence for the local virtual tracking chain. It is still not the official paper tracking teacher, not unpatched official G1 replay/training, not DAgger, and not real-robot evidence.

VAE/diffusion/guidance chain:

- VAE training: `res/level_c/official_csv_loop_teacher_rollout_vae_training/level_c_official_csv_loop_teacher_rollout_vae_training.json`
- State-latent dataset: `res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json`
- Denoiser training: `res/level_c/official_csv_loop_state_latent_diffusion_training/level_c_official_csv_loop_state_latent_diffusion_training.json`
- Offline guidance: `res/level_c/official_csv_loop_state_latent_guidance_eval/level_c_official_csv_loop_state_latent_guidance_eval.json`
- Guided latent action decode: `res/level_c/official_csv_loop_guidance_vae_action_decode_eval/level_c_official_csv_loop_guidance_vae_action_decode_eval.json`
- VAE closed-loop rollout: `res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json`
- Action-guidance rollout: `res/level_c/official_csv_loop_action_guidance_rollout_eval/level_c_official_csv_loop_action_guidance_rollout_eval.json`
- Receding-horizon latent-guidance rollout: `res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json`
- Task-conditioned latent-guidance rollouts: `res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json`

The VAE closed-loop rollout ran a formal local evaluation with 2048 environments over 299 steps per shard. It honestly records that the per-GPU 10GB threshold was not met on both GPUs. The action-guidance and receding-horizon latent-guidance rollouts are single-environment evidence/visualization runs and therefore are not formal GPU experiments.

The task-conditioned latent-guidance rollout runs four 299-step local IsaacLab proxy tasks on GPU 4: joystick, waypoint, obstacle_avoidance, and composed. Each task compares teacher, VAE-base, denoised-latent, and receding-horizon guided-latent variants and saves local MP4/keyframes/metric plots/CSV under `res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/`. This is currently the strongest virtual evidence for the guided-control part of the reading report, but it is still local proxy-cost evidence rather than official Fig. 5/Fig. 6 reproduction.

Report-ready aggregate figures/tables for these four task-conditioned rollouts are saved under `res/report_assets/official_csv_loop_task_conditioned_guidance_summary/`, including an overview comparison plot, a guidance-cost/tracking-error tradeoff plot, guided summary CSV, and full metrics CSV.

The local deployment-path audit now exports the official-csv-loop VAE encoder, VAE decoder, and state-latent denoiser to ONNX and verifies ONNXRuntime CPU inference against PyTorch:

- ONNX/async audit: `res/level_c/official_csv_loop_vae_denoiser_onnx_async/level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json`
- Latency CSV: `res/level_c/official_csv_loop_vae_denoiser_onnx_async/level_c_official_csv_loop_vae_denoiser_onnx_async_latency.csv`

The maximum ONNXRuntime-vs-PyTorch errors are below `6e-7` for the exported components, and the local CPU thread-pool async proxy reports about `2.46x` throughput speedup versus sequential mean latency. This is not TensorRT and not paper Mini-PC latency: the local ONNXRuntime build exposes `AzureExecutionProvider` and `CPUExecutionProvider` only, with no CUDA or TensorRT provider. It is therefore a useful deployment-path audit for the reading report, not a paper-level deployment reproduction.

### What Is Not Yet Reproduced

The following cannot be claimed as complete:

- Official unpatched G1 `csv_to_npz.py` conversion and paper-level official replay.
- Full paper-scale PPO tracking teacher training/evaluation.
- True official BeyondMimic DAgger rollout logs.
- Official BeyondMimic conditional VAE checkpoint.
- Official BeyondMimic state-latent diffusion checkpoint.
- Receding-horizon latent diffusion control in IsaacLab matching the paper.
- Fig. 5 and Fig. 6 paper-level videos and metrics.
- TensorRT/CUDA-provider/Mini-PC deployment of the paper-level policy stack. A local CPU ONNXRuntime async proxy exists, but it is not the paper deployment result.
- Real Unitree G1 robot results.

The required artifact absence audit explicitly records missing official checkpoints, paper-required rollout videos, DAgger logs, and real robot evidence:

```text
res/required_artifact_absence/required_artifact_absence_audit.json
```

## What Can Still Be Verified In Simulation

Without real robot hardware, the project can still make substantial progress in simulation. The useful next simulation targets are:

1. Unpatched official G1 conversion/replay recovery

   Keep attacking the official G1 URDF/USD conversion path so `csv_to_npz.py` and `replay_npz.py` can run without the enriched-USD runtime patch. This is the cleanest way to strengthen the tracking section.

2. More robust official-csv-loop PPO evaluation

   Run multi-seed and longer evaluation for the local official-csv-loop PPO chain. This would still be resource-adjusted, but it would produce stronger evidence than a single reduced run.

3. Closed-loop VAE reconstruction and ablations

   Expand current VAE closed-loop rollout evidence with multiple seeds, autonomous latent sampling variants, and failure/survival analysis. These must remain labeled as local VAE evidence, not official BeyondMimic VAE.

4. Multi-task and multi-seed latent guidance rollouts

   The current guidance stack now includes denoiser training, offline guidance, guided latent decoding, action-space teacher-consistency rollout, and a first local receding-horizon latent-guidance rollout in IsaacLab. The next main paper-facing simulation step is to extend the guidance bridge beyond one composed-cost local run: multiple seeds, task-specific joystick/waypoint/inpainting/obstacle proxy costs, and clearer with/without-guidance success/failure metrics.

5. Guidance task proxies

   Implement virtual joystick/velocity, waypoint, inpainting, obstacle, and composed-objective proxy evaluations in simulation. These can support the English report's method-understanding section, but must not be reported as Fig. 5/Fig. 6 paper reproduction unless they use the paper's closed-loop setup and metrics.

6. TensorRT/asynchronous deployment audit

   Export the local VAE/denoiser components where feasible, measure ONNX/TensorRT/latency, and document what is or is not comparable to the paper deployment stack. This can be useful even without hardware.

7. Visual evidence for the reading report

   Keep producing small, clearly labeled videos/keyframes/plots for local virtual behavior: reference replay, policy rollout, VAE reconstruction rollout, and guidance variants. These are good report evidence if the captions explicitly say they are not paper-level Fig. 5/Fig. 6.

## Recommended Next Step

The next technical step should be:

```text
Extend the successful resource-adjusted receding-horizon latent guidance rollout into task-specific and multi-seed IsaacLab evaluations.
```

The run should:

- reuse the existing local official-csv-loop state-latent denoiser and VAE;
- run at least joystick/velocity and waypoint-style proxy costs before obstacle/inpainting;
- compare teacher, VAE-base, denoised-latent, and guided-latent variants;
- record reward, done, target-body error, action deltas, guidance cost deltas, finite checks, and video/keyframes;
- state clearly that the results are local virtual bridges, not official Fig. 5/Fig. 6.

If that proves too unstable, the fallback should be a TensorRT/ONNX/asynchronous deployment audit for the local VAE and denoiser, because that also directly supports the paper reading report.

## GitHub Status

The repository is on branch `main` with remote:

```text
https://github.com/hunter20041220/BeyondMimic.git
```

The latest committed baseline before this report was:

```text
d86820c feat: add action guidance rollout eval
```

Git ignores large artifacts such as:

```text
download/
other/
envs/
cache/
tmp/
logs/
res/runs/
res/checkpoints/
*.pt
*.pth
*.ckpt
*.onnx
*.engine
*.npz
*.npy
*.mp4
*.usd
*.usda
```

Only small code, documentation, audit JSON/CSV/TSV summaries, and report-facing Markdown should be committed.

## Boundary Statement

This project currently cannot claim a full BeyondMimic reproduction. It can claim a substantial, auditable reproduction and analysis workspace with released-data reproduction, official-code/static audits, restored local environments, resource-adjusted tracking experiments, local VAE/diffusion/guidance experiments, and clear evidence of what remains blocked.

Current不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
