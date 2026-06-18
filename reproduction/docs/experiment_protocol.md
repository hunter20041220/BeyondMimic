# BeyondMimic Experiment Protocol

This protocol turns `goal.md` into executable gates for the current workspace. It must be followed before any result is
reported as a paper reproduction.

## Global Rules

- Raw material under `/mnt/infini-data/test/BeyondMimic/download` is read-only.
- Generated code, logs, configs, figures, metrics, checkpoints, and videos stay under `reproduction`, `logs`, or `res`.
- Do not fabricate metrics, videos, training curves, GPU utilization, missing checkpoints, or missing Fig. 5/Fig. 6 data.
- Do not mark a run `SUCCESS` unless it reaches the declared training/evaluation endpoint.
- Preserve failed runs under `res/failed_runs`.
- Long training is forbidden until the relevant smoke gates pass.

## Required Entry Points

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/resolved_reproduction_config.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py
```

## Phase 0: Inventory, Sources, And Config

Required evidence:

- `reproduction/docs/local_inventory.tsv`
- `reproduction/docs/source_ledger.md`
- `reproduction/docs/paper_parameter_map.md`
- `reproduction/docs/discrepancy_report.md`
- `reproduction/docs/unresolved_details.md`
- `reproduction/docs/environment_plan.md`
- `res/config/resolved_reproduction_config.json`
- `res/artifact_manifest/artifact_manifest.json`

Gate: all key raw sources, selected official repos, environment locks, config files, and final reports must be hashed or
listed before claiming reproducibility.

## Phase 1: Environments And Smoke Tests

Required environments:

- `/mnt/infini-data/test/BeyondMimic/envs/bm_analysis`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion`

Required evidence:

- `res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json`
- `res/setup/gpu_resource_audit/gpu_resource_audit.json`
- `res/blocked_gates/blocked_gate_audit.json`
- `res/failed_runs/failed_run_audit/failed_run_audit.json`

Current blocker: IsaacLab/AppLauncher reaches the live headless sentinel, but official G1 motion preprocessing/replay is
still blocked in the USD conversion path. The latest tracking probes show that direct `Usd.Stage.Export(...)` can write
non-empty local USD files, but routing the URDF importer's initial `Stage.Save()` through `Stage.Export()` still leaves
the generated G1 destination/current stages empty because deeper base/physics/sensor layer saves remain blocked. The
latest deeper probe shows Python-visible `Sdf.Layer.Save` can be monkeypatched for direct test layers, but the URDF
importer's C++/Kit configuration-layer save path is not intercepted by that Python patch and still produces empty
base/physics/sensor layers. `dest_path=""` in-memory import attempts under both AppLauncher and raw `SimulationApp`
with the IsaacLab headless experience reach the importer branch that avoids layered output, but currently record Vulkan
device loss before a current-stage export can be captured. A variant matrix over GPU 5, GPU 6, waitIdle/low-RTX
settings, and the headless-rendering experience produced no valid G1 USD. The next recovery path should prioritize a
trusted preconverted G1 USD or a lower-level/offline URDF conversion path before retrying official `csv_to_npz.py` /
`replay_npz.py`. The local asset audit found official mesh-level G1 USD files but no official full-robot G1 USD; a
reference-code ASAP G1 USD opens as a robot-like stage and can be evaluated only as a clearly labeled
resource-adjusted workaround. The compatibility audit shows that the reference USD has all official target bodies but
locks the six wrist action joints as fixed joints, so it cannot be used as a drop-in 29-DoF BeyondMimic replay asset.
A minimal official-URDF-derived 29-DoF skeleton USD now preserves the official 40-link/29-revolute-joint/14-target-body
contract and opens in Kit, but it is a placeholder scaffold without mesh, collision, inertia, drive, converter, or
replay validation. The URDF physical asset contract audit confirms that all visual mesh references, collision
primitives, non-fixed joint axes/limits, and action-drive rows are available for an offline converter scaffold; the
G1 URDF source-equivalence audit further confirms that the downloaded official LAFAN G1 URDF and reproduction-data copy
are byte-identical and that the official `whole_body_tracking` URDF preserves the same 29 non-fixed/action joints, while
support links/joints and physical bookkeeping differ. The current resource-adjusted enriched USD scaffold authors those
fields as metadata/proxy geometry and reads back the expected counts, but it is not official converter output. The
remaining work is to validate or refine that scaffold through `csv_to_npz.py` / `replay_npz.py`. Do not start long
tracking training until a physically faithful official 29-DoF USD or an explicitly resource-adjusted locked-wrist
contract, `motion.npz`, and replay gate are produced.

The current official `replay_npz.py` entry diagnostic reaches AppLauncher with the unmodified official script, but it
blocks in the official URDF converter layer-save path before fake-WandB artifact download or replay-loop execution.
This confirms that the official replay gate is still blocked by converter/write-path behavior, not only by missing
WandB credentials or a local registry shim.

## Phase 2: Released Data And Figures

Required evidence:

- `res/released_figures/released_figure_summary.tsv`
- `reproduction/docs/paper_panel_map.tsv`
- `res/released_panel_mapping_audit/released_panel_mapping_audit.json`

Each released-data figure must retain source hashes, processing script, PDF/SVG/PNG output, and paper-panel mapping.
Released-data panels may be claimed only for the released-data scope.

## Phase 3: Motion Tracking Smoke

Prerequisites:

- Phase 1 Kit gate cleared.
- `motion_preprocessing_contract_audit` still passes.
- No long-training safety gate is active.

Required outputs for a smoke run:

- valid run directory under `res/runs`
- `gpu_metrics.csv`
- checkpoint if PPO reaches the smoke endpoint
- evaluation metrics and replay video
- failed-run record if it fails

Smoke results are not final paper results.

## Phase 4: Motion Tracking Full Reproduction

Use official `whole_body_tracking` worktree and resolved config unless a discrepancy is explicitly documented.

Minimum reporting requirements:

- at least 3 seeds when feasible
- reward components
- tracking errors
- fall/success rates
- adaptive sampling distributions
- throughput and GPU metrics
- checkpoints and videos

Do not adjust model definitions only to increase GPU utilization.

## Phase 5: Conditional VAE And DAgger

Prerequisites:

- trained tracking teacher checkpoint
- accepted teacher rollout dataset
- run schema and GPU logging active

Required gates:

- VAE forward/backward smoke
- gradient accumulation check
- DAgger manifest with true rollout data
- reconstruction evaluation
- closed-loop VAE rollout

Current status: only debug/schema probes exist. Do not claim paper VAE reproduction.

## Phase 6: State-Latent Dataset

Every sample must record:

- source motion
- teacher/student policy
- start/end timestep
- state frame
- latent
- augmentation
- accept/reject
- split

Required gates:

- no train/validation/test leakage
- paper-state transform checks
- trajectory inverse transform checks
- VAE latents from real rollout, not synthetic placeholders

## Phase 7: Diffusion Training

Required order:

1. single-batch overfit
2. single-motion overfit
3. small-dataset overfit
4. held-out debug evaluation
5. full DDP training only after true dataset exists

The effective global batch size must match the paper config unless a documented resource-adjusted config preserves the
effective batch through accumulation.

Current status: debug overfit and held-out baselines exist, but no full diffusion checkpoint or paper rollout evaluation
exists.

## Phase 8: Guidance Tasks

Task order:

1. unconditional rollout
2. joystick
3. waypoint
4. inpainting
5. obstacle avoidance
6. composed objectives

Each task requires without-guidance and with-guidance comparisons, multiple guidance weights, quantitative metrics, and
success/failure videos. Do not tune guidance weight on a held-out test set.

## Phase 9: Ablations

Change exactly one variable at a time.

Tracking ablations:

- Rot6D / quaternion / axis-angle
- history 1 / 4 / 8 / 25
- armature x0 / x0.1 / original / x10
- delay 0 / 2 / 5 / 10 ms
- adaptive sampling on/off
- PD natural frequency

Diffusion ablations:

- direct state-action diffusion
- latent diffusion
- without OU perturbation
- without symmetry augmentation
- without emphasis projection
- history sensitivity
- horizon sensitivity
- denoising-step sensitivity
- guidance-scale sensitivity

## Phase 10: Comparison And Final Report

Required outputs:

- `res/comparison/paper_vs_reproduction.csv`
- `res/comparison/paper_vs_reproduction.md`
- `res/final_report/final_reproduction_report.json`
- `reproduction/docs/final_reproduction_report.md`

The final report must separate:

- official-code results
- clean-room/debug reimplementation evidence
- released-data redraws
- true retraining results
- qualitative-only comparisons
- not-publicly-reproducible claims
- hardware-required claims
- blocked gates

## Run Directory Contract

Every real or diagnostic run must contain:

- `resolved_config.yaml`
- `command.sh`
- `stdout.log`
- `stderr.log`
- `environment.txt`
- `git_state.txt`
- `gpu_metrics.csv`
- `metrics.json`
- `metrics.csv`
- `checkpoint/`
- `figures/`
- `videos/`
- `status.json`

Allowed statuses are `QUEUED`, `RUNNING`, `SUCCESS`, `FAILED`, `FAILED_OOM`, `INTERRUPTED`, and `INVALID`.

## Failure Handling

On failure:

- keep the run directory
- write a record under `res/failed_runs`
- preserve the last log
- record GPU state
- record checkpoint presence or absence
- record failure reason and resolution plan

The preserved inotify failure is:

`res/failed_runs/phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654`

## Resource-Adjusted Enriched USD Replay Gate

This diagnostic gate may be rerun with:

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_enriched_usd_bounded_replay_metrics_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_task_smoke_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_multi_fixture_eval_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_conversion_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_full_replay_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.py
```

It validates only that the generated resource-adjusted G1 USD scaffold can be loaded as an IsaacLab articulation and can
render bounded debug-fixture replay step sequences and metrics. The task smoke additionally instantiates the official
`Tracking-Flat-G1-v0` manager stack with the generated USD and debug fixture to verify reset, stepping, observation,
action, reward, and termination surfaces. These gates must not be reported as official `csv_to_npz.py` conversion,
official replay, PPO training, DAgger rollout data, or paper-level BeyondMimic tracking performance. The current
automated gates use explicit process exit after success sentinels to return deterministically; this proves bounded
articulation/task surfaces, not clean Kit shutdown.

After the task smoke passes, the multi-fixture eval should be run directly rather than increasing the dataset in tiny
increments. It executes the three available debug fixtures (`walk`, `run`, `jump`) for all `299` steps each, using one
isolated Kit process per fixture and a stall detector based on log progress instead of a fixed short timeout. The
expected aggregate is `fixture_count=3`, `total_steps=897`, action dimension `29`, policy observation dimension `160`,
critic observation dimension `286`, nine reward terms, four termination terms, `29` robot joints, and `40` robot
bodies.

After the resource-adjusted fixture gates pass, the official-CSV conversion gate can be run directly on
`download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv` frame range 1-180. The expected diagnostic output is
a 299-frame `motion.npz` with joint shape `[299, 29]`, body position shape `[299, 40, 3]`, and a matching JSON contract.
The follow-up full replay gate should execute all 299 frames and record zero joint/root write-read errors. These gates
use official downloaded CSV data and the official interpolation/logging schema, but they still use the generated
resource-adjusted USD instead of official URDF-converter output and must be reported as resource-adjusted only.
The follow-up task eval gate feeds the same official-CSV-derived motion into `Tracking-Flat-G1-v0` for all 299 steps and
must verify action dimension `29`, policy observation dimension `160`, critic observation dimension `286`, nine reward
terms, four termination terms, `29` robot joints, and `40` robot bodies. It is a zero-action diagnostic, not policy
performance.

The bounded train-entry diagnostic may be run only after the CSV task eval passes. It instantiates the official
`Tracking-Flat-G1-v0` environment, `RslRlVecEnvWrapper`, and custom `MotionOnPolicyRunner`, then executes one tiny PPO
learning iteration with `num_envs=1`, `num_steps_per_env=4`, no log directory, and no checkpoint write. It is a wiring
gate only. It must not be reported as formal PPO training, a trained tracking teacher, official replay/evaluation,
closed-loop tracking performance, or a paper-level result. Current logs record PhysX GPU kernel warnings before the
success sentinel, so the next formal step still needs a longer controlled GPU run with GPU telemetry and policy
evaluation.

The official G1 URDF ImportConfig probe may be rerun with
`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_import_config_variant_probe.py`. It is a converter
surface diagnostic only. Passing status means the probe recorded the Isaac Sim 4.5 import-config surface and the
baseline official G1 URDF converter output; it does not mean official replay, `motion.npz`, PPO, or paper tracking
metrics succeeded. If the probe again shows no instanceable setters and an empty baseline USD, return to runnable
resource-adjusted/full virtual task or controlled PPO diagnostics unless a new lower-level official converter path is
identified.

After train-entry smoke has passed, the resource-adjusted PPO training harness may be run with
`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`. It is configured
to select available GPUs from physical GPUs 4-7, `torch.distributed` world size matching the selected GPU count,
`512` environments per rank, official PPO rollout length `24`, `100` iterations by default, checkpoint output, and GPU
telemetry. The script first checks GPU memory/utilization; if no candidate GPU is sufficiently free, it must not start
IsaacLab and should report `ok_with_gpu_resource_unavailable_before_training`.
Only `ok_resource_adjusted_ppo_training_completed` plus retained checkpoints can be interpreted as a completed
resource-adjusted PPO run, and even that remains below official BeyondMimic paper-level training because the asset and
motion pipeline are resource-adjusted.

After a resource-adjusted PPO checkpoint evaluation has completed, the teacher-candidate rollout dataset gate may be
run with `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`.
The accepted local gate requires fixed physical GPUs `[4, 7]`, two rollout shards, retained raw `.npz` files under
ignored `res/runs`, GPU telemetry, and a summary JSON under `res/tracking`. This artifact may be used only as local
resource-adjusted state/action evidence for downstream VAE/state-latent experiments. It must not be described as the
paper's official DAgger dataset, official teacher rollout data, or paper-level closed-loop diffusion evidence.

## Current Completion Boundary

The current evidence set is internally audited, but the full goal is incomplete because official replay, official
teacher/Dagger rollouts, trained official VAE/diffusion checkpoints, Fig. 5/Fig. 6 paper reproduction, TensorRT
deployment, and real Unitree G1 execution remain missing or blocked. A local resource-adjusted teacher-candidate
rollout dataset exists, but it does not replace the missing official DAgger evidence.
