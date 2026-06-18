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
Do not start long tracking training until a valid official 29-DoF USD or an explicitly resource-adjusted locked-wrist
contract, `motion.npz`, and replay gate are produced.

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

## Current Completion Boundary

The current evidence set is internally audited, but the full goal is incomplete because live Kit tracking, true teacher
rollouts, DAgger, trained VAE/diffusion checkpoints, Fig. 5/Fig. 6 paper reproduction, TensorRT deployment, and real
Unitree G1 execution remain missing or blocked.
