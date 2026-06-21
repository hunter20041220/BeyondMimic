# BeyondMimic Current Reproduction Summary

Generated: 2026-06-22 03:23 Asia/Shanghai

This document summarizes the current state of the local BeyondMimic reproduction project and is intended as a baseline for updating the next project goal. It uses the latest machine-readable audit artifacts in `/mnt/infini-data/test/BeyondMimic/res/` as the primary source of truth.

## Executive Position

The project is now an audit-heavy, public-resource partial reproduction and local virtual BeyondMimic-like pipeline. It is not an empty repository and not only a reading report. It contains restored local environments, source and artifact audits, released-data reproduction, official IsaacLab/whole-body-tracking source integration, local G1 motion preprocessing/replay, PPO tracking experiments, teacher rollout collection, conditional action VAE training, state-latent denoiser training, guidance proxy evaluation, visualization assets, and English/Chinese report drafts.

The project still cannot claim full paper-level reproduction of BeyondMimic. Excluding real robot deployment, the main remaining gap is not basic environment setup anymore. The main gap is a stable paper-level simulation-side control chain: high-quality tracking teacher, true DAgger-style rollouts, paper-equivalent VAE/diffusion checkpoints, closed-loop Fig. 5/Fig. 6 task metrics and videos, and TensorRT/asynchronous deployment evidence.

Recommended progress estimates:

- Course reading report / project defense readiness: 85-90%.
- Auditable public-resource engineering coverage: 75-80%.
- Strict simulation-side paper-level reproduction excluding real robot: 40-50%.

These are engineering estimates, not paper claims. They deliberately separate "good evidence for a course report" from "paper-level control reproduction."

## Latest Audit Baseline

Latest JSON/audit state read on this machine:

- Master audit: `ok`, `370/370` artifacts passed.
- Artifact manifest: `ok`, `1485` artifacts, all manifest artifacts exist.
- Paper-vs-reproduction table: `ok`, `227` rows.
- Completion matrix: `207` effective rows, `74 complete`, `130 partial`, `2 blocked`, `1 out_of_scope`.
- Required artifact absence audit: `32` rows, including `12 missing_required_artifact`, `18 present_but_not_required_artifact`, and `2 debug_only_not_required_artifact`.
- Progress report audit: `ok`, `202` audited progress rows.
- `goal_complete=false`.

Paper comparison classes:

- `exactly_comparable`: 58 rows.
- `approximately_comparable`: 19 rows.
- `qualitative_only`: 137 rows.
- `not_publicly_reproducible`: 10 rows.
- `requires_real_robot`: 3 rows.

Interpretation:

- Exact rows mostly cover released-data figures/tables, public paper-source values, source contracts, schema checks, and directly comparable public artifacts.
- Approximate rows are useful engineering evidence but do not match hidden paper protocols or checkpoints.
- Qualitative rows dominate because many local experiments are source audits, diagnostics, proxy rollouts, or local reimplementations.
- Missing-required-artifact rows mark the true paper-level blockers, not bookkeeping noise.

## Environment Status

Project root:

```text
/mnt/infini-data/test/BeyondMimic
```

Important directories:

```text
download/       read-only original material
other/          preserved old-server reproduction snapshot
reproduction/   scripts, source, docs, tests, third-party work copies
res/            small results, audits, report assets, summaries
logs/           runtime logs
envs/           project-local conda/venv environments
cache/          project-local caches
tmp/            project-local temporary files
```

Environment layers:

- `bm_analysis`: works for audits, pandas/matplotlib plots, ONNX/ONNXRuntime checks, report generation, artifact manifests, and master audit refresh.
- `bm_diffusion`: works for PyTorch CUDA VAE/diffusion/guidance experiments. Two-GPU runs have used GPUs 4 and 7 in recent experiments.
- `bm_tracking`: works for Isaac Sim/IsaacLab/RSL-RL/official `whole_body_tracking` imports, AppLauncher headless startup, G1 task construction, local PPO training/evaluation, and live tracking diagnostics.

Current IsaacLab status:

- `isaaclab_import_ok=true`.
- `isaacsim_import_ok=true`.
- `isaaclab_live_headless_gate_ok=true`.
- Current headless AppLauncher gate status: `ok`.
- Current G1 task construction gate is available in the tracking evidence set.

Known environment limits:

- `nvcc` shell command is not available, but PyTorch CUDA works.
- ONNXRuntime evidence is CPU/Azure-provider style evidence; TensorRT/CUDA-provider paper latency has not been reproduced.
- Some diffusion/VAE runs use two visible GPUs but do not exceed 10GB/card because the model/harness is smaller than a formal large PPO or deployment workload.
- Storage is tight on the shared filesystem, so large checkpoints, videos, datasets, and raw rollout shards must remain ignored/local while small JSON/CSV/Markdown summaries are tracked.

## Codebase And Script Implementation

Current repository implementation surface:

- `reproduction/scripts/`: 371 Python scripts.
- Tracking-related scripts: about 123.
- Level C VAE/diffusion/guidance scripts: about 115.
- Audit/report/verification/source scripts: about 190.
- Visualization/report-asset scripts: about 51.

Reusable reimplementation modules exist under:

```text
reproduction/src/beyondmimic_reimpl/
```

Implemented module areas:

- `dagger/schema.py`: DAgger-style data schema and audit scaffolding.
- `diffusion/schedules.py`: diffusion schedule utilities.
- `evaluation/metrics.py`: success/failure/tracking metric helpers.
- `geometry/rotations.py`: quaternion/rotation helpers.
- `guidance/costs.py`: task/guidance objective costs.
- `trajectory/dataset.py`: trajectory/window dataset utilities.
- `vae/latent.py`: latent/action VAE helpers.
- `sampling.py`, `state.py`, `validation.py`: shared local pipeline utilities.

Boundary:

- These modules are local paper-faithful or paper-inspired reimplementations, not official BeyondMimic VAE/diffusion source.
- They are useful for demonstrating understanding and for public-resource experiments, but they cannot be described as official checkpoints or official paper-level reproduction.

## What Has Been Reproduced Or Audited

### 1. Paper, Source, And Audit Infrastructure

Completed:

- Root migration to `/mnt/infini-data/test/BeyondMimic`.
- Source inventory and source ledger.
- Paper/source coverage map.
- Paper panel map.
- Formula-to-code trace.
- Paper table value audit.
- Required artifact absence audit.
- Artifact manifest with hashed small artifacts.
- Completion matrix and progress report audits.
- Verification command script/syntax/coverage audits.
- Master audit with zero failing artifacts.
- GitHub-friendly ignore rules and lightweight versioning workflow.
- English reading report, Chinese reading report, and Chinese project report drafts.

Effect:

- This part is strong enough to support the course reading-report requirement.
- It clearly separates official evidence, local reproduction, proxy diagnostics, and unavailable paper artifacts.

### 2. Released-Data Reproduction

Completed:

- Released-data figure/table reproduction and summaries.
- Public paper table value audit.
- Source/panel references for released claims.
- Many exactly comparable rows in `paper_vs_reproduction`.

Boundary:

- Released-data reproduction is not retraining.
- It cannot be reported as a closed-loop paper-level robot result.

### 3. Official Tracking Code And Public G1 Assets

Completed or strongly audited:

- Official `whole_body_tracking` work copy integrated under the reproduction workspace.
- IsaacLab/Isaac Sim/RSL-RL imports restored.
- AppLauncher headless gate passes.
- Observation/action schema audited: 29 action dimensions, 160 policy observation dimensions, 286 critic observation dimensions.
- Reward, termination, randomization, motion preprocessing, ONNX, MuJoCo/ROS launch, and deployment controller contracts audited.
- Official `csv_to_npz.py` loop body exercised under local routing/patches.
- Official `replay_npz.py` loop body exercised under local routing/patches.
- Public 40-motion G1 bundle prepared for local tracking experiments.
- Official-importer-export G1 USD path used for the strongest local virtual experiments.
- FK-repaired and robot-order FK-repaired motion bundles built to address `body_pos_w` and body-order issues.

Boundary:

- Some official entrypoints require runtime output/asset routing or patches.
- The official worktree is preserved rather than directly modified.
- This is much stronger than static inspection, but still not the hidden official training/evaluation setup from the paper.

### 4. Tracking PPO And Evaluation

Strongest current local tracking chain:

- Robot-order FK-repaired full public motion bundle.
- Official-importer-export G1 USD.
- Local PPO training to iteration 999.
- Single-seed checkpoint evaluation.
- Three-seed checkpoint evaluation.
- Policy-vs-reference rollout video/report assets.
- Reset/warmup/target-refresh/termination diagnostics.

Single-seed robot-order checkpoint evaluation:

- Scope: `2048 envs x 299 steps = 612352 env steps`.
- Status: `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_completed`.
- Done count: `109170`.
- Done rate: `0.1782798129180602`.
- Timeout count: `0`.
- Mean reward: `0.02073384587805606`.
- Mean anchor position error: `0.07790673197711191`.
- Mean body position error: `0.36114187777839774`.
- Mean joint position error: `1.5732512252785291`.

Three-seed robot-order checkpoint evaluation:

- Scope: `3 seeds x 2048 envs x 299 steps = 1837056 env steps`.
- Mean done rate: `0.1785340240036232`.
- Done-rate std: `0.001763911381666986`.
- Mean reward: `0.020480790998840676`.
- Mean body position error: `0.3597400628005382`.
- Mean joint position error: `1.5772204704773731`.
- Mean joint velocity error: `16.140327120837295`.

Important reset/termination diagnostic:

- No-advance target refresh reduced the stale reset endpoint/body target issue.
- Endpoint done rate before refresh: `1.0`; after refresh: `0.2734375`.
- Endpoint z error mean before refresh: `0.5298784375190735`; after refresh: `0.104344442486763`.
- But target refresh also exposed or induced an initial velocity/action transient.
- Step-0 body error improved by about `-43.03 m`.
- Step-0 joint velocity error worsened by about `+17.83`.
- First-five-step action mean increased by about `+0.0718`.
- Post-step0 done rate worsened by about `+0.0477`.

Additional reset state/action consistency live probe:

- New 256-env live IsaacLab probe: `robot_order_fk_reset_state_action_consistency_live_probe`.
- Inputs: official-importer-export G1 USDA, robot-order FK-repaired full public motion bundle, iteration-999 local PPO checkpoint.
- Variants tested: baseline, target refresh, target refresh + action reset, target refresh + action-offset alignment, target refresh + motion-state rewrite, and combined rewrite/action variants.
- Each variant was tested under zero action and checkpoint-policy first-step action.
- Target refresh alone: policy done rate `0.28125`, post-step joint velocity error `14.182840347290039`.
- Target refresh + action reset: policy done rate `0.4765625`, post-step joint velocity error `10.899185180664062`.
- Target refresh + action-offset alignment: policy done rate `0.49609375`, post-step joint velocity error `10.263128280639648`.
- Target refresh + motion-state rewrite + action-offset alignment: policy done rate `0.73828125`, post-step joint velocity error `8.305423736572266`.
- Conclusion: action reset/offset and motion-state rewrite reduce joint velocity error, but they worsen done rate. No variant improves both done rate and joint velocity.
- Recommended full-eval variant: none.

Interpretation update:

- The next tracking fix is no longer just stale command target refresh.
- A simple action-history reset, action-offset alignment, or direct motion-state rewrite is not sufficient.
- The likely remaining issue is a coupled reset distribution problem involving target refresh, robot state, initial velocities, action offset/last-action observations, contacts, and `ee_body_pos` termination.
- Because the live probe found no safe candidate, a new full checkpoint eval or PPO rerun should wait until the reset/state/action semantics are repaired more carefully.

Interpretation:

- The local tracking system is real and runnable, not smoke-only.
- However, the current teacher is still weak. It cannot be used as final paper-level DAgger/VAE/diffusion teacher evidence.
- The next tracking target should be reset state/action consistency, endpoint termination, joint velocity transient, and then stronger PPO.

### 5. Teacher Rollouts, Conditional VAE, State-Latent Dataset, Diffusion, Guidance

Completed local downstream chain:

- Teacher rollout dataset from local virtual PPO teacher.
- Conditional action VAE training.
- State/action-latent dataset construction.
- State-latent denoiser training.
- Offline guidance evaluation.
- Task-conditioned latent guidance multi-seed evaluation.
- Local Fig.5/Fig.6-adjacent proxy protocol tables and videos.

Official-importer-export full-bundle conditional VAE:

- Dataset samples: `306176`.
- Train/validation/test split: `244940/30618/30618`.
- Test action MSE: `5.362209958548192e-05`.
- Test action absolute error mean: `0.005292208399623632`.
- Training epochs: `40`.
- Two visible CUDA devices used.

State-latent dataset:

- Samples: `306176`.
- Windows: `285696`.
- Split counts: train `228557`, validation `28570`, test `28569`.
- Weighted posterior reconstruction MSE: `5.118260560266208e-05`.

State-latent denoiser:

- Training epochs: `30`.
- Best validation epoch: `30`.
- Test noisy token MSE: `0.06729835644364357`.
- Test predicted token MSE: `0.013647833040782384`.
- Test denoising improvement ratio: `0.7972040661615378`.

Offline guidance:

- 4 tasks evaluated.
- 48 aggregate rows.
- 57139 selected windows.
- All best costs improve.
- All best guidance gradients nonzero.

Task-conditioned latent guidance multi-seed proxy:

- 20 rollout rows.
- 5 seed groups.
- 4 task groups.
- 23920 rollout-variant steps.
- 20 MP4 paths recorded.

Fig.5/Fig.6-adjacent local task protocol proxy:

- 20 rows.
- 5 seed groups.
- 4 tasks.
- Overall recorded 299-step completion rate: `1.0`.
- Overall endpoint proxy pass rate: `1.0`.
- Overall target-body mean proxy pass rate: `1.0`.
- Overall local task protocol proxy pass rate: `0.65`.
- Reward-improved-vs-denoised rate: `0.45`.
- Tracking-error-not-worse-vs-denoised rate: `0.5`.
- Mean final root XY error: `0.005920683296880743 m`.
- Paper-level reproduced Fig.5/Fig.6 panel count: `0`.

Scaled-PPO variant local task proxy:

- Overall local task protocol proxy pass rate: `0.8`.
- Reward-improved-vs-denoised rate: `0.6`.
- Paper-level reproduced Fig.5/Fig.6 panel count: `0`.

Interpretation:

- This chain is valuable for explaining the BeyondMimic method and for demonstrating a public-resource local implementation of the idea.
- It is still qualitative/proxy evidence because the upstream teacher is weak and the official DAgger/VAE/diffusion checkpoints are missing.
- The proxy rollouts must not be reported as paper Fig.5/Fig.6 success rates.

## What Remains Excluding Real Robot Deployment

### Highest Priority Simulation-Side Gaps

1. High-quality tracking teacher.
   - Current PPO teacher runs but has high done/termination and large joint velocity error.
   - Reset-state/action consistency and target refresh behavior need fixing before a stronger full PPO run.

2. Paper-level closed-loop tracking metrics.
   - Paper-style velocity tracking errors and success/fall/collision metrics are not reproduced at paper level.
   - Current local done/termination proxies are useful but not the paper's exact protocol.

3. True DAgger rollout logs.
   - No official DAgger dataset with teacher query, aggregation, student/teacher comparison, and stability logs exists locally.
   - Local teacher rollout datasets exist but are not true official DAgger evidence.

4. Paper-equivalent VAE checkpoint.
   - Local conditional action VAE is trained and works on local rollouts.
   - It is not the official BeyondMimic VAE checkpoint and not trained on the hidden/true DAgger data.

5. Paper-equivalent state-latent diffusion Transformer checkpoint.
   - Local denoiser training works and improves noisy-token MSE.
   - It is not the official paper checkpoint and not validated in the strict paper closed-loop task protocol.

6. Closed-loop VAE/diffusion/guidance rollout evaluation.
   - Local proxy guidance and task-conditioned rollouts exist.
   - Strict Fig.5/Fig.6 success/fall/collision/tracking metrics and paper-level videos are still missing.

7. TensorRT/asynchronous deployment.
   - ONNX/contract/latency proxy evidence exists.
   - No trained TensorRT engine, Mini-PC asynchronous deployment log, or paper-equivalent deployment latency exists.

8. MuJoCo/ROS sim-to-sim execution.
   - Launch/controller contracts are audited.
   - No actual ROS 2/MuJoCo launch execution logs/bags are present on this host.

### Real Robot Only

These remain out of scope unless hardware is explicitly made available:

- Real Unitree G1 deployment.
- Real-robot Fig.2 motion gallery evidence.
- Real-world Fig.6 obstacle or transition execution evidence.
- Real robot controller logs, videos, safety checks, and hardware latency evidence.

## Required Artifact Absence Items

The latest absence audit records these missing required artifacts:

- Trained motion-tracking policy checkpoint/export for BeyondMimic G1 tracking.
- BeyondMimic motion policy ONNX satisfying the official motion-tracking-controller contract.
- Trained conditional VAE checkpoint from true DAgger teacher/student rollouts.
- True DAgger rollout logs.
- Trained state-latent diffusion Transformer checkpoint.
- TensorRT engine or plan for the trained diffusion policy.
- Closed-loop tracking evaluation logs/metrics from live IsaacLab/Kit rollout.
- Figure 5 joystick/waypoint/latent visualization rollout data, metrics, and visuals.
- Figure 6 inpainting/obstacle rollout data, metrics, and visuals.
- Success and failure videos for all Phase 8 guidance tasks.
- MuJoCo/ROS sim-to-sim execution logs, bags, or deployment evaluation.
- A fully completed paper-facing training run directory with config, logs, metrics, checkpoint, figures, videos, and SUCCESS status.

Some local artifacts with similar names exist, but the audit intentionally classifies them as present-but-not-required or debug-only when they are local proxy outputs rather than paper-level required artifacts.

## Recommended New Goal

The old goal should be updated because "restore IsaacLab import/headless startup" is no longer the central blocker. A more accurate next goal is:

> Build a defensible non-robot BeyondMimic reproduction package: maintain the English/Chinese reading reports, repair the local tracking teacher quality enough for credible simulation-side evidence, rerun downstream VAE/diffusion/guidance from the improved teacher, and keep all results honestly classified as exact, approximate, qualitative proxy, not publicly reproducible, or real-robot-only.

Suggested next milestones:

1. Reset consistency fix and live probe.
   - Fix target refresh/state/action/last-action consistency.
   - Reduce endpoint z error without increasing joint velocity/action transients.
   - Preserve a JSON/CSV diagnostic and failed-run evidence if it fails.

2. Stronger PPO tracking run.
   - Use GPUs 4 and 7.
   - Run full, not smoke-only, once the gate looks promising.
   - Record GPU metrics, config, seed, reward/error curves, termination breakdown, and evaluation videos.

3. Rebuild downstream chain from improved teacher.
   - Teacher rollout dataset.
   - Conditional VAE.
   - State-latent dataset.
   - Denoiser/diffusion training.
   - Closed-loop/proxy guidance rollout.
   - Unified task protocol table.

4. Report consolidation.
   - Keep `english_reading_report.md` as the course-facing deliverable.
   - Use Chinese reports for defense/explanation.
   - Include the current boundaries explicitly: no full paper-level reproduction, no official hidden checkpoints, no real robot.

## Safe Claiming Language

Safe:

> This project reproduces and audits the publicly reproducible parts of BeyondMimic, rebuilds a local virtual BeyondMimic-like pipeline, and identifies which paper-level claims require missing official artifacts or real robot hardware.

Unsafe:

> This project fully reproduces BeyondMimic.

Unsafe:

> The local guidance proxy results reproduce Fig.5/Fig.6 paper success rates.

Unsafe:

> The local VAE/diffusion checkpoints are official BeyondMimic checkpoints.

Current conclusion:

> Current project status: strong public-resource partial reproduction and reading-report evidence; incomplete paper-level non-robot reproduction; real-robot deployment not attempted.
