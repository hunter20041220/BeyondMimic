# BeyondMimic Current Reproduction Baseline

Generated: 2026-06-22 11:05 +08:00.

This document records the current project baseline for updating the next goal, the English reading report, the Chinese reading report, and the defense/project report. It is intentionally claim-bounded: the project is a public-resource partial reproduction plus a local virtual BeyondMimic-like pipeline, not a complete paper-level reproduction.

## Machine-Audited State

Latest local evidence:

| Item | Current value |
|---|---:|
| master audit | ok, 397/397 artifacts passed |
| artifact manifest | ok, 1564 artifacts, missing 0 |
| paper-vs-reproduction rows | 235 |
| exactly comparable rows | 58 |
| approximately comparable rows | 19 |
| qualitative-only rows | 145 |
| not publicly reproducible rows | 10 |
| real-robot-required rows | 3 |
| completion matrix complete | 74 |
| completion matrix partial | 132 |
| completion matrix blocked | 2 |
| completion matrix out of scope | 1 |
| required artifact absence rows | 32 |
| missing required artifacts | 12 |
| goal complete | false |

Recommended progress wording:

| Progress lens | Estimate | Meaning |
|---|---:|---|
| Course reading report / defense readiness | 85-90% | The paper reading, evidence organization, reports, and claim boundary are mostly ready. |
| Public-resource engineering coverage | 75-80% | Most released-data, source-audit, environment, formula, and local virtual pipeline pieces are runnable or audited. |
| Strict non-robot paper-level reproduction | 40-50% | Core closed-loop paper claims still need stronger tracking, true DAgger, official-equivalent VAE/diffusion, strict Fig. 5/Fig. 6 metrics, and TensorRT deployment. |

These three numbers must not be collapsed into one. The project has broad coverage, but strict paper-level reproduction remains incomplete.

## What Has Been Done

The project started from reading the paper and turning the method diagram into reproducible engineering gates:

1. Motion tracking teacher.
2. Teacher rollout / DAgger-style data.
3. Conditional action VAE.
4. State-latent trajectory dataset.
5. State-latent diffusion / denoiser.
6. Test-time guidance for joystick, waypoint, obstacle, transition, inpainting, and composed objectives.
7. Deployment / ONNX / TensorRT / controller path.

The current project has completed or substantially audited:

- local inventory, source ledger, source hashes, environment documentation, and artifact manifest;
- released-data figures, released table values, panel map, paper/source coverage, and formula/code trace;
- IsaacLab, RSL-RL, `whole_body_tracking`, G1 task, reward, termination, observation/action, randomization, ONNX, MuJoCo/ROS contract audits;
- IsaacLab headless AppLauncher and current G1 task construction gates;
- official-loop body audits for public G1 motion conversion and replay on the captured official-importer-export G1 USDA path;
- local PPO training/evaluation, including larger two-GPU 1000-iteration scaled PPO and robot-order FK-repaired checkpoint evaluation;
- teacher rollout collection from local PPO checkpoints;
- conditional action VAE training and action-reconstruction closed-loop checks;
- state-latent dataset construction;
- local denoiser/diffusion training and held-out denoising metrics;
- offline guidance and closed-loop local proxy guidance rollouts;
- unified local task protocol table and report-ready plots/tables/videos;
- English reading report, Chinese reading report, and Chinese project/defense report drafts;
- GitHub version-control workflow with progress Markdown files for traceability.

## Current Best Local Pipeline

The strongest complete local virtual chain is:

```text
captured official-importer-export G1 USDA
-> public 40-motion G1/LAFAN motion bundle
-> local PPO tracking teacher
-> local teacher rollout dataset
-> conditional action VAE
-> state-latent windows
-> denoiser/diffusion training
-> offline guidance
-> IsaacLab task-conditioned local proxy guidance rollouts
-> report assets / videos / tables
```

This is valuable evidence that a BeyondMimic-like pipeline can be reconstructed from public resources. It is not official BeyondMimic reproduction because the teacher, DAgger data, VAE checkpoint, diffusion checkpoint, task metrics, and deployment stack are local/resource-adjusted.

## Current Tracking Bottleneck

The main blocker is no longer package import. It is tracking teacher quality.

Current robot-order FK-repaired PPO evidence:

- single checkpoint evaluation: 2048 envs x 299 steps = 612352 virtual env steps;
- reward mean about 0.02073384587805606;
- done count 109170;
- body-position error mean about 0.36114187777839774;
- three-seed evaluation total: 1837056 virtual env steps;
- three-seed mean done rate: 0.1785340240036232.

Recent diagnostics narrowed the issue:

- stale reset targets create a large step-0 body-position spike;
- no-advance target refresh reduces the step-0 spike but worsens post-step0 done rate;
- reset action/history and deterministic reset variants lower some velocity transients but worsen done rate;
- endpoint-group ablation identifies wrists/endpoints as a dominant `ee_body_pos` termination source;
- full-size wrist endpoint source diagnostics identify motion/phase/body-specific endpoint exceedance;
- endpoint-threshold sweep keeps official endpoint bodies active and finds a candidate threshold 0.5 with done rate 0.08907621760033445 versus target-refresh baseline 0.22340745192307693.

The threshold sweep is a candidate for the next training/evaluation path, but it changes the evaluator and must not be reported as the paper's tracking metric. The next mainline technical action should either repair endpoint/body-target semantics or run a clearly labeled threshold-candidate full PPO/eval before regenerating downstream teacher/VAE/diffusion artifacts.

## Paper Modules And Local Source Implementation

| Paper component | Local code / evidence | Boundary |
|---|---|---|
| Motion tracking teacher | IsaacLab/RSL-RL task gates, PPO wrappers, FK-repaired motion data, robot-order diagnostics | local virtual teacher; weak termination quality |
| Teacher rollout / DAgger | rollout shard schemas and local teacher rollout datasets | local teacher data; not official DAgger logs |
| Conditional VAE | `reproduction/src/beyondmimic_reimpl/vae`, local teacher-rollout VAE training | paper-faithful/local; not official checkpoint |
| State-latent dataset | `reproduction/src/beyondmimic_reimpl/trajectory`, dataset builders | derived from local teacher; not official dataset |
| Diffusion / denoiser | `reproduction/src/beyondmimic_reimpl/diffusion`, Transformer probes, local denoiser training | local denoiser; not official diffusion checkpoint |
| Guidance | `reproduction/src/beyondmimic_reimpl/guidance`, offline and closed-loop proxy rollouts | proxy costs; not strict Fig. 5/Fig. 6 |
| Evaluation | `reproduction/src/beyondmimic_reimpl/evaluation`, comparison and audit scripts | strong audit value; not a substitute for paper metrics |
| Deployment | ONNXRuntime async proxy and controller contract audits | no TensorRT engine, no Mini-PC latency, no robot |

## Still Missing Except Real Robot

Even ignoring real robot deployment, these paper-level pieces remain incomplete:

- unmodified official G1 `csv_to_npz.py` / `replay_npz.py` live-entry success;
- high-quality paper-level PPO tracking teacher;
- paper-level tracking metrics with stronger success/fall/termination behavior and multi-seed protocol;
- true DAgger rollout dataset;
- VAE trained/evaluated from true teacher rollout distribution;
- autonomous or paper-equivalent VAE closed-loop control evidence;
- full diffusion Transformer training/evaluation from a credible teacher/state-latent dataset;
- strict Fig. 5 joystick/waypoint/transition/latent visualization reproduction;
- strict Fig. 6 inpainting/obstacle/composed-task reproduction;
- TensorRT engine / asynchronous deployment benchmark;
- MuJoCo/ROS sim-to-sim execution logs;
- a completed full paper-facing training run directory with config, logs, metrics, checkpoint, figures, videos, and success status.

## Recommended Updated Goal

The next goal should be:

> Maintain an auditable public-resource BeyondMimic partial reproduction and course-report package, while pushing the non-robot simulation-side gates toward paper-level evidence. The immediate technical priority is tracking data quality and teacher quality; once a gate improves done/termination and state/action consistency, run full GPU 4/7 PPO, multi-seed eval, policy video, and then regenerate teacher rollout -> VAE -> state-latent -> denoiser -> guidance results.

The project must continue to state:

```text
This project does not fully reproduce BeyondMimic at paper level.
```

The correct positive claim is:

```text
This project reproduces and audits the publicly reproducible parts, implements a paper-faithful local virtual BeyondMimic-like pipeline, and documents the non-public or hardware-dependent boundaries.
```
