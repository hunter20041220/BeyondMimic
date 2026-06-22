# BeyondMimic Reproduction

This workspace is a structured reproduction attempt for BeyondMimic following `goal.md`.

Raw downloaded materials are kept under `download/` and are treated as read-only. Reproduction code, generated data,
logs, reports, audits, and environment metadata live under `reproduction/`, `logs/`, `res/`, and `envs/`.

## Current Status

- Master audit: see `res/master_audit/reproduction_master_audit.json`.
- Final report: see `res/final_report/reproduction_report.md` and `reproduction/docs/final_reproduction_report.md`.
- Experiment protocol: see `reproduction/docs/experiment_protocol.md`.
- Resolved config manifest: see `res/config/resolved_reproduction_config.json`.
- Artifact hash manifest: see `res/artifact_manifest/artifact_manifest.json`.

The current evidence set is internally audited, but the full paper reproduction is not complete. Latest local audits
record `1564` manifest artifacts, `235` paper-vs-reproduction rows, and `397/397` master-audit artifacts passing.
The paper-vs-reproduction table currently contains `58` exactly comparable, `19` approximately comparable, `145`
qualitative-only, `10` not-publicly-reproducible, and `3` real-robot-required rows. IsaacLab/AppLauncher and G1 task
construction are no longer the main blocker. The current main blocker is tracking teacher quality: the robot-order
FK-repaired PPO teacher runs, but done/termination remains high. The latest endpoint diagnostics identify wrist and
endpoint `ee_body_pos` behavior as dominant termination contributors; a threshold sweep found a lower-done candidate
(`threshold=0.5`, done rate `0.08907621760033445` versus target-refresh baseline `0.22340745192307693`), but this is a
diagnostic evaluator change, not a paper tracking score.

Local official-importer-export G1 USDA tracking, PPO, teacher rollout, VAE, state-latent denoiser, offline guidance,
and task-conditioned guidance rollout bridges have run locally. The full public-motion official `csv_to_npz.py` and
`replay_npz.py` loop bodies also pass on the captured G1 USDA exported by the official Isaac Sim URDF importer (`40/40`
motions, `11960` frames/steps). These results are report/PPT-ready local virtual evidence, not official BeyondMimic
paper-level checkpoints or Fig. 5/Fig. 6 results. Unmodified live official converter-entry success, true official DAgger
rollouts, trained official VAE/diffusion checkpoints, Fig. 5/Fig. 6 paper-level reproduction, TensorRT deployment, and
real Unitree G1 execution remain blocked or missing.

## What Is Complete For Current Scope

- Local inventory, source ledger, paper/source parameter map, unresolved-details docs, and environment docs.
- Released-data figure reproduction for the locally released dataset scope.
- Official tracking/config/controller static audits and ONNX contract checks.
- Level C formula, schema, mask, guidance, overfit, schedule, and package API debug gates.
- Local official-importer-export downstream chain through PPO, teacher rollout, conditional VAE, state-latent denoiser,
  full-split offline guidance, and four task-conditioned closed-loop proxy rollouts with local visualization assets.
  The tracking side now includes a larger 1000-iteration, 4096-env local PPO run/eval on GPUs 4 and 7, but its peak
  memory stayed below 10GB/card and the resulting checkpoint remains local virtual evidence, not an official
  BeyondMimic teacher checkpoint. A 299-frame local policy-vs-reference MP4/keyframe/metrics asset has also been
  captured from that scaled checkpoint for the reading report/PPT, but it is qualitative report media only.
- Robot-order FK-repaired tracking diagnostics that isolate reset/target refresh, reset state/action distribution,
  deterministic reset, `ee_body_pos` termination, endpoint-group behavior, wrist endpoint source attribution, and
  endpoint-threshold candidates. The newest diagnostics point to endpoint/body-target semantics and termination
  calibration as the next repair target rather than blindly rerunning PPO.
- Run-management, failed-run retention, GPU metrics schema, resolved config, and artifact hash manifests.

## What Is Not Complete

- Official G1 `csv_to_npz.py`/`replay_npz.py` conversion and replay from the unmodified official path.
- Paper-level PPO tracking metrics from official assets and paper-scale evaluation.
- Paper-level tracking metrics from newly trained policies.
- True VAE DAgger rollout dataset, trained VAE checkpoint, and closed-loop VAE evaluation.
- Full diffusion Transformer training, checkpoint evaluation, Fig. 5/Fig. 6 results, and deployment benchmarks.
- Paper-level guided diffusion closed-loop metrics remain incomplete: the new task-conditioned videos are local
  virtual proxy rollouts, not official Fig. 5/Fig. 6 evidence.
- Real robot execution or safety validation.

## Current Progress Estimate

Use three separate percentages:

- Course reading report and defense readiness: about `85-90%`.
- Public-resource engineering coverage: about `75-80%`.
- Strict simulation-side paper-level reproduction, excluding real robot: about `40-50%`.

The strict score is lower because the highest-value paper claims still require a stronger tracking teacher, true
DAgger-style rollout data, official-equivalent VAE/diffusion evidence, strict Fig. 5/Fig. 6 closed-loop task metrics,
and TensorRT/asynchronous deployment evidence.

## Key Commands

```bash
python3 reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/resolved_reproduction_config.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/experiment_protocol_audit.py
```

For the full execution protocol and rerun order, read `reproduction/docs/experiment_protocol.md` and
`reproduction/RUNBOOK.md`.

## Safety And Integrity Rules

- Do not write generated files into `download/`.
- Do not fabricate missing metrics, videos, checkpoints, GPU utilization, or paper figures.
- Do not mark a run `SUCCESS` unless it reaches its declared training/evaluation endpoint.
- Preserve failed runs under `res/failed_runs/`.
- Do not start long training while the relevant smoke gates remain blocked.
