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

The current evidence set is internally audited, but the full paper reproduction is not complete. The current
IsaacLab/AppLauncher headless startup gate reaches a success sentinel, and local official-importer-export G1 USDA
tracking, PPO, teacher rollout, VAE, state-latent denoiser, offline guidance, and task-conditioned guidance rollout
bridges have run locally. The full public-motion official `csv_to_npz.py` and `replay_npz.py` loop bodies now also pass
on the captured G1 USDA exported by the official Isaac Sim URDF importer (`40/40` motions, `11960` frames/steps), which
removes the generated enriched-USD scaffold from that specific full-loop test. A representative local kinematic
reference replay video/keyframe asset has also been generated from that full official-importer-export conversion audit
for report/PPT use. Unmodified live official converter-entry success, true official DAgger rollouts, trained official
VAE/diffusion checkpoints, Fig. 5/Fig. 6 paper-level reproduction, TensorRT deployment, and real Unitree G1 execution
remain blocked or missing.

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
