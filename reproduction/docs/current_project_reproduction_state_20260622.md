# BeyondMimic Current Project Reproduction State

Generated: 2026-06-22 02:21 Asia/Shanghai

This document is a current-state baseline for updating the project goal. It records what has actually been done in the workspace, what the current effects are, and what still remains for a non-real-robot BeyondMimic reproduction. It supersedes older goal text that treats IsaacLab import or basic PPO smoke tests as the main blocker.

## Executive Summary

The project is now a large auditable partial reproduction of **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion**. It is not an empty code dump and not just a report draft. The workspace has restored project-local analysis, diffusion, and tracking environments; recovered IsaacLab/Isaac Sim headless startup; exercised the official `whole_body_tracking` source and public G1 motion assets; generated a large set of released-data, source-contract, tracking, PPO, VAE, diffusion, guidance, visualization, and report artifacts; and maintains machine-readable audits plus Git-tracked progress reports.

The project still does **not** fully reproduce BeyondMimic at paper level. The best current robot-control result is a local virtual official-importer-export G1 pipeline with a robot-order FK-repaired 40-motion public bundle, a 1000-iteration PPO run, single-seed and three-seed checkpoint evaluations, a policy-vs-reference video, reset/termination diagnostics, and local VAE/diffusion/guidance proxy chains. These are useful engineering and scientific evidence, but they are not the official BeyondMimic tracking teacher, not official DAgger, not official VAE/diffusion checkpoints, not paper Fig. 5/Fig. 6 closed-loop metrics, not TensorRT deployment, and not real-robot validation.

Recommended progress estimates:

- Course reading report and defense readiness: **85-90%**. The paper understanding, method breakdown, evidence organization, figures/tables, failure boundaries, and reflection material are already strong.
- Auditable public-resource engineering coverage: **75-80%**. Most public data, source contracts, local environments, official-loop preprocessing/replay, and local virtual method components have runnable or audited evidence.
- Strict simulation-side paper-level reproduction excluding real robot: **40-50%**. The highest-weight closed-loop claims still need a stronger tracking teacher, true teacher/DAgger rollouts, paper-equivalent VAE/diffusion evaluation, Fig. 5/Fig. 6 task protocol metrics/videos, and TensorRT/asynchronous deployment evidence.

These percentages are engineering estimates, not official script outputs. They intentionally separate "good course report evidence" from "paper-level control reproduction".

## Latest Machine-Audit Baseline

Latest refreshed baseline before the no-advance reset-target refresh integration:

```text
master_audit: ok, 364/364 artifacts passed
artifact_manifest: ok, 1465 artifacts, 0 missing
paper_vs_reproduction: ok, 225 rows
completion matrix audit: ok, 205 rows
required artifact absence audit: ok, 32 rows
progress report audit: ok, 161 per-round Markdown progress files audited
goal_complete: false
```

An additional no-advance reset-target refresh diagnostic has now been run and is being integrated into the report/audit chain. It is useful tracking-quality evidence, but it does not make the local PPO checkpoint a paper-level teacher.

`paper_vs_reproduction` comparison types:

```text
exactly_comparable: 58
approximately_comparable: 19
qualitative_only: 135
not_publicly_reproducible: 10
requires_real_robot: 3
```

Completion matrix after adding the seed-matched reset-command warmup phase diagnostic row:

```text
complete: 74
partial: 128
blocked: 2
out_of_scope: 1
```

Required artifact absence audit:

```text
debug_only_not_required_artifact: 2
missing_required_artifact: 12
present_but_not_required_artifact: 18
```

Interpretation:

- Exact rows mainly cover released-data tables/figures, paper-source values, static contracts, and directly comparable public artifacts.
- Approximate rows are useful but not paper-equivalent because the local setup lacks official hidden datasets/checkpoints/protocols.
- Qualitative rows are the largest group because many local experiments are proxy or diagnostic evidence.
- The missing-required-artifact rows are not administrative noise. They are the core blockers: official teacher/VAE/diffusion checkpoints, true DAgger logs, paper rollout videos/metrics, deployment artifacts, and robot logs.

## Environment Status

Project root:

```text
/mnt/infini-data/test/BeyondMimic
```

Important directories:

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

`download/` is treated as read-only raw source material. `other/` remains the preserved old-server snapshot. New code, small reports, JSON/CSV/Markdown audits, and generated summaries live under `reproduction/` and `res/`. Large checkpoints, videos, raw rollout shards, datasets, environments, caches, and most logs remain local and are not pushed to GitHub.

Environment layers:

- `bm_analysis`: usable for audits, plots, pandas/matplotlib, ONNX/ONNXRuntime CPU checks, report generation, and manifest/master-audit refresh.
- `bm_diffusion`: usable for PyTorch CUDA local VAE, state-latent diffusion, offline guidance, ONNX export, and proxy evaluation.
- `bm_tracking`: usable for Isaac Sim/IsaacLab/RSL-RL/official `whole_body_tracking` imports, headless AppLauncher, G1 task construction, official-loop preprocessing/replay gates, local PPO training/evaluation, and live diagnostic probes.

Current IsaacLab status:

```text
headless AppLauncher gate: ok
selected physical GPU for gate: 4
G1 task construction gate: ok
```

The active blocker is no longer "cannot import IsaacLab". The active paper-facing blocker is **tracking quality**: the local teacher can run, but it terminates too often and is not yet a reliable teacher for DAgger/VAE/diffusion.

Known environment limits:

- `nvcc` shell command is unavailable, but PyTorch CUDA works.
- ONNXRuntime evidence is CPU/Azure provider based; no TensorRT/CUDA provider paper-latency result is available.
- Current formal tracking experiments have used GPUs 4 and 7. Some runs did not exceed 10GB/card because the local harness/model size did not require it; this is recorded rather than inflated.
- Storage must be managed carefully. Large raw results stay local and should be summarized by small artifacts.

## What Has Been Done

### Paper, Source, And Reporting Infrastructure

Completed or strong:

- Goal/context migration from old root to `/mnt/infini-data/test/BeyondMimic`.
- Local inventory and source ledger.
- Download/source integrity audits.
- Paper source coverage, panel map, formula/code trace, and table value audit.
- Artifact manifest, completion matrix, required-artifact absence audit, failed-run retention, verification-command audit, and master audit.
- English reading report, Chinese reading report, project report, and generated final reproduction report.
- GitHub-friendly `.gitignore`, progress Markdown workflow, and lightweight code/doc/result versioning.

This part supports a strong course report because it shows traceable reading, source mapping, and claim boundaries.

### Released-Data Reproduction

Completed or strong:

- Released-data figure/table reproduction and summaries.
- Paper table value audit.
- Paper panel map and source-code coverage.
- Comparison table entries for exactly/approximately comparable public values.

Boundary:

- Released-data reproduction is not retraining.
- It cannot be reported as closed-loop paper-level robot behavior.

### Official Tracking Code And Assets

Completed or strong:

- Official `whole_body_tracking` source prepared under the reproduction workspace.
- IsaacLab and official tracking import gates restored.
- Observation/action schema audited: 29 action dimensions, 160 policy observation dimensions, 286 critic observation dimensions.
- Reward/termination/randomization contracts audited.
- Motion preprocessing contract audited.
- ONNX and MuJoCo/ROS launch contracts audited.
- G1 URDF/source-equivalence and physical asset contracts audited.
- Captured official-importer-export G1 USDA path built and used in local virtual evaluation.

Official-loop motion evidence:

- Official `csv_to_npz.py` loop body exercised on public G1 motions under local output/fake-WandB routing.
- Official `replay_npz.py` loop body exercised on the 40-motion public bundle.
- Full public bundle contains 40 motions and 11960 frames/steps.
- Full-dataset replay/task diagnostics produce report-ready CSV/PNG/video assets.

Boundary:

- The unmodified official converter/replay entry is still not a clean paper-level success.
- Local wrappers and captured/importer-export assets are used to keep the public path runnable and auditable.

### Tracking Data Repair And PPO

The tracking side evolved through several important fixes:

1. Resource-adjusted/enriched scaffold path: useful for initial task and PPO gates, but not the final asset path.
2. Official-importer-export G1 path: stronger asset path from Isaac Sim's official URDF importer.
3. FK-repaired motion bundle: fixed a degenerate `body_pos_w` issue.
4. Robot-order FK-repaired bundle: fixed a body-order mismatch where target bodies were written in URDF order but read by IsaacLab runtime articulation order.

Current strongest local tracking baseline:

```text
training:
res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/

single-seed eval:
res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/

three-seed eval:
res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/

policy video:
res/visualization/official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout/
```

Single-seed robot-order PPO checkpoint eval:

```text
status: ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_completed
checkpoint iteration: 999
eval scope: 2048 envs x 299 steps = 612352 env steps
done_count_total: 109170
done_rate: 0.1782798129180602
timeout_count_total: 0
reward_mean: 0.02073384587805606
anchor_position_error_mean: 0.07790673197711191
body_position_error_mean: 0.36114187777839774
joint_position_error_mean: 1.5732512252785291
```

Three-seed robot-order PPO checkpoint eval:

```text
status: ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval_completed
scope: 3 seeds x 2048 envs x 299 steps = 1837056 env steps
done_rate_mean/std: 0.1785340240036232 / 0.001763911381666986
reward_mean/std: 0.020480790998840676 / 0.0004249192220635496
body_position_error_mean: 0.3597400628005382
joint_position_error_mean: 1.5772204704773731
```

Interpretation:

- This is real local virtual PPO training/evaluation, not just smoke.
- It is much better than earlier broken data paths.
- It is still not a paper-level teacher because termination remains high and joint/velocity tracking is weak.

### Latest Reset-Command Warmup And Phase Diagnostic

The latest diagnostic reran the iteration-999 robot-order FK-repaired checkpoint with a reset-command warmup:

```text
path:
res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup/

status:
ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed

scope:
2048 envs x 299 steps = 612352 env steps
```

Warmup effects:

```text
old step0 done_count: 2048
warmup step0 done_count: 568
step0 done_count delta: -1480

old step0 body_position_error: 43.294166564941406 m
warmup step0 body_position_error: 0.2640186548233032 m

old total done_rate: 0.1782798129180602
warmup total done_rate: 0.22864463576505017
done_rate_delta: +0.05036482284698998

warmup reward_mean: 0.02111538346968965
warmup body_position_error_mean: 0.2436978654518574
warmup joint_position_error_mean: 1.5415688719239122
```

Interpretation:

- Reset-command warmup clearly fixes a large step-0 bootstrap artifact.
- It does **not** make the checkpoint a usable teacher; total done rate becomes worse.
- Because the first warmup eval used a different seed from the non-warmup baseline, a second full evaluation matched the non-warmup seed (`20260721`) to remove the adaptive-sampling seed confound.

Seed-matched warmup diagnostic:

```text
path:
res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched/

status:
ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched_completed

scope:
2048 envs x 299 steps = 612352 env steps

seed:
20260721
```

Seed-matched phase-analysis effects:

```text
step0 done_count delta: -1452
step0 body_position_error delta: -43.02924793958664 m
same-seed total done_rate delta: +0.04325779943561875
same-seed post-step0 done_rate delta: +0.04578210203439598
same-seed ee_body_pos termination fraction delta: +0.04554896530100333
same-seed sampling top-bin post-step0 delta: 0.0
```

Updated interpretation:

- Reset-command warmup removes the stale reset target at step 0.
- The same-seed run confirms that the worse total done rate is not merely a different random/adaptive-sampling seed.
- The post-step0 regression is concentrated in `ee_body_pos` termination while adaptive-sampling top-bin behavior stays unchanged.
- The next tracking work should target command/observation phase consistency and reset-target refresh, not downstream DAgger/VAE/diffusion collection from this checkpoint.

### No-Advance Reset-Target Refresh Diagnostic

The follow-up diagnostic tested the recommended reset-target refresh directly, without calling `command_manager.compute()` or advancing `MotionCommand.time_steps`.

Live probe:

```text
path:
res/tracking/robot_order_fk_reset_target_refresh_no_advance_live_probe/

status:
ok_robot_order_fk_reset_target_refresh_no_advance_live_probe

endpoint-z done rate:
1.0 -> 0.2734375

endpoint-z error mean:
0.5298784375190735 m -> 0.104344442486763 m

time_steps_unchanged_by_refresh:
true
```

Full same-seed checkpoint eval:

```text
path:
res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/

status:
ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed

scope:
2048 envs x 299 steps = 612352 env steps

seed:
20260721
```

Effects relative to the non-warmup baseline:

```text
step0 done_count delta: -1453
old total done_rate: 0.1782798129180602
target-refresh total done_rate: 0.22340745192307693
done_rate_delta: +0.045127639005016734
old post-step0 done_rate: 0.17552236262583892
target-refresh post-step0 done_rate: 0.22318221738674496
post-step0 done_rate_delta: +0.047659854760906034
```

Updated interpretation:

- Stale reset targets are a real bug: refreshing them sharply reduces reset endpoint-z error without advancing the sampled motion phase.
- The teacher-quality problem is not fixed: total and post-step0 done rates remain worse than the non-refresh baseline.
- The next tracking repair should inspect reset state/action distribution, initial joint velocity mismatch, endpoint thresholds, and `ee_body_pos` termination before launching another expensive PPO/downstream chain.
- This diagnostic remains local virtual evidence, not official BeyondMimic tracking evaluation, not DAgger, not Fig. 5/Fig. 6, not TensorRT, and not real robot evidence.

### Teacher Rollout, VAE, State-Latent, Diffusion, And Guidance

Completed local Level C evidence includes:

- Conditional VAE reimplementation and training on public/local rollout data.
- State-latent trajectory dataset construction.
- Diffusion denoiser training and evaluation.
- Offline guidance and reverse-guidance diagnostics.
- Local closed-loop proxy guidance tasks.
- Multi-seed guidance summaries for joystick, waypoint, obstacle avoidance, and composed tasks.
- Report-ready visualization assets and local videos.
- ONNX export/parity/latency-style audits for local models.

Important caveat:

- These are local paper-faithful or BeyondMimic-like reproductions.
- They are not the official BeyondMimic VAE/diffusion checkpoints.
- They do not use the true paper DAgger rollout distribution.
- They do not reproduce Fig. 5/Fig. 6 paper metrics.

### Fig. 5 / Fig. 6 Proxy Work

The project has a unified local task protocol table covering:

```text
task_count: 6
multiseed_proxy_task_count: 4
single_seed_proxy_task_count: 2
paper_level_reproduced_count: 0
```

Covered local proxy tasks include joystick, waypoint, obstacle avoidance, composed objectives, transition, and inpainting-style tasks.

Interpretation:

- This is useful reading-report evidence because it shows how the paper's guidance idea can be turned into code and local metrics.
- `paper_level_reproduced_count = 0` must be kept explicit.
- The proxy table is not paper Fig. 5/Fig. 6 success/fall/collision reproduction.

### Deployment And ONNX/TensorRT

Completed:

- ONNX contracts and export/parity checks for local components.
- ONNXRuntime CPU evidence.
- Async-proxy deployment-path audits.
- MuJoCo/ROS launch contract audits.

Still missing:

- TensorRT engine generation and benchmark.
- CUDA/TensorRT provider latency.
- Mini-PC deployment latency.
- CppAD/guidance deployment integration.
- Real Unitree G1 deployment.

## What Remains, Excluding Real Robot Deployment

The most important remaining simulation-side paper-level items are:

1. **Stronger tracking teacher**  
   The current teacher runs, but done/termination remains too high. This is the main blocker.

2. **Tracking termination/reset repair**  
   Need to resolve post-refresh `ee_body_pos` termination, reset state/action distribution, and policy-state mismatch before more downstream work.

3. **Official or paper-equivalent teacher rollout dataset**  
   Current rollouts are local and depend on weak/diagnostic teachers. True DAgger-style data remains missing.

4. **VAE closed-loop validation from a strong teacher**  
   Existing VAE evidence is useful but does not prove closed-loop paper-level behavior.

5. **State-latent diffusion from strong teacher rollouts**  
   Existing diffusion is local and proxy; paper-equivalent data distribution is not available.

6. **Fig. 5 / Fig. 6 strict virtual protocol**  
   Need task-specific success/fall/collision metrics and videos under a credible closed-loop stack.

7. **TensorRT/asynchronous deployment audit**  
   Need actual TensorRT engine and GPU latency evidence, not only ONNXRuntime CPU proxy.

8. **Unmodified official converter/replay/training entry equivalence**  
   Current official-loop body evidence is strong, but not a clean unmodified official-entry success.

9. **MuJoCo/ROS runtime execution evidence**  
   Source/launch contracts are audited, but runtime deployment logs are not paper-level.

## Suggested New Goal Direction

The old goal should be replaced by a staged, evidence-aware objective:

1. Maintain claim boundaries and GitHub-light artifact discipline.
2. Treat the English reading report as the course deliverable, not a binary "full reproduction" claim.
3. Keep the strict statement: **this project does not fully reproduce BeyondMimic at paper level**.
4. For engineering progress, prioritize the tracking teacher bottleneck:
   - inspect post-refresh termination and `ee_body_pos`;
   - compare reset state/action distributions and initial joint velocities before/after refresh;
   - decide whether training-time target refresh, reset-state repair, or termination curriculum is justified;
   - rerun robot-order PPO only after the diagnostic gate is better understood.
5. Only after a stronger teacher exists, rerun teacher rollout, VAE, state-latent diffusion, and guidance.
6. For the course report, polish the narrative and figures rather than claiming all non-robot paper results are reproduced.

## Honest Claim Boundary

Current honest statement:

> This project does not fully reproduce BeyondMimic at paper level. It reproduces and audits a large public subset, rebuilds the method as a local virtual pipeline, and documents the exact missing artifacts and technical blockers that prevent a full paper-level reproduction.

Current forbidden statements:

- "BeyondMimic is fully reproduced."
- "The local PPO checkpoint is the official teacher."
- "The warmup eval fixes tracking teacher quality."
- "The local VAE/diffusion results reproduce the official paper checkpoints."
- "The local proxy guidance table reproduces Fig. 5/Fig. 6."
- "The project has TensorRT/Mini-PC/real-robot evidence."
