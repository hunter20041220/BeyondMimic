# Runbook

## Shared Environment

```bash
cd /mnt/infini-data/test/BeyondMimic
source reproduction/scripts/project_env.sh
```

## Phase 0

```bash
bash reproduction/scripts/system_audit.sh
python3 reproduction/scripts/generate_local_inventory.py
```

Outputs:

- `logs/setup/system_audit.txt`
- `reproduction/docs/local_inventory.tsv`
- `reproduction/docs/source_ledger.md`
- `reproduction/docs/environment_plan.md`
- `reproduction/docs/paper_parameter_map.md`
- `reproduction/docs/discrepancy_report.md`
- `reproduction/docs/unresolved_details.md`

## Phase 1

Create project-local prefix environments, then run smoke tests. The exact commands will be appended after the
micromamba/Isaac environment installation is completed.

Current smoke commands:

```bash
source /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh
/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0/python.sh \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/whole_body_tracking_nokit_smoke.py
```

Full IsaacLab/Kit smoke is deferred until the inotify watch limit is increased above the current `8192` value.

Safe tracking smoke rerun audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_smoke_rerun_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/kit_inotify_budget_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/kit_watcher_config_surface_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_import_gate_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_extension_namespace_probe.py
```

This verifies the latest non-Kit `whole_body_tracking` package/asset smoke rerun log and records the current Kit retry
condition from `/proc/sys/fs/inotify/*`. The Kit/inotify budget audit also parses the failed Kit retry log, records the
current watcher limits, proves a bounded Isaac Sim directory lower bound exceeds the current watch budget, and records
available filesystem capacity so the Kit `errno=28` messages are not mistaken for ordinary disk exhaustion. It does not
launch Kit, start PPO, or claim live tracking reproduction. The watcher config surface audit indexes local Kit app
configuration surfaces and extension watch roots. The tracking extension namespace probe appends local
`isaacsim.core.*` extension namespace paths inside a subprocess only; this changes the import failure from missing
`isaacsim.core` to native/Kit runtime dependencies such as `libarch.so` and `omni.kit.commands`, so it is evidence of
the runtime boundary rather than a safe Kit bypass.
extension roots and per-extension `[fswatcher.*]` config sections; it records that no documented global watcher-disable
setting was found in the local app configs. The tracking import gate audit uses the IsaacLab-bound Isaac Sim Python
entrypoint to prove that plain non-Kit imports can load `isaaclab` but not `isaacsim.core` or deep
`whole_body_tracking` config modules, so those modules remain gated by a live Kit extension-manager context.

Non-Kit tracking configuration audit:

```bash
source /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh
/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba run \
  -p /mnt/infini-data/test/BeyondMimic/envs/bm_analysis \
  python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_config_audit.py
```

Output:

`/mnt/infini-data/test/BeyondMimic/res/tracking/smoke_config_audit/tracking_config_audit.json`

Adaptive-sampling paper/source vs official-code discrepancy audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/adaptive_sampling_discrepancy_audit.py
```

Output:

`/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json`

Motion preprocessing producer/consumer contract audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/motion_preprocessing_contract_audit.py
```

Output:

`/mnt/infini-data/test/BeyondMimic/res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json`

Non-Kit tracking `motion.npz` debug fixture builder:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_tracking_motion_npz_fixture.py
```

Output:

`/mnt/infini-data/test/BeyondMimic/res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json`

The generated files under `/mnt/infini-data/test/BeyondMimic/reproduction/data/tracking_motion_npz_fixtures` are
URDF-FK contract fixtures. They pass the local `motion.npz` validator, but they are not official Isaac/Kit
`csv_to_npz.py` articulation exports.

MuJoCo/ROS launch contract audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_ros_launch_contract_audit.py
```

Output:

`/mnt/infini-data/test/BeyondMimic/res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json`

This audit checks the official `motion_tracking_controller` README, `package.xml`, plugin XML, controller YAML, and
MuJoCo/real launch files without executing ROS. It records the current host runtime gate when ROS 2 Jazzy/Noble tools
are unavailable.

## Phase 2

```bash
source /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh

/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba run \
  -p /mnt/infini-data/test/BeyondMimic/envs/bm_analysis \
  python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduce_released_figures.py

/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/convert_adaptive_sampling.py

/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba run \
  -p /mnt/infini-data/test/BeyondMimic/envs/bm_analysis \
  python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/plot_adaptive_sampling_released.py

/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_data_metrics_summary.py

/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_data_statistical_audit.py

python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/metrics_catalog.py

python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_log_config_catalog.py
```

Main output:

`/mnt/infini-data/test/BeyondMimic/res/released_figures/released_figure_summary.tsv`

Released-data numeric table output:

`/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_data_metrics_summary.json`

Released-data statistical audit output:

`/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_data_statistical_audit.json`

Metrics catalog output:

`/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.json`

Run/log/config catalog output:

`/mnt/infini-data/test/BeyondMimic/res/run_log_config_catalog/run_log_config_catalog.json`

The adaptive probability evolution uses:

`/mnt/infini-data/test/BeyondMimic/reproduction/data/Dataset_beyondmimic/adaptive_sample/sampling_prob_over_time.pkl`

## Level C Smoke / Fixture Gates

Master evidence audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py
```

This audit verifies the current reproduction evidence set, checks key JSON/TSV artifacts, and summarizes the completion
matrix counts. It is the fastest way to confirm that the current partial/blocked state is internally consistent.

goal.md directive index audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_directive_index_audit.py
```

This reads all of `/mnt/infini-data/test/BeyondMimic/goal.md`, indexes headings and directive-bearing lines by line number,
and cross-checks the line/heading counts against the goal traceability and requirement-matrix audits. It is coverage
evidence for reading and tracking the instruction document, not proof that the still-blocked paper-level experiments
have been completed.

Blocked-gate audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/blocked_gate_audit.py
```

This audit records the external/system gates that prevent claiming full paper-level reproduction on this host:
IsaacLab/Kit inotify limits, ROS 2 Jazzy/Noble deployment availability, Unitree G1 hardware absence, missing official
Level C code/checkpoints, Fig. 5/Fig. 6 data/checkpoint gaps, and the long-training safety gate.

Paper-source coverage audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_source_coverage_audit.py
```

This audit parses the LaTeX source for figures/tables and maps figures, tables, and key method/formula claims to the
current reproduction evidence or blocked/debug-only status.

Paper LaTeX inventory audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_latex_inventory_audit.py
```

This audit automatically inventories the local LaTeX source files, equations, sections, figures, tables, and key
experiment-setting statements, then cross-checks the inventory against the paper-source coverage and table-value audits.
It indexes the paper/source content; it does not create missing checkpoints, rollout logs, videos, or deployment results.

Paper formula/code trace audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_formula_code_trace_audit.py
```

This maps paper LaTeX equations, experiment settings, and table values to local formula code, tests, static official
source audits, and debug probes. It records debug/static/protocol boundaries explicitly and does not claim trained
policies, TensorRT deployment, Fig. 5/Fig. 6 reproduction, or videos.

Download source integrity audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/download_source_integrity_audit.py
```

This reads the original `/mnt/infini-data/test/BeyondMimic/download/manifests/downloaded_files.tsv` as the full download
inventory, then hashes required paper/source, official dataset/code, dependency, supplemental-manifest, and reference
README files. It is read-only provenance evidence for the raw download bundle, not training, rollout, video, TensorRT,
ROS, or hardware evidence.

Paper PDF/source consistency audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_pdf_source_consistency_audit.py
```

This audit extracts text from the downloaded paper PDF with `pypdf`, checks key PDF anchors against the paper/source
settings, verifies the raw source tar member list against the extracted LaTeX tree, and cross-checks the LaTeX inventory,
source coverage, and table-value audits. It does not reproduce paper-level experiments.

Paper table value audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_table_value_audit.py
```

This audit maps reward, domain-randomization, PPO, VAE, and diffusion table values to official-code audits or
debug-only Level C probes. VAE/diffusion matches are value-level checks, not training/checkpoint reproduction.

Skill-success table data audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/skill_success_table_data_audit.py
```

This audit parses Table `tab:skill_success` and checks the listed LAFAN1 motion names and Real time intervals against
local G1 retargeted CSV data. It does not reproduce sim or real execution success.

Released-data panel mapping audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_panel_mapping_audit.py
```

This audit cross-checks `reproduction/docs/paper_panel_map.tsv`,
`res/released_figures/released_figure_summary.tsv`, the extracted `Dataset_beyondmimic` tree, and immutable
`download/official/Dataset_beyondmimic.zip` so released-data paper panels can be traced to generated artifacts and raw
release files. It does not claim Fig. 5/Fig. 6 or hardware-only results.

Results claims audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/results_claims_audit.py
```

This audit maps Results-section claims to released-data reproductions, debug-only probes, paper-only rows, hardware
requirements, or unavailable checkpoint/rollout evidence.

Goal traceability audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_traceability_audit.py
```

This audit maps the major `goal.md` sections and paper-specific requirements to current evidence, partial/blocked
status, or out-of-scope conditions.

Goal requirement matrix audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_requirement_matrix_audit.py
```

This audit maps finer-grained `goal.md` requirements to evidence and remaining gaps. It intentionally keeps incomplete,
blocked, and out-of-scope items visible.

Level C VAE architecture/loss contract audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_contract_audit.py
```

This audit checks the paper VAE table, method formulas, local debug VAE dimensions, runtime shapes, KL/reparameterization
math, interpolation path, and 15-step gradient accumulation. It is debug-contract evidence only; it is not a trained VAE
checkpoint or true DAgger rollout.

Level C sagittal symmetry mapping audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_symmetry_mapping_audit.py
```

This audit checks the candidate Unitree G1 sagittal mirror mapping against the official controller joint order, URDF
joint names, the paper's high-level sagittal symmetry statement, and the augmentation double-mirror probe. It remains a
candidate/debug audit because the paper does not publish a definitive G1 sign table.

Level C guidance cost coverage audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/guidance_task_coverage_audit.py
```

This maps all 30 Phase 8 guidance-task rows: six tasks times five required evidence types. It explicitly records
debug-only evidence, missing task-specific trained rollouts, and blocked success/failure videos without claiming
paper-level Fig. 5/Fig. 6 reproduction.

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_cost_coverage_audit.py
```

This audit separates paper-explicit classifier guidance, joystick, waypoint, SDF, and relaxed-barrier formulas from the
candidate keyframe inpainting term, then cross-checks them against local debug gradients, the guided reverse loop, and
the guidance-scale sweep.

Core math unit tests:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py
```

This pure-NumPy test entrypoint writes `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json`
and `.tsv`. It covers 18 formula-level checks for Rot6D, current-frame transforms, rewards/termination, adaptive
sampling, OU noise, symmetry, emphasis projection, diffusion, masks, VAE latent math, guidance costs, and smoothness.
It is a unit-test gate for math mechanics, not a trained Isaac/ROS/TensorRT deployment test suite.

Core math checklist coverage audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/core_test_coverage_audit.py
```

This maps the explicit 20-item `goal.md` "at least test" checklist to passed core-math tests and formula tags.

Coding requirements audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/coding_requirements_audit.py
```

This checks the current clean-room formula package and run scaffolding against `goal.md` section 14 coding requirements:
type hints, docstrings, shape/frame documentation, NaN/Inf guards, seeds, CLI/YAML/resolved config, run metadata, and
core math tests. It does not certify missing full training/deployment code.

Lightweight reimplementation package audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reimpl_package_audit.py
```

This checks the reusable formula APIs under `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl` and
verifies that the core math tests import the package API. It is not the unpublished official training implementation.

Reimplementation runtime integration audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reimpl_runtime_integration_audit.py
```

This runs the reusable package APIs over local debug state-latent windows, downstream action outputs, and a motion fixture:
state-latent stacking, split counts, projection reconstruction, diffusion masking/reverse step, VAE latent math, DAgger
sample schema, guidance costs, action metrics, tracking error, and survival-rate checks. It is still fixture/debug evidence,
not official training code, a trained checkpoint, or a closed-loop rollout.

Reimplementation package API contract tests:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/tests/test_reimpl_package_api.py
```

This checks exported package symbols, shape/error contracts, metadata helpers, finite-value guards, and trajectory,
DAgger, guidance, state, diffusion, VAE, sampling, geometry, and evaluation APIs. It is pure NumPy/stdlib evidence,
not an IsaacLab, ROS, TensorRT, long-training, or deployment test suite.

bm_diffusion environment audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/bm_diffusion_env_audit.py
```

This checks the project-local `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion` prefix, exported lock files, the
NumPy/SciPy/PyYAML/tqdm base stack, and a Torch CUDA smoke test. It records environment readiness for Level C debug
mechanics, not completion of VAE/diffusion training.

GPU resource snapshot and metrics schema:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/gpu_resource_audit.py --samples 3 --interval-sec 1
```

This appends goal-schema rows to `/mnt/infini-data/test/BeyondMimic/logs/gpu/gpu_metrics.csv` and writes
`/mnt/infini-data/test/BeyondMimic/res/setup/gpu_resource_audit/gpu_resource_audit.json`. It samples current resources only;
it does not create artificial utilization or prove training throughput.

Run-management schema diagnostic:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/create_run_management_skeleton.py
```

This creates `/mnt/infini-data/test/BeyondMimic/res/runs/setup_run_management_diagnostic_static_000_20260617_050000`
with every file and subdirectory required by goal.md Section 15, then audits it at
`/mnt/infini-data/test/BeyondMimic/res/run_management_audit/run_management_audit.json`. The diagnostic run is marked
`INVALID`; only real runs that reach their training endpoint and evaluation may be marked `SUCCESS`.

Checkpoint/resume smoke:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/checkpoint_resume_smoke.py
```

This creates `/mnt/infini-data/test/BeyondMimic/res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500`,
writes a deterministic diagnostic checkpoint, resumes from it, and verifies exact agreement with an uninterrupted
baseline at `/mnt/infini-data/test/BeyondMimic/res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json`.
It is not a PPO/VAE/diffusion model checkpoint.

Failed-run retention record:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/record_failed_run.py
```

This preserves the observed IsaacLab/Kit inotify failure in
`/mnt/infini-data/test/BeyondMimic/res/failed_runs/phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654` with the
fields required by goal.md Section 9.5. It is failure evidence and a rerun plan, not a fix for the host inotify limit.

Resolved reproduction config manifest:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/resolved_reproduction_config.py
```

This writes `/mnt/infini-data/test/BeyondMimic/res/config/resolved_reproduction_config.json`, `.yaml`, and `.csv` from the
current machine-readable audits. It is the current config contract for reruns, not evidence that the long training
phases have completed.

Artifact hash manifest:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py
```

This writes `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json` and `.tsv`, hashing key raw
sources, environment locks, config, audits, run logs, comparison, final report, and docs for current traceability.

Experiment protocol audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/experiment_protocol_audit.py
```

This verifies that `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md` covers Phase 0-10 gates,
run directory rules, failed-run retention, no-fabrication boundaries, and current blockers.

Top-level README audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/readme_audit.py
```

This verifies that `/mnt/infini-data/test/BeyondMimic/README.md` points to the current evidence, commands, blockers, and
integrity rules without claiming that the full paper reproduction is complete.

Level C timestep/mask source coverage audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_timestep_mask_coverage_audit.py
```

This audit separates the paper-explicit independent `k_s/k_z` timestep mechanics from unpublished deployed mask policy
details. It cross-checks the older 181-D fixture schedules and the separate 99-D paper-state mask/reverse debug probe.

Level C paper-state mask/reverse probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_mask_reverse_probe.py
```

This probe applies the timestep/mask schedules and oracle reverse mechanics to paper-formula 99-D state windows plus
synthetic 32-D latents. It is a debug mechanism check, not a trained diffusion model or Fig. 6 rollout.

Level C diffusion deployment protocol audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_deployment_protocol_audit.py
```

This audit indexes the paper's 25 Hz, 20 ms / 20-step, TensorRT, asynchronous-thread, CPU decoder, CppAD, state-estimation,
and mocap deployment claims, then separates local debug evidence from expected missing deployment boundaries.

Consolidated final evidence report:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py
```

Outputs:

- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`

Final-report requirement audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_report_requirement_audit.py
```

This verifies that the Markdown final report explicitly covers the 12 `goal.md` final-report topics: official code use,
paper-faithful reimplementation, released-data reproduction, retraining status, qualitative-only comparison,
not-publicly-reproducible boundaries, differences, likely sources, credibility, completed/incomplete scope, hardware
cost/training time, and rerun commands.

Final deliverables audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_deliverables_audit.py
```

This checks the `goal.md` environment/code/experiment/documentation deliverable lists, records evidence paths, and
keeps missing checkpoints/videos explicit instead of treating placeholder directories as paper-result artifacts.

Required trained/deployment artifact absence audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py
```

This audit separates downloaded reference-project ONNX/PT/GIF assets from the still-missing BeyondMimic reproduction
artifacts: trained checkpoints, ONNX/TensorRT exports, rollout logs, completed training run directories, and videos.

Patch inventory audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/patch_inventory_audit.py
```

This records the contents of `reproduction/patches` and the tracked state of the local official worktrees. It is patch
hygiene evidence, not a completed training/deployment patch series.

Patch snapshot audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/patch_snapshot_audit.py
```

This exports currently tracked official-worktree diffs into
`/mnt/infini-data/test/BeyondMimic/reproduction/patches/official_worktree_snapshots`. Empty or semantic-empty snapshots are
line-ending/whitespace hygiene evidence, not a functional patch series.

Evaluation metrics coverage audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/evaluation_metrics_coverage_audit.py
```

This maps all 44 `goal.md` Section 12 metrics to current evidence or explicit blocked/debug-only gaps. It is a coverage
audit, not a substitute for missing trained rollout metrics.

Trial/failure accounting audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/trial_failure_accounting_audit.py
```

This consolidates available source-table, released-data, debug-seed, run-catalog, and retained failed-run counts for
Section 12.5. It does not create missing paper rollout trial/failure counts.

Ablation coverage audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/ablation_coverage_audit.py
```

This maps all 15 `goal.md` Phase 9 ablation items to released-data panels, debug probes, or explicit missing
paper-level training/evaluation gaps.

Progress report audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/progress_report_audit.py
```

This verifies that `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md` contains all 21 mandatory section-17 fields,
key progress markers, master-audit progression, and explicit incomplete checkpoint/video boundaries.

Project boundary audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_boundary_audit.py
```

This verifies the `download` top-level allowlist, supplemental-download manifest, source ledgers, generated project
roots, and project-local cache/tmp redirects in `reproduction/scripts/project_env.sh`.

Official/reference Level C artifact audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_artifact_audit.py
```

This audit checks whether local official artifacts include BeyondMimic-specific VAE/diffusion code, checkpoints, ONNX,
or TensorRT engines. It does not download new sources.

Figure 5 / Figure 6 feasibility audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_fig5_fig6_feasibility_audit.py
```

This audit classifies each Fig. 5/Fig. 6 panel as paper-reproduction blocked or debug-mechanism covered based on local
released data, official artifact audit results, and Level C probes.

Synthetic VAE/diffusion/guidance smoke:

```bash
source /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_env.sh
BM_LEVEL_C_DEVICE=cpu /mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_synthetic_smoke.py
```

VAE gradient accumulation probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_accumulation_probe.py \
  --device cpu
```

This probe checks paper VAE dimensions, KL/reconstruction loss wiring, and 15-step gradient accumulation on synthetic
teacher queries. It is not real DAgger rollout or VAE checkpoint reproduction.

DAgger iteration smoke:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_dagger_iteration_smoke.py
```

This runs three debug-only synthetic student-rollout/teacher-query aggregation iterations and checks that the aggregate
dataset grows and held-out teacher/student action discrepancy falls. It is not Isaac rollout or a true BeyondMimic
teacher-policy DAgger run.

VAE checkpoint save/load smoke:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_checkpoint_smoke.py \
  --device cpu
```

This writes a debug-only conditional VAE checkpoint and verifies exact save/load eval consistency. It is explicitly not
a trained BeyondMimic VAE checkpoint from true DAgger rollout data.

Tiny VAE debug overfit latent artifact:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_debug_overfit_latent_artifact.py \
  --device cpu
```

This trains a small CPU-only debug VAE on local paper-state windows with synthetic teacher actions and exports nonzero
32-D latents. It is a state-latent data-path smoke gate, not true DAgger or a paper VAE checkpoint.

VAE motion-split held-out eval:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_motion_split_heldout_eval.py \
  --device cpu
```

This trains the tiny conditional VAE only on the train motion split and reports validation/test action MSE and KL over
held-out local fixture motions. It is not true DAgger rollout, a paper VAE checkpoint, or closed-loop VAE evaluation.

VAE receding-horizon rollout smoke:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_receding_horizon_rollout_smoke.py
```

This rolls the current-action index across the exported tiny-VAE debug state-latent windows and compares decoded
actions to synthetic teacher actions. It validates the local current-latent action interface only; it is not true
DAgger, a paper VAE checkpoint, closed-loop Isaac evaluation, or a success/failure video.

Diffusion-to-VAE action smoke:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_smoke.py
```

This fits a train-split-only surrogate action decoder from clean tiny-VAE debug tokens and evaluates held-out diffusion
token predictions through it. It checks the local diffusion-token to current-action pipe only; it is not the
unpublished trained VAE decoder, trained diffusion Transformer, closed-loop rollout, or paper Fig. 5/Fig. 6 protocol.

Diffusion-to-VAE action multi-seed audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_multiseed_audit.py
```

This repeats the downstream diffusion-token to surrogate-action-decoder pipe over three seeds and reports mean/std/min/max.
It is smoke-level statistics only, not multi-seed paper diffusion training or closed-loop task evaluation.

Diffusion-to-VAE action smoothness audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_smoothness_audit.py
```

This computes 25 Hz first/second-difference smoothness metrics over the target, clean, noisy, and predicted downstream
action windows from the diffusion-to-VAE action smoke artifact. It is offline debug evidence for action-sequence
regularity, not a trained-policy rollout, TensorRT benchmark, or closed-loop action smoothness evaluation.

Direct-vs-latent action ablation audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_direct_vs_latent_action_ablation_audit.py
```

This compares an offline direct 99-D state-only action branch with the existing 131-D state-latent downstream action
pipe on the same debug windows and splits. It is a local data-path ablation, not the paper's trained direct-vs-latent
cartwheel success-rate result.

Receding-horizon decoder probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_receding_horizon_decoder_probe.py \
  --device cpu
```

This probe decodes the current denoised latent into one current action using a synthetic CPU VAE decoder. It is not a
trained decoder checkpoint, paper-exact proprioception layout, deployment integration, or closed-loop rollout.

Full paper Transformer architecture probe:

```bash
BM_LEVEL_C_TORCH_THREADS=2 /mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_full_transformer_arch_probe.py \
  --device cuda:0 \
  --batch-size 1
```

This probe instantiates the paper diffusion Transformer dimensions (`embedding=512`, `heads=8`, `layers=6`) and runs
one clean-trajectory forward/backward pass. It is not long training, checkpoint reproduction, TensorRT deployment, or
guided rollout evaluation.

Paper-state Transformer architecture probe:

```bash
BM_LEVEL_C_TORCH_THREADS=2 /mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_transformer_arch_probe.py \
  --device cuda:0 \
  --batch-size 1
```

This probe repeats the paper-sized Transformer forward/backward check on 99-D paper-formula state windows plus
synthetic 32-D latents, yielding tau dimension 131. It is not long training or checkpoint reproduction.

VAE-latent Transformer architecture probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_transformer_arch_probe.py \
  --device cpu \
  --batch-size 1
```

This repeats the paper-sized Transformer forward/backward check using nonzero tiny-VAE debug latents instead of random
latents. It is not true VAE rollout data, long training, checkpoint reproduction, or rollout evaluation.

VAE-latent Transformer AdamW/EMA smoke:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_transformer_ema_smoke.py \
  --device cpu --steps 2
```

This executes two debug optimizer/EMA updates on one paper-sized state-latent batch using nonzero tiny-VAE debug
latents. It validates local training mechanics on the intended token shape, not true rollout latents, full diffusion
training, or a trained/EMA checkpoint.

Transformer parameter-count audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_parameter_count_audit.py
```

This audit compares the two local paper-sized Transformer probes against the paper's approximate `~19.8M` parameter
statement. It records approximate consistency and the remaining exact-implementation boundary.

Transformer state-dict hash manifest:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_state_dict_manifest.py
```

This writes a lightweight tensor-hash manifest for the paper-state Transformer initialization. It does not write a
trained diffusion checkpoint or an EMA checkpoint.

Diffusion checkpoint save/load/resume smoke:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_checkpoint_smoke.py \
  --device cpu
```

This writes a debug-only paper-state Transformer checkpoint, reloads model/optimizer/EMA state, resumes one step, and
checks exact agreement with an uninterrupted path. The `.pt` is explicitly excluded from trained checkpoint counts.

Transformer AdamW/EMA smoke:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_ema_smoke.py \
  --device cpu --steps 2
```

This executes two debug optimizer/EMA updates on one paper-state batch. It is a training-mechanics smoke, not full
diffusion training or a trained/EMA checkpoint.

Diffusion training schedule probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_training_schedule_probe.py
```

This probe records the paper diffusion optimizer/LR/EMA parameters and checks a local warmup/cosine/EMA schedule. It is
not full training, checkpoint reproduction, or validation evaluation.

Retargeted-motion non-Kit state fixture:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_motion_state_fixture.py

/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_level_c_motion_state_fixture.py \
  /mnt/infini-data/test/BeyondMimic/reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz \
  --manifest-json \
  /mnt/infini-data/test/BeyondMimic/res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json
```

The motion-state fixture is `debug_only`. It is a schema/provenance/transform gate for later Level C work, not a
replacement for official Isaac/Kit `motion.npz`, teacher rollouts, DAgger data, or paper-exact state-latent trajectories.

Window provenance/split manifest for the fixture:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_fixture_split_manifest.py
```

This manifest is `debug_only`. It checks required provenance fields and guards against overlapping-window leakage in the
single-motion fixture, but it is not the paper-exact train/validation/test split for real teacher/VAE rollouts.

OU/sagittal augmentation probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_augmentation_probe.py
```

This probe is also `debug_only`. It validates executable OU perturbation parameters and a candidate G1 sagittal
left/right mapping, but it is not a VAE rollout perturbation pipeline, episode rejection pipeline, or trainable
augmented dataset.

Dataset collection protocol audit:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_dataset_collection_protocol_audit.py
```

This audit indexes the paper/goal OU perturbation, approximate 100x sample coverage, 2.5 s rollout, 5 s stability
verification, episode rejection, sagittal augmentation, and provenance requirements against the current debug-only
evidence. It does not replace true VAE rollout collection.

Rollout rejection manifest probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_rollout_rejection_manifest_probe.py
```

This probe creates a debug-only manifest for the paper's OU perturbation collection protocol: 2.5 s recorded windows,
5 s stability windows, 100 synthetic OU seeds per valid start, and accept/reject fields. It does not run a trained VAE
policy or make real rejection decisions.

Guidance cost formula probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_formula_probe.py
```

This probe uses PyTorch autograd in `bm_tracking` to validate joystick, waypoint, SDF relaxed-barrier, composed
guidance, and a candidate keyframe term on the motion-derived fixture. It is not denoising-loop guidance, a scale sweep,
or guided rollout evaluation.

Independent timestep/mask schedule probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_timestep_mask_probe.py
```

This probe validates `[sequence, state_or_latent]` denoising-step schedules for training, history conditioning, and
future-keyframe inpainting. It is not the paper-exact deployed mask policy or reverse denoising controller.

Trajectory inverse-transform audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_trajectory_inverse_transform_audit.py
```

This audit checks the paper-formula root current-character-frame transform and body local-frame transform, then inverts
them over the motion-derived fixture windows. It is a math gate for the trajectory representation, not an official
teacher rollout or trainable state-latent dataset.

State representation source audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_representation_source_audit.py
```

This audit compares the debug fixture state windows with the paper/source hybrid character-yaw formulas. It confirms
the body local-position formula and records the non-paper-exact boundaries for root current-frame features, body
velocity subtraction, full root relative position, and emphasis projection.

Paper-formula state-window builder:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_paper_state_windows.py
```

This builder creates 99-D paper-formula state windows from the debug motion fixtures: root relative pose/twist in the
window-current character frame and body local positions/velocities in each local root frame. It is still a debug state
artifact and does not include VAE latents or teacher rollout provenance.

State-latent dataset consistency audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_latent_dataset_consistency_audit.py
```

This cross-checks the paper-state window NPZ files, debug tiny-VAE latent/action NPZ, VAE JSON rows, and downstream
diffusion-to-action NPZ ordering. It verifies shape, split, start-timestep, state-array, decoded-action, current-root
zero, and boundary-declaration consistency. It is still a debug consistency audit, not the paper's true DAgger/VAE
rollout dataset.

Paper-state diffusion overfit gate:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_overfit_probe.py
```

This gate checks that the clean-trajectory denoising path can memorize the paper-formula state windows plus synthetic
latents. It intentionally uses an overparameterized token basis and is not full diffusion training.

Debug-VAE-latent diffusion overfit gate:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_diffusion_overfit_probe.py
```

This repeats the paper-state clean-trajectory memorization gate using the nonzero tiny-VAE debug latents instead of
random synthetic latents. It is a data-path smoke gate, not true VAE rollout data or a trained diffusion checkpoint.

Paper-state held-out debug evaluation:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_heldout_eval.py
```

Debug-VAE-latent held-out debug evaluation:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_heldout_eval.py
```

This repeats the non-memorizing held-out ridge baseline with nonzero tiny-VAE debug latents and motion-level splits. It
is not true VAE rollout data, trained diffusion evaluation, or the paper validation protocol.

This gate trains a non-memorizing ridge baseline on the motion-level train split using paper-formula state windows plus
synthetic latents. It does not use token identity features, but it is still a debug baseline rather than a trained
diffusion model.

Paper-state held-out multi-seed statistics:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_heldout_multiseed_audit.py
```

This gate repeats the non-memorizing paper-state held-out baseline over at least three seeds and reports mean/std/min/max
statistics. It remains smoke-level evidence over synthetic latents, not paper multi-seed diffusion training.

Debug-VAE-latent held-out multi-seed statistics:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_heldout_multiseed_audit.py
```

This repeats the same non-memorizing held-out baseline over at least three seeds using nonzero tiny-VAE debug latents.
It remains smoke-level statistics, not true VAE rollout data or paper multi-seed diffusion evaluation.

Emphasis projection and pseudoinverse audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_emphasis_projection_audit.py
```

This gate implements the paper formula `P=[AB I]^T` with `A_ij ~ N(0,1)`, root pose/twist coefficient `c=6`, and
pseudoinverse reconstruction over the local paper-state windows. It is a formula-level audit, not trained diffusion.

VAE latent reparameterization and interpolation probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_probe.py
```

This gate checks paper-dimension conditional VAE latent reparameterization, KL computation, and latent interpolation
continuity over three seeds. It is synthetic debug evidence, not true DAgger rollout or checkpoint reproduction.

Smoothness and latency metric audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_smoothness_latency_audit.py
```

This gate computes debug trajectory/latent smoothness, schema action delta, guidance-cost reduction, and paper latency
budget checks from existing reverse-loop and decoder artifacts. It is not a trained TensorRT latency benchmark.

Tracking ONNX export contract audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_onnx_export_contract_audit.py
```

This gate cross-checks the official Python ONNX exporter with the C++ motion-tracking controller consumer and confirms
that the Unitree reference ONNX is not a valid BeyondMimic motion export.

Reverse denoising mechanics probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_reverse_denoising_probe.py
```

This probe validates a debug-only oracle reverse loop with independent state/latent timesteps and future-keyframe
clamping. It is not the paper-exact alpha/gamma/sigma schedule, a trained denoising network, TensorRT deployment,
denoising-loop guidance, or guided rollout evaluation.

Diffusion equation/coefficient-boundary audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_equation_audit.py
```

This audit verifies that the local paper/source contains the DDPM forward posterior, clean-prediction losses, reverse
`alpha/gamma/sigma` form, and `K=20` denoising steps. It also records that the exact numeric beta or
`alpha/gamma/sigma` coefficient schedule is not published in the current paper/source artifacts.

Single-batch diffusion overfit gate:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_single_batch_overfit_probe.py
```

This NumPy probe checks that the local clean-trajectory loss/data path can overfit one debug fixture batch with
independent state/latent denoising steps. It is not full diffusion training or checkpoint reproduction.

Single-motion diffusion overfit gate:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_single_motion_overfit_probe.py
```

This NumPy probe checks that all windows from the single debug fixture motion can be memorized with the local
clean-trajectory target and independent state/latent denoising steps. It intentionally uses an overparameterized
memorization basis and is not a multi-motion small-dataset overfit or validation result.

Small-dataset diffusion overfit gate:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_overfit_probe.py
```

This NumPy probe checks that three short debug fixture motions can be memorized with the local clean-trajectory target
and independent state/latent denoising steps. It intentionally uses an overparameterized memorization basis and is not
full diffusion training, held-out validation, or paper-result reproduction.

Small-dataset provenance/split manifest:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_small_dataset_split_manifest.py
```

This manifest records motion-level train/validation/test splits for the three debug fixture motions and marks missing
VAE latents plus debug-only accept/reject status. It is not the paper-exact rollout split.

Small-dataset multi-seed memorization statistics:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_multiseed_audit.py
```

This audit repeats the debug small-dataset memorization gate over three seeds and reports mean/std/min/max. It is not
multi-seed full diffusion training or paper evaluation.

Small-dataset held-out debug evaluation:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_heldout_eval.py
```

This audit trains a simple non-memorizing ridge baseline on the motion-level train split and reports validation/test
losses. It does not use token identity features, but it is still only a debug baseline, not the paper diffusion model.

Small-dataset held-out multi-seed statistics:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_heldout_multiseed_audit.py
```

This audit repeats the non-memorizing held-out debug baseline over three seeds and reports mean/std/min/max. It is
smoke-level statistical evidence only, not multi-seed paper diffusion training or evaluation.

Guided reverse-loop probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guided_reverse_loop_probe.py \
  --schedule history_conditioning \
  --guidance-scale 0.002
```

This probe wires joystick classifier guidance into the debug-only oracle reverse loop. It is not a trained guided
diffusion controller, paper-exact guidance-scale protocol, rollout evaluation, TensorRT deployment, or Fig. 5/Fig. 6
reproduction.

Guidance scale sweep probe:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_scale_sweep_probe.py \
  --schedule history_conditioning
```

This probe sweeps a small local joystick-guidance scale grid on the debug-only oracle reverse loop. It is not the
paper-exact validation protocol or guided rollout evaluation.

Guidance task formula-level scale sweep:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_task_scale_sweep.py
```

This sweeps formula-level scales for joystick, waypoint, obstacle, inpainting, and composed objectives on one
motion-derived future-state tensor. It is not a closed-loop rollout, video, or task validation protocol.

Guidance debug visualization:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_debug_visualization.py
```

This writes PNG/SVG/PDF/GIF visualizations of local formula-guidance effects for joystick, waypoint, obstacle,
inpainting, and composed objectives on one fixture trajectory. The GIF is debug-only and is explicitly excluded from
the required paper success/failure video count.

Guidance task metric audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_task_metric_audit.py
```

This consolidates the five debug primary task metrics from the visualization with the formula-level scale sweep. It is
not a closed-loop rollout metric, success-rate table, or Fig. 5/Fig. 6 reproduction.

Resource-adjusted G1 tracking gates:

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_csv_conversion_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_csv_full_replay_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python \
  /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.py
```

These gates use the official downloaded G1 LAFAN CSV where applicable, but still route through the generated
resource-adjusted enriched USD. The train-entry diagnostic verifies `Tracking-Flat-G1-v0` -> `RslRlVecEnvWrapper` ->
`MotionOnPolicyRunner` -> one tiny four-step PPO update with no checkpoint. It is a wiring diagnostic, not formal PPO
training, a trained teacher, official replay/evaluation, or paper-level tracking performance.

Paper-vs-reproduction comparison table:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py
```

This generates `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv`,
`/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.md`, and
`/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`. Rows use the goal.md comparison types:
`exactly_comparable`, `approximately_comparable`, `qualitative_only`, `not_publicly_reproducible`, and
`requires_real_robot`.

Verification command coverage audit:

```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_coverage_audit.py
```

This audits the final report verification-command list, categorizes lightweight versus environment/heavy commands, and
executes a bounded five-command smoke subset. It does not run IsaacLab/Kit, ROS 2, long VAE/diffusion training,
TensorRT, Fig. 5/Fig. 6 evaluation, or real Unitree G1 deployment.
