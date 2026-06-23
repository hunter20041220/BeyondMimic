# Progress Update

## Goal

Audit the Stage-1 motion-tracking teacher implementation before launching more teacher, VAE, diffusion, guidance, or video work. The specific question was whether the local code follows BeyondMimic paper/supplement contracts for RL rewards, observations, PD/action scale, armature, material/domain randomization, termination, PPO, and teacher gating.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`
- `res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/paper_contract_tracking_parameters.json`
- `res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/tracking_g1_official_importer_export_paper_contract_ppo_training_run.json`
- `res/tracking/stage1_multisource_paper_contract_ppo_training_run/tracking_stage1_multisource_paper_contract_ppo_training_run.json`
- `res/tracking/stage1_teacher_checkpoint_quality_selector/stage1_teacher_checkpoint_quality_selector.json`
- `res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json`

## Files Modified

- Added `reproduction/scripts/stage1_tracking_parameter_contract_audit.py`
- Added `res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.json`
- Added `res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.tsv`
- Added `res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.md`
- Added this progress record.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/stage1_tracking_parameter_contract_audit.py
python3 reproduction/scripts/stage1_tracking_parameter_contract_audit.py
```

## Results

The new audit produced 15 Stage-1 tracking contract rows:

- `pass`: 8
- `pass_with_caution`: 5
- `fail_or_unverified`: 2

Main conclusion: the official/public `whole_body_tracking` Stage-1 code and the 4/7 paper-contract wrapper largely match the paper/supplement formulas for PD gains, action scale, armature, rewards, termination, observation dimensions, and PPO hyperparameters. However, current teachers still fail downstream quality gates, so VAE/diffusion/guidance should not be treated as final.

Important cautions:

- Public code uses adaptive sampling `adaptive_kernel_size=1`, while the supplement describes a look-back kernel with `K=3`.
- Supplement prose describes larger ankle default-joint offset randomization, but public code appears to randomize all joints with the compact `[-0.01, 0.01]` range.
- The 5/6 multi-source training summary inherited a resource-adjusted USD metadata flag even though the wrapper points to the official importer/export USD path. This is a reporting/metadata cleanup item.
- The current checkpoint selector reports no downstream-ready teacher checkpoint.

## Verification

Completed:

```bash
python3 -m py_compile reproduction/scripts/stage1_tracking_parameter_contract_audit.py
python3 reproduction/scripts/stage1_tracking_parameter_contract_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

All verification commands returned `rc=0`. Logs were written under:

`logs/verification/stage1_tracking_parameter_contract_20260624_044608/`

Refreshed summary:

- `res/artifact_manifest/artifact_manifest.json`: status `ok`, 1818 artifacts, including the new Stage-1 tracking parameter contract audit script and JSON/TSV/MD outputs.
- `res/master_audit/reproduction_master_audit.json`: status `ok`, failed artifacts `[]`.

## Failed / Blocked Items

- Stage-1 teacher quality remains blocked: no current checkpoint is ready for final downstream VAE/diffusion dataset collection.
- Existing poor MuJoCo teacher/VAE/diffusion videos should be interpreted as expected failures from weak teacher data and adapter limitations, not as paper-level BeyondMimic behavior.
- Do not launch final VAE/diffusion/guidance training until the teacher quality gate passes.

## Effect on English Reading Report

This audit strengthens the report by separating formula/parameter fidelity from empirical policy quality. It supports the claim that many official Stage-1 contracts were checked against the paper, while the current reproduction still cannot claim a high-quality teacher or paper-level closed-loop controller.

## Next Step

Patch or explicitly label the 5/6 multi-source metadata mismatch, then run focused Stage-1 diagnosis: per-motion checkpoint evaluation, adaptive sampling `K=3` ablation, ankle-offset ablation, reset-sampling checks, and MuJoCo adapter normalization/order checks.

## Git Commit

Pending.

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
