# Progress Update

## Goal

Audit the BeyondMimic teacher/VAE/diffusion/guidance/video chain against the paper formulas and supplement hyperparameters before any further training or success-video claims. The immediate concern was that teacher/VAE/diffusion videos lean forward or stay in generic standing poses instead of reproducing lifted-leg or walking reference poses.

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
- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py`
- `reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py`
- `reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.json`
- `res/level_c/paper_contract_teacher_rollout_vae_training/level_c_paper_contract_teacher_rollout_vae_training.json`
- `res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json`
- `res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json`
- `res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json`
- `res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json`

## Files Modified

- Updated `reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Added `res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
- Added `res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.tsv`
- Added `res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.md`
- Added this progress update.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
```

## Results

The model-chain audit status is:

`blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready`

Route decision:

- Disabled for success claims: legacy `resource_adjusted_teacher_rollout_vae`, legacy `resource_adjusted_state_latent_diffusion`, and the current clean-walk/single-leg video chain.
- Preferred local diagnostic route: Stage-1 tracking parameter contract audit, paper-contract VAE, paper-contract state-latent dataset, paper-contract denoising output, and paper-contract offline guidance.
- Still blocked because Stage-1 teacher quality has not passed, the state-latent dataset still uses local `policy_obs` rather than the full paper hybrid state, the current preferred denoiser is inherited from an MLP implementation rather than the paper Transformer, guidance is offline rather than receding-horizon closed-loop control, and existing videos use blending/root assist or weak teacher actions.

Important correction versus earlier interpretation:

- The older resource-adjusted VAE is indeed wrong for paper-contract claims because it encodes `obs+action`.
- A newer `paper_contract_teacher_rollout_vae_training` path does repair the main VAE interface: encoder uses reference intent terms and decoder uses proprioception plus latent.
- That repair is useful but not sufficient for success videos because the teacher source and closed-loop gates remain weak.

## Verification

Completed:

```bash
python3 -m py_compile reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
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

`logs/verification/model_chain_paper_contract_20260624_045629/`

Refreshed summary:

- `res/artifact_manifest/artifact_manifest.json`: status `ok`, 1822 artifacts, missing 0.
- `res/master_audit/reproduction_master_audit.json`: status `ok`, failed artifacts `[]`.

## Failed / Blocked Items

- Do not use existing weak teacher/VAE/diffusion videos as successful single-leg or walking evidence.
- Do not train final downstream VAE/diffusion from the current teacher until the teacher gate passes.
- Do not use the legacy resource-adjusted VAE/diffusion route for success claims.
- The current guidance evidence is offline cost-gradient evidence, not closed-loop task control.

## Effect on English Reading Report

This gives the report a precise explanation for why the current videos are poor: the project has progressed from generic diagnostics to a better paper-contract VAE route, but it still lacks a high-quality teacher, paper Transformer diffusion implementation, and closed-loop guided control. It prevents overstating the current MuJoCo videos while preserving the real engineering progress.

## Next Step

Patch or replace the local state-latent diffusion route with the paper Transformer/per-token denoising implementation, then focus Stage-1 teacher diagnosis and high-throughput retraining only after the code contract is ready.

## Git Commit

Pending.

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
