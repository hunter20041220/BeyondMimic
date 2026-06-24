# Progress Update

## Goal

Before launching more teacher/VAE/diffusion training, repair the state-latent dataset code path so paper-contract runs cannot silently train diffusion on 160-D Stage-1 `policy_obs` or reset-spliced rollout windows.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/src/beyondmimic_reimpl/state.py`
- `reproduction/tests/test_core_math.py`
- `reproduction/tests/test_reimpl_package_api.py`
- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
- `reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`

## Files Modified

- `reproduction/src/beyondmimic_reimpl/state.py`
- `reproduction/tests/test_core_math.py`
- `reproduction/tests/test_reimpl_package_api.py`
- `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
- `reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `res/tests/core_math_unit_tests/core_math_unit_tests.json`
- `res/tests/core_math_unit_tests/core_math_unit_tests.tsv`
- `res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json`
- `res/tests/reimpl_package_api_tests/reimpl_package_api_tests.tsv`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.tsv`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.md`

## Commands Run

```bash
python -m py_compile reproduction/src/beyondmimic_reimpl/state.py reproduction/tests/test_core_math.py reproduction/tests/test_reimpl_package_api.py reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py reproduction/scripts/reproduction_master_audit.py
python reproduction/tests/test_core_math.py
python reproduction/tests/test_reimpl_package_api.py
python reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py
```

## Results

- Added `valid_contiguous_window_mask()` to reject state-latent windows that cross `done`, optional `timeout`, or discontinuous `motion_time_steps`.
- Added unit coverage for reset/jump rejection so discontinuous rollout snippets cannot be treated as continuous training windows.
- Extended the state-latent builder with a paper-contract mode that requires raw root/body world-state fields and builds 99-D hybrid state windows using each window's own current frame.
- Added window-level `state_windows` / `latent_windows` outputs for paper-contract diffusion, avoiding the previous global-shard current-frame mistake.
- Changed the paper-contract state-latent wrapper so it forces `paper_hybrid`, raw-state mode, paper-contract VAE, and done-discontinuity rejection.
- Changed the paper-contract VAE wrapper to call the actual paper-contract VAE trainer rather than the old action-VAE trainer.
- Changed the paper-contract Transformer diffusion loader so it requires `state_windows`/`latent_windows` instead of reading `source_shard["policy_obs"]`.

## Verification

- `test_core_math.py`: `ok`, 27 rows, 0 failed.
- `test_reimpl_package_api.py`: `ok`, 8 rows, 0 failed.
- `beyondmimic_state_latent_dataset_source_contract_audit.py`: still `blocked`, now 6 pass / 4 blocked.

The remaining block is expected: current generated state-latent artifacts are still old `policy_obs` artifacts, and old teacher rollout shards do not contain the raw world-state fields needed for paper hybrid-state reconstruction.

## Failed / Blocked Items

- Existing paper-contract state-latent dataset still reports `state_source='policy_obs in local paper-contract best-teacher rollout shards'`, `obs_dim=160`, `token_dim=192`.
- Existing paper-contract teacher rollout shard probe still finds no raw root/body world-state arrays.
- Existing source shards still include nonzero done/reset events; the old generated window index was not rebuilt with the new discontinuity filter.
- OU perturbation collection and sagittal symmetry augmentation are still not recorded as a trainable dataset protocol.

## Effect on English Reading Report

This gives a stronger and more honest explanation for why current MuJoCo teacher/VAE/diffusion videos collapse to forward-leaning or weak standing behavior: the downstream diffusion path was previously allowed to learn from policy observations and reset-spliced windows rather than the paper's continuous accepted hybrid state-latent trajectories.

## Next Step

Recollect teacher rollout shards with raw root/body world-state fields from the current best teacher, then rebuild the paper-contract VAE and state-latent dataset using the new `paper_hybrid` path before any long diffusion or guidance training.

## Git Commit

Pending at the time this progress note was written.
