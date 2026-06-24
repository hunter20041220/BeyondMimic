# Progress Update

## Goal

Before starting any new long teacher/VAE/diffusion/guidance training, re-check the BeyondMimic paper/appendix contracts and repair local gates that could allow another weak forward-leaning controller chain to be trained by mistake.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/known_limitations.md`
- `res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json`
- `res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
- `res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
- `res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py`
- `reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`

## Files Modified

- `reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py`
- `reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/progress/20260624_080414_pretraining_formula_gate_alignment.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_training_run.py reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_training_run.py reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_training_run.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
python3 reproduction/scripts/beyondmimic_state_latent_dataset_source_contract_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 reproduction/scripts/paper_formula_code_trace_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
python3 reproduction/tests/test_core_math.py
python3 reproduction/tests/test_reimpl_package_api.py
```

The standard manifest/report/verification commands were started after the edits and their refreshed outputs are recorded in the final response for this turn.

## Results

- The code/formula audit now separates three different states:
  - paper-contract hybrid-state builder exists;
  - old generated trainable state-latent dataset is still blocked;
  - teacher rollout shards still need raw root/body world-state fields before downstream training.
- The paper-contract VAE row no longer repeats the obsolete claim that the script is still a single-hidden-dim model. It now records the current `[2048, 1024, 512]` hidden dims and `GRAD_ACCUM_STEPS=15`.
- Future Stage-1 paper-contract training wrappers now explicitly set `adaptive_kernel_size=3` through `BM_ADAPTIVE_KERNEL_SIZE`, matching the paper supplement look-back `u={0,1,2}` instead of silently inheriting the upstream public-code default `1`.
- Master audit was updated to expect the corrected semantics: code-level hybrid-state support is present, but the generated dataset gate remains blocked until regenerated.

## Verification

- `py_compile`: passed.
- Core math tests: passed, `27` rows, `0` failed.
- Reimplementation API tests: passed, `8` rows, `0` failed.
- `beyondmimic_code_formula_appendix_contract_audit`: status remains `blocked_code_formula_appendix_contract_has_required_fixes_before_training`, now with `17` rows.
- `beyondmimic_state_latent_dataset_source_contract_audit`: expected blocked status; old generated data still uses policy obs and lacks raw-state shards.
- `beyondmimic_pretraining_hard_gate_audit`: expected blocked status; no long downstream training is allowed from the current teacher chain.
- `reproduction_master_audit`: restored to `ok` after updating the audit assertion to the new semantics.

## Failed / Blocked Items

- The current generated state-latent dataset is still not ready for paper-facing diffusion training.
- Existing teacher rollout shards still lack the raw world-state fields required for the paper hybrid state.
- Guidance is still offline/proxy rather than receding-horizon closed-loop MuJoCo control.
- MuJoCo videos still do not prove native no-root-assist policy/VAE/diffusion control.
- Current teacher quality is still blocked for downstream VAE/diffusion training.

## Effect on English Reading Report

This strengthens the report's reproducibility section: the project can now honestly say that the paper-contract VAE and hybrid-state code paths were audited, while still explaining why existing videos fail and why long downstream training was not restarted prematurely.

## Next Step

Collect fresh teacher rollout shards with raw root/body states and continuous accepted windows, then rebuild the hybrid state-latent dataset. Only after that should VAE/diffusion/guidance training or new success videos be attempted.

## Git Commit

Pending.

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
