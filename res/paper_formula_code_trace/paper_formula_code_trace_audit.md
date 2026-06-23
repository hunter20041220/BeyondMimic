# Paper Formula Code Trace Audit

- Status: `ok`
- Rows: `11`
- Missing evidence rows: `0`
- Status counts: `{"covered_debug_architecture": 2, "covered_debug_formula": 3, "covered_protocol_only": 1, "covered_public_data_formula": 3, "covered_static_source": 1, "indexed_blocked_or_partial": 1}`
- Source counts: `{"api_test_row_count": 8, "core_math_test_row_count": 26, "core_test_required_count": 20, "latex_equation_count": 8, "latex_experiment_setting_count": 14, "paper_table_value_mismatch_rows": 0, "paper_table_value_rows": 58, "reimpl_symbol_row_count": 37}`

## Trace Rows
- `eq_root_current_frame`: `covered_debug_formula`; Formula and paper-state window checks exist; no trained state-latent rollout dataset uses this end-to-end.
- `eq_body_local_frame`: `covered_debug_formula`; Body local positions match; audit records remaining non-paper-exact debug boundaries.
- `eq_ou_noise`: `covered_debug_formula`; OU mechanics are tested/debugged; no true VAE rollout perturbation data exists.
- `eq_joystick_cost`: `covered_public_data_formula`; Formula/gradient tests and full-split public-data offline/reverse joystick guidance metrics exist; no closed-loop joystick rollout exists.
- `eq_waypoint_cost`: `covered_public_data_formula`; Cost behavior is audited in formula/debug probes and full-split public-data offline/reverse waypoint guidance metrics; paper scene protocol and no closed-loop videos remain missing.
- `eq_sdf_cost`: `covered_public_data_formula`; Barrier formula is implemented/tested and full-split public-data offline/reverse obstacle guidance metrics exist; no closed-loop obstacle trial videos exist.
- `setting_tracking_ppo_domain_reward`: `covered_static_source`; Official config/table values match where public; live Kit rollout/training is blocked.
- `setting_vae_hyperparameters`: `covered_debug_architecture`; Architecture/loss settings are debug-matched; no true DAgger-trained checkpoint exists.
- `setting_diffusion_hyperparameters`: `covered_debug_architecture`; Architecture/schedule probes exist; no long diffusion training or trained checkpoint exists.
- `setting_deployment_protocol`: `covered_protocol_only`; Deployment claims are indexed and debug-budgeted; no TensorRT/async/Mini-PC deployment exists.
- `setting_results_claims`: `indexed_blocked_or_partial`; Claims are compared or blocked; Fig.5/Fig.6 paper reproduction remains unavailable.
