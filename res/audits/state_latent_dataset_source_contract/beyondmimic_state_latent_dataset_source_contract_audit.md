# BeyondMimic State-Latent Dataset Source Contract Audit

- Status: `blocked_state_latent_dataset_source_uses_policy_obs_and_missing_rollout_state`
- Rows: `10` pass `4` blocked `6`
- Current paper-contract dataset state source: `policy_obs in local paper-contract best-teacher rollout shards`
- Current dims: obs/state `160`, token `192`
- Expected dims: hybrid state `99`, projected state `163`, token `131` or `195` with latent `32`
- Teacher rollout missing fields: `['body_lin_vel_w', 'body_pos_w', 'root_ang_vel_w', 'root_lin_vel_w', 'root_pos_w', 'root_quat_w_or_rot']`
- Permission: `{"allowed_next_step": "modify teacher rollout collection to save raw root/body world state, then rebuild continuous accepted 99-D hybrid-state windows", "start_downstream_vae_training": false, "start_guided_closed_loop_video_generation": false, "start_state_latent_diffusion_training": false, "use_existing_policy_obs_state_latent_dataset_for_long_training": false}`

## Blocking Rows

### Current paper-contract state-latent dataset is not 160-D policy observation
- Observed: state_source='policy_obs in local paper-contract best-teacher rollout shards', obs_dim=160, token_dim=192
- Required fix: Rebuild state-latent shards from raw simulator/root/body state, not from Stage-1 policy_obs.

### Teacher rollout shards contain raw world-state fields required to construct paper hybrid state
- Observed: inspected_shards=2, missing_required_fields=['body_lin_vel_w', 'body_pos_w', 'root_ang_vel_w', 'root_lin_vel_w', 'root_pos_w', 'root_quat_w_or_rot'], keys=[['policy_obs', 'critic_obs', 'actions', 'rewards', 'dones', 'timeouts', 'motion_time_steps', 'final_policy_obs', 'final_critic_obs', 'rank', 'world_size', 'seed'], ['policy_obs', 'critic_obs', 'actions', 'rewards', 'dones', 'timeouts', 'motion_time_steps', 'final_policy_obs', 'final_critic_obs', 'rank', 'world_size', 'seed']]
- Required fix: Modify teacher rollout collection to save root/body world states and velocities before building state-latent data.

### State-latent builder does not read policy_obs as the state token
- Observed: builder_reads_policy_obs=True, wrapper_overrides_state_source_to_policy_obs=True
- Required fix: Replace policy_obs encoding path with explicit hybrid-state construction and record schema version.

### Window index excludes done/reset discontinuities and implements paper 5s rejection
- Observed: done_count_in_source_shards=47200, window_count=285696, builder_has_done_rejection_filter=False
- Required fix: Implement accepted-episode filtering and reject/omit all windows crossing dones/resets before diffusion training.

### OU-noise collection and symmetry augmentation are recorded in the trainable dataset protocol
- Observed: ou_collection_recorded=False, symmetry_recorded=False
- Required fix: Add explicit OU perturbation metadata and symmetry augmentation outputs for train/val/test splits.

### Diffusion training scripts consume paper hybrid/projected state, not policy_obs
- Observed: diffusion_reads_policy_obs=True
- Required fix: Gate full diffusion training until scripts read corrected state-latent arrays instead of source_shard['policy_obs'].
