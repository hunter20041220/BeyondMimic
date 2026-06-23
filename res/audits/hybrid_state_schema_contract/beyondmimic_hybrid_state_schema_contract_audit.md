# Hybrid State Schema Contract Audit

- Status: `blocked_hybrid_state_schema_ready_but_trainable_dataset_missing`
- Schema: `{"body_feature_dim": 84, "coefficient": 6, "gaussian_rows": 64, "projected_dim": 163, "root_dim": 15, "slices": {"body_lin_vel_local_root_frame": [57, 99], "body_pos_local_root_frame": [15, 57], "root_ang_vel_rel_current_frame": [12, 15], "root_lin_vel_rel_current_frame": [9, 12], "root_pos_rel_current_frame": [0, 3], "root_rot6d_rel_current_frame": [3, 9]}, "state_dim": 99, "target_body_count": 14}`
- Checks: `{"body_feature_dim_84": true, "does_not_allow_long_training_yet": true, "does_not_claim_goal_complete": true, "emphasis_coefficient_6": true, "gaussian_emphasis_rows_64": true, "hybrid_state_dim_99": true, "paper_sources_readable": true, "projected_state_dim_163": true, "projection_pseudoinverse_roundtrip": true, "rejects_160d_policy_obs": true, "root_state_dim_15": true, "trainable_dataset_rebuilt_with_corrected_schema": false}`

## Required Fix Before Long Training

Rebuild teacher rollout state-latent windows from continuous MuJoCo/IsaacLab rollouts using the paper hybrid state representation, then rerun the code/formula appendix gate.
