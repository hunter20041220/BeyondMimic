# MuJoCo Native Observation Adapter Contract

- Status: `blocked_native_mujoco_observation_adapter_not_validated`
- Generated: `2026-06-24T00:35:52.583254+00:00`
- Scope: official 160-D observation contract and native MuJoCo reconstruction gate; no physics rollout.
- 结论：当前不能把任意 160 维拼接 obs 喂给 IsaacLab PPO actor 后声称 MuJoCo native policy rollout 成功。
- 当前不得声称完整复现 BeyondMimic；本审计只给出后续修 native obs/action adapter 的逐项合同。

## Failed / Blocking Checks

- `native_adapter_validated_against_isaaclab_observation_manager`
- `native_adapter_validated_against_deployment_controller`
- `native_adapter_all_terms_numerically_validated`
- `native_adapter_has_no_root_assist_rollout_success`
- `native_action_adapter_rollout_ready`
- `native_action_adapter_ctrlrange_allows_rollout`
- `native_rollout_preconditions_ready`

## Policy Observation Layout

- `0:58` `command` (58D): MotionCommand.command = cat([joint_pos, joint_vel]) | status=`contract_known_not_runtime_validated`
- `58:61` `motion_anchor_pos_b` (3D): subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w) | status=`approximate_probe_exists_not_validated_against_isaaclab`
- `61:67` `motion_anchor_ori_b` (6D): relative anchor orientation via subtract_frame_transforms, then first two rotation-matrix columns | status=`approximate_probe_exists_not_validated_against_isaaclab`
- `67:70` `base_lin_vel` (3D): IsaacLab mdp.base_lin_vel from ArticulationData.root_lin_vel_b | status=`approximate_probe_exists_not_validated_against_isaaclab`
- `70:73` `base_ang_vel` (3D): IsaacLab mdp.base_ang_vel from ArticulationData.root_ang_vel_b | status=`approximate_probe_exists_not_validated_against_isaaclab`
- `73:102` `joint_pos` (29D): mdp.joint_pos_rel | status=`partial_default_pose_warning`
- `102:131` `joint_vel` (29D): mdp.joint_vel_rel | status=`contract_known_not_runtime_validated`
- `131:160` `actions` (29D): mdp.last_action | status=`bug_fixed_in_clean_walk_suite_but_not_global_gate`

## Normalizer Gate

- Checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_paper_contract_ppo_training/resource_adjusted_ppo_20260622_114146_seed20260851/rank_0/model_29999.pt`
- `obs_norm_state_dict_present`: `True`
- `obs_norm__mean_shape`: `[1, 160]`
- `obs_norm__std_shape`: `[1, 160]`
- `obs_norm__mean_trailing_dim_160`: `True`
- `obs_norm__std_trailing_dim_160`: `True`
- `actor_input_dim_160`: `True`
- `actor_output_dim_29`: `True`

## Runtime Validation Matrix

- `command` `0:58`: isaaclab=`False`, deployment=`False`, ready=`False`. wrong phase/time-step or reset-spliced commands can make a good policy chase discontinuous targets
- `motion_anchor_pos_b` `58:61`: isaaclab=`False`, deployment=`False`, ready=`False`. meter-scale fake anchor error can drive the actor into a persistent leaning/recovery posture
- `motion_anchor_ori_b` `61:67`: isaaclab=`False`, deployment=`False`, ready=`False`. rot6D column/order mismatch can make the actor believe the target is rotated even while standing
- `base_lin_vel` `67:70`: isaaclab=`False`, deployment=`False`, ready=`False`. world/body-frame velocity mismatch looks like a constant disturbance and biases recovery actions
- `base_ang_vel` `70:73`: isaaclab=`False`, deployment=`False`, ready=`False`. angular velocity convention mismatch can suppress leg-lift behavior and over-trigger torso stabilization
- `joint_pos` `73:102`: isaaclab=`False`, deployment=`False`, ready=`False`. default-pose mismatch shifts every policy input and can make neutral standing look like crouching/leaning
- `joint_vel` `102:131`: isaaclab=`False`, deployment=`False`, ready=`False`. joint-order or qvel-index mismatch corrupts feedback even if the action order is correct
- `actions` `131:160`: isaaclab=`False`, deployment=`False`, ready=`False`. feeding teacher or reference actions as last_action contaminates VAE/diffusion closed-loop observations

## Required Next Implementation Steps

- Export an official motion policy ONNX with metadata and embedded normalizer, or load the checkpoint obs normalizer exactly.
- Implement a native MuJoCo observation builder that returns the exact eight policy terms and slices in this audit.
- Validate that builder numerically against IsaacLab observation_manager output for the same reset state, motion time_step, and last_action.
- Validate frame-alignment semantics against motion_tracking_controller worldToInit_/Pinocchio local-frame formulas.
- Validate body-frame base velocity, Rot6D column ordering, default_joint_pos source, and previous-action semantics with finite numeric fixtures.
- Resolve or justify MuJoCo ctrlrange clipping in the native action adapter before allowing no-root-assist policy videos.
- Combine the validated obs builder with the native action adapter fixture, disable root assist, and log raw/clipped normalized actions plus PD setpoints.
- Only after the above gates pass should a native MuJoCo PPO rollout video be treated as motion-control evidence.

## Claim Boundary

Passing this audit in the future would only clear the observation-adapter precondition. It would still not prove teacher quality, VAE reconstruction quality, diffusion closed-loop stability, guided task success, true Isaac rendering, or real-robot deployment.
