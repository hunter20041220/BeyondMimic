# MuJoCo Native Observation Adapter Contract

- Status: `blocked_native_mujoco_observation_adapter_not_validated`
- Generated: `2026-06-24T05:15:57.833648+00:00`
- Scope: official 160-D observation contract and native MuJoCo reconstruction gate; no physics rollout.
- 结论：当前不能把任意 160 维拼接 obs 喂给 IsaacLab PPO actor 后声称 MuJoCo native policy rollout 成功。
- 当前不得声称完整复现 BeyondMimic；本审计只给出后续修 native obs/action adapter 的逐项合同。

## Failed / Blocking Checks

- `native_adapter_validated_against_isaaclab_observation_manager`
- `native_adapter_validated_against_deployment_controller`
- `native_adapter_all_terms_numerically_validated`
- `native_adapter_has_no_root_assist_rollout_success`
- `runtime_observation_all_slices_pass`
- `runtime_observation_anchor_pose_matches_isaaclab`
- `runtime_observation_any_candidate_model_anchor_frame_matches`
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

## Same-State Formula Parity

- Status: `ok_same_state_observation_formula_slices_match_official_sample_but_mujoco_runtime_pending`
- Claim level: `same-state formula parity only; no MuJoCo runtime rollout claim`
- 解释：这里比较的是同一个 IsaacLab captured state 下，本地 NumPy 公式重算值 vs. official critic/noise-free shared observation terms；它不是 MuJoCo runtime rollout。
- `command` dim=58 max_abs_error=0.000000e+00 passed=`True`
- `motion_anchor_pos_b` dim=3 max_abs_error=1.462929e-09 passed=`True`
- `motion_anchor_ori_b` dim=6 max_abs_error=1.143636e-07 passed=`True`
- `base_lin_vel` dim=3 max_abs_error=0.000000e+00 passed=`True`
- `base_ang_vel` dim=3 max_abs_error=0.000000e+00 passed=`True`
- `joint_pos` dim=29 max_abs_error=2.700835e-08 passed=`True`
- `joint_vel` dim=29 max_abs_error=0.000000e+00 passed=`True`
- `actions` dim=29 max_abs_error=0.000000e+00 passed=`True`

## MuJoCo Runtime Injected-State Parity

- Status: `blocked_mujoco_injected_state_observation_runtime_parity_mismatch`
- Claim level: `MuJoCo injected-state observation adapter audit only; no policy rollout, no training, no video`
- 解释：这里加载 MuJoCo G1 XML，把 IsaacLab captured root/joint/qvel 状态注入 MuJoCo，执行 `mj_forward` 后再构造 160-D observation；它仍不是 policy rollout。
- `command` dim=58 max_abs_error=0.000000e+00 passed=`True`
- `motion_anchor_pos_b` dim=3 max_abs_error=5.219149e-03 passed=`False`
- `motion_anchor_ori_b` dim=6 max_abs_error=3.175157e-01 passed=`False`
- `base_lin_vel` dim=3 max_abs_error=3.092577e-08 passed=`True`
- `base_ang_vel` dim=3 max_abs_error=8.795058e-08 passed=`True`
- `joint_pos` dim=29 max_abs_error=2.700835e-08 passed=`True`
- `joint_vel` dim=29 max_abs_error=0.000000e+00 passed=`True`
- `actions` dim=29 max_abs_error=0.000000e+00 passed=`True`
- Anchor frame diagnostic: position_m=`0.0005856326823376577`, quat_sign_invariant=`0.1336460034751546`
- Candidate MJCF torso frame errors:
  - `mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.0005856326823376577`
  - `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.0005856326823376577`
  - `mujoco_mp4/assets/work_g1/pbhc_g1/g1_29dof_rev_1_0.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.043775235909656596`
  - `mujoco_mp4/assets/work_g1/pbhc_g1/g1_29dof_rev_1_0_with_toe.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.0005856326823376577`
  - `download/reference_code/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.0005856326823376577`
  - `download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.0005856326823376577`
  - `download/reference_code/ASAP/humanoidverse/data/robots/g1/g1_29dof_old.xml` loaded=`True` torso_quat_err=`0.1336460034751546` torso_pos_err=`0.009492369703265169`

## MuJoCo Torso Frame Offset Hypothesis

- Status: `blocked_torso_frame_offset_hypothesis_single_terminated_sample_requires_walk_validation`
- Claim level: `single-sample MuJoCo/IsaacLab torso frame offset hypothesis only; no rollout, no training`
- Raw anchor pos/orient error: `0.005219148958698112` / `0.3175156624836241`
- Corrected anchor pos/orient error: `1.4629287503620247e-09` / `1.1436359326211232e-07`
- Candidate quaternion offset: `[0.981741168614011, 0.10665968775146022, -0.1357829358628406, -0.07981843888240901]`
- Sample terminated after zero step: `True`
- 解释：该 offset 只支持 frame-mismatch 假设；因为样本已 terminated，不能直接作为 rollout adapter 修复。

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
- Validate that builder numerically against the captured IsaacLab observation_manager sample for the same reset state, motion time_steps, and last_action.
- Resolve the MuJoCo MJCF versus IsaacLab USD/URDF torso_link frame mismatch before feeding native MuJoCo observations to the actor.
- Capture a non-terminated low-dynamic walk observation_manager sample and verify whether the same MuJoCo-to-IsaacLab torso frame offset restores anchor terms.
- Validate frame-alignment semantics against motion_tracking_controller worldToInit_/Pinocchio local-frame formulas.
- Validate body-frame base velocity, Rot6D column ordering, default_joint_pos source, and previous-action semantics with finite numeric fixtures.
- Use the no-action-clipping MuJoCo actuator XML from the action adapter audit for any later no-root-assist policy videos.
- Combine the validated obs builder with the native action adapter fixture, disable root assist, and log raw/clipped normalized actions plus PD setpoints.
- Only after the above gates pass should a native MuJoCo PPO rollout video be treated as motion-control evidence.

## Claim Boundary

Passing this audit in the future would only clear the observation-adapter precondition. It would still not prove teacher quality, VAE reconstruction quality, diffusion closed-loop stability, guided task success, true Isaac rendering, or real-robot deployment.
