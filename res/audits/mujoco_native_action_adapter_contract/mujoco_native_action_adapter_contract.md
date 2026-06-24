# MuJoCo Native Action Adapter Contract

- Status: `ok_native_action_adapter_formula_and_no_clip_ctrlrange_patch_ready`
- Generated: `2026-06-24T01:10:38.930280+00:00`
- Scope: formula/order/default-pose fixture only; no physics rollout and no success video claim.
- Formula: `theta_sp = theta_default + action_scale * normalized_action`.
- 当前不得声称完整复现 BeyondMimic；本审计只证明 action-to-PD 公式和顺序可作为后续 native rollout 的前置条件。

## Checks

- `paper_formula_available`: `True` - Paper text contains the normalized setpoint/action-scale formula.
- `isaaclab_affine_joint_action_semantics_available`: `True` - IsaacLab JointPositionAction applies raw_action * scale + default offset, and the RSL-RL wrapper can clip normalized actions.
- `official_action_scale_rows_29`: `True` - The official G1 action-scale audit expands all 29 controllable joints.
- `controller_default_pose_available`: `True` - Deployment controller standby default_position is available and nonzero.
- `zero_default_fallback_not_used`: `True` - The adapter fixture uses an official default pose source, not all-zero fallback.
- `mujoco_mapping_order_matches_action_rows`: `True` - MuJoCo mapping joint order matches action-scale row order.
- `pd_actuator_order_matches_action_rows`: `True` - MuJoCo position actuator order matches action-scale row order.
- `deployment_no_action_clip_xml_written`: `True` - A deployment-compatible MuJoCo XML patch was written for no pre-physics action clipping.
- `patched_pd_actuator_order_matches_action_rows`: `True` - The patched MuJoCo actuator order still matches action-scale row order.
- `zero_action_returns_default_pose`: `True` - A zero normalized action maps exactly to theta_default.
- `unit_action_delta_matches_action_scale`: `True` - A +1 action changes each joint by +action_scale.
- `negative_unit_action_delta_matches_action_scale`: `True` - A -1 action changes each joint by -action_scale.
- `large_action_clips_to_unit_scale`: `True` - A larger action clips to the configured normalized-action bound for the fixture.
- `unit_targets_inside_mujoco_ctrlrange`: `True` - The fixture's unit action setpoints stay inside the patched MuJoCo actuator ctrlrange.
- `does_not_claim_rollout_or_success`: `True` - This audit does not run physics or claim policy/VAE/diffusion success.

## Default Pose Sources

- Selected source: `official_motion_tracking_controller_standby_controller.default_position`
- IsaacLab vs deployment max abs delta: `0.032999999999999974`
- Delta note: The official deployment standby default differs from the IsaacLab InitialStateCfg mainly at ankle pitch (-0.33 versus -0.363 rad). A real rollout should prefer the exported ONNX metadata default_joint_pos when available.

## Original Ctrlrange Violations

- `left_ankle_roll_joint`: raw setpoint range `[-0.438577, 0.438577]` exceeds MuJoCo ctrlrange `[-0.261800, 0.261800]` by max `0.176777` rad.
- `right_ankle_roll_joint`: raw setpoint range `[-0.438577, 0.438577]` exceeds MuJoCo ctrlrange `[-0.261800, 0.261800]` by max `0.176777` rad.

## Patched Ctrlrange

- Patched XML: `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/g1_clean_walk_control_suite_pd_no_action_clip.xml`
- Written: `True`
- Expanded joints: `2`
- Remaining patched violations: `0`

## Rollout-Readiness Warnings

- The original MuJoCo actuator ctrlrange clipped legal unit-action setpoints for left_ankle_roll_joint, right_ankle_roll_joint. This audit now writes a no-action-clipping XML patch; future rollout/video code must use the patched range or regenerate the same policy.
- IsaacLab InitialStateCfg and motion_tracking_controller standby default_position differ at ankle pitch by about 0.033 rad; rollout code should prefer exported ONNX metadata when available.

## Claim Boundary

Passing this audit does not mean the teacher policy can walk, the VAE reconstructs lifted-leg poses, or the diffusion/guidance controller is stable. It only means future native MuJoCo/Isaac rollout code has a verified action-setpoint adapter to call.
