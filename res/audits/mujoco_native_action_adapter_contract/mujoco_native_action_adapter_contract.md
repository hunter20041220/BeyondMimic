# MuJoCo Native Action Adapter Contract

- Status: `ok_native_action_adapter_formula_contract_ready_not_rollout`
- Generated: `2026-06-23T21:34:36.362645+00:00`
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
- `zero_action_returns_default_pose`: `True` - A zero normalized action maps exactly to theta_default.
- `unit_action_delta_matches_action_scale`: `True` - A +1 action changes each joint by +action_scale.
- `negative_unit_action_delta_matches_action_scale`: `True` - A -1 action changes each joint by -action_scale.
- `large_action_clips_to_unit_scale`: `True` - A larger action clips to the configured normalized-action bound for the fixture.
- `unit_targets_inside_mujoco_ctrlrange`: `False` - The fixture's unit action setpoints stay inside MuJoCo actuator ctrlrange.
- `does_not_claim_rollout_or_success`: `True` - This audit does not run physics or claim policy/VAE/diffusion success.

## Default Pose Sources

- Selected source: `official_motion_tracking_controller_standby_controller.default_position`
- IsaacLab vs deployment max abs delta: `0.032999999999999974`
- Delta note: The official deployment standby default differs from the IsaacLab InitialStateCfg mainly at ankle pitch (-0.33 versus -0.363 rad). A real rollout should prefer the exported ONNX metadata default_joint_pos when available.

## Rollout-Readiness Warnings

- MuJoCo actuator ctrlrange clips unit action setpoints for ankle_roll joints; native rollout code must record raw setpoint versus MuJoCo-clipped setpoint.
- IsaacLab InitialStateCfg and motion_tracking_controller standby default_position differ at ankle pitch by about 0.033 rad; rollout code should prefer exported ONNX metadata when available.

## Claim Boundary

Passing this audit does not mean the teacher policy can walk, the VAE reconstructs lifted-leg poses, or the diffusion/guidance controller is stable. It only means future native MuJoCo/Isaac rollout code has a verified action-setpoint adapter to call.
