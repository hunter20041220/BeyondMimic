# MuJoCo Control Contract Audit

- Status: `blocked_mujoco_control_semantics_not_native_policy_control`
- Generated: `2026-06-24T00:19:29.609417+00:00`
- Conclusion: the MuJoCo video stack contains useful local visualization assets, but it is not yet a native BeyondMimic control reproduction path.
- 当前不得声称完整复现 BeyondMimic，也不得把 root-assist/IK/absolute-target MuJoCo 视频作为 teacher/VAE/diffusion 成功结果。

## Blocking Gaps

- `mujoco_floor_material_gap`: The official IsaacLab terrain uses static/dynamic friction 1.0 and randomized material buckets during training. The local MuJoCo base XML floor records friction around 0.6 in at least one asset path, and material randomization is not reproduced in the video controller.
- `mujoco_pd_video_control_semantics`: The MuJoCo PD video script directly drives absolute joint targets or IK-fitted targets with 29 position actuators. It does not reconstruct the IsaacLab observation manager, run PPO/VAE/diffusion to produce normalized actions, or apply theta0 + alpha * action as the native control path.
- `mujoco_root_assist_semantics`: The script applies an external pelvis force/torque root-assist stabilizer by default. This is useful for centered videos but prevents any claim of unassisted humanoid balance/control.
- `mujoco_trace_mesh_video_semantics`: Trace mesh videos fit MuJoCo qpos to captured body positions via IK or write qpos for reference replay. They are visualization assets, not closed-loop policy/diffusion controllers.
- `current_report_video_claims`: Current summaries already mark clean-walk/MuJoCo outputs as diagnostic/local evidence, not paper-level success.

## Component Rows

### paper_action_contract
- Status: `available`
- Matches paper/control contract: `True`
- Notes: Paper action is normalized policy output mapped to PD setpoint theta_sp = theta0 + alpha * action.

### official_stage1_pd_action_scale
- Status: `ok`
- Matches paper/control contract: `True`
- Notes: Official G1 action scale, stiffness, damping, effort limits, and armature are available and match the paper formula audit.

### official_stage1_material_randomization
- Status: `available`
- Matches paper/control contract: `True`
- Notes: Official IsaacLab training randomizes contact coefficients, default joint positions, torso COM, and pushes as described in the paper.

### mujoco_pd_xml_parameter_patch
- Status: `available`
- Matches paper/control contract: `True`
- Notes: The local MuJoCo PD XML is numerically patched with official 29-DoF stiffness/damping/armature/effort settings, but this only covers actuator parameters.

### mujoco_floor_material_gap
- Status: `diagnostic_gap`
- Matches paper/control contract: `False`
- Notes: The official IsaacLab terrain uses static/dynamic friction 1.0 and randomized material buckets during training. The local MuJoCo base XML floor records friction around 0.6 in at least one asset path, and material randomization is not reproduced in the video controller.

### mujoco_nominal_floor_friction
- Status: `available`
- Matches paper/control contract: `True`
- Notes: The generated MuJoCo PD XML should use nominal floor friction 1.0 to match the official flat terrain. This does not reproduce IsaacLab's robot material randomization.

### mujoco_pd_video_control_semantics
- Status: `diagnostic_not_native_policy_control`
- Matches paper/control contract: `False`
- Notes: The MuJoCo PD video script directly drives absolute joint targets or IK-fitted targets with 29 position actuators. It does not reconstruct the IsaacLab observation manager, run PPO/VAE/diffusion to produce normalized actions, or apply theta0 + alpha * action as the native control path.

### mujoco_root_assist_semantics
- Status: `blocks_success_claim`
- Matches paper/control contract: `False`
- Notes: The script applies an external pelvis force/torque root-assist stabilizer by default. This is useful for centered videos but prevents any claim of unassisted humanoid balance/control.

### mujoco_trace_mesh_video_semantics
- Status: `kinematic_or_ik_visualization_only`
- Matches paper/control contract: `False`
- Notes: Trace mesh videos fit MuJoCo qpos to captured body positions via IK or write qpos for reference replay. They are visualization assets, not closed-loop policy/diffusion controllers.

### current_report_video_claims
- Status: `diagnostic_claims_recorded`
- Matches paper/control contract: `False`
- Notes: Current summaries already mark clean-walk/MuJoCo outputs as diagnostic/local evidence, not paper-level success.

## Next Gate

Before producing final single-leg/walk success videos, implement or verify a native MuJoCo/IsaacLab adapter that uses the policy/VAE/diffusion output action as normalized action, applies `theta0 + alpha * action`, disables root assist, and logs observation/action contract consistency.
