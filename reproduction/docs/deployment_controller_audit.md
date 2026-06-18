# Deployment Controller Audit

This document records the current Level B deployment/sim-to-sim evidence for the official
`motion_tracking_controller` repository.

## Work Copy

- Raw official source:
  `/mnt/infini-data/test/BeyondMimic/download/official/motion_tracking_controller`
- Working copy:
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/motion_tracking_controller`
- Commit:
  `cbdb4a80d5ea506b2045bdd39cdfb4058084aeb4`
- Static audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.tsv`

The raw checkout reports modified files, so reproduction work treats the copied tree plus file hashes as the auditable
source state rather than editing the raw download.

## Confirmed Static Evidence

- ROS package dependencies from `package.xml`:
  `ament_cmake`, `ament_cmake_auto`, `legged_rl_controllers`, `legged_bringup`, `rosbag2-storage-mcap`.
- CMake uses `ament_cmake_auto`, links Eigen3, exports a controller plugin, and installs `config` and `launch`.
- `config/g1/controllers.yaml` sets:
  - controller manager update rate: `500 Hz`
  - walking controller: `motion_tracking_controller/MotionTrackingController`
  - walking controller update rate: `50 Hz`
  - standby joint/default/kp/kd vector lengths: `29`
- MuJoCo launch supports `policy_path`, `wandb_path`, `start_step`, `robot_type`, and external position correction
  arguments, uses `mujoco_sim_ros2`, and resolves `unitree_description`.
- Real launch supports the same policy/start-step path, starts `ros2_control_node`, records MCAP bags, and requires a
  network interface for the Unitree system.

## ONNX Export / Inference Contract

The official `whole_body_tracking` exporter writes a motion-aware ONNX with:

- Inputs: `obs`, `time_step`.
- Outputs: `actions`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`.
- Metadata keys:
  `run_path`, `joint_names`, `joint_stiffness`, `joint_damping`, `default_joint_pos`, `command_names`,
  `observation_names`, `observation_history_lengths`, `action_scale`, `anchor_body_name`, `body_names`.

The official C++ controller consumes this contract:

- `MotionOnnxPolicy` sends `time_step` into the ONNX model and reads the reference-motion outputs.
- `MotionOnnxPolicy::parseMetadata()` reads `anchor_body_name` and `body_names`.
- `MotionCommandTerm` aligns the first reference motion frame with the current robot position and yaw.
- Motion observations use anchor position, Rot6D anchor orientation, local body positions, and local body orientations.

Machine-readable ONNX contract inspection script:

`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/inspect_motion_onnx_contract.py`

Exporter-to-controller contract matrix:

`/mnt/infini-data/test/BeyondMimic/res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json`

This static audit cross-checks the official `whole_body_tracking` exporter against the C++ consumer and records that a
real BeyondMimic motion-policy ONNX still has not been exported in this environment.

## Reference ONNX Caveat

`/mnt/infini-data/test/BeyondMimic/download/dependencies/unitree_bringup/config/g1/policy.onnx`
exists and is hashed in the audit, but it is a `unitree_bringup` reference policy, not a proven BeyondMimic exported
motion-tracking policy. It must not be used as evidence that BeyondMimic ONNX export has passed.

Machine check:

- Summary:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_tracking_controller_audit/unitree_g1_policy_onnx_contract.json`
- Required-contract check log:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/unitree_g1_policy_motion_contract_check.log`

Result: the reference ONNX has input `obs` and output `actions`, but lacks `time_step`, motion reference outputs, and
metadata keys `run_path`, `observation_history_lengths`, `anchor_body_name`, and `body_names`.

## Not Executed Yet

Official MuJoCo sim-to-sim and real-robot deployment have not been run on this host.

Reasons:

- `motion_tracking_controller` targets ROS 2 Jazzy / Ubuntu Noble, while this host is Ubuntu 20.04.5.
- Real Unitree G1 hardware is unavailable and must remain out of scope.
- The ONNX policy expected by this controller should be exported from the tracking stack after IsaacLab/Kit smoke and
  local tracking smoke pass.

## Future Gate

After IsaacLab/Kit and local tracking smoke pass:

1. Export a BeyondMimic motion policy ONNX from `whole_body_tracking`.
2. Validate that the ONNX contains the metadata and outputs listed above.
3. In a ROS 2 Jazzy/Noble workspace with `legged_control2`, `unitree_bringup`, and `unitree_description`, build
   `motion_tracking_controller`.
4. Run MuJoCo sim-to-sim only:

```bash
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=/absolute/path/to/policy.onnx
```

No real-robot command should be run in this reproduction.
