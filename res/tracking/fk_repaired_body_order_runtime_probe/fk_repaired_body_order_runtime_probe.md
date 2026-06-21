# FK-Repaired Body Order Runtime Probe

Status: `ok_fk_repaired_body_order_runtime_probe`

This live IsaacLab probe checks whether the FK-repaired motion arrays are ordered the way the official `MotionLoader` expects when it indexes `body_pos_w` with IsaacLab robot body indexes.

## Key Findings

- Robot body order exactly matches URDF order: `False`
- Target robot indexes equal URDF indexes by name: `False`
- MotionLoader matches named FK targets: `False`
- Misindexed targets present: `True`
- Max named-vs-loader z delta: `0.9731605760753155` m
- Endpoint z error after one zero-action step max: `1.0134339332580566` m

## Target Rows

| Body | Robot index | URDF index | Raw body at robot index | Loader z mean | Named z mean | Abs z delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `pelvis` | 0 | 0 | `pelvis` | 0.793049 | 0.793049 | 0.000000 |
| `left_hip_roll_link` | 6 | 3 | `left_ankle_pitch_link` | 0.070644 | 0.664048 | 0.593404 |
| `left_knee_link` | 12 | 5 | `right_ankle_pitch_link` | 0.074609 | 0.363620 | 0.289011 |
| `left_ankle_roll_link` | 24 | 7 | `left_shoulder_yaw_link` | 1.026369 | 0.053209 | 0.973161 |
| `right_hip_roll_link` | 7 | 9 | `left_ankle_roll_link` | 0.053209 | 0.657113 | 0.603904 |
| `right_knee_link` | 13 | 11 | `right_ankle_roll_link` | 0.057585 | 0.355506 | 0.297920 |
| `right_ankle_roll_link` | 25 | 13 | `left_elbow_link` | 0.983800 | 0.057585 | 0.926215 |
| `torso_link` | 11 | 16 | `right_knee_link` | 0.355506 | 0.836725 | 0.481220 |
| `left_shoulder_roll_link` | 22 | 23 | `left_shoulder_pitch_link` | 1.083490 | 1.080878 | 0.002612 |
| `left_elbow_link` | 30 | 25 | `right_shoulder_pitch_link` | 1.084259 | 0.983800 | 0.100459 |
| `left_wrist_yaw_link` | 36 | 28 | `right_wrist_yaw_link` | 0.924250 | 0.937065 | 0.012815 |
| `right_shoulder_roll_link` | 23 | 31 | `left_shoulder_roll_link` | 1.080878 | 1.082103 | 0.001226 |
| `right_elbow_link` | 31 | 33 | `right_shoulder_roll_link` | 1.082103 | 0.979859 | 0.102244 |
| `right_wrist_yaw_link` | 37 | 36 | `right_rubber_hand` | 0.916132 | 0.924250 | 0.008118 |

## Claim Boundary

This is a runtime diagnostic only. It does not train PPO, collect DAgger data, reproduce Fig. 5/Fig. 6, benchmark TensorRT, or validate real robot deployment.
