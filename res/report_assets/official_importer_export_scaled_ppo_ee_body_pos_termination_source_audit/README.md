# Scaled PPO ee_body_pos Termination Source Audit

The official tracking config terminates ee_body_pos when any of four distal bodies exceeds a 0.25 m z-only error threshold. The local scaled PPO best and final checkpoints both trip this gate for more than 99% of env-steps, making endpoint body tracking the next mainline blocker.

Threshold: `0.25` m.
Termination body names: `['left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']`.

Claim level: local virtual source-linked diagnostic only. This is not a paper-level BeyondMimic result.
