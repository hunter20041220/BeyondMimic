# ee_body_pos Termination Source Audit

This audit links the local scaled PPO termination diagnostic back to the official tracking source.

- Official termination function: `bad_motion_body_pos_z_only` = `True`.
- Official threshold: `0.25` meters on z-only distal body tracking.
- Configured termination body names: `['left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']`.
- G1 command body-name count: `14`.
- Motion bundle shape: body_pos_w `[11960, 40, 3]`, joint_pos `[11960, 29]`.
- Best checkpoint ee_body_pos fraction: `0.9985874137750836`.
- Final checkpoint ee_body_pos fraction: `0.9988405361622074`.

Interpretation: the weak local scaled PPO teacher is not merely a checkpoint-selection issue; both
full-size evaluated checkpoints terminate almost entirely through the official z-only endpoint body
position gate. The next debugging step should inspect retargeted endpoint z trajectories, policy
stability around the four distal links, and whether the public-data/importer-export setup needs
additional warm start, curriculum, or termination scheduling before downstream DAgger/VAE/diffusion
rollouts are trustworthy.

Claim level: local virtual source-linked diagnostic only. This is not a paper-level BeyondMimic
closed-loop result, not an official checkpoint, and not real-robot evidence.
