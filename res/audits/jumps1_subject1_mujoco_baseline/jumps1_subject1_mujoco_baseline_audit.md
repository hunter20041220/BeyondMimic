# jumps1_subject1 MuJoCo Baseline Audit

- Status: `ok_jumps1_subject1_mujoco_baseline_audit`
- Passed cases: `3/3`
- EGL 300-frame attempt status: `empty_log_after_abort_observed_in_command_output`

## Case Summary

| Case | Category | Frames | MP4 Frames | Passed | Claim Boundary |
|---|---:|---:|---:|---:|---|
| original_csv_reference_replay_osmesa | kinematic_reference_replay | 300 | 300 | True | Official released 36-column G1 LAFAN1 reference rendered in MuJoCo via frame-by-frame mj_forward; not policy control. |
| fk_repaired_npz_reference_replay_osmesa | kinematic_reference_replay | 299 | 299 | True | FK-repaired local motion NPZ rendered in MuJoCo via frame-by-frame mj_forward; not policy control. |
| reference_action_control_osmesa | pd_reference_action_control | 299 | 299 | True | MuJoCo mj_step PD tracking of FK-repaired reference joint targets with root assist; not a learned teacher/VAE/diffusion controller. |

## Interpretation

- The original CSV and FK-repaired NPZ videos are reference replays; they prove that the source motion can be rendered on the G1 mesh in MuJoCo, not that a policy controls the robot.
- `reference_action_control` uses MuJoCo `mj_step` and 29 position actuators, but it also uses a pelvis root-assist stabilizer. It is a PD control baseline, not a teacher/RL, VAE, diffusion, or guidance result.
- The EGL 300-frame attempt is retained as a rendering-backend failure. OSMesa is the current stable backend for this local H20 report-video path.
- This audit does not claim paper-level BeyondMimic Fig. 5/Fig. 6 reproduction and does not claim real-robot deployment.
