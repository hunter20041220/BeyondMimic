# Robot-Order FK Phase-Alignment Live Probe

Generated: 2026-06-21T21:39:47.519896+00:00

## Result

- Status: `ok_robot_order_fk_phase_alignment_live_probe`
- Worker status: `ok_robot_order_fk_phase_alignment_live_probe`
- Diagnosis: `no_phase_alignment_variant_improves_both_done_and_joint_velocity`
- Recommended full-eval variant: ``
- Baseline refresh-t policy done rate: `0.38671875`
- Baseline refresh-t policy joint velocity: `11.769760131835938`

## Policy-Step Comparison

| variant | dt | rewritten | pre endpoint done | step done | ee term | body error | joint vel | reward |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| official_stale | 0.0 | False | 0.54296875 | 0.53125 | 136 | 0.14436224102973938 | 9.643479347229004 | 0.027484934777021408 |
| refresh_t | 0.0 | False | 0.3828125 | 0.38671875 | 99 | 0.14631174504756927 | 11.769760131835938 | 0.033730506896972656 |
| compute_dt0 | 1.0 | False | 0.4921875 | 0.49609375 | 127 | 0.15390613675117493 | 11.086766242980957 | 0.031612578779459 |
| update_command_t_plus_1 | 1.0 | False | 0.5390625 | 0.5390625 | 138 | 0.1541636437177658 | 10.390115737915039 | 0.03189649432897568 |
| phase_minus_1_then_step_target_t | -1.0 | False | 0.60546875 | 0.6015625 | 154 | 0.15591561794281006 | 8.599140167236328 | 0.03178621456027031 |
| phase_plus_1_target_only | 1.0 | False | 0.66796875 | 0.66015625 | 169 | 0.16144219040870667 | 6.945001602172852 | 0.03104715421795845 |
| phase_plus_1_rewrite_state | 1.0 | True | 0.71875 | 0.71484375 | 183 | 0.16026751697063446 | 9.412028312683105 | -0.01600092649459839 |
| phase_minus_1_rewrite_state | -1.0 | True | 0.72265625 | 0.72265625 | 185 | 0.16024701297283173 | 8.142926216125488 | -0.0137783819809556 |

This is a local virtual live diagnostic. It does not train PPO, does not claim paper-level tracking, does not claim DAgger/VAE/diffusion success, and does not use real hardware.
