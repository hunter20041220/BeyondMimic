# Robot-Order FK Deterministic Reset Live Probe

Generated: 2026-06-21T20:38:56.510944+00:00

## Result

- Status: `ok_robot_order_fk_deterministic_reset_live_probe`
- Recommended full-eval variant: ``
- Official refresh policy done rate: `0.33203125`
- Deterministic refresh policy done rate: `0.453125`
- Deterministic joint-velocity delta: `-3.836276054382324`
- Any variant improves done and joint velocity: `False`

## Policy-Step Rows

| Mode | Action | Done rate | Reward | Endpoint done after | Body error | Joint pos error | Joint vel error |
|---|---|---:|---:|---:|---:|---:|---:|
| official_reset | zero | 1.0 | 0.0237712562084198 | 0.26953125 | 0.1509433537721634 | 0.9327948689460754 | 0.644939661026001 |
| official_reset | policy | 0.5078125 | 0.027123596519231796 | 0.1796875 | 0.14942720532417297 | 0.9564528465270996 | 12.31811809539795 |
| official_reset_target_refresh | zero | 0.28515625 | 0.033379875123500824 | 0.078125 | 0.14974839985370636 | 0.998088002204895 | 16.9989013671875 |
| official_reset_target_refresh | policy | 0.33203125 | 0.032367199659347534 | 0.11328125 | 0.14815855026245117 | 0.9855404496192932 | 15.347245216369629 |
| deterministic_reset_target_refresh | zero | 0.3984375 | 0.038006480783224106 | 0.1171875 | 0.14359880983829498 | 0.8943828344345093 | 11.851893424987793 |
| deterministic_reset_target_refresh | policy | 0.453125 | 0.03521870821714401 | 0.17578125 | 0.1480797678232193 | 0.8861483335494995 | 11.510969161987305 |
| deterministic_motion_state | zero | 0.4921875 | -0.02209804207086563 | 0.21484375 | 0.15324905514717102 | 0.8386430144309998 | 17.332820892333984 |
| deterministic_motion_state | policy | 0.55078125 | -0.024065259844064713 | 0.28125 | 0.1534680277109146 | 0.8028693199157715 | 14.773418426513672 |

This is a local virtual live gate. It does not train, does not claim paper-level tracking, does not claim DAgger/VAE/diffusion success, and does not use real hardware.
