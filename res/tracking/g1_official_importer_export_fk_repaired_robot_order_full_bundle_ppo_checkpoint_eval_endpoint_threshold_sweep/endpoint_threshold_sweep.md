# Robot-Order FK Endpoint Threshold Sweep

Generated: `2026-06-22T00:05:34.910103+00:00`

## Result

- Status: `ok_endpoint_threshold_sweep_completed`
- Target-refresh baseline done rate: `0.22340745192307693`
- Best variant: `all_endpoint_threshold_0p5`
- Best done rate: `0.08907621760033445`
- Recommended next action: `evaluate_threshold_candidate_before_full_ppo`

## Rows

| threshold | done rate | post-step0 done | active ee rate | manual original all-endpoint rate |
|---:|---:|---:|---:|---:|
| 0.3 | 0.16474184782608695 | 0.16484243917785235 | 0.13900613464765102 | 0.29601968854865773 |
| 0.35 | 0.13060951870819398 | 0.13076597892197986 | 0.1005908530830537 | 0.3617344798657718 |
| 0.4 | 0.11093782660953178 | 0.11109217701342282 | 0.07708781197567115 | 0.4000956900167785 |
| 0.5 | 0.08907621760033445 | 0.08923421770134228 | 0.04678815803271812 | 0.44215505715184567 |

This sweep keeps all official endpoint bodies active and changes only the z threshold. It remains a local diagnostic candidate, not a paper tracking metric, not DAgger/VAE/diffusion evidence, and not real-robot evidence.
