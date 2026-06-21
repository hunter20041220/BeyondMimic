# Official Importer-Export Scaled PPO Fig. 5/Fig. 6 Task-Protocol Proxy

This asset converts existing local closed-loop importer-export traces into task-level proxy metrics
for the reading report. It is not an official BeyondMimic Fig. 5/Fig. 6 protocol, not TensorRT
deployment evidence, and not real-robot evidence.

## Thresholds

```json
{
  "positive_guidance_cost_delta_threshold": 0.0,
  "recorded_step_count": 299,
  "reward_improvement_threshold": 0.0,
  "root_final_xy_error_threshold_m": 0.02,
  "target_body_frame_error_threshold": 0.36,
  "target_body_mean_error_threshold": 0.35,
  "tracking_error_not_worse_threshold": 0.0
}
```

## Summary

| task | seeds | 299-step | local proxy pass | reward improved | error not worse | final root err mean (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| composed | 5 | 1.00 | 0.60 | 0.60 | 0.40 | 0.0094 |
| joystick | 5 | 1.00 | 0.80 | 0.40 | 0.60 | 0.0050 |
| obstacle_avoidance | 5 | 1.00 | 1.00 | 0.80 | 0.60 | 0.0053 |
| waypoint | 5 | 1.00 | 0.80 | 0.60 | 0.40 | 0.0046 |

## Key Interpretation

- Rows analyzed: 20.
- Seed groups: 5.
- Overall local task-protocol proxy pass rate: 0.800.
- Overall reward-improved-vs-denoised rate: 0.600.
- Overall tracking-error-not-worse-vs-denoised rate: 0.500.

The proxy pass requires a 299-step local trace, a present MP4 path, positive guidance-cost
decrease, a root endpoint proxy within 2 cm of the local reference endpoint, mean target-body error
under 0.35, and either reward improvement or non-worse target-body error relative to the local
denoised baseline. These thresholds are local analysis thresholds, not paper thresholds.

## Remaining Gap

The paper-level Fig. 5/Fig. 6 gates still require exact task protocols, public official checkpoints
or reproduced teacher-derived state-latent rollouts, fall/collision/success definitions, TensorRT or
asynchronous deployment traces where applicable, and real robot/mocap evidence for real-world panels.
