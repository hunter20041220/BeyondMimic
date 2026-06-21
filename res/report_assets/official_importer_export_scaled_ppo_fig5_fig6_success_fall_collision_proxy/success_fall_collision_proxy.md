# Scaled PPO Fig. 5/Fig. 6 Success/Fall/Collision Proxy

This asset summarizes existing scaled PPO closed-loop local virtual traces. It does not use the paper's
official Fig. 5/Fig. 6 success, fall, or collision labels.

## Thresholds

```json
{
  "body_error_spike_anomaly_threshold": 0.5,
  "expected_steps": 299,
  "fall_relative_root_height_drop_threshold_m": -0.25,
  "positive_guidance_cost_delta_threshold": 0.0,
  "root_final_xy_success_threshold_m": 0.02,
  "target_body_mean_success_threshold": 0.35,
  "target_body_p95_success_threshold": 0.36
}
```

## Summary

| task | seeds | success proxy | fall proxy | body-error spike | 299-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| composed | 5 | 1.00 | 0.00 | 0.00 | 1.00 |
| joystick | 5 | 1.00 | 0.00 | 0.00 | 1.00 |
| obstacle_avoidance | 5 | 1.00 | 0.00 | 0.00 | 1.00 |
| waypoint | 5 | 0.60 | 0.40 | 0.20 | 1.00 |

## Collision Boundary

The traces do not include contact or obstacle-collision sensor channels. The reported body-error spike
rate is an anomaly proxy only and must not be described as a paper collision rate.
