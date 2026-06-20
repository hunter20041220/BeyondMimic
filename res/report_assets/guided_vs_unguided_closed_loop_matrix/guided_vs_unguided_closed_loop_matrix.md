# Guided vs Unguided Closed-Loop Matrix

This report-facing matrix aggregates existing local virtual guidance rollouts. It is not a paper-level Fig. 5/Fig. 6 reproduction and it is not real-robot evidence.

- Status: `ok`
- Matrix rows: `43`
- Multiseed rows: `12`
- Video-linked rows: `43`
- Aggregate task rows: `12`

## Multiseed Task Aggregate

| Task | Seeds | Reward Delta Mean | Error Delta Mean | Cost Delta Mean | Grad Norm Mean |
|---|---:|---:|---:|---:|---:|
| `composed` | `5` | `-0.0011751788` | `-0.0018588811` | `5.4913301e-05` | `0.073589255` |
| `joystick` | `5` | `0.00013478333` | `9.2482567e-05` | `6.3416452e-05` | `0.079243665` |
| `obstacle_avoidance` | `5` | `0.0029722271` | `-0.00068134367` | `1.0189125e-05` | `0.030970361` |
| `waypoint` | `5` | `-0.0018419905` | `0.0002269268` | `1.6568337e-05` | `0.040233939` |
| `composed` | `1` | `` | `` | `5.6947653e-05` | `` |
| `joystick` | `1` | `` | `` | `6.4107586e-05` | `` |
| `obstacle_avoidance` | `1` | `` | `` | `9.606066e-06` | `` |
| `waypoint` | `1` | `` | `` | `1.7337974e-05` | `` |
| `composed` | `3` | `0.00085719891` | `-0.0019960403` | `8.5288299e-05` | `0.091541441` |
| `joystick` | `3` | `-0.00020336677` | `0.0017043923` | `8.419073e-05` | `0.091257416` |
| `obstacle_avoidance` | `3` | `-0.0017583435` | `0.00076482445` | `1.1612673e-05` | `0.032876954` |
| `waypoint` | `3` | `-0.00076659086` | `-0.00061245759` | `1.8331717e-05` | `0.042392154` |

## Claim Boundary

Rows tagged `approximately_comparable` or `qualitative_only` are local virtual/resource-adjusted evidence. They should support the reading report's reproduction discussion, but they must not be described as official BeyondMimic checkpoints, unpatched official replay, paper Fig. 5/Fig. 6 metrics, or robot results.

## Files

- Matrix CSV: `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_matrix.csv`
- Aggregate CSV: `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_aggregate.csv`
- JSON: `res/report_assets/guided_vs_unguided_closed_loop_matrix/guided_vs_unguided_closed_loop_matrix.json`
- Plot: `res/report_assets/guided_vs_unguided_closed_loop_matrix/task_conditioned_multiseed_guided_deltas.png`
- Plot: `res/report_assets/guided_vs_unguided_closed_loop_matrix/task_conditioned_guidance_signal_strength.png`
