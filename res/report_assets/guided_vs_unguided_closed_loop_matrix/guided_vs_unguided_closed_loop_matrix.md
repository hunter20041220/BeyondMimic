# Guided vs Unguided Closed-Loop Matrix

This report-facing matrix aggregates existing local virtual guidance rollouts. It is not a paper-level Fig. 5/Fig. 6 reproduction and it is not real-robot evidence.

- Status: `ok`
- Matrix rows: `35`
- Multiseed rows: `12`
- Video-linked rows: `35`
- Aggregate task rows: `12`

## Multiseed Task Aggregate

| Task | Seeds | Reward Delta Mean | Error Delta Mean | Cost Delta Mean | Grad Norm Mean |
|---|---:|---:|---:|---:|---:|
| `composed` | `3` | `-0.0013421524` | `-0.00099969407` | `5.5002163e-05` | `0.073630703` |
| `joystick` | `3` | `0.00055500476` | `0.00064565241` | `6.2737211e-05` | `0.078794787` |
| `obstacle_avoidance` | `3` | `0.001432763` | `-0.0011488721` | `1.0619561e-05` | `0.031576155` |
| `waypoint` | `3` | `-0.0025075359` | `0.0010578309` | `1.632443e-05` | `0.039974159` |
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
