# Stage-1 Single-Leg / Jumps Teacher Failure Audit

- Status: `blocked_teacher_quality_not_motion_absence`
- Claim level: `audit_only_no_training_no_video_no_success_claim`

## Key Findings

- The inspected single-leg motion contains a visible lifted-ankle target; the source motion is not empty.
- The inspected jumps1 motion contains endpoint-height variation suitable for a later reference/teacher target.
- The latest refreshed single-leg checkpoint evidence available to this audit still fails the quality gate.
- Downstream VAE/diffusion/guidance videos remain blocked until a continuous teacher rollout quality gate passes.

## Outputs

- JSON: `/mnt/infini-data/test/BeyondMimic/res/audits/stage1_singleleg_jumps_motion_teacher_failure/stage1_singleleg_jumps_motion_teacher_failure_audit.json`
- TSV: `/mnt/infini-data/test/BeyondMimic/res/audits/stage1_singleleg_jumps_motion_teacher_failure/stage1_singleleg_jumps_motion_teacher_failure_audit.tsv`
