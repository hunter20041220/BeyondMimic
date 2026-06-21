# Progress Update

## Goal

Advance the tracking data-quality mainline before another PPO rerun by scaling the wrist/ankle endpoint source question from a 256-env live probe to a full 2048-env x 299-step checkpoint-evaluation diagnostic.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/robot_order_fk_wrist_endpoint_alignment_live_probe/robot_order_fk_wrist_endpoint_alignment_live_probe.json`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json`

## Files Modified

- `reproduction/scripts/robot_order_fk_wrist_endpoint_source_full_diagnostic.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/update_course_reports.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/robot_order_fk_wrist_endpoint_source_full_diagnostic.py
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
python3 reproduction/scripts/robot_order_fk_wrist_endpoint_source_full_diagnostic.py
python3 -m py_compile reproduction/scripts/robot_order_fk_wrist_endpoint_source_full_diagnostic.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/update_course_reports.py
python3 reproduction/scripts/update_course_reports.py
```

## Results

The full diagnostic completed successfully on physical GPU 4. It used the official-importer-export G1 USDA, the robot-order FK-repaired 40-motion bundle, and the current robot-order FK PPO checkpoint. It did not train PPO.

Key metrics:

- Scope: `2048` envs x `299` steps, `612352` env steps.
- Overall done rate: `0.21957958821070234`.
- `ee_body_pos` rate: `0.19802009301839466`.
- Mean pre-step wrist exceed rate: `0.06626907399665552`.
- Mean pre-step ankle exceed rate: `0.057275880539297656`.
- Mean post-step wrist exceed rate: `0.06591470265468227`.
- Mean post-step ankle exceed rate: `0.05699989548494983`.
- Mean pre-step wrist z-error: `0.11623241832124748` m.
- Mean pre-step ankle z-error: `0.0531279238619932` m.
- Top wrist-heavy motions include `fallAndGetUp1_subject4`, `dance1_subject2`, and `walk3_subject5`.

## Verification

Initial syntax checks and report regeneration passed. Full artifact/comparison/final-report/master verification is the next step in this same round.

## Failed / Blocked Items

This diagnostic improves source attribution but does not fix the tracking teacher. The tracking teacher still has high done/termination rates, so this is not a paper-level PPO result and should not trigger downstream teacher rollout/VAE/diffusion reruns yet.

Remaining blocked or incomplete paper-level items include official tracking policy quality, true DAgger rollout logs, official VAE/diffusion checkpoints, Fig. 5/Fig. 6 protocol metrics/videos, TensorRT deployment, sim-to-sim ROS logs, and real robot evidence.

## Effect on English Reading Report

The English reading report now has a stronger explanation for why tracking remains the bottleneck: the problem is not just a vague reset issue, but a measurable endpoint/body/motion/phase source problem. The report frames this as a reproducibility lesson: BeyondMimic's generative stages depend on a physically reliable tracking teacher.

## Next Step

Use the by-motion and by-phase rows to decide whether the next repair should target specific wrist-heavy public motions, wrist endpoint body selection, or `ee_body_pos` termination semantics. Only after done/endpoint metrics become reasonable should a new full PPO run start.

## Git Commit

Pending.
