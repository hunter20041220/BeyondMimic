# Progress Update

## Goal

Diagnose why the current official-importer-export scaled PPO chain has near-total `ee_body_pos` termination and large ankle z-error, without starting new training or overstating local diagnostic evidence as paper-level BeyondMimic reproduction.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_motion_state_fixture.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle_clips.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_motion_bundle_body_position_degeneracy_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- Regenerated comparison, manifest, final-report, completion-matrix, verification-command, and master-audit outputs under `/mnt/infini-data/test/BeyondMimic/res/`.

## Commands Run

```bash
git status --short && git log -3 --oneline
find res/tracking -maxdepth 3 -type f \( -name '*clips*.csv' -o -name '*conversion*.csv' -o -name '*bundle*.csv' -o -name '*manifest*.csv' \) | sort | head -n 80
rg -n "sim\.render\(|sim\.step\(|body_pos_w|write_npz|np.savez|root_state|joint_pos" reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py
sed -n '1,620p' reproduction/scripts/build_level_c_motion_state_fixture.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_motion_bundle_body_position_degeneracy_audit.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_motion_bundle_body_position_degeneracy_audit.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
git diff --check
```

Full verification log:

`/mnt/infini-data/test/BeyondMimic/logs/verification/20260621_motion_bundle_body_position_degeneracy_verification.log`

## Results

The audit confirms that the current full public-motion official-loop bundle has a valid outer schema but invalid target-body positions for tracking diagnosis:

- Current bundle `body_pos_w` shape: `[11960, 40, 3]`.
- Current bundle max body-minus-root spread: `7.152557373046875e-07` m.
- Current bundle max z spread: `4.76837158203125e-07` m.
- Non-Kit URDF-FK candidate mean z spread on `walk1_subject1`: `1.2378766166891086` m.
- Bundle left/right ankle mean z: `0.7737202890316978` / `0.7737202915982857` m.
- FK-candidate left/right ankle mean z: `0.054083834997340555` / `0.05807142947630243` m.

This explains the scaled PPO endpoint trace: the local policy is being compared against ankle targets at root-like height, producing approximately `0.71` m ankle z-errors and near-unit official threshold exceed rates. The immediate repair path is motion-preprocessing/body-position generation, not longer PPO training on the current bundle.

## Verification

- `artifact_manifest.py`: passed, `1337` artifacts, `0` missing.
- `paper_vs_reproduction_comparison.py`: passed, `209` rows; comparison counts are `58` exactly comparable, `19` approximately comparable, `119` qualitative only, `10` not publicly reproducible, and `3` requires real robot.
- `final_reproduction_report.py`: passed, `goal_complete=false`.
- `completion_matrix_status_audit.py`: passed, `199` rows; status counts are `73` complete, `122` partial, `3` blocked, `1` out of scope.
- `verification_command_syntax_audit.py`: passed, `199` scripts, `0` failed.
- `verification_command_script_manifest.py`: passed, `199` scripts.
- `verification_command_coverage_audit.py`: passed, `207` commands, `10/10` lightweight smoke pass.
- `reproduction_master_audit.py`: passed, `340/340` artifacts, `0` failed.
- `git diff --check`: passed.

## Failed / Blocked Items

- No new failed run was created.
- The current official-importer-export full-bundle `body_pos_w` is degenerate and should not be used as trusted target-body evidence for teacher-quality PPO, DAgger, VAE, diffusion, or paper-level closed-loop evaluation.
- The FK probe is diagnostic repair-direction evidence only. It is not official `csv_to_npz.py` output and not an official motion fix.
- Still missing: official BeyondMimic VAE/diffusion checkpoints, true DAgger rollout logs, true Fig. 5/Fig. 6 paper-level rollout metrics/videos, TensorRT deployment evidence, and real robot validation.

## Effect on English Reading Report

This adds a high-value negative result for the reproduction section: the project did not merely fail to match tracking behavior; it found a concrete data-generation failure mode in the recovered virtual pipeline. The English report can now explain why honest reproduction requires auditing intermediate motion targets, not just running PPO longer.

## Next Step

Build and validate a corrected full public-motion body-position candidate. Two possible routes are:

1. Find the proper IsaacLab articulation-state refresh path after root/joint writes, then regenerate the official-loop bundle and rerun replay/task diagnostics.
2. Produce a clearly labeled non-Kit URDF-FK repaired bundle candidate, rerun task replay/zero-action diagnostics, and only then decide whether PPO should be retrained.

Do not continue teacher rollout, VAE, or diffusion claims from the degenerate bundle as paper-level evidence.

## Git Commit

Pending at time of writing.
