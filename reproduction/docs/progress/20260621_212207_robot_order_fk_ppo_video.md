# Progress Update

## Goal

Promote the latest robot-order FK-repaired PPO run from loose local output into the audited BeyondMimic reproduction chain, update the English/Chinese reports, generate report-ready policy video evidence, and keep claim boundaries explicit.

## Files Read

- `prompt06211658.txt`
- `reproduction/scripts/tracking_fk_repaired_data_quality_gate.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json`
- `res/report_assets/official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_assets.json`

## Files Modified

- `reproduction/scripts/tracking_fk_repaired_data_quality_gate.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`

## New Files

- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.py`
- `reproduction/scripts/official_importer_export_fk_repaired_robot_order_full_bundle_ppo_eval_report_assets.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture.py`
- `res/storage_cleanup/storage_cleanup_20260621_robot_order_fk_ppo_video.md`
- `reproduction/docs/progress/20260621_212207_robot_order_fk_ppo_video.md`

## Commands Run

- `python3 -m py_compile ...`
- `python3 reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture.py`
- `python3 reproduction/scripts/tracking_fk_repaired_data_quality_gate.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_fk_repaired_robot_order_full_bundle_ppo_eval_report_assets.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- storage inspection with `du`, `find`, and `rg`

## Results

- Robot-order FK-repaired PPO training is now represented as a mainline local tracking baseline: 1000 iterations, 4096 total envs, 21 checkpoints, GPUs 4/7.
- Robot-order checkpoint evaluation is recorded: 2048 envs x 299 steps, 612352 env steps, done rate `0.1782798129180602`, reward mean `0.02073384587805606`, anchor/body/joint error mean `0.07790673197711191` / `0.36114187777839774` / `1.5732512252785291`.
- Policy rollout video capture succeeded for 299 frames. The video asset records target-body error mean `0.15473534166812897`, target-body error max `0.2961389124393463`, reward mean `0.024368468672037125`, and done count `44`.
- `paper_vs_reproduction` now has `219` rows: exactly comparable `58`, approximately comparable `19`, qualitative-only `129`, not publicly reproducible `10`, requires real robot `3`.

## Verification

Verification is run after this progress file is written. The expected chain is:

- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`

## Failed / Blocked Items

- The robot-order PPO checkpoint is still not paper-level: done rate remains nontrivial and joint/velocity errors are high.
- No official BeyondMimic tracking teacher checkpoint, true DAgger logs, official VAE/diffusion checkpoints, Fig.5/Fig.6 paper-level rollout videos, TensorRT engine, or real-robot evidence is claimed.
- Formal GPU memory target of 10 GB/card was not reached by this RSL-RL/IsaacLab model configuration; this is recorded as local full-run evidence rather than a fabricated memory claim.

## Effect on English Reading Report

The report can now describe a stronger tracking story: old degenerate body positions, URDF-vs-runtime body-order bug, robot-order FK repair, full PPO training/eval, and a small policy video. This gives a clearer independent reproduction narrative while preserving the claim boundary that the project remains a public-resource partial reproduction.

## Next Step

Run robot-order checkpoint sweep and multi-seed evaluation, then decide whether to train longer or adjust termination/curriculum before using this teacher for downstream VAE/diffusion/guidance.

## Git Commit

Pending at the time this progress note is written.
