# Progress Update

## Goal

Add a multi-seed local virtual evaluation for the official-csv-loop PPO checkpoint and connect the evidence to the comparison table, artifact manifest, final reports, and English reading report. This is meant to strengthen stability evidence for the virtual tracking chain, not to claim paper-level BeyondMimic completion.

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
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json`
- `res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json`

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.py`
- `reproduction/scripts/official_csv_loop_ppo_multiseed_eval_report_assets.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/current_environment_and_reproduction_status.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/progress/20260619_175558_official_csv_loop_multiseed_eval.md`

## Commands Run

```bash
envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_ppo_multiseed_eval_report_assets.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/final_deliverables_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Three full local virtual checkpoint eval seeds completed successfully.
- Seeds: `20260640`, `20260641`, `20260642`.
- GPU assignment: `4`, `7`, `4`.
- Per-seed evaluation: `512` envs x `299` steps.
- Total simulated env steps: `459264`.
- Aggregate reward mean/std: `0.025978426701298924` / `0.00010146760409522878`.
- Aggregate body-position error mean/std: `0.18423418407697012` / `0.000271408645496586`.
- Aggregate joint-position error mean/std: `1.2231450603159773` / `0.0027425904840304373`.
- Report-ready plots and tables were generated under `res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/`.

## Verification

All required verification commands passed after adding the multi-seed evidence:

- `artifact_manifest.py`: `ok`, `482` artifacts, `0` missing.
- `paper_vs_reproduction_comparison.py`: `ok`, `151` rows.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `169` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `185` scripts, `0` failed.
- `verification_command_script_manifest.py`: `ok`, `185` scripts hashed.
- `verification_command_coverage_audit.py`: `ok`, `193` commands, `10/10` lightweight smokes passed.
- `required_artifact_absence_audit.py`: `ok`, `26` rows.
- `final_deliverables_audit.py`: `ok`, `38` rows.
- `reproduction_master_audit.py`: `ok`, `257/257` master artifacts passed.

## Failed / Blocked Items

- No new IsaacLab headless blocker appeared during this multi-seed eval.
- The result still depends on the enriched-USD runtime patch and the reduced iteration-299 local PPO checkpoint.
- It is not unpatched official G1 replay/training, not the paper-scale official tracking teacher, not official DAgger, not Fig. 5/Fig. 6 guided diffusion evidence, and not real robot evidence.

## Effect on English Reading Report

The English report now has a stronger tracking reproduction section: it can cite a three-seed full virtual evaluation rather than relying only on a single checkpoint-eval run. The report still explicitly states that this project does not fully reproduce BeyondMimic at paper level.

## Next Step

Use this stabilized local tracking-eval evidence to decide whether to continue improving the unpatched official G1 conversion/replay path or to run a more formal paper-facing virtual validation sweep over local VAE/diffusion guidance rollouts.

## Git Commit

Commit message: `feat: add official csv-loop multiseed eval`. The final immutable hash should be read from `git log -1 --oneline` after the commit is created.
