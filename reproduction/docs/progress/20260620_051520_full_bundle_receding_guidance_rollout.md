# Progress Update

## Goal

Advance the simulation-side BeyondMimic reproduction evidence beyond single-motion/local-summary artifacts by running a full-bundle closed-loop receding-latent guidance rollout with video, metrics, and audit integration.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Existing single-motion receding-latent guidance rollout script and full-bundle PPO/VAE/diffusion/guidance summaries.

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Refreshed generated comparison, manifest, final report, visual evidence index, required absence audit, verification manifests, and master audit outputs.

## Commands Run

```bash
python3 reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/visual_evidence_index.py
python3 reproduction/scripts/artifact_manifest.py
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

- New rollout status: `ok_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval`.
- Selected GPU: physical GPU `4`.
- Motion source: 40-motion public official-csv-loop bundle, `11960` total frames.
- Variants: `teacher`, `vae_base`, `denoised_latent`, `receding_latent_guided`.
- Rollout length: `299` steps per variant.
- Guided-latent reward mean: `0.023252945707917812`.
- Guided-latent target-body error mean: `0.08156827092170715`.
- Guided-latent guidance cost delta mean: `5.999496549268231e-05`.
- Artifact manifest count: `648`.
- Visual evidence index video count: `10`.
- Paper-vs-reproduction row count: `164`.
- Master audit artifact count: `277`.

## Verification

All required verification commands passed:

- `paper_vs_reproduction_comparison.py`: `ok`
- `visual_evidence_index.py`: `ok`, `10` videos indexed
- `artifact_manifest.py`: `ok`, `648` artifacts
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, `170` rows
- `verification_command_syntax_audit.py`: `ok`, `186` scripts, `0` failed
- `verification_command_script_manifest.py`: `ok`, `186` scripts
- `verification_command_coverage_audit.py`: `ok`, `194` commands, `10/10` smoke pass
- `required_artifact_absence_audit.py`: `ok`, `26` rows
- `final_deliverables_audit.py`: `ok`, `38` rows
- `reproduction_master_audit.py`: `ok`, `277` artifacts

## Failed / Blocked Items

- The first preflight caught a wrong guessed full-bundle NPZ filename. The script was corrected to use `official_csv_loop_full_public_motion_bundle.npz` from the full-bundle audit.
- `reproduction_master_audit.py` initially failed after the new MP4 because `required_artifact_absence_audit.py` classified the new local virtual video as an unclassified paper-level video. I fixed the classification by explicitly treating the full-bundle receding-latent rollout MP4 as a local virtual/reference video, not as required paper-level Fig. 5/Fig. 6 evidence.
- This is a single-environment report/evidence rollout, not formal PPO or diffusion training. It therefore does not use both GPUs or meet the 10GB/GPU formal-experiment threshold.
- Official unpatched replay, official BeyondMimic VAE/diffusion checkpoints, paper Fig. 5/Fig. 6 task protocol/results, TensorRT deployment, and real-robot evidence remain incomplete.

## Effect on English Reading Report

The English report now has a stronger simulation-side guidance result: a full-bundle closed-loop receding-latent guidance MP4 and metrics, instead of only single-motion guidance videos or offline guidance tables. The report still clearly states that this is local virtual/resource-adjusted evidence, not paper-level reproduction.

## Next Step

The next mainline step is either to broaden closed-loop guidance over more tasks/motions, or to use this full-bundle video plus existing PPO/teacher/VAE/diffusion assets to finalize a polished English reading report/PPT evidence section. Official unpatched replay can also be retried, but current evidence says the converter/write path remains the blocker.

## Git Commit

Pending at time of writing this progress file.
