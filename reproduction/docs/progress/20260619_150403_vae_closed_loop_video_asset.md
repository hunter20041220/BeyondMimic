# Progress Update

## Goal

Add a report-ready visual asset for the local official-csv-loop VAE action-reconstruction closed-loop result. The previous round produced a full two-rank numerical rollout; this round adds a single-environment 299-frame robot/reference video so the English report and PPT have a concrete motion visualization.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json`
- `res/level_c/official_csv_loop_teacher_rollout_vae_training/level_c_official_csv_loop_teacher_rollout_vae_training.json`
- `res/visualization/official_csv_loop_policy_rollout/official_csv_loop_policy_rollout_video_asset.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_video_capture.py`
- `reproduction/scripts/visual_media_inventory_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`

Regenerated audit/report outputs under `res/artifact_manifest/`, `res/comparison/`, `res/final_report/`, `res/master_audit/`, `res/required_artifact_absence/`, `res/visual_media_inventory/`, `res/final_deliverables_audit/`, and verification audit directories.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_video_capture.py
python3 reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_video_capture.py
python3 reproduction/scripts/visual_media_inventory_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/final_deliverables_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

New visualization directory:

```text
res/visualization/official_csv_loop_vae_closed_loop_rollout/
```

New local MP4/keyframe/metric assets:

```text
official_csv_loop_vae_closed_loop_rollout_vs_reference.mp4
official_csv_loop_vae_closed_loop_rollout_keyframes.png
official_csv_loop_vae_closed_loop_rollout_metrics.csv
official_csv_loop_vae_closed_loop_rollout_video_asset.json
tracking_g1_official_csv_loop_vae_closed_loop_rollout_capture.json
```

Recorded metrics:

```text
status: ok_official_csv_loop_vae_closed_loop_rollout_video_capture
selected physical GPU: 4
frames: 299
target body count: 14
reward mean: 0.026958581060171127
target body error mean: 0.08216936886310577
target body error max: 0.249213308095932
teacher/VAE action MSE mean: 0.0034388084895908833
teacher/VAE action abs error mean: 0.04385554417967796
MP4 size: 145250 bytes
```

The MP4 is intentionally a local visualization asset and is not committed as a paper-level video. Its path and SHA256 are recorded in the asset JSON.

## Verification

Initial integration checks passed:

```text
visual_media_inventory_audit: ok, 133 visual rows, 3 videos
required_artifact_absence_audit: ok, 25 rows
final_deliverables_audit: ok, 38 rows
artifact_manifest: ok, 410 artifacts
paper_vs_reproduction_comparison: ok
final_reproduction_report: ok
completion_matrix_status_audit: ok, 164 rows
reproduction_master_audit: ok
verification_command_syntax_audit: ok, 185 scripts, 0 failed
verification_command_script_manifest: ok, 185 scripts
verification_command_coverage_audit: ok, 193 commands
```

## Failed / Blocked Items

- The video is not paper-level Fig. 5/Fig. 6 evidence.
- It does not use the unreleased official BeyondMimic VAE checkpoint.
- It is not autonomous VAE control and not receding-horizon guided diffusion.
- It is not real-robot evidence.
- GPU4 had a `wangjc` process during preflight, but the script selected GPU4 based on the instantaneous preflight threshold and completed. This run is a single-env visualization asset, not a formal two-GPU training/evaluation experiment.

## Effect on English Reading Report

The English report now has a concrete VAE closed-loop robot motion video and keyframe image, in addition to JSON metrics and static plots. This makes the reproduction section easier to present: the reader can see the local robot/reference skeleton motion while the text keeps the evidence boundary honest.

## Next Step

Proceed toward receding-horizon closed-loop guidance: use the current observation to generate or guide a state-latent/action plan, decode the current action through the local VAE, step IsaacLab, and compare guided vs unguided rollouts. That would be closer to the paper's Fig. 5/Fig. 6 mechanism, while still remaining local virtual evidence unless official checkpoints and paper task logs become available.

## Git Commit

This progress file is intended to be included in the same Git commit as the VAE closed-loop video asset code, report, and audit updates. The final commit hash is reported in the user-facing turn summary.
