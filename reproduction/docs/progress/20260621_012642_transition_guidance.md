# Progress Update

## Goal

Add a conservative, paper-facing virtual diagnostic for the BeyondMimic Fig. 5B walking-to-running transition idea on the recovered official-importer-export G1 IsaacLab chain. The goal was to produce auditable closed-loop evidence for a local velocity-ramp transition proxy without claiming paper-level Fig. 5B transition reproduction or Fig. 5D t-SNE reproduction.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/fig5_fig6_proxy_protocol_matrix.json`
- `res/report_assets/official_importer_export_full_bundle_latent_projection/official_importer_export_full_bundle_latent_projection_report_assets.json`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_receding_latent_guidance_rollout_eval.py`
- `reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`

## Files Modified

- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval.py`
- `reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`
- Generated audit/report artifacts under `res/artifact_manifest/`, `res/comparison/`, `res/docs/completion_matrix_status_audit/`, `res/final_report/`, `res/master_audit/`, `res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/`, `res/report_assets/official_importer_export_full_bundle_transition_guidance/`, and `res/verification_command_*`.

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py
envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py
envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

## Results

- Added `tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval.py`, which reuses the validated importer-export task-conditioned guidance runner and injects a single local `transition` task.
- The local transition cost ramps target root x velocity from slow walking to faster running over the 21-step guidance horizon, with path, smoothness, and latent regularization terms.
- Ran one 299-step closed-loop diagnostic on physical GPU 4. This was a single-env virtual diagnostic, not a formal two-GPU training run.
- Result status: `ok_official_importer_export_full_bundle_transition_guidance_rollout_eval`.
- Main result file: `res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json`.
- Report assets were generated under `res/report_assets/official_importer_export_full_bundle_transition_guidance/`.
- MP4 visualization was retained locally under `res/visualization/official_importer_export_full_bundle_transition_guidance_rollout/transition/official_csv_loop_task_conditioned_latent_guidance_rollout_vs_reference.mp4` and is intentionally not staged for Git by default.

Key diagnostic metrics for `receding_latent_guided`:

```text
rollout_steps: 299
selected_physical_gpu: 4
guided_reward_mean: 0.024728728436481794
guided_target_body_error_mean: 0.3448648750782013
guidance_cost_delta_mean: 0.00012286637838070208
guided_late_minus_early_speed_mps: 2.0195484379946684
guided_target_speed_rmse_mps: 21.66667160938303
guided_speed_target_corr: 0.016159506113184546
```

Interpretation: this is mixed diagnostic evidence. The guided rollout shows a positive late-minus-early speed change, but the target-speed correlation is weak and the target-speed RMSE is high. It is useful for debugging the local virtual transition objective, not for claiming paper Fig. 5B success.

## Verification

Final verification was rerun after this progress file was added to the artifact manifest:

- `artifact_manifest.py`: passed.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed.
- `verification_command_syntax_audit.py`: passed.
- `verification_command_script_manifest.py`: passed.
- `verification_command_coverage_audit.py`: passed.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- The transition task is a local velocity-ramp proxy, not the paper's original Fig. 5B walking-to-running protocol.
- The latent visualization remains PCA-based local evidence, not paper Fig. 5D t-SNE.
- No official BeyondMimic VAE/diffusion checkpoint was added.
- No true DAgger rollout logs were recovered.
- No paper Fig. 5/Fig. 6 closed-loop videos were reproduced.
- No TensorRT asynchronous deployment audit was completed.
- No real Unitree G1 robot result was produced.

## Effect on English Reading Report

The English reading report can now describe a concrete Fig. 5B-adjacent closed-loop virtual experiment in the reproduction section. It strengthens the independent analysis because the report can discuss not only what worked, but also where local classifier-guided latent control remains fragile: a guidance objective may create measurable motion changes without cleanly matching a commanded transition speed profile.

## Next Step

The next technical step should be either a multi-seed transition proxy with better normalized speed targets and success/failure thresholds, or a return to the IsaacLab live headless gate if the goal is to move closer to official task-level replay and training rather than local report-facing diagnostics.

## Git Commit

Planned commit message: `feat: add importer transition guidance proxy`.

Commit hash: to be recorded by Git after this progress file is committed.
