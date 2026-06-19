# Progress Update

## Goal

Move beyond the earlier 21-step decoded-action bridge by running a longer local closed-loop action-guidance rollout in IsaacLab. The target was to produce report-ready robot motion evidence while keeping the claim boundary clear: local virtual action-space guidance, not official BeyondMimic latent diffusion guidance.

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
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Existing policy, VAE, and decoded-action rollout scripts/results.

## Files Modified

- `reproduction/scripts/tracking_g1_official_csv_loop_action_guidance_rollout_eval.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/required_artifact_absence_audit.py`
- `reproduction/scripts/visual_media_inventory_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `res/final_report/english_reading_report.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_g1_official_csv_loop_action_guidance_rollout_eval.py
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7
python3 reproduction/scripts/tracking_g1_official_csv_loop_action_guidance_rollout_eval.py
python3 reproduction/scripts/visual_media_inventory_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/reproduction_master_audit.py
```

The final full verification chain is recorded in the user-facing turn summary.

## Results

New result directories:

```text
res/level_c/official_csv_loop_action_guidance_rollout_eval/
res/visualization/official_csv_loop_action_guidance_rollout/
res/runs/tracking_g1_official_csv_loop_action_guidance_rollout_eval/action_guidance_rollout_20260619_071538_seed20260639/
logs/tracking_g1_official_csv_loop_action_guidance_rollout_eval/
```

The run executed three 299-step variants in `Tracking-Flat-G1-v0`:

```text
teacher
vae_base
action_guided
```

The action-guided variant used:

```text
a_guided = a_vae + 0.35 * (a_teacher - a_vae)
```

Selected metrics:

```text
status: ok_official_csv_loop_action_guidance_rollout_eval
selected physical GPU: 4
rollout steps: 299
action_guided reward mean: 0.02557246644286607
action_guided target-body error mean: 0.07946009188890457
action_guided done count total: 25
action_guided guided-vs-teacher action MSE mean: 0.001721034897277194
VAE-base target-body error mean: 0.07934186607599258
teacher target-body error mean: 0.08290184289216995
MP4 size: 905475 bytes
```

New local visualization assets:

```text
official_csv_loop_action_guidance_rollout_vs_reference.mp4
official_csv_loop_action_guidance_rollout_keyframes.png
official_csv_loop_action_guidance_rollout_metrics.png
official_csv_loop_action_guidance_rollout_metrics.csv
official_csv_loop_action_guidance_rollout_asset.json
```

The MP4 is intentionally local-only and is not committed to GitHub. Its path, size, and SHA256 are recorded in the asset JSON.

## Verification

Initial integration checks passed:

```text
visual_media_inventory_audit: ok, 136 visual rows, 4 videos
required_artifact_absence_audit: ok, 25 rows
artifact_manifest: ok, 418 artifacts
final_reproduction_report: ok
paper_vs_reproduction_comparison: ok
reproduction_master_audit: ok
```

## Failed / Blocked Items

- This is not the paper receding-horizon latent diffusion controller.
- It does not use the official BeyondMimic diffusion checkpoint.
- It does not evaluate joystick, waypoint, inpainting, or obstacle success rates from Fig. 5/Fig. 6.
- It is not real-robot evidence.
- It is a single-environment report/evidence rollout on GPU4, not a formal two-GPU training/evaluation run; the >=10GB per GPU formal-experiment threshold is not applicable.

## Effect on English Reading Report

The English report now has a stronger visual and quantitative bridge between offline guidance and closed-loop simulation: a robot video plus teacher/VAE/action-guided comparison curves. It helps explain the reproduction trajectory honestly: the project can execute guided local actions in IsaacLab, but it still cannot claim official paper-level guided diffusion reproduction.

## Next Step

The natural next technical step is true receding-horizon latent/action guidance: generate a short horizon from the local state-latent denoiser at each control step, decode only the current latent/action, step IsaacLab, and compare guided versus unguided rollouts over at least one paper-style task proxy.

## Git Commit

Pending at the time this progress file was written.
