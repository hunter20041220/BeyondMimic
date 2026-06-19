# Current Environment And Reproduction Status

This is the final-report copy of:

```text
/mnt/infini-data/test/BeyondMimic/reproduction/docs/current_environment_and_reproduction_status.md
```

Generated: 2026-06-19 15:34 Asia/Shanghai

## Executive Summary

The BeyondMimic workspace is now a substantial, auditable reproduction project, but not a complete paper-level reproduction. `bm_analysis` and `bm_diffusion` are usable; `bm_tracking` has passed the current IsaacLab/AppLauncher headless gate, but unpatched official G1 conversion/replay remains blocked.

Current audit counts:

```text
master_audit: ok, 257/257 artifacts passed
artifact_manifest: 418 artifacts
paper_vs_reproduction: 147 rows
completion matrix: complete 73, partial 88, blocked 3, out_of_scope 1
goal_complete: false
```

Paper-vs-reproduction comparison:

```text
exactly_comparable:          58
approximately_comparable:    19
qualitative_only:            57
not_publicly_reproducible:   10
requires_real_robot:          3
```

The strongest current evidence includes released-data reproduction, official tracking static/config audits, restored IsaacLab gates, resource-adjusted official-csv-loop PPO/evaluation/teacher rollouts, local VAE training, local state-latent diffusion training, offline guidance, guided latent action decoding, local VAE closed-loop rollout, local teacher-consistency action-guidance rollout, and a first local receding-horizon latent-guidance rollout.

The missing paper-level gates remain official unpatched G1 replay, full paper-scale PPO teacher training/evaluation, true official DAgger logs, official BeyondMimic VAE/diffusion checkpoints, receding-horizon latent diffusion control matching Fig. 5/Fig. 6, TensorRT/asynchronous deployment at paper level, and real robot results.

## Environment Snapshot

- Host Python: `/usr/bin/python3`, Python `3.10.12`
- Driver: `570.124.06`
- CUDA in `nvidia-smi`: `12.8`
- GPUs: 8 x NVIDIA H20, about `95.09 GiB` each
- Disk: `/mnt/infini-data` has about `293G` available; inodes are healthy
- `bm_analysis`: numpy, pandas, matplotlib, onnx, onnxruntime import successfully
- `bm_diffusion`: PyTorch `2.5.1+cu121`, CUDA available, two visible H20 GPUs under `CUDA_VISIBLE_DEVICES=5,6`
- `bm_tracking`: IsaacLab/AppLauncher headless gate is `ok`; direct plain-Python deep Kit imports still require live Kit runtime

## Simulation Work Still Worth Doing

1. Recover unpatched official G1 `csv_to_npz.py` / `replay_npz.py`.
2. Run longer/multi-seed local official-csv-loop PPO evaluation.
3. Expand closed-loop VAE reconstruction and autonomous latent tests.
4. Extend the first receding-horizon latent guidance rollout to multi-seed and task-specific joystick/waypoint/inpainting/obstacle proxy costs.
5. Add explicit with/without-guidance success/failure metrics with non-paper-level labels.
6. Run local ONNX/TensorRT/asynchronous deployment audits for VAE/denoiser components.
7. Keep small visual/report assets for the English reading report, with honest captions.

## Boundary

This project currently cannot claim a full BeyondMimic reproduction. It can claim a large, auditable reproduction workspace and a useful reading-report evidence base.

Current不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
