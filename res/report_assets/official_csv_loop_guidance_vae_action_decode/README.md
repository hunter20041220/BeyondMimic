# Official-Loop Guidance VAE Action Decode Assets

These assets visualize the offline bridge from guided state-latent denoiser outputs to decoded
29D VAE actions. They are report/PPT assets, not paper-level closed-loop rollout evidence.

## Source

- Eval JSON: `/mnt/infini-data/test/BeyondMimic/res/level_c/official_csv_loop_guidance_vae_action_decode_eval/level_c_official_csv_loop_guidance_vae_action_decode_eval.json`
- Rows TSV: `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_official_csv_loop_guidance_vae_action_decode_eval/official_csv_loop_guided_decode_20260619_122046_seed20260636/official_csv_loop_guidance_vae_action_decode_rows.tsv`
- Status: `ok_official_csv_loop_guidance_vae_action_decode_eval`
- Total windows: `57140`
- Tasks with finite decoded actions: `4`

## Assets

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_guidance_vae_action_decode/guided_vs_base_action_decode_metrics.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_guidance_vae_action_decode/guided_action_teacher_mse_by_split.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_guidance_vae_action_decode/guided_action_decode_metrics.csv`

## Claim Level

qualitative_only / offline action-decode gate. This does not claim IsaacLab closed-loop guidance,
Fig. 5/Fig. 6 reproduction, TensorRT deployment, or real robot validation.
