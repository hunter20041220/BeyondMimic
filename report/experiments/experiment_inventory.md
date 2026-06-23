# Experiment Results

## Stage 1 Multi-Source Teacher

- Training run: `res/tracking/stage1_multisource_paper_contract_ppo_training_run/tracking_stage1_multisource_paper_contract_ppo_training_run.json`
- Checkpoint sweep: `res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json`
- Best checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_paper_contract_ppo_training/resource_adjusted_ppo_20260622_114146_seed20260851/rank_0/model_29999.pt`
- Best iteration: `29999`
- Best reward mean: `0.024131401152315747`
- Best body-position error mean: `1.0095036663737982`
- Best joint-position error mean: `1.6739522380175`

Interpretation: the 5/6 training completed and the checkpoint sweep is real, but the best teacher is still weak.

## Teacher Rollout and VAE

- Teacher rollout samples: `612352`
- Rollout done count: `118220`
- VAE test action MSE: `0.003289680986199528`
- VAE test absolute action error mean: `0.04251094348728657`

## State-Latent Diffusion

The denoiser reduces token prediction error from `0.072816` to `0.043221`, corresponding to approximately `40.6%` relative denoising improvement.

This indicates that the diffusion model has learned a non-trivial denoising mapping at the token level. However, token-level MSE improvement does not imply closed-loop humanoid control success. The current videos still show unstable or incomplete motion, so the diffusion model is not yet a successful BeyondMimic controller.

Figures:

- `report/figures/denoising_mse_improvement.png`
- `report/figures/metric_plots/stage1_checkpoint_sweep.png`

## Guidance

- Guidance status: `ok_stage1_multisource_state_latent_guidance_eval`
- Selected windows: `8192`
- Tasks with all best costs improved: `4`
- Tasks with nonzero best gradients: `4`

This is offline guidance evidence only, not paper-level Fig. 5/Fig. 6 closed-loop success.
