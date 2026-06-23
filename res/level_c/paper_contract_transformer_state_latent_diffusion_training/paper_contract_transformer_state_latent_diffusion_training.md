# Paper-Contract Transformer State-Latent Diffusion

- Status: `ok_paper_contract_transformer_diffusion_dry_run`
- Dry run: `True`
- Parameter count: `19142848`
- Sequence length: `21`
- Embedding/head/layer: `512` / `8` / `6`
- Denoising steps: `20`

## Checks

- `does_not_claim_closed_loop_guidance`: `True`
- `does_not_claim_official_diffusion_checkpoint`: `True`
- `does_not_claim_paper_level_diffusion`: `True`
- `does_not_claim_real_robot`: `True`
- `forward_backward_ok`: `True`
- `paper_attention_heads_8`: `True`
- `paper_batch_size_512`: `True`
- `paper_denoising_steps_20`: `True`
- `paper_ema_max_0_9999`: `True`
- `paper_ema_power_0_75`: `True`
- `paper_embedding_dim_512`: `True`
- `paper_history_horizon_4_16`: `True`
- `paper_learning_rate_1e_4`: `True`
- `paper_transformer_layers_6`: `True`
- `paper_warmup_steps_10000`: `True`
- `paper_weight_decay_0_001`: `True`
- `uses_individual_state_and_latent_denoising_steps`: `True`
- `uses_transformer_encoder`: `True`

## Claim Boundary

This is a local code-contract route over local teacher/VAE rollouts. It is not an official BeyondMimic diffusion checkpoint, not a closed-loop guidance result, not Isaac-rendered evidence, and not real-robot evidence.

当前不得声称完整复现 BeyondMimic；该脚本只证明 Transformer diffusion 代码合同开始对齐论文结构。
