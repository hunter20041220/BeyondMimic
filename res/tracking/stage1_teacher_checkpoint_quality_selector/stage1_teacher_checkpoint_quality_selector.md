# Stage 1 Teacher Checkpoint Quality Selector

- Status: `blocked_no_downstream_ready_teacher_checkpoint`
- Candidates scanned: `227`
- Usable diagnostic candidates: `0`
- Downstream-ready candidates: `0`

## Thresholds

- `downstream_ready_done_rate`: `0.05`
- `candidate_done_rate`: `0.1`
- `max_body_pos_error_mean`: `0.2`
- `max_joint_pos_error_mean`: `1.0`
- `max_anchor_pos_error_mean`: `0.12`
- `max_action_abs_mean`: `1.2`
- `min_eval_steps`: `250`
- `min_num_envs`: `128`

## Best Ranked Candidate

- Category: `other_tracking_eval`
- Iteration: `299`
- Decision: `not_ready_for_vae_diffusion_due_to_quality_gate`
- Done rate: `0.08585258152173914`
- Body position error mean: `0.2212558212786614`
- Joint position error mean: `1.5075325571175004`
- Checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_ppo_training/resource_adjusted_ppo_20260618_224626_seed20260629/rank_0/model_299.pt`
- Blockers: `done_rate_above_downstream_threshold; error_body_pos_mean_above_threshold; error_joint_pos_mean_above_threshold; error_anchor_pos_mean_above_threshold`

## Top 10 Candidates

- `other_tracking_eval` iter `299`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.08585258152173914`, body `0.2212558212786614`, joint `1.5075325571175004`
- `other_tracking_eval` iter `299`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.0857480664715719`, body `0.22126627990035308`, joint `1.5127941934161362`
- `other_tracking_eval` iter `299`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.08579379180602006`, body `0.22124746835550735`, joint `1.5209383613688492`
- `other_tracking_eval` iter `99`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.08604201505016723`, body `0.22102177402247553`, joint `1.5326235804669435`
- `other_tracking_eval` iter `299`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.0859505643812709`, body `0.22112115166059704`, joint `1.5322400187967613`
- `other_tracking_eval` iter `299`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.08735498536789298`, body `0.2203947813764065`, joint `1.7990627396465544`
- `official_importer_export_paper_contract_4_7_gpu` iter `29500`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.15436872909698995`, body `0.2534336016130288`, joint `1.7281810890472056`
- `fk_repaired_robot_order_diagnostic` iter `999`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.11093782660953178`, body `0.3621071465936393`, joint `2.5801463800927866`
- `official_importer_export_paper_contract_4_7_gpu` iter `25500`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.15858852424749165`, body `0.3348708672567355`, joint `1.898236128398806`
- `official_importer_export_paper_contract_4_7_gpu` iter `25500`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.15858852424749165`, body `0.3348708672567355`, joint `1.898236128398806`

## Best Candidate Per Family

- `fk_repaired_robot_order_diagnostic` iter `999`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.11093782660953178`, body `0.3621071465936393`, joint `2.5801463800927866`
- `legacy_scaled_ppo_diagnostic` iter `999`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.997857441471572`, body `0.5973028396084955`, joint `0.9173021256724329`
- `official_importer_export_paper_contract_4_7_gpu` iter `29500`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.15436872909698995`, body `0.2534336016130288`, joint `1.7281810890472056`
- `other_tracking_eval` iter `299`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.08585258152173914`, body `0.2212558212786614`, joint `1.5075325571175004`
- `stage1_multisource_5_6_gpu` iter `29999`: decision `not_ready_for_vae_diffusion_due_to_quality_gate`, done `0.19413670568561872`, body `1.0095036663737982`, joint `1.6739522380175`

## Claim Boundary

This selector ranks local tracking checkpoints for downstream data collection only. It does not certify official BeyondMimic paper-level tracking, DAgger, VAE, diffusion, Fig.5/Fig.6, Isaac rendered rollout, or real-robot results.

当前不得声称完整复现 BeyondMimic；该选择器只证明本地 teacher checkpoint 是否适合继续采集 VAE/diffusion 数据。
