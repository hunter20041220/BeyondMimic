# Unresolved Details

These items are not yet proven from local paper/source/config evidence:

- Paper-source figure/table/formula coverage is now indexed by
  `/mnt/infini-data/test/BeyondMimic/res/paper_source_coverage/paper_source_coverage_audit.json`; the remaining unresolved
  items below are the rows that require live execution, official artifacts, or paper-level rollout evidence rather than
  only LaTeX/source indexing.
- Official BeyondMimic VAE/diffusion implementation, checkpoints, TensorRT engine, and state-latent deployment artifacts
  are not present in the current local official inventory according to
  `/mnt/infini-data/test/BeyondMimic/res/level_c/official_artifact_audit/level_c_official_artifact_audit.json`.
- Figure 5 / Figure 6 paper-level reproduction is blocked from current local artifacts. The panel-level feasibility
  audit covers all 6 panels and found no released Fig.5/Fig.6 joystick, inpainting, SDF, latent, rollout, or closed-loop
  execution data: `/mnt/infini-data/test/BeyondMimic/res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json`.
- Results-section claims now have a source-indexed audit at
  `/mnt/infini-data/test/BeyondMimic/res/results_claims_audit/results_claims_audit.json`. It separates released-data Fig.3B/Fig.4C
  evidence from debug-only Fig.5/Fig.6 mechanisms, paper-only user-study claims, hardware-required claims, and unavailable
  checkpoint/rollout metrics. This audit is not a substitute for trained controller, Fig.5/Fig.6, MuJoCo, or hardware
  reproduction.
- Table `tab:skill_success` sim/real execution success is not reproduced. The local data audit checks LAFAN table rows
  against the G1 CSV release and records one missing listed motion plus two Real interval rows extending slightly past
  local CSV duration:
  `/mnt/infini-data/test/BeyondMimic/res/paper_skill_success_table_audit/skill_success_table_data_audit.json`.
- Conditional VAE architecture/loss details are now audited at debug-contract level by
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_contract_audit/level_c_vae_contract_audit.json`, which checks table
  values, source formulas, dimensions, runtime shapes, KL/reparameterization/interpolation math, and 15-step gradient
  accumulation. What remains unresolved is the paper-level trained VAE checkpoint, true DAgger rollout, closed-loop
  stability evaluation, and latent-space analysis.
- DAgger schedule, teacher query rule, aggregation cadence, and stopping criteria. A debug-only synthetic VAE
  accumulation probe now checks 15-step gradient accumulation and records a synthetic DAgger-batch manifest, but it is
  explicitly not true DAgger rollout over environment states.
- State-latent trajectory dataset exact split, rejection criteria, and full provenance schema. A debug-only
  retargeted-motion fixture now covers partial schema/provenance/split leakage checks, but not teacher/student rollout
  provenance or paper-exact dataset splits.
- Full diffusion training with paper batch size, epoch count, LR schedule, EMA updates, checkpoints, and validation
  metrics. A full paper Transformer architecture probe now verifies embed `512`, heads `8`, layers `6`, steps `20`, and
  one clean-trajectory backward pass on both the older `213`-D fixture token and the `131`-D paper-state token; a
  schedule probe records batch `512`, epochs `1000`, LR `1e-4`, weight decay `0.001`, warmup `10000`, cosine decay,
  and EMA `0.75/0.9999`; the parameter-count audit records both local variants are within 5% of the paper's approximate
  `~19.8M` statement but do not exactly match it. None of these probes is a training reproduction or exact checkpoint
  architecture.
- VAE rollout perturbation and episode rejection. The paper/source gives OU parameters and stability/rejection windows,
  and debug-only OU plus rollout-window manifests now exist, including
  `/mnt/infini-data/test/BeyondMimic/res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json`
  with 2.5 s/5 s windows, 100 synthetic OU seeds per valid start, and accept/reject fields. No live trained-VAE
  rollout/rejection data, true latent recording, or real 5 s stability verification has been collected.
- Paper-exact sagittal symmetry mapping for Unitree G1 is now audited at candidate/debug level:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json` confirms all
  29 controller joints are covered by 13 left/right pairs plus 3 center joints, match the official controller order and
  URDF names, and pass double-mirror checks. What remains unresolved is a definitive paper/official sign table and a
  trainable augmented VAE/diffusion dataset.
- Hybrid character-yaw-centric representation equations now have a source-aligned audit:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json`.
  The audit confirms body local positions match the paper formula, but it also records that the debug fixture is not the
  full paper state for root current-frame features, body velocity root-velocity subtraction, or full root relative
  position.
- Emphasis projection matrix definition beyond coefficient `c=6` remains unresolved for a paper-exact trainable state.
  The audit records the current debug fixture uses diagonal weights, while the paper/source describes a random Gaussian
  matrix and pseudoinverse projection.
- Full diffusion reverse sampler coefficients and TensorRT deployment details beyond the clean-trajectory training
  objective and reverse update text in `method.tex:187-206`. A debug-only oracle reverse-denoising probe exists, but it
  does not prove the paper-exact alpha/gamma/sigma schedule, a trained network, TensorRT deployment, or guided rollout.
- Trained VAE decoder checkpoint, paper-exact proprioception layout, asynchronous inference integration, TensorRT engine,
  CppAD deployment gradients, onboard RTX 4060 Mobile Mini PC latency, and live state-estimation/mocap task context. A
  debug-only decoder probe verifies current-latent receding-horizon decoding on CPU at the schema level, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json` indexes the
  paper deployment protocol plus local latency-budget/debug evidence, but it is not deployment reproduction.
- Paper-exact timestep masking policy for concrete history conditioning, future inpainting, and keyframe tasks. The
  smoke, debug schedule probe, oracle reverse-denoising probe, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json`
  now support independent state/latent denoising-step tensors and candidate masks, and the coverage audit separates the
  paper-explicit `k_s/k_z` mechanics from the unpublished deployment policy. The exact deployed task policy remains
  unresolved. A separate 99-D paper-formula state-window debug probe now exists at
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json`,
  but it uses synthetic latents and an oracle clean predictor, so it is still not paper-exact deployment behavior.
- Guidance scale selection protocol, validation/test scene split, and denoising-loop rollout evaluation. Formula-level
  joystick, waypoint, and SDF costs now have source-coverage and debug-gradient evidence in
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json`,
  and joystick guidance is wired into a debug-only oracle reverse-loop probe with a small local scale sweep. No
  paper-exact validation protocol or trained guided diffusion controller metrics have been produced.
- Paper-exact keyframe inpainting cost remains unresolved. The guidance coverage audit records that the paper
  demonstrates future keyframes but the local paper/source does not provide a unique keyframe cost equation; the local
  keyframe term is therefore marked as a candidate differentiable term.
- Released Zenodo file-to-panel traceability is now machine-audited for Level A released-data panels:
  `/mnt/infini-data/test/BeyondMimic/res/released_panel_mapping_audit/released_panel_mapping_audit.json`. The remaining
  panel-level blockers are paper-only or live-execution items, especially Fig. 5/Fig. 6 and real-hardware evidence.
