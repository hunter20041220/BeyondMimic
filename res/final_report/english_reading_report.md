# BeyondMimic Reading Report

## Abstract

BeyondMimic addresses a central tension in humanoid control: motion tracking can produce physically grounded behavior, but tracking a fixed reference clip is not the same as versatile task-directed control. The paper proposes a pipeline that first learns a strong motion-tracking teacher, then distills teacher behavior into a conditional latent action space, trains a state-latent diffusion model, and finally uses test-time guidance to satisfy new objectives.

This report combines paper reading with a public-resource reproduction. The local project does not fully reproduce BeyondMimic at paper-level. Instead, it provides an auditable partial reproduction: released-data figures and tables, official tracking-code audits, IsaacLab task gates, full public-motion replay diagnostics, local PPO/VAE/diffusion/guidance experiments, proxy closed-loop rollouts, and a clear record of what remains non-public or hardware-dependent.

## 1. Why The Paper Matters

Humanoid control is difficult because balance, contacts, high-dimensional joints, embodiment constraints, and task objectives are coupled. A controller that only imitates a library motion may look impressive but remains tied to reference trajectories. BeyondMimic is interesting because it treats imitation as a source of competence rather than the final goal. The tracking teacher supplies physical behavior, the VAE compresses action choices, diffusion models feasible state-latent trajectories, and guidance turns the learned prior toward new tasks.

## 2. Method Summary

I understand the method as six connected modules:

1. Motion tracking teacher: a PPO/RSL-RL/IsaacLab policy learns to track Unitree G1 motions.
2. Teacher rollout and DAgger-style data: the teacher's closed-loop state-action distribution becomes the downstream dataset.
3. Conditional action VAE: high-dimensional actions are compressed into a low-dimensional latent conditioned on robot state.
4. State-latent trajectory dataset: states and VAE latents are organized into temporal windows.
5. Latent diffusion: a denoiser learns the distribution of feasible future trajectories.
6. Test-time guidance: task costs such as velocity, waypoint, obstacle, transition, or inpainting objectives guide diffusion samples.

The elegant part is the division of labor. Reinforcement learning handles physical execution, the VAE gives a compact controllable action interface, diffusion handles sequence generation, and guidance injects task objectives without training a new policy for every task.

## 3. Reproduction Setup

The local project uses three project-local environments: an analysis environment for audits and plots, a diffusion environment with PyTorch CUDA, and a tracking environment for Isaac Sim, IsaacLab, RSL-RL, and the official `whole_body_tracking` stack. Raw downloaded materials are kept read-only, while scripts, reports, small JSON/CSV/Markdown evidence, and GitHub-tracked code live under the reproduction workspace. Large checkpoints, videos, raw rollout shards, and datasets stay local and are summarized through manifests rather than pushed to GitHub.

The current environment state is no longer "import-only". The headless IsaacLab AppLauncher gate is `ok`, and the G1 task construction gate is `ok_current_task_env_construction_gate`. This means the project can create and reset the local G1 tracking task, but that gate alone is not a paper-level tracking result.

## 4. Current Audit State

The current machine-readable evidence set is internally consistent:

- master audit: `ok`, `370/370` artifacts passing.
- artifact manifest: `1485` hashed artifacts, missing `0`.
- paper-vs-reproduction table: `227` rows.
- comparison types: exactly comparable `58`, approximately comparable `19`, qualitative-only `137`, not publicly reproducible `10`, requires real robot `3`.
- completion matrix: complete `74`, partial `130`, blocked `2`, out of scope `1`.
- required-artifact absence audit: `32` rows, with debug_only_not_required_artifact: 2, missing_required_artifact: 12, present_but_not_required_artifact: 18.

These numbers are useful because they prevent overclaiming. A large number of artifacts and passing audits does not mean the paper is fully reproduced. It means the current evidence is traceable and the remaining gaps are explicitly documented.

My current progress estimate has three layers. For the course reading report and defense, the material is about `85-90%` ready: the paper is understood, the evidence is organized, and the claim boundary is clear. For public-resource engineering coverage, the project is about `75-80%` complete: most released-data, source-audit, environment, and local virtual components are runnable or audited. For strict simulation-side paper-level reproduction, excluding the real robot, I would estimate only `40-50%`: the highest-weight closed-loop claims still need a stronger tracking teacher, true DAgger-style data, official-equivalent VAE/diffusion evidence, Fig. 5/Fig. 6 protocol metrics, and TensorRT deployment evidence.

## 5. What Has Been Reproduced Or Audited

The strongest exact evidence is in released-data and source-level reproduction. The project checks paper table values, released-data figures, panel mappings, formula/code traces, tracking observation and action schemas, reward and termination contracts, motion preprocessing contracts, ONNX interface contracts, and MuJoCo/ROS launch surfaces. This part is valuable because it tells us what the paper and public code actually specify.

On the tracking side, the project recovered a useful IsaacLab path. The official `csv_to_npz.py` and `replay_npz.py` loop bodies have been exercised over the full public G1 motion bundle with 40 motions and 11960 frames/steps. The captured official-importer-export G1 USDA path is stronger than the earlier generated scaffold because it comes from the Isaac Sim importer, but it is still a captured local asset path rather than a clean unmodified official converter entry.

This is the main official-loop virtual chain in the project: official-loop tracking/PPO eval begins with public G1 motions converted through the official-loop preprocessing body, replayed through the official-loop reference path, loaded into the local IsaacLab tracking task, used for local PPO training/evaluation, then connected to local teacher rollout, VAE, state-latent diffusion, and guidance experiments. I use the phrase "official-loop virtual chain" deliberately. It means the public code path and local simulation chain are substantially exercised, but the result is still virtual and resource-adjusted. It does not mean the unmodified official entrypoint, official teacher checkpoint, official DAgger dataset, or paper deployment stack has been reproduced.

Local PPO training and evaluation have been run on the public-motion bundle. The most important tracking-side finding is that motion data semantics matter as much as the policy. An FK-repaired motion bundle first fixed a degenerate `body_pos_w` problem, but later diagnostics found a subtler body-order mismatch: target body positions were written in URDF body order, while IsaacLab's runtime `MotionLoader` indexes `body_pos_w` by simulator articulation body order. The robot-order FK-repaired bundle is therefore the current mainline data path.

The current robot-order PPO checkpoint evaluation completed with status `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_completed`. It evaluated `612352` virtual environment steps and recorded reward mean about `0.02073384587805606`, done count `109170`, anchor-position error mean `0.07790673197711191`, body-position error mean `0.36114187777839774`, and joint-position error mean `1.5732512252785291`. The three-seed eval totals `1837056` virtual environment steps; its mean done rate is `0.1785340240036232`, reward mean `0.020480790998840676`, body-position error mean `0.3597400628005382`, and joint-position error mean `1.5772204704773731`. This is stable local virtual evidence, but it is not a paper-level teacher.

The latest tracking diagnostic explains why. Every multi-seed eval reports a step-0 done rate of `1.0` and a step-0 body-position error around `43.29219436645508` meters. Removing step 0 reduces mean body-position error to `0.2156714241976706`, but the post-step0 done rate remains around `0.175777426768736`. A reset-command warmup live probe found that command warmup `command_warmup_partially_reduces_reset_endpoint_z_spike`.

I then ran a full 2048-env x 299-step checkpoint evaluation with reset-command warmup: `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed`. It reduced the step-0 done count from `2048.0` to `568.0` and the step-0 body-position error from `43.294166564941406` m to `0.2640186548233032` m. However, the total done rate worsened from `0.1782798129180602` to `0.22864463576505017`. This is important negative evidence: reset warmup fixes a visible bootstrap artifact, but the checkpoint is still not a usable teacher. The next tracking fix should focus on post-warmup termination/policy-state mismatch before another downstream teacher rollout is collected.

A seed-matched follow-up made this conclusion stronger: `ok_robot_order_fk_warmup_seed_matched_phase_diagnostic`. With the same seed as the non-warmup baseline, step-0 done count and body error still improved, but total done rate worsened by `0.04325779943561875`, post-step0 done rate worsened by `0.04578210203439598`, and the `ee_body_pos` termination fraction increased by `0.04554896530100333` while the sampling top-bin delta stayed `0.0`. My current interpretation is: Seed matching confirms that reset-command warmup is not a teacher-quality fix. It removes the stale step-0 body target spike, but post-step0 done rate and ee_body_pos termination increase while the adaptive-sampling top bin stays unchanged. The likely next target is command/observation phase consistency: refresh motion targets after reset without introducing a one-command-step mismatch, or apply the same reset warmup consistently during training and evaluation.

The next diagnostic tested that recommendation directly with a no-advance reset-target refresh. The live probe status is `ok_robot_order_fk_reset_target_refresh_no_advance_live_probe`: endpoint-z done rate moved from `1.0` to `0.2734375`, endpoint-z error mean moved from `0.5298784375190735` to `0.104344442486763`, and `time_steps_unchanged_by_refresh` is `True`. The full 2048-env x 299-step eval status is `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed`. It reduced the step-0 done count by `-1453.0` and avoided the command-time advance, but the total done rate still moved from `0.1782798129180602` to `0.22340745192307693` and the post-step0 done-rate delta was `0.047659854760906034`. This narrows the tracking bottleneck: stale reset targets are real, but they are not sufficient to explain the weak teacher. The next repair should inspect reset state/action distribution, initial joint velocity mismatch, endpoint thresholds, and `ee_body_pos` termination before another large PPO/downstream chain.

A follow-up static trace diagnostic made that bottleneck measurable: `ok_robot_order_fk_reset_state_action_distribution_diagnostic`. It compares the same-seed baseline, reset-command warmup, and no-advance target-refresh full eval traces. Target refresh reduces the step-0 body-position error by `-43.02953788638115` m, but the step-0 joint-velocity error increases by `17.829124450683594`, the first-five-step action mean increases by `0.07184725403785702`, the post-step0 done-rate delta is `0.047659854760906034`, and the `ee_body_pos` termination fraction delta is `0.0478825904055184`. My current conclusion is: No-advance target refresh removes the stale step-0 body target, but it exposes or creates a large initial joint-velocity/action transient and still worsens post-step0 done rate. The current teacher should not be used as the final DAgger/VAE/diffusion data source.

For Level C, the project implements a paper-faithful local chain: teacher rollout, conditional VAE, state-latent windows, denoiser/diffusion training, offline guidance, and local proxy closed-loop guidance. This proves that the method can be studied and partially recreated from public resources, but it is not the official BeyondMimic VAE/diffusion checkpoint chain.

## 6. From Paper Equations To Code

I treated the paper formulas as software contracts. The tracking objective became reward and termination checks over anchor pose, target body positions, endpoint height, action regularization, and contact-like events. The VAE objective became a state-conditioned encoder/decoder with reparameterization, reconstruction error, KL regularization, finite-output checks, and checkpoint save/load tests. The diffusion objective became noisy state-latent sequence prediction with train/validation/test splits and denoising-improvement metrics. The guidance equations became task-cost gradients over sampled trajectories.

This matters for a reading report because it shows independent exploration rather than only summarizing the paper. Implementing the formulas forced me to decide which variables are directly public, which are inferred from source code, and which are local proxies because the paper's exact dataset or checkpoint is not available.

## 7. Local Fig. 5 / Fig. 6 Proxy Evidence

The project has consolidated the local guidance tasks into a unified protocol table. It covers `6` local proxy tasks with `4` multi-seed proxy groups and `2` single-seed proxy groups. The important number is `paper_level_reproduced_count = 0`. This means the local protocol is useful for analysis and presentation, but it must not be described as reproducing the paper's Fig. 5 or Fig. 6.

The current protocol is best described as a local virtual BeyondMimic-like pipeline. It covers joystick, waypoint, obstacle avoidance, composed objectives, transition, and inpainting-style proxies. The next scientific step is to make task metrics stronger: velocity error for joystick, final distance and success rate for waypoint, clearance and collision counts for obstacle avoidance, keyframe error for inpainting, transition smoothness and fall rate for transitions, and guided-vs-unguided improvement for each task.

## 8. Storage And Artifact Management

The project deliberately keeps GitHub lightweight. Large environments, checkpoints, videos, raw rollout shards, datasets, and caches are not committed. The latest conservative cleanup audit is `10` deleted-or-previously-deleted bulky candidates and `4853459410` managed bytes removed or confirmed absent. Current disk free space is about `137.16` GiB of `249856.0` GiB on the project filesystem. The policy is conservative: delete failed, duplicate, or rebuildable bulky directories; keep current active run directories and preserve JSON/CSV/Markdown/log evidence.

## 9. Limitations

The major missing pieces are not cosmetic. They are the pieces that make the original paper a closed-loop humanoid-control result:

- no official BeyondMimic tracking teacher checkpoint.
- no official motion-policy ONNX export from a reproduced trained teacher.
- no true DAgger rollout logs from a mature teacher/student loop.
- no official conditional VAE checkpoint.
- no official state-latent diffusion Transformer checkpoint.
- no paper-level Fig. 5/Fig. 6 closed-loop task logs, success/failure videos, or metrics.
- no TensorRT engine, Mini-PC latency benchmark, or asynchronous deployment reproduction.
- no real Unitree G1 hardware validation.

The largest current technical blocker, excluding real robot work, is tracking quality. The pipeline runs, but the local teacher terminates too often and does not yet provide the stable rollout distribution needed for convincing DAgger, VAE, diffusion, and guidance reproduction.

This boundary also shapes how I would present the result in class. I would not say "I reproduced BeyondMimic." I would say: this project does not fully reproduce BeyondMimic at paper-level, but it reproduces and audits a large public subset, rebuilds the method as a local virtual pipeline, and identifies the exact missing artifacts needed to close the gap. That is a more useful scientific statement than a vague success claim.

## 10. Personal Reflection

This reproduction changed how I read the paper. At first the method looks like a clean sequence of modules: tracking, VAE, diffusion, guidance. In practice, every module depends on embodied details: robot assets, body names, endpoint heights, reset logic, termination thresholds, observation history, simulation stability, and data provenance. A small coordinate or body-position issue can invalidate a beautiful downstream model.

The most important lesson is that robotics reproducibility is not only about code availability. It needs assets, checkpoints, datasets, evaluation scripts, logs, videos, and deployment details. BeyondMimic is technically compelling, but the public artifact boundary makes exact reproduction impossible at several points. A good reproduction report should therefore avoid a binary "success/failure" story. The honest story is that many public components can be reproduced and analyzed, a local virtual pipeline can be built, and the remaining paper-level claims require non-public artifacts or hardware.

## 11. Conclusion

This project currently supports a strong course reading report and defense: it explains the paper, audits the public code and data, implements the main ideas in a local pipeline, and identifies where paper-level reproduction is blocked. It does not fully reproduce BeyondMimic at paper level. The next research step is to repair tracking quality, train a more reliable teacher, then rerun the downstream VAE, state-latent diffusion, and guidance experiments from that stronger teacher.
