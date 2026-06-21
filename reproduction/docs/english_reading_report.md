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

- master audit: `ok`, `345/345` audited artifacts passing, with the robot-order FK PPO multiseed gates included in the verification chain.
- artifact manifest: `1415` hashed artifacts, including the robot-order FK PPO multiseed eval script, summary tables, and report assets.
- paper-vs-reproduction table: `220` rows after adding the robot-order FK PPO multiseed eval row.
- comparison types: exactly comparable `58`, approximately comparable `19`, qualitative-only `130`, not publicly reproducible `10`, requires real robot `3`.
- completion matrix: complete `74`, partial `123`, blocked `2`, out of scope `1`.
- required-artifact absence audit: `32` rows, with debug_only_not_required_artifact: 2, missing_required_artifact: 12, present_but_not_required_artifact: 18.

These numbers are useful because they prevent overclaiming. A large number of artifacts and passing audits does not mean the paper is fully reproduced. It means the current evidence is traceable and the remaining gaps are explicitly documented.

## 5. What Has Been Reproduced Or Audited

The strongest exact evidence is in released-data and source-level reproduction. The project checks paper table values, released-data figures, panel mappings, formula/code traces, tracking observation and action schemas, reward and termination contracts, motion preprocessing contracts, ONNX interface contracts, and MuJoCo/ROS launch surfaces. This part is valuable because it tells us what the paper and public code actually specify.

On the tracking side, the project recovered a useful IsaacLab path. The official `csv_to_npz.py` and `replay_npz.py` loop bodies have been exercised over the full public G1 motion bundle with 40 motions and 11960 frames/steps. The captured official-importer-export G1 USDA path is stronger than the earlier generated scaffold because it comes from the Isaac Sim importer, but it is still a captured local asset path rather than a clean unmodified official converter entry.

This is the main official-loop virtual chain in the project: official-loop tracking/PPO eval begins with public G1 motions converted through the official-loop preprocessing body, replayed through the official-loop reference path, loaded into the local IsaacLab tracking task, used for local PPO training/evaluation, then connected to local teacher rollout, VAE, state-latent diffusion, and guidance experiments. I use the phrase "official-loop virtual chain" deliberately. It means the public code path and local simulation chain are substantially exercised, but the result is still virtual and resource-adjusted. It does not mean the unmodified official entrypoint, official teacher checkpoint, official DAgger dataset, or paper deployment stack has been reproduced.

Local PPO training and evaluation have been run on the public-motion bundle. The scaled official-importer-export PPO chain ran through a larger local training/evaluation protocol, and the FK-repaired chain fixed an earlier `body_pos_w` degeneracy in the motion bundle. However, the first FK-repaired bundle still hid a more subtle but important indexing problem: the motion targets were written in URDF body order, while IsaacLab's runtime `MotionLoader` indexes `body_pos_w` using the simulator articulation body order. A live probe showed misindexed targets, including endpoint height errors larger than one meter after a single zero-action step.

The latest repair reorders the full 40-motion FK bundle into IsaacLab robot body order. This is now the strongest tracking data-quality result in the project. On the full split zero-action task diagnostic, the old FK bundle produced `11958/11960` done or termination events. The robot-order bundle reduced this to `2166/11960`, reduced mean anchor error from about `0.494` to `0.084`, and reduced mean body-position error from about `0.516` to `0.214`.

I then trained and evaluated a new local PPO baseline from this robot-order FK-repaired bundle. The run used GPUs 4 and 7 for 1000 PPO iterations, 4096 total environments, and produced 21 checkpoints. The iteration-999 checkpoint evaluation used 2048 environments for 299 steps, giving `612352` virtual environment steps. Its done rate is about `0.178`, reward mean is `0.0207`, anchor-position error mean is `0.0779`, body-position error mean is `0.3611`, and joint-position error mean is `1.5733`. This is much better than the older URDF-order FK checkpoint, whose done rate was almost one, but it is still not a paper-level tracking teacher.

The tracking gate now treats the robot-order FK PPO checkpoint as the strongest local virtual baseline for report curves and video, not as final downstream data. I also generated a 299-frame policy-vs-reference rollout video from this checkpoint. In that single-env rollout, the asset records target-body error mean `0.1547`, target-body error max `0.2961`, reward mean `0.0244`, and done count `44`. This is useful visual evidence for the current baseline, but it remains local virtual media, not a paper metric.

I then ran a full three-seed checkpoint evaluation for this same robot-order FK-repaired iteration-999 policy. The multiseed gate used seeds `20260730`, `20260731`, and `20260732`, each with 2048 environments for 299 steps, totaling `1,837,056` virtual environment steps. The result is stable across seeds but still weak as a teacher: mean done rate `0.1785`, reward mean `0.02048`, anchor-position error mean `0.07762`, body-position error mean `0.35974`, and joint-position error mean `1.57722`. This closes the immediate multi-seed eval gap, but it strengthens the negative conclusion: the next tracking step should be checkpoint sweep, termination diagnostics, and stronger or longer PPO training before using this policy as final DAgger/VAE/diffusion data.

A follow-up diagnostic made the bottleneck more precise. All three multi-seed evals report `2048/2048` done at step 0 with a body-position error spike around `43.29` m. If step 0 is removed, body-position error drops from about `0.360` to about `0.216`, but the post-step0 done rate remains about `0.176`. This suggests that the next mainline tracking work should first inspect reset/target alignment and `ee_body_pos` termination, then rerun PPO only after that source of early termination is understood.

For Level C, the project implements a paper-faithful local chain: teacher rollout, conditional VAE, state-latent windows, denoiser/diffusion training, offline guidance, and local proxy closed-loop guidance. This proves that the method can be studied and partially recreated from public resources, but it is not the official BeyondMimic VAE/diffusion checkpoint chain.

## 6. Local Fig. 5 / Fig. 6 Proxy Evidence

    The project has consolidated the local guidance tasks into a unified protocol table. It covers `6` local proxy tasks with `4` multi-seed proxy groups and `2` single-seed proxy groups. The important number is `paper_level_reproduced_count = 0`. This means the local protocol is useful for analysis and presentation, but it must not be described as reproducing the paper's Fig. 5 or Fig. 6.

The current protocol is best described as a local virtual BeyondMimic-like pipeline. It covers joystick, waypoint, obstacle avoidance, composed objectives, transition, and inpainting-style proxies. The next scientific step is to make task metrics stronger: velocity error for joystick, final distance and success rate for waypoint, clearance and collision counts for obstacle avoidance, keyframe error for inpainting, transition smoothness and fall rate for transitions, and guided-vs-unguided improvement for each task.

## 7. Limitations

The major missing pieces are not cosmetic. They are the pieces that make the original paper a closed-loop humanoid-control result:

- no official BeyondMimic tracking teacher checkpoint.
- no official motion-policy ONNX export from a reproduced trained teacher.
- no true DAgger rollout logs from a mature teacher/student loop.
- no official conditional VAE checkpoint.
- no official state-latent diffusion Transformer checkpoint.
- no paper-level Fig. 5/Fig. 6 closed-loop task logs, success/failure videos, or metrics.
- no TensorRT engine, Mini-PC latency benchmark, or asynchronous deployment reproduction.
- no real Unitree G1 hardware validation.

The largest current technical blocker, excluding real robot work, is tracking quality. The pipeline runs, and the robot-order FK repair plus the new PPO baseline made the tracking evidence much stronger, but the teacher is still not mature enough for a paper-level DAgger/VAE/diffusion chain.

This boundary also shapes how I would present the result in class. I would not say "I reproduced BeyondMimic." I would say: this project does not fully reproduce BeyondMimic at paper-level, but it reproduces and audits a large public subset, rebuilds the method as a local virtual pipeline, and identifies the exact missing artifacts needed to close the gap. That is a more useful scientific statement than a vague success claim.

## 8. Personal Reflection

This reproduction changed how I read the paper. At first the method looks like a clean sequence of modules: tracking, VAE, diffusion, guidance. In practice, every module depends on embodied details: robot assets, body names, endpoint heights, reset logic, termination thresholds, observation history, simulation stability, and data provenance. A small coordinate or body-position issue can invalidate a beautiful downstream model.

The most important lesson is that robotics reproducibility is not only about code availability. It needs assets, checkpoints, datasets, evaluation scripts, logs, videos, and deployment details. BeyondMimic is technically compelling, but the public artifact boundary makes exact reproduction impossible at several points. A good reproduction report should therefore avoid a binary "success/failure" story. The honest story is that many public components can be reproduced and analyzed, a local virtual pipeline can be built, and the remaining paper-level claims require non-public artifacts or hardware.

## 9. Conclusion

This project currently supports a strong course reading report and defense: it explains the paper, audits the public code and data, implements the main ideas in a local pipeline, and identifies where paper-level reproduction is blocked. It does not fully reproduce BeyondMimic at paper level. The next research step is to fix the robot-order FK PPO reset/target-alignment and `ee_body_pos` termination bottleneck, then rerun stronger tracking PPO before rerunning downstream VAE, state-latent diffusion, and guidance experiments.
