# Discrepancy Report

The following discrepancies and execution gaps are active:

1. `goal.md` lists adaptive sampling look-back `u in {0,1,2}`, while the inspected official code default is
   `adaptive_kernel_size=1` in `mdp/commands.py`. Paper source confirms `u in {0,1,2}` in
   `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:450-456`. The local probe in
   `/mnt/infini-data/test/BeyondMimic/res/tracking/smoke_config_audit/tracking_config_audit.json` shows the default
   code kernel and paper three-bin kernel produce different probability distributions for a single failure bin
   (`l1_difference=1.0730253353204173`, pre-failure mass `0.02272727272727273 -> 0.5592399403874814`). Probe outputs are stored in
   `/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_probe`, and the machine-readable discrepancy audit is
   `/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json`.
   No local official-tree runtime override to kernel size 3 has been found.
2. Host has no `conda`, `mamba`, `micromamba`, or `nvcc`, while official tracking requires a Python 3.10 Isaac Sim /
   Isaac Lab stack. Phase 1 will use project-local micromamba unless a better local environment is discovered.
3. `motion_tracking_controller` requires ROS 2 Jazzy on Ubuntu Noble according to its README, but the host is Ubuntu
   20.04.5. This affects official sim-to-sim deployment and must be separated from Isaac training reproduction.
4. Isaac Sim pip installation attempt failed in `bm_tracking` on Ubuntu 20.04. The placeholder package looked for
   `isaacsim-4.5.0.0-cp310-none-manylinux_2_34_x86_64.whl`, which is not compatible with the host platform. This
   confirms the local IsaacLab documentation recommendation to use Isaac Sim binary installation on Ubuntu 20.04.
5. IsaacLab/Kit headless smoke is blocked by inotify watch exhaustion, shown as `Failed to create change watch ... errno=28`.
   This is not a disk-capacity error. Current host limits are `fs.inotify.max_user_watches=8192` and
   `fs.inotify.max_user_instances=128`. A root/admin change such as increasing `max_user_watches` is likely required
   before full Kit-based tracking environment smoke can be adjudicated.
6. The IsaacLab editable installation is sufficient for plain `import isaaclab` and non-Kit asset checks, but optional
   dependencies required by the package metadata are not fully reconciled yet (`dex-retargeting`, `pin-pink`,
   `transformers`). This must be revisited before claiming complete training-stack equivalence.
7. A non-Kit tracking configuration audit now verifies official-code parameters, G1 target bodies, and assets, but it is not a substitute
   for an executable IsaacLab environment rollout. It should be treated as a smoke-stage bridge while the Kit/inotify
   blocker remains unresolved.
8. Level C now has a synthetic PyTorch VAE/diffusion/guidance smoke under
   `/mnt/infini-data/test/BeyondMimic/res/level_c/synthetic_smoke`, but this is not evidence of faithful guided-diffusion
   reproduction. The official inventory still lacks BeyondMimic-specific VAE/diffusion code, trained checkpoints,
   TensorRT engines, teacher rollouts, or state-latent datasets needed for Fig. 5/Fig. 6 reproduction.
9. Table `tab:skill_success` LAFAN data availability does not exactly match the local G1 CSV release. The audit
   `/mnt/infini-data/test/BeyondMimic/res/paper_skill_success_table_audit/skill_success_table_data_audit.json` records one
   listed LAFAN motion missing locally (`run1_subject4`) and two Real interval rows whose end time slightly exceeds the
   corresponding local 30 Hz CSV duration (`walk1_subject5`, `walk2_subject4`). This is separate from the larger fact
   that the table's `Sim=Full` and Real execution claims are not reproduced on this host.
