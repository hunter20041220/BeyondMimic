# BeyondMimic Reproduction Workspace

This workspace follows `/mnt/infini-data/test/BeyondMimic/goal.md`.

Raw downloaded materials live in `/mnt/infini-data/test/BeyondMimic/download` and are treated as read-only. Reproduction
code, worktrees, environments, caches, logs, and results stay under `/mnt/infini-data/test/BeyondMimic`.

Current status:

- Level A released-data reproduction is complete for the locally released dataset scope.
- Level B official tracking/deployment evidence is partial: source/config/controller audits pass, but live IsaacLab/Kit
  execution is blocked by host inotify limits and ROS 2 Jazzy/Noble deployment is not available on this host.
- Level C VAE/diffusion evidence is debug/mechanics-only: paper parameters, schemas, formulas, masks, probes, and
  blocking conditions are audited, but official Level C code/checkpoints and paper-level Fig. 5/Fig. 6 rollouts are not
  available locally.
- Consolidated report: `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`.

Long PPO, VAE, or diffusion training must not start until the smoke gates and external blockers recorded in
`/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json` are resolved.
