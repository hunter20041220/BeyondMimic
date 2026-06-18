# Known Limitations

- Real-robot deployment is out of scope on this machine because no Unitree G1 hardware is available or connected.
- Full IsaacLab/Kit smoke is not passing yet because the host inotify watch limit is too low for the extracted Isaac Sim
  extension tree.
- Current user permission cannot raise `fs.inotify.max_user_watches` or `fs.inotify.max_user_instances`; this needs
  administrator action before live IsaacLab smoke/training can pass.
- The current tracking smoke evidence includes source/config/asset/data audits, but not a live IsaacLab simulation
  rollout because of the Kit/inotify blocker.
- ROS 2 Jazzy deployment from `motion_tracking_controller` targets Ubuntu Noble, while the host is Ubuntu 20.04.5.
- Long PPO, VAE, or diffusion training has not started because the required smoke-test gate is not fully satisfied.
- The paper/code discrepancy around adaptive sampling look-back remains unresolved; the current machine-readable audit is
  `/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json`.
- Released-data figures cover the Level A checklist and are now mapped against the arXiv source figure captions in
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/paper_panel_map.tsv`.
- Guided-diffusion results in Fig. 5 and Fig. 6 are not reproduced yet because the released official repository currently
  covers motion tracking only; VAE/diffusion evidence is paper-level hyperparameters, not executable official code.
- `motion_tracking_controller` has been statically audited, but MuJoCo sim-to-sim has not been executed because this
  host does not provide the ROS 2 Jazzy/Noble deployment stack required by the official README.
- The current blocked conditions are also summarized in machine-readable form at
  `/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json`.
