# Known Limitations

- Real-robot deployment is out of scope on this machine because no Unitree G1 hardware is available or connected.
- Full IsaacLab/Kit tracking replay is not passing yet. The old inotify failure is still retained as historical
  evidence, but the current host limits have been raised to `fs.inotify.max_user_watches=1048576` and
  `fs.inotify.max_user_instances=10240`. The current official replay blocker is localized to the IsaacLab/Kit USD
  conversion/write path: `layer.Save()` is blocked by `permissionToSave=False`, while the latest API probe shows
  direct `Usd.Stage.Export(...)` can write non-empty local USD files and should be tested as a conversion workaround.
  A first G1-specific workaround routed the importer's initial `Stage.Save()` through `Stage.Export()`, but the
  destination/current stages remained empty. A deeper `Sdf.Layer.Save` probe confirmed that Python monkeypatching works
  for direct layers, but the URDF importer's C++/Kit base/physics/sensor configuration-layer save path is not
  intercepted by that Python patch; the three generated configuration layers remain empty and no valid G1 USD or
  official `motion.npz` has been produced.
- The current Vulkan/USD evidence is tracked in
  `/mnt/infini-data/test/BeyondMimic/res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json` and
  `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`, plus the
  tracking USD probes under `/mnt/infini-data/test/BeyondMimic/res/tracking`.
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
