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
  official `motion.npz` has been produced. In-memory import attempts with `dest_path=""` under both AppLauncher and raw
  `SimulationApp` plus the IsaacLab headless experience reach the importer branch that avoids layered output, but both
  currently hit Vulkan `ERROR_DEVICE_LOST` before an exported robot stage can be captured. A follow-up variant matrix
  reproduced the failure on GPU 5 and GPU 6 and under waitIdle/low-RTX settings; the headless-rendering experience
  crashes even earlier. This localizes the current blocker below the AppLauncher wrapper and makes simple GPU switching
  or basic renderer downgrades insufficient. A newer ImportConfig surface probe confirms that Isaac Sim 4.5 exposes no
  `set_make_instanceable` or instanceable-USD-path setters through `URDFCreateImportConfig`; the baseline official G1
  URDF conversion still writes an openable but empty USD with zero prims, joints, or rigid bodies. This closes the
  Python-level instanceable-patch route and keeps official replay blocked. A local preconverted-asset audit found official mesh-level G1 USD files but
  no official full-robot preconverted G1 USD. It also found a structurally valid ASAP reference-code G1 USD, which may
  inform a resource-adjusted workaround but must not be reported as an official BeyondMimic asset. A follow-up
  compatibility audit shows that this reference USD contains all official target bodies and an articulation root, but
  its six wrist joints are fixed rather than revolute; it is therefore not a drop-in 29-DoF BeyondMimic replay asset.
  A minimal official-URDF-derived 29-DoF skeleton USD has now been generated and validated for the official link,
  action-joint, and target-body naming contract, but it has placeholder transforms and lacks physical fidelity
  (meshes, collisions, inertias, and drives), so it is a conversion scaffold only and official replay remains blocked.
  A follow-up URDF physical asset contract audit confirms that the official URDF provides all visual mesh references,
  collision primitives, non-fixed joint axes/limits, and local action-drive rows needed for an offline converter
  scaffold, while three sensor/IMU links lack inertial tags. A G1 URDF source-equivalence audit confirms that the
  downloaded official LAFAN G1 URDF and reproduction-data copy are byte-identical and that the official
  `whole_body_tracking` URDF keeps the same 29 non-fixed/action joints, but it also records support-link/joint and
  physical-bookkeeping differences that prevent treating the sources as identical. A resource-adjusted enriched USD
  scaffold has been
  authored and read back with mass/inertia metadata, mesh references, collision proxies, joint limits, and drive
  metadata. Its revolute joint limits now use USD Physics degree units while preserving URDF radian limits in custom
  metadata. A bounded resource-adjusted replay preflight can load this USD through IsaacLab `UsdFileCfg`, reach
  `num_joints=29` and `num_bodies=40`, and render four debug-fixture steps on `cuda:6`. The gate now returns
  successfully by explicitly exiting after the success sentinel, but clean Kit shutdown is still not verified; this is
  not official converter output, official `csv_to_npz.py`, official replay, PPO, DAgger, or paper-level closed-loop
  evidence. A follow-up bounded replay metrics gate runs 64 debug-fixture steps on `cuda:6` and records root/joint
  state consistency metrics, but it still uses the generated scaffold and debug fixture and remains below official
  replay/evaluation. A further resource-adjusted `Tracking-Flat-G1-v0` task smoke reaches reset and eight zero-action
  steps through the official manager stack, verifying obs/action/reward/termination dimensions; it is still not
  official replay/evaluation, PPO, DAgger, or paper-level evidence because the robot asset and motion source are
  resource-adjusted/debug fixtures. The smoke has been extended to a full available debug-fixture task eval: walk,
  run, and jump each run all `299` available steps in isolated Kit processes (`897` total steps), with action
  dimension `29`, policy observation dimension `160`, critic observation dimension `286`, nine reward terms, four
  termination terms, `29` robot joints, and `40` robot bodies verified. This confirms a stronger local task-contract
  gate, but it still does not produce official conversion/replay, PPO, teacher rollout data, or paper-level tracking
  metrics. A newer resource-adjusted official-CSV conversion gate takes the downloaded official G1 LAFAN
  `walk1_subject1.csv` frame range 1-180, runs the official interpolation/logging schema through the generated enriched
  USD, and writes a 299-frame `motion.npz` contract. A full replay gate then replays that official-CSV-derived motion
  for all 299 steps with zero joint/root write-read error. This moves the evidence closer to official replay, but the
  artifact is still not official `csv_to_npz.py` output because the official URDF/USD converter path remains bypassed.
  The same official-CSV-derived motion has also been fed into the official `Tracking-Flat-G1-v0` ManagerBasedRLEnv stack
  for all 299 available steps, verifying action dimension `29`, policy observation dimension `160`, critic observation
  dimension `286`, nine reward terms, four termination terms, `29` robot joints, and `40` robot bodies. This is still a
  zero-action diagnostic with generated USD, not a trained-policy evaluation. A bounded RSL-RL train-entry diagnostic
  now constructs the same official task, wraps it with `RslRlVecEnvWrapper`, instantiates `MotionOnPolicyRunner`, and
  completes one tiny PPO update (`num_envs=1`, `num_steps_per_env=4`, one learning iteration) on `cuda:6`. This confirms
  train-entry wiring, but it writes no checkpoint, logs PhysX GPU kernel warnings, uses the generated resource-adjusted
  USD and resource-adjusted motion path, and is not formal PPO training or paper-level tracking performance. A newer
  official `replay_npz.py` entry diagnostic runs the unmodified official replay entry with a local fake-WandB artifact
  and bounded AppLauncher wrapper; it reaches AppLauncher but blocks in the official URDF converter layer-save path
  before artifact download or replay-loop execution, leaving an empty robot prim. This preserves the active official
  replay blocker as converter/write-path evidence rather than a missing registry-only issue.
- The current Vulkan/USD evidence is tracked in
  `/mnt/infini-data/test/BeyondMimic/res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json` and
  `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`, plus the
  tracking USD probes under `/mnt/infini-data/test/BeyondMimic/res/tracking`.
- The current tracking smoke evidence includes source/config/asset/data audits plus resource-adjusted live IsaacLab
  task/train-entry diagnostics. Historical Kit/inotify failures remain retained, but current inotify limits and the
  AppLauncher live gate are no longer the active blocker. The active official tracking blocker is the official G1
  USD/conversion/replay path, followed by formal PPO training/evaluation.
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
