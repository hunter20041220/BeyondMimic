# BeyondMimic Reproduction Progress

阶段：Phase 0 / Phase 1 / Phase 2 - local inventory, environment setup, released-data figures
状态：RUNNING
开始时间：2026-06-16T17:13:01+08:00
结束时间：TBD
使用环境：host shell, bm_analysis, bm_tracking, Isaac Sim 4.5.0 standalone Python
使用代码：/mnt/infini-data/test/BeyondMimic/reproduction/scripts, /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official
官方/重新实现：audit tooling
Git commit：TBD
配置：ROOT=/mnt/infini-data/test/BeyondMimic
执行命令：bash reproduction/scripts/system_audit.sh; python3 reproduction/scripts/generate_local_inventory.py
GPU：8 x NVIDIA GeForce RTX 4090 D, driver 560.35.03, CUDA capability 12.6
峰值显存：GPU 6 reported 3988 MiB used during audit; other GPUs 5-7 MiB
平均 GPU-Util：0% during audit
平均功耗：37.98W-63.35W idle range during audit
运行时间：TBD
输出文件：docs/local_inventory.tsv, logs/setup/system_audit.txt, docs/source_ledger.md, docs/environment_plan.md, docs/paper_parameter_map.md, docs/discrepancy_report.md, docs/unresolved_details.md, env locks, smoke figures, res/released_figures, res/level_c/synthetic_smoke
主要指标：local inventory has 653 rows plus header; released-data summary has 13 figure rows with 20 PDF, 20 SVG, and 20 PNG figure files; paper_panel_map.tsv maps released-data artifacts to Fig. 3B, Fig. 4C, Fig. 8A/B, Fig. S1, and Fig. S2 evidence; completion_matrix.md tracks goal-level status
与论文一致性：official tracking stack requires Isaac Sim 4.5.0, Isaac Lab 2.1.0, Python 3.10; official code confirms key PPO and tracking hyperparameters
发现的差异：host lacks conda/mamba/nvcc; default python is 2.7; adaptive sampling look-back is a quantified paper/code mismatch; ROS 2 Jazzy deployment does not match Ubuntu 20.04 host
失败与风险：IsaacLab Kit headless smoke is blocked by low host inotify limits; current user cannot raise sysctl inotify limits; IsaacLab optional dependencies are not fully reconciled; GPU 6 background memory use needs recheck before training
下一阶段：Resolve/route around Kit smoke blocker, then run executable tracking env smoke; use real tracking rollouts to replace Level C synthetic fixtures when available

## Phase 1 Setup Notes

- Installed project-local micromamba 2.8.1 at `/mnt/infini-data/test/BeyondMimic/envs/_micromamba/bin/micromamba`.
- Created `bm_analysis`; CSV/matplotlib smoke passed and produced PDF/SVG/PNG smoke figures.
- Created `bm_tracking`; PyTorch 2.5.1+cu121 CUDA smoke passed on 8 GPUs.
- Isaac Sim pip installation failed on Ubuntu 20.04 because the 4.5.0 pip placeholder looked for a manylinux_2_34 wheel.
- Downloaded official Isaac Sim 4.5.0 standalone binary to `/mnt/infini-data/test/BeyondMimic/download/_supplemental/isaac-sim-standalone-4.5.0-linux-x86_64.zip`.
- Extracted Isaac Sim to `/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0`; `python.sh -c "import isaacsim"` passed.
- Prepared IsaacLab v2.1.0 work copy at `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0` and symlinked `_isaac_sim`.
- Installed IsaacLab editable packages and `rsl_rl`; plain `import isaaclab` passes.
- IsaacLab/Kit headless smoke currently stalls/fails with `Failed to create change watch ... errno=28`, caused by inotify watch exhaustion (`fs.inotify.max_user_watches=8192`), not disk.
- A direct `sysctl -w fs.inotify.max_user_watches=524288 fs.inotify.max_user_instances=1024` attempt was denied for the
  current user.
- Prepared official `whole_body_tracking` work copy, extracted Unitree assets, installed it editable into Isaac Sim Python, and passed non-Kit package/asset smoke.
- Added non-Kit tracking configuration audit at
  `/mnt/infini-data/test/BeyondMimic/res/tracking/smoke_config_audit/tracking_config_audit.json`.
- Tracking audit verified G1 PPO hidden dimensions `[512, 256, 128]`, `sim.dt=0.005`, `decimation=4`,
  control frequency `50.0 Hz`, 4096 envs, G1 URDF mesh references, and a LAFAN1 G1 CSV sample.
- Tracking audit now also records G1 anchor body `torso_link`, the 14 target body names, termination end-effectors,
  and an adaptive-sampling paper-vs-code kernel probe.
- Adaptive-sampling probe plot/table:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_probe`.
- Added adaptive-sampling discrepancy audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/adaptive_sampling_discrepancy_audit.py`.
- Adaptive-sampling discrepancy outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json`
  and `/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.tsv`.
- Adaptive-sampling discrepancy evidence: paper/source and `goal.md` specify look-back `u in {0,1,2}`, while raw
  download and reproduction official code both default to `adaptive_kernel_size=1`; no local official-tree numeric
  runtime override to kernel size 3 was found. The official/PyTorch-kernel-direction single-failure-bin probe records
  `l1_difference=1.0730253353204173` and pre-failure mass `0.02272727272727273 -> 0.5592399403874814`.

## Phase 2 Released-Data Notes

- Extracted `Dataset_beyondmimic.zip` to `/mnt/infini-data/test/BeyondMimic/reproduction/data/Dataset_beyondmimic`.
- Generated dataset inventory at `/mnt/infini-data/test/BeyondMimic/reproduction/docs/released_dataset_inventory.tsv` with 414 files.
- Added released-data reproduction script:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduce_released_figures.py`.
- Added adaptive conversion/plot scripts:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/convert_adaptive_sampling.py`
  and `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/plot_adaptive_sampling_released.py`.
- Current released-data summary:
  `/mnt/infini-data/test/BeyondMimic/res/released_figures/released_figure_summary.tsv`.
- Paper/source panel mapping:
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/paper_panel_map.tsv`.
- Reproduced released-data figure groups:
  IMU orientation/acceleration/angular velocity; observation history, armature, PD gain, orientation representation,
  and latency ablations; walking/running human GRF; walking/running robot GRF; adaptive sampling with/without matrices;
  adaptive sampling probability evolution.
- Every generated figure group includes processed CSV data, source hashes, a run log, paper panel mapping notes, and
  PDF/SVG/PNG outputs.
- Added released-data numeric table summary:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_data_metrics_summary.py`.
- Released-data table outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_data_metrics_summary.json`,
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv`,
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_grf_metrics.csv`,
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_imu_metrics.csv`, and source hashes.
- Released-data table result: status `ok`, `10` source processed CSVs hashed, `30` tracking-ablation summary rows,
  `12` GRF rows, and `10` IMU rows. The best global-position ablation row is
  `ablation_orientation_representation/quat` with mean `0.2161546877936451` versus baseline `origin`
  `0.2350160799679482`; peak vertical GRF absolute value is `2.316238655046181`; IMU duration is
  `6.318044000072405` s.
- Boundary: these are Level A released-data summary tables derived from reproduced CSVs, not new official PPO training,
  VAE/diffusion training, Fig. 5/Fig. 6 rollout metrics, or real-robot reproduction.
- Added released-data statistical audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_data_statistical_audit.py`.
- Released-data statistical outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_data_statistical_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv`,
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_grf_confidence_intervals.csv`,
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_imu_confidence_intervals.csv`,
  and source hashes/Markdown.
- Released-data statistical result: status `ok`; `10` source processed CSVs hashed, `30` ablation comparison rows,
  `12` GRF confidence-interval rows, and `11` IMU confidence-interval rows. Best relative ablation improvement is
  PD gain local orientation error `origin -> wn25` with relative improvement `0.12405441378347314` and
  range-based effect size `6.661494579260983`. The released IMU norm comparison records peak acceleration
  `36.18653061127687` m/s^2, peak angular velocity `16.764945820586494` rad/s, and mean angular velocity
  `4.854256896352452` rad/s against the paper-highlighted `31/20/7.01` values.
- Boundary: these are uncertainty/effect-size summaries over released processed CSVs. They do not produce new
  tracking/VAE/diffusion training statistics, user-study statistics, hardware logs, or Fig. 5/Fig. 6 rollout metrics.

## Level B Tracking Notes

- Official tracking protocol and current gate are documented at
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/level_b_tracking_protocol.md`.
- `bm_tracking` lock files are exported under `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking`.
- Isaac Sim standalone Python package list is exported to
  `/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0/pip-freeze.txt`.
- Prepared local, non-WandB official-script smoke variants under
  `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local`.
- Added:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/prepare_tracking_local_smoke.py`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_tracking_local_smoke.sh`, and
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_motion_npz_contract.py`.
- `run_tracking_local_smoke.sh` now validates the generated `motion.npz` contract immediately after local
  `csv_to_npz` conversion and before replay/training smoke.
- Added motion preprocessing producer/consumer contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/motion_preprocessing_contract_audit.py`.
- Motion preprocessing contract outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json`
  and `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.tsv`.
- Motion preprocessing contract evidence: all `40` official G1 retargeted CSV files have the expected `36` columns
  (`root xyz + xyzw quaternion + 29 joints`), finite values, and near-unit quaternions with max error
  `8.877706059173818e-07`; official `csv_to_npz.py`, replay, training `MotionLoader`, and local validator agree on
  the seven required `motion.npz` keys.
- Added non-Kit tracking `motion.npz` debug fixture builder:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_tracking_motion_npz_fixture.py`.
- Tracking fixture outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json`
  and `/mnt/infini-data/test/BeyondMimic/reproduction/data/tracking_motion_npz_fixtures`.
- Tracking fixture evidence: three URDF-FK debug `motion.npz` files for walk/run/jump clips pass
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_motion_npz_contract.py`; the set contains `897` total
  frames, `29` joints, `40` URDF bodies, and all official `14` tracking body names. These fixtures are not official
  Isaac/Kit `csv_to_npz.py` articulation exports and do not unblock rendered replay or PPO training.
- Prepared official `motion_tracking_controller` work copy and static deployment audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/deployment_controller_audit.md`
  and `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_tracking_controller_audit`.
- Added MuJoCo/ROS launch contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_ros_launch_contract_audit.py`.
- MuJoCo/ROS launch contract outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json`
  and `/mnt/infini-data/test/BeyondMimic/res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.tsv`.
- MuJoCo/ROS launch contract evidence: README Jazzy instructions, package dependencies, pluginlib declaration,
  500 Hz controller manager, 50 Hz walking controller, 29-joint standby impedance arrays, MuJoCo sim node/plugin,
  policy/WandB path resolution, real-launch network interface wiring, and MCAP rosbag recording with exclusions all
  pass static checks. Host runtime gate remains explicit: Ubuntu `20.04`, no `ros2`, no `colcon`, no `rosdep`.
- Added ONNX contract inspector:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/inspect_motion_onnx_contract.py`.
- Checked the Unitree bringup G1 reference ONNX and confirmed it is not a BeyondMimic motion-policy ONNX.

## Level C Diffusion Notes

- Added Level C VAE/diffusion plan:
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/level_c_diffusion_plan.md`.
- No BeyondMimic-specific official VAE/diffusion code, checkpoint, TensorRT engine, or state-latent dataset has been
  found in the current official inventory; reference diffusion repositories remain reference-only.
- Added machine-readable local artifact audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_artifact_audit.py`.
- Artifact audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/official_artifact_audit/level_c_official_artifact_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/official_artifact_audit/level_c_official_artifact_audit.tsv`.
- Audit evidence: `6472` files scanned, `832` matched VAE/diffusion/deployment-related rows, `0` official
  BeyondMimic-specific VAE/diffusion code candidates, `0` official checkpoint/ONNX/TensorRT engine candidates, and `516`
  reference diffusion-related rows.
- Added Fig. 5 / Fig. 6 feasibility audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_fig5_fig6_feasibility_audit.py`.
- Fig. 5 / Fig. 6 audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.tsv`.
- Fig. 5 / Fig. 6 audit evidence: `6` panels audited, `2` paper PDF figures present, `0` released data matches, all
  `6` panels blocked for paper reproduction because trained Level C checkpoints, state-latent rollout logs,
  closed-loop joystick/inpainting/obstacle execution logs, and real/mocap evidence are absent. Existing Level C probes
  cover debug mechanisms only.
- Added synthetic PyTorch Level C smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_synthetic_smoke.py`.
- The smoke ran successfully on CPU and produced:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/synthetic_smoke/level_c_synthetic_smoke.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/synthetic_smoke/level_c_synthetic_smoke.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/level_c_synthetic_smoke.log`.
- Smoke evidence after aligning with paper/source clean-trajectory training: VAE total loss `0.01444791816174984`,
  diffusion clean-trajectory MSE `1.8675247430801392`, guidance total `5.266604900360107`, independent state/latent
  denoising-step tensor shape `[2,21,2]`, and nonzero gradient norms for VAE, diffusion, and guidance tensors.
- This smoke uses synthetic tensors only. It does not reproduce Fig. 5/Fig. 6 or replace the missing teacher rollout /
  state-latent dataset.
- Added debug-only VAE gradient accumulation and synthetic DAgger-batch schema probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_accumulation_probe.py`.
- VAE accumulation probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.npz`.
- VAE accumulation evidence: latent dim `32`, encoder/decoder hidden dimensions `[2048,1024,512]`, teacher hidden
  dimensions `[512,256,128]`, LR `0.0005`, KL coefficient `0.01`, gradient accumulation steps `15`, micro-batch size
  `2`, effective batch size `30`, one optimizer step, VAE parameter count `5697117`, teacher parameter count `251933`,
  grad norm before step `0.09209133684635162`, parameter update norm `1.1804022789001465`, and zero-grad check `0.0`.
- The synthetic DAgger manifest explicitly marks `is_true_dagger_rollout=false`; this remains a mechanics/schema probe,
  not real DAgger rollout, VAE checkpoint reproduction, rollout stability evaluation, or latent analysis.
- Added debug-only receding-horizon decoder probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_receding_horizon_decoder_probe.py`.
- Receding-horizon decoder outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.npz`.
- Decoder probe evidence: guided denoised tau `[21,213]`, current index `4`, latent dim `32`, candidate proprioception
  dim `96`, decoder input dim `128`, action dim `29`, CPU device, control frequency `25.0` Hz, control dt `0.04` s,
  current action norm `0.3012312948703766`, and wrong-future-latent contrast delta `0.3690456449985504`.
- This probe validates receding-horizon current-latent decoding schema. It is not a trained decoder checkpoint,
  paper-exact proprioception layout, asynchronous TensorRT deployment, closed-loop rollout, or latency benchmark.
- Added debug-only full paper Transformer architecture probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_full_transformer_arch_probe.py`.
- Full architecture probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.npz`.
- Full architecture probe evidence: device `cuda:0`, clean/noisy/predicted tau shape `[1,21,213]`, independent timestep
  tensor `[1,21,2]`, state dim `181`, latent dim `32`, embedding `512`, heads `8`, layers `6`, denoising steps `20`,
  parameter count `19164373`, clean-trajectory MSE `1.7915563583374023`, gradient norm `4.83304500579834`, and CUDA
  peak memory `388.63671875` MB.
- This probe verifies one paper-sized architecture forward/backward pass. It is not long diffusion training, a trained
  checkpoint, TensorRT deployment, guided rollout evaluation, or Fig. 5/Fig. 6 reproduction.
- Added debug-only paper-state Transformer architecture probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_transformer_arch_probe.py`.
- Paper-state Transformer architecture outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.npz`.
- Paper-state Transformer architecture evidence: device `cuda:0`, clean/noisy/predicted tau shape `[1,21,131]`,
  independent timestep tensor `[1,21,2]`, paper state dim `99`, latent dim `32`, embedding `512`, heads `8`, layers `6`,
  denoising steps `20`, parameter count `19080323`, clean-trajectory MSE `1.708655595779419`, gradient norm
  `5.51327657699585`, and CUDA peak memory `386.9970703125` MB.
- This probe verifies one paper-sized architecture forward/backward pass on the 99-D paper-state artifact. It is not
  long diffusion training, a trained checkpoint, TensorRT deployment, guided rollout evaluation, or Fig. 5/Fig. 6
  reproduction.
- Added Level C Transformer parameter-count audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_parameter_count_audit.py`.
- Transformer parameter-count outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.tsv`.
- Transformer parameter-count result: status `ok`; paper source line `root.tex:589` states approximately `19.8M`
  parameters. Local `213`-D tau variant has `19164373` parameters (`-3.2102373737373735%` vs `19.8M`), paper-state
  `131`-D tau variant has `19080323` parameters (`-3.634732323232323%`), and the `84050` local-variant difference is
  exactly explained by token input/output projection dimensions. The audit records approximate consistency but no exact
  paper checkpoint architecture claim.
- Added debug-only diffusion training schedule probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_training_schedule_probe.py`.
- Training schedule probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/training_schedule_probe/level_c_training_schedule_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/training_schedule_probe/level_c_training_schedule_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/training_schedule_probe/level_c_training_schedule_probe.npz`.
- Schedule evidence: batch size `512`, epochs `1000`, LR `0.0001`, weight decay `0.001`, cosine warmup scheduler,
  warmup gradient steps `10000`, EMA power/max `0.75/0.9999`, warmup final LR `0.0001`, first cosine LR `0.0001`,
  local-probe final LR `1.5421257049119676e-13`, local-probe final EMA `0.9997009302437557`, and checks pass for
  monotone warmup, monotone cosine decay, LR peak, monotone EMA, and EMA cap.
- This schedule probe is not official optimizer code, full diffusion training, checkpoint reproduction, validation/test
  metrics, or Fig. 5/Fig. 6 reproduction.
- Added non-Kit retargeted-motion Level C state fixture:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_motion_state_fixture.py` and
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_level_c_motion_state_fixture.py`.
- The fixture uses official LAFAN1 G1 `walk1_subject1.csv` plus the official G1 URDF to generate:
  `/mnt/infini-data/test/BeyondMimic/reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz`.
- Manifest and validation evidence:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.tsv`,
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/build_level_c_motion_state_fixture.log`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/validate_level_c_motion_state_fixture.log`.
- Fixture evidence: `joint_pos [299,29]`, `body_pos_w [299,14,3]`,
  candidate hybrid state `[299,181]`, candidate windows `[28,21,181]`, global XY/yaw invariance error
  `1.2698175844150228e-15`, and emphasis pseudoinverse error `1.1102230246251565e-16`.
- This fixture is marked `debug_only`; it is not official Kit `motion.npz`, not a teacher rollout, and not a
  paper-exact state-latent dataset.
- Added debug-only fixture window provenance/split manifest:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_fixture_split_manifest.py`.
- Manifest outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/fixture_split_manifest/fixture_window_provenance.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/fixture_split_manifest/fixture_split_manifest_summary.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/fixture_split_manifest/fixture_split_manifest_summary.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/build_level_c_fixture_split_manifest.log`.
- Split/leakage evidence: `28` window rows, train/validation/test/excluded-guard counts `17/3/4/4`, no cross-split
  overlap, no near-gap violation, missing latents explicitly marked, and live stability accept/reject explicitly marked
  as unavailable debug fixture evidence.
- Added debug-only OU/sagittal augmentation probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_augmentation_probe.py`.
- Probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/augmentation_probe/level_c_augmentation_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/augmentation_probe/level_c_augmentation_probe.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/augmentation_probe/level_c_augmentation_probe.npz`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/augmentation_probe/ou_perturbation_probe.pdf`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/augmentation_probe/sagittal_symmetry_probe.pdf`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/level_c_augmentation_probe.log`.
- Probe evidence: OU parameters `theta=0.8`, `mu=0`, `dt=1.0`, `sigma=0.1`; OU lag-1 autocorrelation
  `0.1897790054129594` versus iid Gaussian `-0.004747132016480871`; candidate G1 sagittal double-mirror error `0.0`.
- This probe is not VAE rollout perturbation, episode rejection, a paper-exact symmetry implementation, or a trainable
  augmented diffusion dataset.
- Added Level C dataset collection protocol audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_dataset_collection_protocol_audit.py`.
- Dataset collection protocol outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json`
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.tsv`.
- Dataset collection protocol evidence: paper/source and `goal.md` requirements for OU perturbation, about `100x`
  sample coverage, `2.5 s` rollout, `5 s` stability verification, episode rejection, sagittal augmentation, and
  provenance fields are indexed; current debug evidence covers `3` motions, `84` debug samples, and `1764` paper-state
  tokens, while `8` paper dataset-collection requirements remain explicitly missing for true VAE rollout collection.
- Added Level C rollout rejection manifest probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_rollout_rejection_manifest_probe.py`.
- Rollout rejection manifest outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.npz`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/rollout_rejection_manifest_probe/rollout_rejection_manifest_rows.tsv`.
- Rollout rejection manifest result: status `ok`, `3` debug fixture motions, `150` valid 5-second starts, `100`
  synthetic OU seeds per valid start, `15000` manifest rows, 2.5-second recorded windows (`125` frames at 50 Hz),
  5-second stability windows (`250` frames), nonzero recorded coverage minimum `100`, and accept/reject fields marked
  `accepted_debug_fixture_no_live_failure_signal`. It explicitly does not claim trained VAE rollout, true latent
  recording, live 5-second stability verification, or real rejection decisions.
- Added debug-only guidance formula probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_formula_probe.py`.
- Guidance probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/level_c_guidance_formula_probe.log`.
- Guidance probe evidence: joystick gradient norm `1.4540001695947904`, waypoint gradient norm
  `0.08946569697214952`, SDF relaxed-barrier gradient norm `444.1331090906918`, composed paper-formula gradient norm
  `444.1357487225691`, and candidate keyframe gradient norm `0.004506473481857317`.
- This probe is not denoising-loop guidance, a guidance-scale sweep, a validation/test split, guided rollout metrics, or
  Fig. 5/Fig. 6 reproduction. Geometry plots were not written because `bm_tracking` has PyTorch but not matplotlib.
- Added debug-only independent timestep/mask schedule probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_timestep_mask_probe.py`.
- Timestep/mask probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.npz`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/level_c_timestep_mask_probe.log`.
- Timestep/mask probe evidence: schedule tensors `[21,2]`, `state_dim=181`, `latent_dim=32`, `tau_dim=213`,
  training schedule has independent state/latent steps on `21` tokens, history-conditioning prefix is clean for
  state/latent tokens, and future-keyframe schedule has `8` clean state tokens versus `5` clean latent tokens.
- This probe is not the paper-exact deployed inpainting mask policy, reverse denoising implementation, guided rollout
  evaluation, or Fig. 6 reproduction.
- Added debug-only reverse denoising mechanics probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_reverse_denoising_probe.py`.
- Reverse denoising probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.npz`, and
  `/mnt/infini-data/test/BeyondMimic/logs/level_c/level_c_reverse_denoising_probe.log`.
- Reverse denoising probe evidence: future-keyframe schedule, `tau_shape=[21,213]`, `state_dim=181`, `latent_dim=32`,
  one-step MSE improves from `0.11735366650108521` to `0.02061608879839087`, full reverse MSE reaches
  `5.253923735449295e-10`, observed/keyframe clamp error is `0.0`, and max denoising steps decrease `19 -> 0`.
- This probe uses an oracle clean predictor and candidate deterministic reverse update. It is not the paper-exact
  alpha/gamma/sigma schedule, a trained diffusion network, TensorRT deployment, denoising-loop guidance, guided rollout
  evaluation, or Fig. 5/Fig. 6 reproduction.
- Added debug-only guided reverse-loop probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guided_reverse_loop_probe.py`.
- Guided reverse-loop outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.npz`.
- Guided reverse-loop evidence: history-conditioning schedule, `tau_shape=[21,213]`, guidance scale `0.002`, command
  velocity `[0.35,0.0]`, initial joystick guidance cost `4.187819789824388`, unguided final cost
  `1.0463021188199848`, guided final cost `1.0423308869100696`, clamp error `0.0`, and max denoising steps `19 -> 0`.
- This probe connects classifier-style guidance to the reverse loop, but still uses an oracle clean predictor and a
  candidate guidance update. It is not a trained guided diffusion controller, guidance-scale protocol, rollout metric,
  TensorRT deployment, or Fig. 5/Fig. 6 reproduction.
- Added debug-only guidance scale sweep probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_scale_sweep_probe.py`.
- Guidance scale sweep outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.npz`.
- Scale sweep evidence: scales `[0,0.0005,0.001,0.002,0.005,0.01,0.02]`, all rows valid, all rows keep clamp error
  `0.0`, all rows reach final max step `0`, zero-scale final joystick cost `1.0463021188199848`, selected debug best
  scale `0.02`, selected final joystick cost `1.006958747664373`, and selected final MSE `1.801002874448887e-07`.
- This sweep is local to the debug fixture and oracle reverse loop. It is not the paper-exact guidance-scale protocol,
  a validation/test scene split, rollout metrics, TensorRT deployment, or Fig. 5/Fig. 6 reproduction.
- Added diffusion equation/coefficient-boundary audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_equation_audit.py`.
- Diffusion equation audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.npz`.
- Diffusion equation evidence: paper/source contains the DDPM forward posterior, clean latent/trajectory prediction
  losses, reverse `alpha/gamma/sigma` form, state-latent reverse form, and `20` denoising steps; candidate
  clean-prediction DDPM posterior coefficients are algebraically equivalent to the paper reverse form with max absolute
  error `4.440892098500626e-16`; forward noisy-vs-clean MSE increases from `9.993764986922505e-05` at the first
  probe step to `0.1929149275521341` at the last probe step.
- Boundary: the local paper/source artifacts do not publish the exact numeric beta or `alpha/gamma/sigma` coefficient
  schedule used by the trained BeyondMimic diffusion model. The audit intentionally marks the linear beta schedule as
  a candidate equivalence probe, not a paper claim.
- Added single-batch diffusion overfit gate:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_single_batch_overfit_probe.py`.
- Single-batch overfit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.npz`.
- Single-batch overfit evidence: one debug fixture batch with independent state/latent denoising steps is fit using a
  NumPy ridge clean-trajectory denoising map; noisy identity baseline loss `0.06463638649755082` falls to
  `4.557876969502246e-16`, loss reduction ratio `0.9999999999999929`.
- Boundary: this is a single-batch pre-training gate for loss/data-path correctness, not full diffusion training,
  validation/test evaluation, checkpoint reproduction, or Fig. 5/Fig. 6 reproduction.
- Added single-motion diffusion overfit gate:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_single_motion_overfit_probe.py`.
- Single-motion overfit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.npz`.
- Single-motion overfit evidence: all `28` windows from the debug fixture motion are fit with a NumPy ridge
  clean-trajectory denoising map using independent state/latent denoising steps and an overparameterized memorization
  basis; noisy baseline loss `0.06594478121987789` falls to `7.383799715088834e-19`, loss reduction ratio `1.0`.
- Boundary: this is a single-motion memorization gate, not a multi-motion small-dataset overfit, validation result,
  full diffusion training, checkpoint reproduction, or Fig. 5/Fig. 6 reproduction.
- Added two additional debug-only Level C motion-state fixtures for small-dataset gates:
  `/mnt/infini-data/test/BeyondMimic/reproduction/data/level_c_fixtures/run2_subject1_frames_1_180_state_fixture.npz` and
  `/mnt/infini-data/test/BeyondMimic/reproduction/data/level_c_fixtures/jumps1_subject1_frames_1_180_state_fixture.npz`.
- Added small-dataset diffusion overfit gate:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_overfit_probe.py`.
- Small-dataset overfit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.npz`.
- Small-dataset overfit evidence: `84` windows from `3` debug fixture motions are fit with a NumPy ridge
  clean-trajectory denoising map using independent state/latent denoising steps, motion-id basis, and an
  overparameterized token memorization basis; noisy baseline loss `0.06878844770150597` falls to
  `1.1634015638380633e-18`, loss reduction ratio `1.0`.
- Boundary: this is a small multi-motion memorization gate over short debug fixtures, not a paper-scale VAE rollout
  dataset, held-out validation, full diffusion training, checkpoint reproduction, or Fig. 5/Fig. 6 reproduction.
- Added small-dataset provenance/split manifest:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_small_dataset_split_manifest.py`.
- Small-dataset split outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_split_manifest/small_dataset_window_provenance.tsv`.
- Small-dataset split evidence: motion-level split assigns walk/run/jump fixtures to train/validation/test with
  motion counts `1/1/1` and sample counts `28/28/28`; no motion crosses splits; VAE latents are explicitly marked
  `missing_no_teacher_or_vae_latent`; live stability accept/reject is marked debug-only.
- Boundary: this manifest prevents overlap leakage in the debug fixture set, but it is not the paper-exact VAE rollout
  train/validation/test split.
- Added small-dataset multi-seed memorization statistics audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_multiseed_audit.py`.
- Small-dataset multi-seed outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.tsv`.
- Small-dataset multi-seed evidence: seeds `20260904`, `20260905`, and `20260906` all reduce loss and finish below
  `1e-8`; final overfit loss mean `1.1825253467675343e-18`, std `1.6954088866134628e-20`; loss reduction ratio
  mean `1.0`, std `0.0`.
- Boundary: this satisfies a debug reporting habit of not picking only the best seed, but it is not multi-seed full
  diffusion training, held-out evaluation, or paper-result statistics.
- Added small-dataset held-out debug evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_heldout_eval.py`.
- Small-dataset held-out evaluation outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.npz`.
- Held-out debug evidence: a non-memorizing ridge baseline trained on the motion-level train split does not use token
  identity features; validation loss falls from `0.06930764133540507` to `0.03443011209674649` and test loss falls
  from `0.0682719253409472` to `0.0379766888712374`.
- Boundary: this is a debug held-out baseline on synthetic latents and short fixtures, not a trained diffusion
  Transformer, paper validation/test protocol, checkpoint, or Fig. 5/Fig. 6 result.
- Added small-dataset held-out multi-seed statistics audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_heldout_multiseed_audit.py`.
- Small-dataset held-out multi-seed outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.tsv`.
- Held-out multi-seed evidence: seeds `20260907`, `20260908`, and `20260909` all reduce train, validation, and test
  losses without token identity features; validation loss-reduction ratio mean/std is
  `0.5027735618904363/0.007619892358679882`, and test loss-reduction ratio mean/std is
  `0.4483171668248292/0.024525236825074737`.
- Boundary: this improves debug statistical reporting and avoids best-seed-only reporting, but it is not paper
  multi-seed diffusion training, the paper validation/test protocol, checkpoint reproduction, or Fig. 5/Fig. 6
  reproduction.
- Added trajectory inverse-transform audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_trajectory_inverse_transform_audit.py`.
- Trajectory inverse-transform outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.npz`.
- Trajectory inverse-transform evidence: `28` motion-derived windows checked; paper-formula root current-character-frame
  position/velocity inverse max errors are `1.1102230246251565e-16`; root quaternion round-trip angle is `0.0`;
  body local-frame position inverse max error is `1.6653345369377348e-16`; body velocity inverse max error is
  `4.440892098500626e-16`.
- Boundary: the existing debug fixture's `candidate_hybrid_state` still does not encode the full paper
  window-current-frame root translation state. The inverse-transform audit is a math/representation gate, not an
  official trainable state-latent dataset.
- Added state representation source audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_representation_source_audit.py`.
- State representation source audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.npz`.
- State representation evidence: method/source patterns for hybrid character-yaw state, state-latent trajectory,
  history/horizon, and emphasis projection are found; body local positions in all three debug fixtures match the paper
  local-frame formula with max error `1.3877787807814457e-17`.
- Boundary: the audit explicitly records non-paper-exact differences in the current debug fixture: root
  current-frame feature difference max `1.1687931299897365`, body velocity missing root-velocity subtraction max
  `1.8108774263211476`, missing full root relative position, and diagonal debug emphasis weights instead of the
  paper random Gaussian/pseudoinverse projection.
- Added paper-formula state-window builder:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_paper_state_windows.py`.
- Paper-formula state-window outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_windows/level_c_paper_state_windows.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_windows/level_c_paper_state_windows.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_windows/level_c_paper_state_windows_summary.npz`, and
  per-motion NPZ files under `/mnt/infini-data/test/BeyondMimic/reproduction/data/level_c_paper_state_windows`.
- Paper-state window evidence: `84` windows from `3` debug motions, sequence length `21`, state dim `99`, and
  sample count `1764`; each window has root relative position and root relative linear velocity exactly zero at the
  current timestep, matching the current-frame normalization.
- Boundary: these are paper-formula state windows from debug FK fixtures only. They still lack trained VAE latents,
  teacher/student rollouts, live accept/reject collection, paper-exact emphasis projection, and checkpoint/evaluation
  evidence.
- Added paper-state diffusion overfit gate:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_overfit_probe.py`.
- Paper-state overfit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.npz`.
- Paper-state overfit evidence: paper-formula state dim `99`, token dim `131` after adding synthetic `32`-D latents,
  `84` windows and `1764` tokens; noisy baseline loss `0.06703983482819055` falls to
  `1.830447187210791e-18`, loss reduction ratio `1.0`, with independent state/latent denoising steps.
- Boundary: this proves the local clean-trajectory denoising path can consume the paper-formula state artifact, but it
  uses synthetic latents and an overparameterized memorization basis; it is not full diffusion training, held-out
  evaluation, a teacher/VAE rollout dataset, or paper Fig. 5/Fig. 6 reproduction.
- Added paper-state held-out debug evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_heldout_eval.py`.
- Paper-state held-out outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.npz`.
- Paper-state held-out evidence: using paper-formula 99-D state windows plus synthetic latents, a non-memorizing ridge
  baseline trained on the motion-level train split does not use token identity features; validation loss falls from
  `0.06610149404036221` to `0.030197348642920434`, and test loss falls from `0.06511901426400574` to
  `0.03338779974210115`.
- Boundary: this is a debug held-out sanity gate over synthetic latents and short motion fixtures, not trained
  Transformer diffusion, paper validation/test protocol, checkpoint reproduction, or Fig. 5/Fig. 6 reproduction.
- Added paper-state held-out multi-seed statistics:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_heldout_multiseed_audit.py`.
- Paper-state held-out multi-seed outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.tsv`.
- Paper-state held-out multi-seed evidence: the non-memorizing ridge baseline is repeated over at least 3 seeds with
  the same motion-level split manifest, no token identity basis, paper-state dim `99`, and explicit debug-only boundary.
  Validation loss-reduction ratio mean/std is `0.5475202271651117/0.009720682743715466`; test loss-reduction ratio
  mean/std is `0.48924922500559204/0.008383799449541141`.
- Boundary: this strengthens statistical reporting for the debug paper-state held-out sanity gate, but it remains
  synthetic-latent smoke evidence rather than trained Transformer diffusion, paper validation/test protocol, checkpoint
  reproduction, or Fig. 5/Fig. 6 reproduction.
- Added emphasis projection and pseudoinverse audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_emphasis_projection_audit.py`.
- Emphasis projection outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.npz`.
- Emphasis projection evidence: the paper formula `P=[AB I]^T` is implemented with random Gaussian `A`, root
  pose/twist emphasis coefficient `c=6`, full-column-rank projection, and pseudoinverse reconstruction over all local
  99-D paper-state tokens. Projection shape is `163 x 99`, rank is `99`, and max reconstruction error is
  `2.4077961846558082e-15`.
- Boundary: this verifies projection mechanics over debug paper-state windows, not an unpublished paper trainable
  dataset, official sampled projection matrix, trained diffusion checkpoint, or paper evaluation.
- Added VAE latent reparameterization and interpolation probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_probe.py`.
- VAE latent probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_probe/level_c_vae_latent_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_probe/level_c_vae_latent_probe.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_probe/level_c_vae_latent_probe.npz`.
- VAE latent evidence: the paper-dimension conditional VAE path checks latent dim `32`, encoder/decoder hidden dims,
  reparameterization `z = mu + std * eps`, KL formula, and latent interpolation endpoints over at least three seeds.
  KL mean/std is `0.16656828920046488/0.01607387453652661`, mean latent std is `1.0006367762883503`,
  reparameterization max error is `0.0`, and interpolation endpoint errors are below `1e-6`.
- Boundary: this is synthetic debug evidence for VAE latent math and interpolation, not true DAgger rollout, trained VAE
  checkpoint, closed-loop survival, or paper latent analysis.
- Added smoothness and latency metric audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_smoothness_latency_audit.py`.
- Smoothness/latency outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.npz`.
- Smoothness/latency evidence: computes debug state/latent trajectory smoothness, schema action delta, guidance-cost
  reduction, and the paper latency budget using 25 Hz control, 20 ms diffusion target, and <1 ms CPU decoder target.
  Guided final state second-difference mean norm is `0.027660824669436108`, guided final latent second-difference mean
  norm is `13.538484354740202`, guidance-cost reduction is `3.1454889029143187`, and the paper diffusion latency
  target is `0.5` of the 40 ms control period.
- Boundary: this is a metric audit over debug oracle/guided artifacts and paper targets, not a TensorRT benchmark,
  closed-loop action sequence, trained diffusion rollout, or paper Fig. 5/Fig. 6 evaluation.
- Added tracking ONNX exporter-to-controller contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_onnx_export_contract_audit.py`.
- Tracking ONNX contract outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.tsv`.
- Tracking ONNX evidence: the official exporter provides `obs`/`time_step`, seven motion outputs, and the required
  metadata keys; the C++ controller consumes `time_step`, motion reference outputs, and anchor/body metadata; the
  Unitree reference ONNX remains rejected as not satisfying the BeyondMimic motion contract. Exporter missing required
  fields count is `0`; the reference Unitree ONNX is missing `11` required motion-contract fields; `body_lin_vel_w` and
  `body_ang_vel_w` are exported but unused by the current C++ consumer path.
- Boundary: this is a static contract audit, not live Kit export, a trained BeyondMimic ONNX, ROS 2 Jazzy/Noble
  MuJoCo launch, or real robot deployment.

## Completion Tracking

- Goal-level completion matrix:
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`.
- Added master evidence audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Master audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.tsv`.
- Added blocked-gate audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/blocked_gate_audit.py`.
- Blocked-gate audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.tsv`.
- Blocked-gate audit result: status `ok`, six gates audited, with `blocked=4`, `clear=1`, and `out_of_scope=1`.
  The blocked gates are IsaacLab/Kit inotify, ROS 2 Jazzy/Noble deployment, missing official Level C artifacts, and
  Fig. 5/Fig. 6 paper-result evidence; real Unitree G1 deployment remains out of scope; the long-training safety gate
  is clear for the current BeyondMimic root.
- Added paper-source coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_source_coverage_audit.py`.
- Paper-source coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/paper_source_coverage/paper_source_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/paper_source_coverage/paper_source_coverage_audit.tsv`.
- Paper-source coverage result: status `ok`, 10 LaTeX figures, 8 LaTeX tables, and 9 key method/formula claims are
  indexed; missing expected labels `0`, unmapped rows `0`, missing evidence paths `0`; bucket counts are
  `strong=5`, `partial=4`, `indexed=3`, `debug_only=10`, and `blocked_or_unreproduced=5`.
- Added paper LaTeX inventory audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_latex_inventory_audit.py`.
- Paper LaTeX inventory outputs:
  `/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_inventory_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_equations.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_tables.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_figures.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_experiment_settings.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_inventory_audit.md`.
- Paper LaTeX inventory result: status `ok`; `5` TeX files are hashed, `51` section headings, `8` equation blocks,
  `10` figures, `8` tables, and `14` key experiment-setting statements are indexed with zero missing expected settings.
  Equation topics are state representation `3`, OU perturbation `1`, joystick cost `1`, waypoint cost `1`,
  SDF obstacle cost `1`, and SDF barrier function `1`.
- Boundary: this is an automatic source inventory and cross-check against existing paper-source/table-value audits. It
  strengthens evidence that the LaTeX formulas, hyperparameters, and experiment settings were read and indexed, but it
  does not create missing trained checkpoints, closed-loop rollout logs, videos, TensorRT engines, or real-robot runs.
- Added paper PDF/source consistency audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_pdf_source_consistency_audit.py`.
- Paper PDF/source consistency outputs:
  `/mnt/infini-data/test/BeyondMimic/res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/paper_pdf_source_consistency/paper_pdf_anchor_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/paper_pdf_source_consistency/paper_source_tar_audit.tsv`.
- Paper PDF/source consistency result: status `ok`; the downloaded PDF has `59` pages and all `59` pages produced text
  with `pypdf`; `20/20` PDF anchors for title, control frequencies, latency, TensorRT/RTX4060/CppAD, OU parameters,
  rollout windows, keyframe interval, velocity-error claims, and long-track claim are present. The source tar contains
  all `19/19` expected source members, all expected members exist in the extracted LaTeX tree, and there are `0`
  unexpected tar file members.
- Boundary: this proves raw PDF/source integrity and source/text-anchor consistency for the current audits. It does not
  produce trained checkpoints, rollout logs, deployment engines, videos, Fig.5/Fig.6 local results, or robot evidence.
- Added paper table value audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_table_value_audit.py`.
- Paper table value outputs:
  `/mnt/infini-data/test/BeyondMimic/res/paper_table_values/paper_table_value_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/paper_table_values/paper_table_value_audit.tsv`.
- Paper table value result: status `ok`, `58` reward/domain/PPO/VAE/diffusion table-value rows audited, mismatch rows
  `0`; table counts are reward `9`, domain randomization `13`, PPO `14`, VAE `8`, diffusion `14`. VAE/diffusion rows
  are explicitly `debug_match`, not paper-level training/checkpoint reproduction.
- Added skill-success table data audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/skill_success_table_data_audit.py`.
- Skill-success table outputs:
  `/mnt/infini-data/test/BeyondMimic/res/paper_skill_success_table_audit/skill_success_table_data_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/paper_skill_success_table_audit/skill_success_table_data_audit.tsv`.
- Skill-success table evidence: `36` table rows parsed (`7` short-sequence, `29` LAFAN), `40` local G1 CSVs scanned,
  and available LAFAN CSVs are finite 36-column files. The audit records one listed LAFAN CSV missing locally
  (`run1_subject4`) and `2` Real interval rows extending slightly past the corresponding local 30 Hz CSV duration; it
  does not claim sim or real success reproduction.
- Added released-data panel mapping audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_panel_mapping_audit.py`.
- Released-data panel mapping outputs:
  `/mnt/infini-data/test/BeyondMimic/res/released_panel_mapping_audit/released_panel_mapping_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/released_panel_mapping_audit/released_panel_mapping_audit.tsv`.
- Released-data panel mapping result: status `ok`, `15` released-data panel-map rows checked against `13` generated
  released-figure groups, source hashes, the extracted `Dataset_beyondmimic` tree, and raw
  `Dataset_beyondmimic.zip`; failed rows `0`. It keeps Fig. 5/Fig. 6 and hardware-only outputs as non-claims.
- Added Level C VAE architecture/loss contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_contract_audit.py`.
- VAE contract outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_contract_audit/level_c_vae_contract_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_contract_audit/level_c_vae_contract_audit.tsv`.
- VAE contract result: status `ok`, `35` contract rows, failed rows `0`; paper/source formulas, 8 VAE table values,
  encoder/decoder/teacher dimensions, runtime shapes, KL/reparameterization/interpolation math, and debug 15-step
  gradient accumulation all pass. The audit remains explicitly debug-only and does not claim a trained VAE checkpoint or
  true DAgger rollout.
- Added Level C sagittal symmetry mapping audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_symmetry_mapping_audit.py`.
- Symmetry mapping outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.tsv`.
- Symmetry mapping result: status `ok`, all `29` official controller action joints are covered by `13` left/right pairs
  plus `3` center joints, controller order matches the augmentation probe, controller joints exist in the URDF, and the
  double-mirror checks pass. The audit records that the paper does not publish a definitive G1 sign table.
- Added Level C guidance cost coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_cost_coverage_audit.py`.
- Guidance cost coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.tsv`.
- Guidance cost coverage result: status `ok`, `8` rows, failed rows `0`; `5` paper-explicit classifier/joystick/waypoint/SDF/barrier rows have source evidence and debug gradients, while the keyframe inpainting cost is explicitly recorded as paper-demonstrated but formula-missing in the local source and implemented only as a candidate term.
- Added Level C timestep/mask source coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_timestep_mask_coverage_audit.py`.
- Timestep/mask coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.tsv`.
- Timestep/mask coverage result: status `ok`, `8` rows, failed rows `0`; paper-explicit independent `k_s/k_z`
  denoising steps, uniform training sampling, and high-level inpainting noise mechanics are source-covered, while the
  concrete deployed mask/keyframe policy is recorded as unpublished. The audit cross-checks the older `181`-D debug
  fixture plus `32`-D synthetic latent and the separate `99`-D paper-state mask/reverse debug artifact.
- Added Level C paper-state mask/reverse probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_mask_reverse_probe.py`.
- Paper-state mask/reverse outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.npz`.
- Paper-state mask/reverse result: status `ok`, paper state dim `99`, latent dim `32`, tau dim `131`; all schedule
  tensors are `[21,2]`, future-keyframe inpainting keeps sparse future state tokens clean, oracle reverse reaches max
  step `0`, clamp error is `0.0`, and MSE falls from `0.11580444302027998` to `5.232890366443532e-10`. This closes the
  prior local 99-D artifact gap but remains debug-only with synthetic latents and an oracle clean predictor.
- Added consolidated final evidence report:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- Final report outputs:
  `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json` and
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`.
- Final report result: status `ok`, Level A/B/C evidence, paper-source coverage, table-value audit, blocked gates, and
  verification commands are consolidated. It explicitly keeps `goal_complete=false`.
- Added goal traceability audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_traceability_audit.py`.
- Goal traceability outputs:
  `/mnt/infini-data/test/BeyondMimic/res/goal_traceability/goal_traceability_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/goal_traceability/goal_traceability_audit.tsv`.
- Goal traceability result: status `ok`, `goal.md` line count `1951`, heading count `80`, trace rows `25`, missing
  evidence rows `0`; status counts are `covered=6`, `partial=17`, `blocked=1`, and `out_of_scope=1`.
- Added paper-vs-reproduction comparison generator:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`.
- Paper-vs-reproduction comparison outputs:
  `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv`,
  `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.md`, and
  `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`.
- Paper-vs-reproduction comparison result: status `ok`, `110` comparison rows, type counts
  `exactly_comparable=58`, `approximately_comparable=19`, `qualitative_only=20`,
  `not_publicly_reproducible=10`, and `requires_real_robot=3`; required goal checkpoint rows for walking velocity
  error `12.14%`, running velocity error `13.65%`, direct diffusion cartwheel success `5%`, and latent diffusion
  cartwheel success `95%` are present and explicitly marked unreproduced when current artifacts are insufficient.
- Goal traceability after adding comparison outputs: status `ok`, status counts are
  `covered=7`, `partial=16`, `blocked=1`, and `out_of_scope=1`.
- Added Level C diffusion deployment protocol audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_deployment_protocol_audit.py`.
- Deployment protocol outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.tsv`.
- Deployment protocol result: status `ok`, `9` paper deployment rows, failed rows `0`; it indexes 25 Hz control,
  20 ms / 20-step diffusion, asynchronous thread, TensorRT, RTX 4060 Mobile Mini PC, synchronous CPU decoder, CppAD
  guidance gradients, proprioceptive state estimation, and mocap context. The audit records `5` expected deployment
  boundaries and does not claim TensorRT/async/CppAD/Mini-PC/live-task reproduction.
- Added Results claims audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/results_claims_audit.py`.
- Results claims outputs:
  `/mnt/infini-data/test/BeyondMimic/res/results_claims_audit/results_claims_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/results_claims_audit/results_claims_audit.tsv`.
- Results claims result: status `ok`, `14` source-indexed Results claims, failed rows `0`; `2` rows are
  partial/released-data reproductions, `5` are debug-only, and the remainder are paper-only, hardware-required, or
  not-publicly-reproducible with explicit missing evidence. The audit does not claim Fig.5/Fig.6 reproduction.
- Added fine-grained goal requirement matrix audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_requirement_matrix_audit.py`.
- Goal requirement matrix outputs:
  `/mnt/infini-data/test/BeyondMimic/res/goal_requirement_matrix/goal_requirement_matrix_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/goal_requirement_matrix/goal_requirement_matrix_audit.tsv`.
- Goal requirement matrix result: status `ok`, `28` fine-grained `goal.md` requirement rows over `1951` goal lines,
  missing evidence rows `0`, status counts `complete=11`, `partial=15`, `blocked=1`, `out_of_scope=1`; it explicitly
  keeps `goal_complete=false`.
- Previous master audit before the core math unit-test addition had all tracked artifacts passing and explicitly kept
  `goal_complete=false` because live Kit tracking, teacher rollouts, true DAgger, trained Level C checkpoints,
  Fig. 5/6 paper reproduction, and real robot deployment are not completed.
- Added pure-NumPy core math unit tests:
  `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py`.
- Core math unit-test outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.tsv`.
- Core math unit-test result: status `ok`, `18` rows, failed rows `0`; covered formula-level checks include Rot6D,
  current-frame inverse transforms, height preservation, reward/termination, adaptive sampling, OU noise, sagittal
  symmetry involution, emphasis projection/pseudoinverse, diffusion forward/reverse, independent timestep and
  inpainting masks, VAE reparameterization/KL, joystick/waypoint/SDF guidance gradients, and smoothness. This is a
  math unit-test gate, not a full trained Isaac/ROS/TensorRT deployment test suite.
- Master audit result after adding core math unit-test evidence: status `ok`, `64/64` key artifacts pass, completion
  matrix counts are `complete=44`, `partial=39`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The audit explicitly
  keeps `goal_complete=false` because live Kit tracking, teacher rollouts, true DAgger, trained Level C checkpoints,
  Fig. 5/6 paper reproduction, and real robot deployment are not completed.
- Created project-local `bm_diffusion` prefix environment:
  `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion`.
- `bm_diffusion` setup and lock artifacts:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/create_bm_diffusion.log`,
  `/mnt/infini-data/test/BeyondMimic/logs/setup/install_bm_diffusion_torch.log`,
  `/mnt/infini-data/test/BeyondMimic/logs/setup/export_bm_diffusion_locks.log`,
  `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/environment.yml`,
  `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/requirements-lock.txt`,
  `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/pip-freeze.txt`, and
  `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/conda-list-explicit.txt`.
- Added `bm_diffusion` environment audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/bm_diffusion_env_audit.py`.
- `bm_diffusion` environment audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.tsv`.
- `bm_diffusion` environment result: status `ok`, Python `3.10.20`, NumPy `2.2.6`, SciPy `1.15.2`, PyYAML `6.0.3`,
  tqdm `4.68.2`, Torch `2.5.1+cu121`, torchvision `0.20.1+cu121`, CUDA available with `8` devices, and CUDA tensor
  smoke passed. The pip install wrapper log is incomplete because it was interrupted after a long shared-filesystem IO
  wait, but exported locks and runtime smoke prove the environment is usable for Level C debug mechanics. This does
  not imply VAE/diffusion training or paper metrics are complete.
- Master audit result after adding `bm_diffusion` environment evidence: status `ok`, `65/65` key artifacts pass,
  completion matrix counts are `complete=44`, `partial=39`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added GPU resource monitoring snapshot/audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/gpu_resource_audit.py`.
- GPU monitoring outputs:
  `/mnt/infini-data/test/BeyondMimic/logs/gpu/gpu_metrics.csv`,
  `/mnt/infini-data/test/BeyondMimic/res/setup/gpu_resource_audit/gpu_resource_audit.json`, and
  `/mnt/infini-data/test/BeyondMimic/res/setup/gpu_resource_audit/gpu_resource_audit.tsv`.
- GPU monitoring result: status `ok`, `24` rows over `8` GPUs for the setup snapshot, goal-required columns are
  present, no artificial load or power/clock modification is used, and existing nontrivial memory use on GPU `6`
  (`3988` MiB) is recorded. This is a monitoring/schema gate, not proof of sustained training utilization or
  throughput targets.
- Master audit result after adding GPU resource monitoring evidence: status `ok`, `66/66` key artifacts pass,
  completion matrix counts are `complete=44`, `partial=40`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added run-management schema diagnostic:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/create_run_management_skeleton.py`.
- Diagnostic run directory:
  `/mnt/infini-data/test/BeyondMimic/res/runs/setup_run_management_diagnostic_static_000_20260617_050000`.
- Run-management audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/run_management_audit/run_management_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/run_management_audit/run_management_audit.tsv`.
- Run-management result: status `ok`; the diagnostic run contains all goal-required files and directories
  (`resolved_config.yaml`, `command.sh`, `stdout.log`, `stderr.log`, `environment.txt`, `git_state.txt`,
  `gpu_metrics.csv`, `metrics.json`, `metrics.csv`, `checkpoint/`, `figures/`, `videos/`, and `status.json`),
  has an allowed `INVALID` status, includes GPU metrics, and explicitly does not mark success without training.
  This is a schema gate, not a completed training run with real checkpoints or videos.
- Master audit result after adding run-management schema evidence: status `ok`, `67/67` key artifacts pass,
  completion matrix counts are `complete=44`, `partial=41`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added lightweight reimplementation package modules under
  `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl`.
- Package APIs now cover geometry/current-frame transforms, sampling/OU/symmetry, diffusion forward/reverse/masks,
  VAE latent math, guidance costs/finite differences, and state emphasis/smoothness helpers. The core math unit tests
  now import these package APIs instead of defining all formulas locally in the test file.
- Added reimplementation package audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reimpl_package_audit.py`.
- Reimplementation package audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/code/reimpl_package_audit/reimpl_package_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/code/reimpl_package_audit/reimpl_package_audit.tsv`.
- Reimplementation package audit result: status `ok`, `11` Python files, `17` expected API symbols, all modules import,
  all expected symbols exist, and tests use the package API. This is reusable formula-level code, not the unpublished
  official training implementation or trained deployment code.
- Master audit result after adding reimplementation package evidence: status `ok`, `68/68` key artifacts pass,
  completion matrix counts are `complete=44`, `partial=42`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added resolved reproduction config manifest generator:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/resolved_reproduction_config.py`.
- Resolved config outputs:
  `/mnt/infini-data/test/BeyondMimic/res/config/resolved_reproduction_config.json`,
  `/mnt/infini-data/test/BeyondMimic/res/config/resolved_reproduction_config.yaml`, and
  `/mnt/infini-data/test/BeyondMimic/res/config/resolved_reproduction_config.csv`.
- Resolved config result: status `ok`; it consolidates tracking 50 Hz/PPO/target-body values, VAE latent and training
  constants, diffusion history/horizon/Transformer/training schedule, GPU policy, run status schema, source evidence,
  and formula package metadata from existing audits. It is a rerun/config contract, not proof of completed long
  training or paper-level metrics.
- Master audit result after adding resolved config evidence: status `ok`, `69/69` key artifacts pass, completion
  matrix counts are `complete=44`, `partial=43`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added failed-run retention script:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/record_failed_run.py`.
- Preserved failed run:
  `/mnt/infini-data/test/BeyondMimic/res/failed_runs/phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654`.
- Failed-run audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/failed_runs/failed_run_audit/failed_run_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/failed_runs/failed_run_audit/failed_run_audit.tsv`.
- Failed-run retention result: status `ok`; it records the IsaacLab/Kit inotify failure with run ID, error, config,
  checkpoint absence, last log containing `errno=28`/`No space left on device`, GPU status, failure reason, resolution
  plan, and `FAILED` status. This preserves failure evidence but does not resolve the host inotify blocker.
- Master audit result after adding failed-run retention evidence: status `ok`, `70/70` key artifacts pass, completion
  matrix counts are `complete=44`, `partial=44`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added key artifact hash manifest:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Artifact manifest outputs:
  `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json` and
  `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.tsv`.
- Artifact manifest result: status `ok`, `27` key artifacts hashed, missing count `0`; categories include raw sources,
  environment locks, config, code audit, run logs, paper audits, tracking, Level C, comparison, final report, and docs.
  This improves current deliverable traceability but does not create missing trained checkpoints, videos, TensorRT
  engines, Fig. 5/6 paper results, or hardware deployment evidence.
- Master audit result after adding artifact manifest evidence: status `ok`, `71/71` key artifacts pass, completion
  matrix counts are `complete=45`, `partial=44`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added experiment protocol document:
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`.
- Added experiment protocol audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/experiment_protocol_audit.py`.
- Experiment protocol audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/docs/experiment_protocol_audit/experiment_protocol_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/docs/experiment_protocol_audit/experiment_protocol_audit.tsv`.
- Experiment protocol result: status `ok`, `19` required protocol patterns, missing count `0`; it covers Phase 0-10,
  no-fabrication rules, run directory contract, failed-run handling, GPU metrics, current Kit/inotify boundary, final
  report separation rules, and explicitly states the full goal remains incomplete.
- Artifact manifest after adding experiment protocol evidence: status `ok`, `29` key artifacts hashed, missing count
  `0`.
- Master audit result after adding experiment protocol evidence: status `ok`, `72/72` key artifacts pass, completion
  matrix counts are `complete=46`, `partial=44`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added top-level README entry point:
  `/mnt/infini-data/test/BeyondMimic/README.md`.
- Added top-level README audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/readme_audit.py`.
- README audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/docs/readme_audit/readme_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/docs/readme_audit/readme_audit.tsv`.
- README audit result: status `ok`, `19` required entry-point patterns, missing count `0`; it points to current
  evidence, commands, blockers, raw-download read-only rules, no-fabrication rules, and failed-run retention rules
  without claiming full paper reproduction completion.
- Artifact manifest after adding README evidence: status `ok`, `31` key artifacts hashed, missing count `0`.
- Master audit result after adding README evidence: status `ok`, `73/73` key artifacts pass, completion matrix counts
  are `complete=47`, `partial=44`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report still keeps
  `goal_complete=false`.
- Added goal.md-specified final report Markdown path:
  `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`, generated from the same final-report script
  as `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`.
- Final-report path result: the generator now writes both Markdown paths plus
  `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`; README audit verifies both report
  links with `20` required entry-point patterns and zero missing rows.
- Artifact manifest after adding the goal-specified final report path: status `ok`, `32` key artifacts hashed, missing
  count `0`.
- Master audit result after adding direct goal-report evidence: status `ok`, `74/74` key artifacts pass, completion
  matrix counts remain `complete=47`, `partial=44`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added final-report requirement coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_report_requirement_audit.py`.
- Final-report requirement audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/final_report/final_report_requirement_audit/final_report_requirement_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/final_report/final_report_requirement_audit/final_report_requirement_audit.tsv`.
- Final-report requirement audit result: status `ok`, `12` explicit `goal.md` final-report requirements checked, missing
  count `0`; the two Markdown report paths are identical and the JSON summary points to the goal-specified report.
- Artifact manifest after adding the final-report requirement audit: status `ok`, `33` key artifacts hashed, missing
  count `0`.
- Master audit result after adding final-report requirement evidence: status `ok`, `75/75` key artifacts pass,
  completion matrix counts remain `complete=47`, `partial=44`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added final deliverables audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_deliverables_audit.py`.
- Final deliverables audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/final_deliverables_audit/final_deliverables_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/final_deliverables_audit/final_deliverables_audit.tsv`.
- Final deliverables audit result: status `ok`, `37` deliverable rows over environment/code/experiment/documentation,
  zero missing listed evidence paths. Status counts are `complete=17`, `partial=11`, `blocked_or_missing=2`, plus
  scoped complete statuses for local copies, released/debug figures, current failures, released-data plotting, and core
  math. Checkpoints and videos are explicitly recorded as `blocked_or_missing` with zero model/video files.
- Artifact manifest after adding final deliverables evidence: status `ok`, `34` key artifacts hashed, missing count `0`.
- Master audit result after adding final deliverables evidence: status `ok`, `76/76` key artifacts pass, completion
  matrix counts are `complete=47`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added mandatory progress report audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/progress_report_audit.py`.
- Progress report audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/progress_report_audit/progress_report_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/progress_report_audit/progress_report_audit.tsv`.
- Progress report audit result: status `ok`, all `21` mandatory `goal.md` section-17 fields are present and nonempty,
  `17` key progress markers are present, and the report records master-audit progression plus incomplete
  checkpoint/video boundaries rather than only saying complete or success.
- Artifact manifest after adding progress-report evidence: status `ok`, `35` key artifacts hashed, missing count `0`.
- Master audit result after adding progress-report evidence: status `ok`, `77/77` key artifacts pass, completion matrix
  counts are `complete=48`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report still
  keeps `goal_complete=false`.
- Added project boundary/raw-download policy audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_boundary_audit.py`.
- Project boundary audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/project_boundary_audit/project_boundary_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/project_boundary_audit/project_boundary_audit.tsv`.
- Project boundary audit result: status `ok`, `8` path/download/cache checks, failed count `0`; it verifies the
  `download` top-level allowlist, supplemental manifest, download manifests, source docs, project-local generated roots,
  and `project_env.sh` cache/tmp redirects.
- Artifact manifest after adding project-boundary evidence: status `ok`, `36` key artifacts hashed, missing count `0`.
- Master audit result after adding project-boundary evidence: status `ok`, `78/78` key artifacts pass, completion
  matrix counts are `complete=49`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`. The final report
  still keeps `goal_complete=false`.
- Added core math checklist coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/core_test_coverage_audit.py`.
- Core math checklist coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_test_coverage_audit/core_test_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_test_coverage_audit/core_test_coverage_audit.tsv`.
- Core math checklist coverage result: status `ok`, all `20` explicit `goal.md` "at least test" checklist items
  have passed-test evidence, missing count `0`, core math test rows `18`, and failed core-test rows `0`. The audit
  remains explicitly scoped to pure-NumPy formula mechanics and does not claim Isaac/ROS/TensorRT training or
  deployment completion.
- Artifact manifest after adding core checklist coverage evidence: status `ok`, `37` key artifacts hashed, missing
  count `0`.
- Master audit result after adding core checklist coverage evidence: status `ok`, `79/79` key artifacts pass,
  completion matrix counts are `complete=50`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added shared finite-value validation for the clean-room formula package:
  `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/validation.py`.
- Updated core formula utilities to reject NaN/Inf inputs and document shape/frame contracts across geometry,
  diffusion, sampling, state projection, guidance, and VAE latent helpers.
- Added core math NaN/Inf guard unit test in
  `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py`.
- Added coding requirements audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/coding_requirements_audit.py`.
- Coding requirements audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/code/coding_requirements_audit/coding_requirements_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/code/coding_requirements_audit/coding_requirements_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/code/coding_requirements_audit/coding_requirements_functions.tsv`.
- Coding requirements audit result: status `ok`, `13` goal.md section-14 coding rows checked, failed requirement
  count `0`, and `19` public formula-package function rows checked for type hints, docstrings, shape/frame
  documentation, finite guards, CLI/YAML/resolved-config evidence, run metadata, and unit tests.
- Core math unit-test result after adding NaN/Inf guards: status `ok`, `19` rows, failed rows `0`, covered goal-item
  tags `26`.
- Core checklist coverage audit after the new guard test: status `ok`, all `20` explicit checklist items still covered,
  missing count `0`, core test rows `19`, and failed core-test rows `0`.
- Reimplementation package audit after adding validation helpers: status `ok`, `12` Python files and `17` expected API
  symbols checked.
- Artifact manifest after adding coding-requirements evidence: status `ok`, `38` key artifacts hashed, missing count
  `0`.
- Master audit result after adding coding-requirements evidence: status `ok`, `80/80` key artifacts pass, completion
  matrix counts are `complete=51`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added executable checkpoint/resume diagnostic:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/checkpoint_resume_smoke.py`.
- Checkpoint/resume smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json`,
  `/mnt/infini-data/test/BeyondMimic/res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.tsv`, and run
  directory `/mnt/infini-data/test/BeyondMimic/res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500`.
- Checkpoint/resume smoke result: status `ok`; it writes diagnostic checkpoint
  `/mnt/infini-data/test/BeyondMimic/res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500/checkpoint/step_0005.npz`,
  reloads seed/step/state, resumes to the final step, and matches an uninterrupted deterministic baseline with max
  absolute resume error `0.0`. The run is explicitly `SUCCESS` for diagnostic plumbing only with `is_training_run=false`
  and does not claim a PPO/VAE/diffusion model checkpoint.
- Coding requirements audit now uses the executable checkpoint/resume smoke as evidence for the `checkpoint_resume`
  row while still recording that no completed training checkpoint exists.
- Artifact manifest after adding checkpoint/resume smoke evidence: status `ok`, `39` key artifacts hashed, missing count
  `0`.
- Master audit result after adding checkpoint/resume smoke evidence: status `ok`, `81/81` key artifacts pass,
  completion matrix counts are `complete=51`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added evaluation metrics coverage audit for `goal.md` Section 12:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/evaluation_metrics_coverage_audit.py`.
- Evaluation metrics coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.tsv`.
- Evaluation metrics coverage result: status `ok`, all `44` required Section 12 metrics mapped with existing evidence
  paths and explicit status labels. Section counts are motion tracking `10`, adaptive sampling `6`, VAE `8`,
  diffusion/guidance `14`, and statistics `6`; missing evidence rows `0`. The motion-tracking local/global error rows
  point to released-data numeric summary tables under
  `/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary`, while paper-level missing metrics remain
  explicit rather than overclaimed.
- Added trial/failure accounting audit for `goal.md` Section 12.5:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/trial_failure_accounting_audit.py`.
- Trial/failure accounting outputs:
  `/mnt/infini-data/test/BeyondMimic/res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.tsv`.
- Trial/failure accounting result: status `ok`, `14` rows; it accounts for `36` source-table skill rows, `24` source
  real segments, `53` released metric rows, `15` debug seed runs, `1` retained failed IsaacLab smoke run, and `0`
  valid completed training runs. It explicitly records missing paper-level rollout trial/failure counts instead of
  treating debug or released-data rows as rollout trials.
- Added patch inventory audit for final code deliverables:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/patch_inventory_audit.py`.
- Patch inventory outputs:
  `/mnt/infini-data/test/BeyondMimic/res/code/patch_inventory_audit/patch_inventory_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/code/patch_inventory_audit/patch_inventory_audit.tsv`.
- Patch inventory result records `reproduction/patches`, the three local official worktrees, official HEAD hashes,
  tracked-change counts or timeout/status samples, and the current absence of explicit patch files. It is hygiene and
  boundary evidence only, not a complete training/deployment patch series.
- Added patch snapshot audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/patch_snapshot_audit.py`.
- Patch snapshot outputs:
  `/mnt/infini-data/test/BeyondMimic/res/code/patch_snapshot_audit/patch_snapshot_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/code/patch_snapshot_audit/patch_snapshot_audit.tsv`, and patch files under
  `/mnt/infini-data/test/BeyondMimic/reproduction/patches/official_worktree_snapshots`.
- Patch snapshot result exports the tracked official-worktree diffs into reproducible `.patch` files and records
  whether semantic diffs are empty. It preserves the boundary that these snapshots are not a curated functional patch
  series for full training, deployment, or paper-result reproduction.
- Added consolidated metrics catalog:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/metrics_catalog.py`.
- Metrics catalog outputs:
  `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.json`,
  `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.csv`, and
  `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.md`.
- Metrics catalog result is regenerated by `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/metrics_catalog.py`;
  current final counts are recorded near the end of this file after all later metric-bearing audits are added. This
  centralizes released-data, debug-only, comparison, coverage, and blocked-boundary metrics without converting them
  into paper-level training/evaluation results.
- Added run/log/config catalog:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_log_config_catalog.py`.
- Run/log/config catalog outputs:
  `/mnt/infini-data/test/BeyondMimic/res/run_log_config_catalog/run_log_config_catalog.json`,
  `/mnt/infini-data/test/BeyondMimic/res/run_log_config_catalog/run_log_config_catalog.csv`, and
  `/mnt/infini-data/test/BeyondMimic/res/run_log_config_catalog/run_log_config_catalog.md`.
- Run/log/config catalog result: status `ok`, `76` files indexed with hashes, including `53` log/text files, `5`
  config files, `4` run/failure IDs, and `0` valid paper-scale training runs. It strengthens traceability for setup,
  data, Level C, failed-run, and diagnostic run evidence, but does not create long training stdout/stderr logs.
- Artifact manifest after adding evaluation metrics coverage evidence: status `ok`, `40` key artifacts hashed,
  missing count `0`.
- Master audit result after adding evaluation metrics coverage evidence: status `ok`, `82/82` key artifacts pass,
  completion matrix counts are `complete=52`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added Phase 9 ablation coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/ablation_coverage_audit.py`.
- Ablation coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/ablation_coverage/ablation_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/ablation_coverage/ablation_coverage_audit.tsv`.
- Ablation coverage result: status `ok`, all `15` `goal.md` Phase 9 ablation items mapped with existing evidence
  paths and explicit status labels. Group counts are motion tracking `6` and diffusion `9`; missing evidence rows `0`.
  Motion-tracking ablations are covered by released-data panels and adaptive-sampling code audit where applicable.
  Diffusion ablations remain explicitly marked as debug/protocol/formula/blocked rather than overclaimed as trained
  paper-result ablations.
- Artifact manifest after adding ablation coverage evidence: status `ok`, `41` key artifacts hashed, missing count `0`.
- Master audit result after adding ablation coverage evidence: status `ok`, `83/83` key artifacts pass, completion
  matrix counts are `complete=53`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added Phase 8 guidance task coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/guidance_task_coverage_audit.py`.
- Guidance task coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/guidance_task_coverage/guidance_task_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/guidance_task_coverage/guidance_task_coverage_audit.tsv`.
- Guidance task coverage result: status `ok`, all `30` `goal.md` Phase 8 task/requirement rows mapped:
  six tasks x five evidence types. Task counts are `5` each for unconditional rollout, joystick, waypoint,
  inpainting, obstacle avoidance, and composed objectives. Requirement counts are `6` each for without-guidance,
  with-guidance, multiple guidance weights, success/failure videos, and quantitative metrics. Missing evidence rows
  are `0`.
- The guidance task audit explicitly separates debug evidence from paper-level evidence: joystick has debug
  unguided/guided/scale-sweep coverage; inpainting has mask/reverse metrics and candidate keyframe guidance evidence;
  waypoint, obstacle avoidance, and composed objectives have formula-gradient evidence. All success/failure video rows
  remain `blocked_missing_videos`, and trained closed-loop rollouts are not overclaimed.
- Artifact manifest after adding guidance task coverage evidence: status `ok`, `42` key artifacts hashed, missing
  count `0`.
- Master audit result after adding guidance task coverage evidence: status `ok`, `84/84` key artifacts pass,
  completion matrix counts are `complete=54`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added required trained/deployment artifact absence audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`.
- Required artifact absence outputs:
  `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.tsv`.
- Required artifact absence result: status `ok`, `15` required artifact classes audited. Status counts are
  `missing_required_artifact=12`, `debug_only_not_required_artifact=2`, and
  `present_but_not_required_artifact=1`. It records local reproduction model files excluding diagnostic checkpoints
  `0`, local reproduced videos `0`, diagnostic checkpoint files `1`, downloaded reference model files `31`, and
  downloaded reference videos `39`.
- The audit explicitly separates downloaded reference-project assets from BeyondMimic reproduction artifacts:
  reference ONNX/PT/GIF files from Unitree/ASAP/PBHC/motion-diffusion/IsaacLab docs are not counted as reproduced
  BeyondMimic tracking/VAE/diffusion checkpoints, TensorRT engines, rollout logs, Fig.5/Fig.6 artifacts, completed
  training run directories, or success/failure videos.
- Artifact manifest after adding required-artifact absence evidence: status `ok`, `43` key artifacts hashed, missing
  count `0`.
- Master audit result after adding required-artifact absence evidence: status `ok`, `85/85` key artifacts pass,
  completion matrix counts are `complete=55`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added lightweight clean-room DAgger, trajectory, and evaluation package APIs:
  `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/dagger/schema.py`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/trajectory/dataset.py`, and
  `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/evaluation/metrics.py`.
- New package APIs cover DAgger sample validation with explicit `teacher_queried` flags, teacher/student discrepancy
  metrics, state-latent token concatenation, train/validation/test split counts, action MSE, tracking error, and
  survival rate. These are schema/metric implementations for future real rollouts; they do not claim true DAgger
  rollout, a trained VAE checkpoint, or closed-loop policy evaluation.
- Core math unit tests now run `22` pure-NumPy formula/schema tests with failed rows `0`, adding tests for DAgger
  teacher-query metrics, state-latent window/split schema, and tracking/survival metrics. Covered goal-item tags now
  include `35` items.
- Reimplementation package audit after adding DAgger/trajectory/evaluation APIs: status `ok`, `18` Python files and
  `25` expected API symbols checked.
- Coding requirements audit after adding these APIs: status `ok`, `28` public functions checked against type hints,
  docstrings, shape/frame documentation, finite guards where applicable, CLI/YAML/resolved-config evidence, run
  metadata, and unit tests; failed requirement count `0`.
- Core checklist coverage audit remains status `ok`: all `20` explicit core math checklist rows are still covered,
  now backed by `22` passing core-test rows.
- Artifact manifest after adding DAgger/trajectory/evaluation API evidence remains status `ok`, `43` key artifacts
  hashed, missing count `0`.
- Master audit result after adding DAgger/trajectory/evaluation API evidence remains status `ok`, `85/85` key artifacts
  pass, completion matrix counts are `complete=55`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added package-backed state-latent schema audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/state_latent_schema_audit.py`.
- State-latent schema audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_schema_audit/state_latent_schema_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_schema_audit/state_latent_schema_audit.tsv`.
- State-latent schema audit result: status `ok`, `84` paper-state debug windows validated through the
  `beyondmimic_reimpl.trajectory` package API. Split counts are `train=28`, `validation=28`, and `test=28`; motion
  counts are `28` windows each for walk/run/jump fixtures; every concatenated state-latent token has shape `[21,131]`.
- The audit explicitly marks all `84` latent sequences as `synthetic_zero_placeholder_for_schema_only`, so this proves
  schema/provenance/split wiring for the current debug dataset but does not claim true trained VAE latents, teacher
  rollout data, or a paper-level trainable state-latent dataset.
- Added package-backed DAgger schema audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/dagger_schema_audit.py`.
- DAgger schema audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_schema_audit/dagger_schema_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_schema_audit/dagger_schema_audit.tsv`.
- DAgger schema audit result: status `ok`, `30` deterministic synthetic teacher-query samples validated through the
  `beyondmimic_reimpl.dagger` and `beyondmimic_reimpl.evaluation` package APIs. Split counts are `train=20`,
  `validation=5`, and `test=5`; all samples have 163-D teacher input states, 29-D teacher/student actions,
  `teacher_queried=true`, finite arrays, and finite discrepancy metrics with action MSE `0.0012922259414464622`.
- The DAgger audit explicitly marks `is_true_dagger_rollout=false`, so it proves schema and metric wiring for the
  current synthetic VAE/DAgger debug path but does not claim Isaac rollout, a real teacher policy query, true DAgger
  aggregation over environment states, or a trained VAE checkpoint.
- Added debug-only iterative DAgger aggregation smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_dagger_iteration_smoke.py`.
- DAgger iteration smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.tsv`.
- DAgger iteration result: status `ok`; it runs 3 synthetic student-rollout/teacher-query aggregation iterations with
  teacher queries `96 -> 192 -> 288`, aggregate dataset sizes `96 -> 192 -> 288`, 163-D states, 29-D actions, and
  held-out action MSE falling from `0.3170884047315886` to `6.111142407404412e-12`.
- Boundary: this proves local iterative DAgger mechanics (query, aggregate, update, held-out discrepancy reduction) in a
  deterministic synthetic setting. It is not Isaac rollout, a true BeyondMimic teacher policy query, environment-state
  aggregation, a trained conditional VAE checkpoint, or closed-loop VAE survival/fall evaluation.
- Added debug-only VAE checkpoint save/load smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_checkpoint_smoke.py`.
- VAE checkpoint smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_checkpoint_smoke/debug_conditional_vae_checkpoint_smoke.pt`.
- VAE checkpoint smoke result: status `ok`; it runs the paper-dimension synthetic conditional VAE for one accumulated
  optimizer step, saves model and optimizer state, reloads them into a fresh model/optimizer, and verifies deterministic
  eval-action max error `0.0`. The checkpoint file is `68387282` bytes, VAE parameter count is `5697117`, and
  optimizer state tensor count is `11394250`.
- The VAE checkpoint smoke explicitly marks `is_trained_paper_checkpoint=false`. The required-artifact absence audit
  excludes this debug `.pt` from trained-model counts, so it does not claim a true DAgger-trained VAE checkpoint,
  closed-loop VAE survival evaluation, or paper metrics.
- Added tiny VAE debug overfit latent artifact:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_debug_overfit_latent_artifact.py`.
- Tiny VAE debug latent outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz`.
- Tiny VAE debug latent result: status `ok`, `84` windows / `1764` tokens, split counts `28/28/28`, all latents
  nonzero and finite, token shape `[21,131]`, reconstruction MSE `0.08905366063117981 -> 0.000199218382476829`,
  and latent abs mean `0.1456600978669161`.
- Boundary: this replaces some zero-placeholder schema evidence with nonzero debug latents from synthetic teacher
  projections, but it is not true DAgger, not a trained paper VAE checkpoint, and not an accepted VAE rollout dataset.
- Added VAE motion-split held-out evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_motion_split_heldout_eval.py`.
- VAE motion-split held-out outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.npz`.
- VAE motion-split held-out result: status `ok`; the tiny conditional VAE trains only on the train motion split
  (`588` tokens) and evaluates held-out validation/test motions (`588/588` tokens). Action MSE falls
  `0.10267462700281195 -> 0.004535351618306475` on validation and
  `0.1096794881829601 -> 0.008726497973125745` on test; validation/test reduction ratios are
  `0.955827922139106` and `0.9204363722178501`.
- Boundary: this is a debug-only held-out VAE training/evaluation gate using synthetic teacher projections from local
  paper-state fixtures. It is not true DAgger rollout, not a trained paper conditional VAE checkpoint, and not
  closed-loop VAE stability evaluation.
- Added VAE receding-horizon rollout smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_receding_horizon_rollout_smoke.py`.
- VAE receding-horizon rollout smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.tsv`,
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.npz`.
- VAE receding-horizon rollout result: status `ok`; it rolls the current-action index over all `84` exported
  tiny-VAE debug state-latent windows from 3 motions, with 99-D states, 32-D latents, 29-D actions, and current index
  `4`. Current-action MSE mean/max are `0.0001923113866260608` and `0.0008571597683028913`; full-window action MSE
  mean is `0.00019921838212813082`; next-latent action delta mean is `0.06758337096089406`.
- Boundary: this validates the local current-latent action inference contract over tiny-VAE debug exports only. It is
  not true DAgger, not a trained paper VAE checkpoint, not closed-loop Isaac evaluation, and not a required
  success/failure video.
- Added diffusion-to-VAE action smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_smoke.py`.
- Diffusion-to-VAE action smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.tsv`,
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.npz`.
- Diffusion-to-VAE action result: status `ok`; a train-split-only surrogate action decoder maps held-out diffusion
  predicted `[21,131]` state-latent tokens into 29-D actions. Validation/test current-action MSE are
  `0.00911132119759807` and `0.0039371201747736865`, improving over noisy-token current-action baselines by
  `0.8536799905511543` and `0.9435052428100379`.
- Boundary: this is a debug-only pipe from held-out diffusion token prediction to a surrogate tiny-VAE action decoder.
  It is not the unpublished trained VAE decoder, not a trained diffusion Transformer checkpoint, not Isaac closed-loop
  rollout, and not a paper Fig. 5/Fig. 6 protocol.
- Added diffusion-to-VAE action multi-seed audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_multiseed_audit.py`.
- Diffusion-to-VAE action multi-seed outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.tsv`.
- Diffusion-to-VAE action multi-seed result: status `ok`; seeds `20260919/20260920/20260921` all improve held-out
  validation/test full and current action MSE vs noisy-token baselines. Validation/test predicted current-action MSE
  mean/std are `0.009255110754950491/0.0008873826936215285` and
  `0.003998468510847306/0.00032580665084176963`; validation/test current-action reduction means are
  `0.8466026289638074` and `0.9298716385610665`.
- Boundary: this is smoke-level multi-seed reporting for a surrogate downstream action decoder, not true VAE rollout,
  trained VAE decoder, trained diffusion Transformer, closed-loop Isaac evaluation, or paper multi-seed evaluation.
- Added diffusion-to-VAE action smoothness audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_smoothness_audit.py`.
- Diffusion-to-VAE action smoothness outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.tsv`.
- Diffusion-to-VAE action smoothness result: status `ok`; over the downstream 21-step, 29-D action windows, validation
  and test predicted smoothness penalties are `0.038083277980476095` and `0.01579856790234264`, reducing the noisy-token
  smoothness penalties by `0.8984023373821292` and `0.9548695448720198`. The predicted action-rate mean norms at 25 Hz
  are `14.520518537468638` and `9.46269689007263`; predicted action-acceleration mean norms are
  `620.1071585816564` and `401.59270034038593`.
- Boundary: this is offline debug smoothness evidence for the surrogate diffusion-to-VAE action stream, not a trained
  closed-loop policy, TensorRT/deployment benchmark, real robot run, or paper-level action smoothness evaluation.
- Added debug-VAE-latent diffusion overfit gate:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_diffusion_overfit_probe.py`.
- Debug-VAE-latent diffusion outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.npz`.
- Debug-VAE-latent diffusion result: status `ok`, `84` windows / `1764` tokens, paper-state dim `99`, tiny-VAE latent
  dim `32`, token dim `131`, latent source `debug_tiny_vae_mu_nonzero_synthetic_teacher`, and baseline clean-trajectory
  loss `0.06669130873528159` falling to `3.341915582546504e-19`.
- Boundary: this is an overparameterized CPU debug gate using nonzero tiny-VAE latents; it is not true VAE rollout,
  not a trained diffusion Transformer checkpoint, and not a Fig. 5/Fig. 6 reproduction.
- Added debug-VAE-latent held-out diffusion evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_heldout_eval.py`.
- Debug-VAE-latent held-out outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.npz`.
- Debug-VAE-latent held-out result: status `ok`, no token identity basis, motion-level split, paper-state dim `99`,
  tiny-VAE latent dim `32`, token dim `131`; validation loss `0.07110373914021167 -> 0.007872506845332104`, and
  test loss `0.06889640177242959 -> 0.007374265934471514`.
- Boundary: this is a non-memorizing held-out ridge debug baseline over nonzero tiny-VAE latents; it is not true VAE
  rollout, a trained diffusion Transformer checkpoint, or paper validation/test evaluation.
- Added debug-VAE-latent held-out multi-seed audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_heldout_multiseed_audit.py`.
- Debug-VAE-latent held-out multi-seed outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json`
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.tsv`.
- Debug-VAE-latent held-out multi-seed result: status `ok`, seeds `20260919/20260920/20260921`, all losses finite,
  no token identity basis, nonzero tiny-VAE latents, state/latent/token dims `99/32/131`, and all seeds reduce
  validation/test loss. Validation loss-reduction ratio mean/std is `0.8860020143594977/0.0038682645702139994`;
  test loss-reduction ratio mean/std is `0.8925315111873378/0.0003842527523094619`.
- Boundary: this is smoke-level statistics for a debug ridge baseline, not true VAE rollout data, a trained diffusion
  Transformer checkpoint, or paper multi-seed evaluation.
- Added debug-only Transformer state-dict hash manifest:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_state_dict_manifest.py`.
- Transformer state-dict manifest outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.tsv`.
- Transformer state-dict manifest result: status `ok`; it rebuilds the paper-state Transformer initialization with
  token dim `131`, embedding `512`, heads `8`, layers `6`, and denoising steps `20`, records `79` state-dict tensor
  hashes, confirms state-dict numel/parameter count `19080323`, verifies same-seed deterministic hash
  `ff12bb75791998a31e227c11eda0f7a2721b3d25d2b2e3ede1c9b2e4903ce756`, and confirms a different seed changes the
  hash.
- The Transformer state-dict manifest explicitly does not write a trained weight checkpoint or EMA checkpoint. It is
  architecture/init traceability evidence only, not diffusion training, TensorRT deployment, or rollout evaluation.
- Added debug-only diffusion Transformer checkpoint save/load/resume smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_checkpoint_smoke.py`.
- Diffusion checkpoint smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_checkpoint_smoke/debug_diffusion_transformer_checkpoint_smoke.pt`.
- Diffusion checkpoint smoke result: status `ok`, paper-state token dim `131`, Transformer parameter count `19080323`,
  checkpoint size `305417863` bytes, resumed model/EMA/optimizer SHA-256 hashes exactly match the uninterrupted path,
  eval max abs error after resume is `0.0`, and final resumed/uninterrupted loss after step 2 is
  `1.7795637845993042`.
- Boundary: the `.pt` file is marked `is_trained_paper_checkpoint=false` and `is_ema_paper_checkpoint=false`. It proves
  debug save/load/resume mechanics only, and required-artifact absence audit excludes it from trained-checkpoint counts.
- Added paper-sized Transformer architecture probe using nonzero tiny-VAE debug latents:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_transformer_arch_probe.py`.
- VAE-latent Transformer architecture outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.json`,
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.tsv`,
  and `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.npz`.
- VAE-latent Transformer architecture result: status `ok`, token dim `131`, state/latent dims `99/32`, paper
  Transformer hyperparameters (`512` embed, `8` heads, `6` layers, `20` steps), parameter count `19080323`,
  clean-trajectory MSE `1.5802749395370483`, and gradient norm `6.366533279418945`.
- Boundary: this proves the paper-sized Transformer forward/backward path can consume nonzero tiny-VAE debug latents,
  but it is not true VAE rollout data, diffusion training, a trained checkpoint, TensorRT deployment, or rollout
  evaluation.
- Added debug-only VAE-latent Transformer AdamW/LR/EMA smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_transformer_ema_smoke.py`.
- VAE-latent Transformer EMA smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.tsv`.
- VAE-latent Transformer EMA smoke result: status `ok`; it runs two debug optimizer steps on one paper-sized
  state-latent batch with nonzero tiny-VAE debug latents, token dim `131`, state/latent dims `99/32`, latent abs mean
  `0.1456600978669161`, paper Transformer parameter count `19080323`, AdamW weight decay `0.001`, LR warmup values
  `1e-08` and `2e-08`, and EMA decay values `0.0` and `0.4053964424986395`. Grad norms are positive, model
  parameters change, EMA shadow separates from final model with L2 `6.398524821094043e-09`, and the last-batch loss
  changes from `1.5763062238693237` to `1.5761398077011108`.
- Boundary: this proves optimizer/LR/EMA mechanics can consume the nonzero tiny-VAE debug latent token stream. It does
  not write a trained diffusion checkpoint or EMA checkpoint and is not true VAE rollout data, paper-scale training,
  validation/test metrics, TensorRT deployment, or rollout evaluation.
- Added debug-only Transformer AdamW/LR/EMA smoke:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_ema_smoke.py`.
- Transformer EMA smoke outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.tsv`.
- Transformer EMA smoke result: status `ok`; it runs two debug optimizer steps on one paper-state batch with token dim
  `131`, paper Transformer hyperparameters, AdamW weight decay `0.001`, LR warmup values `1e-08` and `2e-08`, and EMA
  decay values `0.0` and `0.4053964424986395`. Grad norms are positive (`5.38038969039917`,
  `4.348121643066406`), model parameters change, EMA shadow separates from final model with L2
  `6.1767519987654396e-09`, and the last-batch loss changes from `1.5448564291000366` to `1.5447429418563843`.
- The Transformer EMA smoke does not write a weight checkpoint or EMA checkpoint. It validates local training mechanics
  only, not paper-scale diffusion training, validation/test metrics, TensorRT deployment, or rollout evaluation.
- Added formula-level task-specific guidance scale sweep:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_task_scale_sweep.py`.
- Guidance task scale sweep outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.tsv`.
- Guidance task scale sweep result: status `ok`, `40` rows from 5 tasks x 8 scales over one motion-derived future
  state tensor. Tasks are joystick, waypoint, obstacle avoidance, inpainting, and composed objectives. All rows are
  finite, all task gradients are nonzero, and every task has an improving positive scale. Best cost deltas are
  joystick `2.0925944100191884e-05`, waypoint `4.105169410864584e-07`, obstacle `1.2331344537504947`,
  inpainting `1.6731278190164345e-10`, and composed objectives `1.2331578715416072`.
- Guidance task coverage now points waypoint/obstacle/composed/inpainting `multiple_guidance_weights` rows to this
  formula-level task sweep instead of only recording a missing task-specific sweep. This still does not claim
  closed-loop guided diffusion rollout, task validation/test scale selection, success/failure videos, or Fig. 5/6
  reproduction.
- Added debug-only guidance trajectory visualization:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_debug_visualization.py`.
- Guidance debug visualization outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json`,
  `.tsv`, `.npz`, `.png`, `.svg`, `.pdf`, and `.gif`.
- Guidance debug visualization result: status `ok`; it visualizes joystick, waypoint, obstacle avoidance,
  inpainting, and composed formula effects on one local fixture trajectory. Primary metrics improve for all five
  debug tasks: joystick velocity-command MSE `0.12435979371656938 -> 0.05254195714655683`, waypoint terminal distance
  `0.09053375319717637 -> 0.06804232128257154`, obstacle clearance `-0.1598505830727198 -> 0.03950747084636402`,
  inpainting keyframe error `0.04599400237472454 -> 0.013521512305151584`, and composed clearance
  `-0.1598505830727198 -> -0.06329020028220764`.
- Boundary: the GIF is a debug formula-effect visualization and is explicitly excluded from the required paper
  success/failure video count. It is not a trained guided diffusion rollout, not closed-loop simulation, and not
  Fig. 5/Fig. 6 reproduction.
- Added guidance task metric audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_task_metric_audit.py`.
- Guidance task metric audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.tsv`.
- Guidance task metric audit result: status `ok`, `5` rows, all five debug primary metrics improve and all five rows
  are linked to formula-level scale-sweep summaries. It centralizes joystick velocity-command MSE, waypoint terminal
  distance, obstacle clearance, inpainting keyframe error, and composed clearance before/after metrics for the local
  fixture.
- Boundary: these are fixture-level debug quantitative metrics. They do not claim task success rates, closed-loop
  guided diffusion rollouts, success/failure videos, or Fig. 5/Fig. 6 paper reproduction.
- Added verification command coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_coverage_audit.py`.
- Verification command coverage outputs:
  `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.tsv`.
- This audit categorizes the final report verification-command list and executes a bounded five-command lightweight
  smoke subset. It is evidence for command hygiene and repeatability only; it does not execute IsaacLab/Kit, ROS 2,
  long VAE/diffusion training, TensorRT, Fig. 5/Fig. 6 evaluation, or real Unitree G1 deployment.
- Artifact manifest after adding state-latent, DAgger schema, VAE checkpoint-smoke, Transformer state-dict,
  DAgger iteration smoke, Transformer EMA, VAE-latent Transformer EMA, released-data metric tables, metrics catalog,
  run/log/config catalog, guidance task-sweep, final-report, and artifact-absence evidence: status `ok`, `62` key artifacts hashed, missing
  count `0`.
- Master audit result after adding state-latent, DAgger schema, VAE checkpoint-smoke, Transformer state-dict,
  DAgger iteration smoke, Transformer EMA, VAE-latent Transformer EMA, released-data metric tables, metrics catalog,
  run/log/config catalog, guidance task-sweep, final-report, and artifact-absence evidence: status `ok`, `104/104` key artifacts pass,
  completion matrix counts are `complete=55`, `partial=45`, `blocked=7`, `out_of_scope=1`, and `pending=0`.
  The final report still keeps `goal_complete=false`.
- Added reimplementation runtime integration audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reimpl_runtime_integration_audit.py`.
- Runtime integration audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.tsv`.
- Runtime integration audit result: status `ok`; the clean-room `beyondmimic_reimpl` package APIs run together over
  local debug state-latent windows, downstream diffusion-to-action outputs, and a motion fixture. It verifies `84`
  windows, `[84,21,131]` state-latent tokens, split counts `train=28`, `validation=28`, `test=28`, state/latent/action
  dims `99/32/29`, projection reconstruction max error `1.8908485888147197e-15`, diffusion reverse MSE
  `0.5864538340928773 -> 0.2814410265203397`, decoded-teacher action MSE `0.00019921838212813084`,
  downstream current-action MSE `0.007938830058739797`, DAgger teacher-query count `30`, tracking mean error
  `0.017320508075688783`, and survival rate `0.75`.
- Boundary: this is fixture/debug runtime integration evidence for the current clean-room package. It does not claim
  official BeyondMimic VAE/diffusion training code, a trained checkpoint, closed-loop Isaac rollout, TensorRT engine,
  or hardware deployment.
- Added Level C state-latent dataset consistency audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_latent_dataset_consistency_audit.py`.
- State-latent dataset consistency outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.tsv`.
- Consistency result: status `ok`, `84` rows, dimensions state/latent/token/action `99/32/131/29`, split counts
  `train=28`, `validation=28`, `test=28`, max state-array difference between paper-state windows and tiny-VAE NPZ
  `1.1409430689113265e-07`, decoded-action target max error `0.0`, latent abs mean `0.1456600978669161`, current
  root position and current root linear velocity both exactly zero at the history index.
- Boundary: this proves the local paper-state, debug latent/action, and downstream action artifacts are internally
  aligned. It is not the paper's true DAgger/VAE rollout dataset, not a trained checkpoint, and not a closed-loop
  rollout.
- Added offline direct-vs-latent action ablation audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_direct_vs_latent_action_ablation_audit.py`.
- Direct-vs-latent action ablation outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json`,
  `.tsv`, and `.npz`.
- Direct-vs-latent action ablation result: status `ok`; direct branch uses only 99-D state features plus tanh state
  and bias, while latent branch uses the existing 131-D state-latent downstream action pipe. Held-out validation
  current-action MSE is direct `0.009613393533001568` vs latent `0.009111321118349816` (ratio
  `0.9477736542326911`), and held-out test current-action MSE is direct `0.006074739836526304` vs latent
  `0.003937120162097986` (ratio `0.6481133790166287`).
- Boundary: this upgrades the direct-vs-latent ablation from paper-only to local debug data-path evidence. It still
  does not reproduce the paper's 5% vs 95% cartwheel success rates, train direct/latent diffusion checkpoints, or run
  closed-loop Isaac/robot evaluation.
- Added package-level API contract tests for the clean-room reimplementation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_reimpl_package_api.py`.
- Package API test outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.tsv`.
- Package API test result: status `ok`, `7` rows, failed `0`; covered items are `api_surface`, `dagger`,
  `diffusion`, `evaluation`, `finite_guards`, `fixed_seed`, `geometry`, `guidance`, `mask_shape`,
  `package_exports`, `sampling`, `shape_errors`, `state`, `trajectory`, and `vae`.
- Boundary: these are pure NumPy/stdlib API-contract tests for exported symbols, shape/error paths, metadata helpers,
  finite-value guards, and local reusable formulas. They do not execute IsaacLab, ROS 2, TensorRT, long training,
  paper-level evaluation, or hardware deployment.
- After adding the package API tests, refreshed the dependent audits and final reports. Current verification state:
  artifact manifest status `ok` with `80` hashed artifacts and missing count `0`; master audit status `ok` with
  `122/122` artifacts passing and completion matrix counts `complete=61`, `partial=47`, `blocked=7`,
  `out_of_scope=1`; metrics catalog status `ok` with `21` metric-bearing sources, `11` debug-only sources,
  `3` coverage-audit sources, and `425` indexed rows; verification command coverage status `ok` with `99` commands
  and `5/5` lightweight smoke commands passing. The final report remains `goal_complete=false`.
- Added download source integrity audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/download_source_integrity_audit.py`.
- Download source integrity outputs:
  `/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_manifest.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_required.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_audit.md`.
- Download source integrity result: status `ok`; the raw download manifest records `6391` rows totaling
  `6577530557` bytes. The audit hashes `17` required paper/official/dependency/manifest files and `8` reference-code
  README files, including the paper PDF/source tar, official BeyondMimic dataset zip, LAFAN retargeting metadata,
  whole-body tracking source, motion-tracking controller source, Unitree description archive, IsaacLab/rsl_rl/unitree
  dependency markers, Zenodo/resource/git/download manifests, supplemental manifest, and diffusion/reference-code
  README anchors.
- Boundary: this is read-only source-bundle provenance evidence. It does not provide missing Level C official code,
  trained checkpoints, Fig. 5/6 rollout data, videos, TensorRT engines, ROS deployment evidence, or hardware results.
- After adding the download source integrity audit, refreshed the dependent audits and final reports. Current
  verification state: artifact manifest status `ok` with `81` hashed key artifacts and missing count `0`; master audit
  status `ok` with `123/123` artifacts passing and completion matrix counts `complete=62`, `partial=47`,
  `blocked=7`, `out_of_scope=1`; verification command coverage status `ok` with `100` commands and `5/5`
  lightweight smoke commands passing. The final report remains `goal_complete=false`.
- Added goal.md directive index audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_directive_index_audit.py`.
- goal.md directive index outputs:
  `/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_directive_index_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_directive_rows.tsv`,
  `/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_heading_rows.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_directive_index_audit.md`.
- goal.md directive index result: status `ok`; it reads `1951` lines and `80` headings from `goal.md`, indexes `258`
  directive-bearing rows, and records tag counts `boundary=29`, `deliverable=72`, `execution=131`, `mandatory=41`,
  and `prohibition=42`. It cross-checks line and heading counts against the goal traceability audit and verifies the
  traceability/requirement-matrix evidence lists still have zero missing evidence paths.
- Boundary: this proves line-level reading/indexing/tracking coverage for `goal.md`; it does not complete blocked
  Isaac/ROS execution, true DAgger, trained checkpoints, Fig. 5/Fig. 6, videos, TensorRT, or hardware deployment.
- After adding the goal.md directive index audit, refreshed the dependent audits and final reports. Current verification
  state: artifact manifest status `ok` with `82` hashed key artifacts and missing count `0`; master audit status `ok`
  with `124/124` artifacts passing and completion matrix counts `complete=63`, `partial=47`, `blocked=7`,
  `out_of_scope=1`; verification command coverage status `ok` with `101` commands and `5/5` lightweight smoke
  commands passing. The final report remains `goal_complete=false`.
- Added paper formula/code trace audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_formula_code_trace_audit.py`.
- Paper formula/code trace outputs:
  `/mnt/infini-data/test/BeyondMimic/res/paper_formula_code_trace/paper_formula_code_trace_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/paper_formula_code_trace/paper_formula_code_trace_audit.tsv`, and
  `/mnt/infini-data/test/BeyondMimic/res/paper_formula_code_trace/paper_formula_code_trace_audit.md`.
- Paper formula/code trace result: status `ok`, `11` trace rows, missing evidence `0`. It maps the `8` LaTeX
  equations, `14` experiment-setting statements, `58` paper table-value rows, `20` core-test checklist items, `25`
  reimplementation API symbols, and `7` package API tests to local code/tests/audits. Status counts are
  `covered_debug_formula=6`, `covered_debug_architecture=2`, `covered_static_source=1`, `covered_protocol_only=1`,
  and `indexed_blocked_or_partial=1`.
- Boundary: this strengthens formula/source-to-code traceability. It does not provide trained tracking/VAE/diffusion
  checkpoints, true DAgger rollouts, TensorRT deployment, Fig. 5/Fig. 6 paper results, reproduced videos, or hardware
  results.
- After adding the paper formula/code trace audit, refreshed the dependent audits and final reports. Current
  verification state: artifact manifest status `ok` with `83` hashed key artifacts and missing count `0`; master audit
  status `ok` with `125/125` artifacts passing and completion matrix counts `complete=64`, `partial=47`,
  `blocked=7`, `out_of_scope=1`; verification command coverage status `ok` with `102` commands and `5/5`
  lightweight smoke commands passing. The final report remains `goal_complete=false`.
- Reran the safe non-Kit `whole_body_tracking` package/asset smoke:
  `/mnt/infini-data/test/BeyondMimic/logs/setup/whole_body_tracking_nokit_smoke_rerun.log`.
- Added tracking smoke rerun audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_smoke_rerun_audit.py`.
- Tracking smoke rerun audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.tsv`.
- Tracking smoke rerun result: status `ok`, `3` rows, failed `0`. The non-Kit package/asset smoke passes and checks
  the official `whole_body_tracking` package, G1 config files, G1 URDF, and first mesh reference. Current retry
  condition for Kit remains blocked: `fs.inotify.max_user_watches=8192`, `fs.inotify.max_user_instances=128`, with
  `17` previous `errno=28` change-watch failures recorded.
- Boundary: this is a safe Level-B rerun that does not launch Kit, start PPO, export ONNX, run replay, or complete
  motion-tracking reproduction. Full IsaacLab/Kit gates still require raised inotify limits.
- After adding the tracking smoke rerun audit, refreshed the dependent audits and final reports. Current verification
  state: artifact manifest status `ok` with `84` hashed key artifacts and missing count `0`; master audit status `ok`
  with `126/126` artifacts passing and completion matrix counts `complete=64`, `partial=47`, `blocked=7`,
  `out_of_scope=1`; verification command coverage status `ok` with `103` commands and `5/5` lightweight smoke
  commands passing. The final report remains `goal_complete=false`.
- Added Kit/inotify budget audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/kit_inotify_budget_audit.py`.
- Kit/inotify budget audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.tsv`.
- Kit/inotify budget result: status `ok`, `5` rows, failed `0`. It parses `17` unique Kit change-watch paths from the
  retry log, records current limits `fs.inotify.max_user_watches=8192` and `fs.inotify.max_user_instances=128`,
  proves a bounded Isaac Sim directory lower bound of `8193` directories exceeds the current watch budget, and records
  `/shared_disk` available capacity `5618234793394176` bytes. This strengthens the blocker evidence: Kit's
  `errno=28/No space left on device` messages are attributable to inotify watcher budget, not ordinary disk-capacity
  exhaustion.
- Boundary: this audit does not launch Kit, start PPO, run official `csv_to_npz.py`/`replay_npz.py`, export ONNX, or
  complete motion-tracking reproduction. Full IsaacLab/Kit gates still require host inotify limits to be raised and
  live smoke rerun.
- After adding the Kit/inotify budget audit, refreshed the dependent audits and final reports. Current verification
  state: artifact manifest status `ok` with `85` hashed key artifacts and missing count `0`; master audit status `ok`
  with `127/127` artifacts passing and completion matrix counts `complete=64`, `partial=47`, `blocked=7`,
  `out_of_scope=1`; verification command coverage status `ok` with `104` commands and `5/5` lightweight smoke
  commands passing. The final report remains `goal_complete=false`.
- Added Kit watcher config surface audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/kit_watcher_config_surface_audit.py`.
- Kit watcher config surface audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.tsv`.
- Kit watcher config surface result: status `ok`, `6` rows, failed `0`. It indexes the local
  `isaacsim.exp.base.python.kit` app extension roots (`5` folders), maps `11/17` failed retry watch paths to known
  Kit/Isaac Sim/IsaacLab watch roots, finds `12` local extension configs with `[fswatcher.*]` sections, and records
  `24` local `omni.kit.watched_config` references. It did not find a documented global watcher-disable setting in the
  local Kit app configs.
- Boundary: this is a source/config audit only. It does not modify the Isaac Sim installation, launch Kit, start PPO,
  run official preprocessing/replay, or prove that the inotify blocker can be bypassed safely under current
  `8192/128` limits.
- After adding the Kit watcher config surface audit, refreshed the dependent audits and final reports. Current
  verification state: artifact manifest status `ok` with `86` hashed key artifacts and missing count `0`; master audit
  status `ok` with `128/128` artifacts passing and completion matrix counts `complete=64`, `partial=47`,
  `blocked=7`, `out_of_scope=1`; verification command coverage status `ok` with `105` commands and `5/5`
  lightweight smoke commands passing. The final report remains `goal_complete=false`.
- Added tracking import gate audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_import_gate_audit.py`.
- Tracking import gate audit outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.tsv`.
- Tracking import gate result: status `ok`, `6` rows, failed `0`. It uses the IsaacLab-bound Isaac Sim Python entrypoint
  and shows plain non-Kit Python imports `isaaclab`, but `isaacsim.core.api` plus `5` official `whole_body_tracking`
  deep config/MDP modules fail with `ModuleNotFoundError: No module named 'isaacsim.core'`. The audit also verifies the
  `isaacsim.core.api` extension source exists locally and that `27` official tracking Python source files are present.
- Boundary: this is a deeper Level-B import gate, not a live Kit smoke. It does not run `SimulationApp`/`AppLauncher`,
  official `csv_to_npz.py`, replay, PPO, ONNX export, or evaluation. Deep official tracking config import remains gated
  by the live Kit extension-manager context, which is still blocked by the host inotify limit.
- After adding the tracking import gate audit, refreshed the dependent audits and final reports. Current verification
  state: artifact manifest status `ok` with `87` hashed key artifacts and missing count `0`; master audit status `ok`
  with `129/129` artifacts passing and completion matrix counts `complete=63`, `partial=48`, `blocked=7`,
  `out_of_scope=1`; verification command coverage status `ok` with `106` commands and `5/5` lightweight smoke
  commands passing. The final report remains `goal_complete=false`.
- Added tracking extension namespace probe:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_extension_namespace_probe.py`.
- Tracking extension namespace probe outputs:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json`
  and
  `/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.tsv`.
- Tracking extension namespace result: status `ok`, `4` rows, failed `0`. It appends `9` local `isaacsim.core.*`
  extension namespace paths to `isaacsim.__path__` inside a subprocess only, growing the observed namespace path length
  to `10`. That changes the core import failure from missing `isaacsim.core` to `ImportError: libarch.so: cannot open
  shared object file`; official tracking deep imports then fail on `ModuleNotFoundError: No module named
  'omni.kit.commands'`.
- Boundary: this proves the earlier failure is not only a source-tree/PYTHONPATH discovery problem. Deep official
  tracking imports still require Kit/native runtime initialization, so this probe does not launch Kit, run
  `SimulationApp`/`AppLauncher`, preprocess/replay motions, start PPO, export ONNX, or complete Level-B tracking.
- After adding the tracking extension namespace probe, refreshed the dependent audits and final reports. Current
  verification state: artifact manifest status `ok` with `88` hashed key artifacts and missing count `0`; master audit
  status `ok` with `130/130` artifacts passing and completion matrix counts `complete=63`, `partial=48`, `blocked=7`,
  `out_of_scope=1`; verification command coverage status `ok` with `107` commands and `5/5` lightweight smoke
  commands passing. The final report remains `goal_complete=false`.
- Added completion matrix status hygiene audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/completion_matrix_status_audit.py`.
- Completion matrix status hygiene outputs:
  `/mnt/infini-data/test/BeyondMimic/res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/docs/completion_matrix_status_audit/completion_matrix_status_audit.tsv`.
- Completion matrix status result: status `ok`, `123` requirement rows, `7` table header rows, invalid status count
  `0`, invalid row count `0`. Four rows that previously used scoped prose in the Status column are now normalized to
  allowed enum values while preserving the scope qualifier in Evidence. This makes master-audit completion counts
  countable instead of silently ignoring non-enum statuses.
- After adding the completion matrix status audit, refreshed the dependent audits and final reports. Current
  verification state: artifact manifest status `ok` with `89` hashed key artifacts and missing count `0`; master audit
  status `ok` with `131/131` artifacts passing and completion matrix counts `complete=67`, `partial=48`, `blocked=7`,
  `out_of_scope=1`; verification command coverage status `ok` with `108` commands and `5/5` lightweight smoke
  commands passing. The final report remains `goal_complete=false`.
- Extended the clean-room evaluation metric API under
  `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/evaluation/metrics.py`.
- New metric helpers cover goal-level scalar metric accounting without requiring rollout fabrication: `success_rate`,
  `fall_rate`, `velocity_tracking_error`, and `split_metric_summary`, in addition to existing action MSE, position
  tracking error, and survival-rate helpers.
- Updated package API and core math tests:
  `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_reimpl_package_api.py` now runs `8` rows with goal metric
  contracts, and `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py` now runs `23` rows including
  success/fall/velocity-tracking metrics. Both write refreshed machine-readable outputs under
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/` and
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/`.
- Updated reimplementation package audit:
  `/mnt/infini-data/test/BeyondMimic/res/code/reimpl_package_audit/reimpl_package_audit.json` now verifies `29` expected API
  symbols across the lightweight package.
- Boundary: these are metric formula/API contracts only. They do not create rollout episodes, paper success/fall
  rates, trained checkpoints, Fig. 5/Fig. 6 logs, IsaacLab/Kit evaluation, ROS/TensorRT deployment, or hardware
  evidence.
- Updated Section 12 evaluation metric coverage to link the new formula APIs without overclaiming rollout results:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/evaluation_metrics_coverage_audit.py`.
- Evaluation metric coverage result remains status `ok` with all `44` Section 12 metrics mapped and zero missing
  evidence paths. Status counts are now `released_data=8`, `debug_only=14`, `debug_or_released=5`,
  `debug_or_paper_budget=2`, `formula_api_only=6`, `partial=2`, and `blocked_or_missing=7`. The six formula-only rows
  are motion-tracking success/fall rate, VAE tracking/fall metrics, and diffusion-guidance velocity/fall metrics; they
  link `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json`, while still recording that
  paper-level rollout/evaluation evidence is missing or blocked.
- Refreshed the consolidated metrics catalog after adding formula-API metric evidence:
  `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.json`,
  `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.csv`, and
  `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.md`.
- Metrics catalog result: status `ok`, `23` metric-bearing sources, `456` indexed rows, level counts
  `released_data=4`, `formula_api=2`, `debug_only=11`, `comparison=1`, `coverage_audit=3`, and
  `blocked_boundary=2`. The new formula-API sources are the package API and core math test outputs for
  success/fall/velocity metric contracts. The catalog still preserves the boundary that formula/API/debug metrics are
  not paper-level rollout or deployment results.
- Updated final deliverables audit to include formula-API metric evidence in the Section 16 experiment-metrics
  deliverable:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_deliverables_audit.py`.
- Final deliverables result remains status `ok`, `38` rows, zero missing evidence rows. The experiment `metrics`
  deliverable now lists `/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.json`,
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json`,
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json`,
  `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv`, and
  `/mnt/infini-data/test/BeyondMimic/res/results_claims_audit/results_claims_audit.json`. It records `321` metric files
  while preserving that paper-scale training/evaluation metrics are missing.
- Updated paper-vs-reproduction checkpoint rows:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`.
- The four required goal checkpoint rows remain present and `not reproduced`: walking velocity tracking error
  `12.14%`, running velocity tracking error `13.65%`, direct diffusion cartwheel success `5%`, and latent diffusion
  cartwheel success `95%`. Their `run_id` fields now point to formula/API evidence
  (`evaluation_metrics_coverage_audit.json`, `reimpl_package_api_tests.json`, `core_math_unit_tests.json`, and for
  cartwheel rows also `level_c_fig5_fig6_feasibility_audit.json`) so the comparison table separates metric-formula
  reproducibility from missing paper-level rollout results.
- Updated the Results claims audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/results_claims_audit.py`.
- The `velocity_tracking_errors` and `latent_diffusion_cartwheel_ablation` rows now link the same formula/API metric
  evidence used by the comparison table: `/mnt/infini-data/test/BeyondMimic/res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json`,
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json`, and
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json`; the cartwheel row also
  links `/mnt/infini-data/test/BeyondMimic/res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json`.
- Results claims audit status remains `ok` with `14` rows, `0` failures, `2` formula/API-linked paper metric claim
  rows, and `2` paper metric claim rows still marked `not_publicly_reproducible_currently`. New checks confirm the
  paper metric claims have formula/API evidence and that formula/API evidence is not overclaimed as paper-level
  rollout/evaluation results.
- Updated the fine-grained goal requirement matrix:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_requirement_matrix_audit.py`.
- The Section 12 `evaluation_metrics` requirement now links the formula/API metric contract artifacts
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json` and
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json`, references the required
  goal-checkpoint values `12.14%/13.65%` and `5%/95%`, and keeps the row `partial` because paper-level rollout
  metrics, trained checkpoints, and live evaluation logs are still missing or blocked.
- Goal requirement matrix result remains status `ok`: `28` requirement rows, zero missing evidence rows, status counts
  `complete=11`, `partial=15`, `blocked=1`, `out_of_scope=1`. New checks confirm the evaluation-metrics requirement
  links formula/API contracts, preserves Results-claim formula links, and does not claim goal completion.
- Refreshed and strengthened the paper formula/code trace audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_formula_code_trace_audit.py`.
- The trace audit now records current downstream package/test counts instead of stale values: `23` core math test rows,
  `29` reimplementation API symbols, and `8` package API test rows. New checks require metric helper symbols,
  goal-metric API contracts, and core metric tests to remain present.
- Paper formula/code trace result remains status `ok` with `11` trace rows, zero missing evidence rows, `8` LaTeX
  equations, `14` experiment settings, `58` paper table-value rows, and zero table mismatches. It still only proves
  formula/source-to-code traceability and does not claim full training, deployment, Fig. 5/Fig. 6, or video
  reproduction.
- Strengthened the core math checklist coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/core_test_coverage_audit.py`.
- Core test coverage result remains status `ok` with `20` explicit checklist requirements, zero missing rows, `23`
  core math unit-test rows, and `38` covered goal items. New coverage checks confirm the goal-metric items
  `success_rate`, `fall_rate`, `velocity_tracking_error`, and `evaluation_metrics` have passed pure-NumPy test
  evidence, while still not claiming training/deployment completion.
- Updated the final deliverables audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_deliverables_audit.py`.
- The code `tests` deliverable now records `/mnt/infini-data/test/BeyondMimic/res/tests/core_test_coverage_audit/core_test_coverage_audit.json`
  in addition to raw core math and package API test outputs. The experiment `metrics` deliverable now records
  `/mnt/infini-data/test/BeyondMimic/res/tests/core_test_coverage_audit/core_test_coverage_audit.json` and
  `/mnt/infini-data/test/BeyondMimic/res/paper_formula_code_trace/paper_formula_code_trace_audit.json` alongside the
  metrics catalog, package API tests, core math tests, comparison table, and Results-claims audit.
- Final deliverables result remains status `ok` with `38` rows and zero missing listed evidence paths. New checks
  confirm the tests deliverable records core coverage and the metrics deliverable records formula/API, core-coverage,
  and paper-trace evidence while preserving that checkpoints, videos, and paper-scale training/evaluation metrics are
  missing.
- Updated the required-artifact absence audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`.
- The audit now has `17` rows and explicitly excludes 2 bundled official/reference documentation GIFs under
  `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/docs/source/_static/` from
  reproduced BeyondMimic video counts. Status counts are now `missing_required_artifact=12`,
  `debug_only_not_required_artifact=2`, and `present_but_not_required_artifact=3`.
- Required-artifact checks remain status `ok`: zero missing evidence rows, zero local reproduced model checkpoints,
  zero local reproduced videos, diagnostic/debug checkpoints excluded, local debug guidance GIF excluded, official
  reference documentation GIFs excluded, and no BeyondMimic-named model/video found in the downloaded reference
  assets.
- Strengthened the Level-B motion preprocessing contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/motion_preprocessing_contract_audit.py`.
- The audit still verifies the official `csv_to_npz.py` producer, replay/training `MotionLoader` consumer, local
  validator, 40 G1 CSV files, 36-column input layout, 29 joints, and required motion keys. It now additionally checks
  that `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_tracking_local_smoke.sh` has valid shell syntax and is
  executable, and that `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/manifest.tsv`,
  `csv_to_npz_local.py`, `replay_npz_local.py`, `rsl_rl/train_local.py`, and `rsl_rl/cli_args.py` exist with executable
  generated scripts.
- Motion preprocessing contract result remains status `ok`, with local runner hygiene checks all true. This improves
  Level-B smoke readiness evidence but still does not execute IsaacLab/Kit `csv_to_npz.py`, rendered replay, PPO
  smoke training, or paper tracking rollout metrics while the inotify gate is unresolved.
- Hardened the progress report audit writer:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/progress_report_audit.py`.
- The audit now writes `progress_report_audit.json` and `progress_report_audit.tsv` via temporary files followed by
  atomic replacement, and records `atomic_write_used=true` in machine-readable checks. This addresses the observed
  race where a concurrent final-report refresh could read the progress audit JSON while it was being overwritten.
- Progress report audit remains status `ok` with `38` rows, all 21 required fields present/nonempty, all 17 key
  progress markers present, incomplete checkpoint/video boundaries recorded, and no leftover `.tmp` output files.
- Updated the artifact manifest:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- The manifest now hashes the Level-B local tracking smoke runner and generated local script chain:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_tracking_local_smoke.sh`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/prepare_tracking_local_smoke.py`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/manifest.tsv`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/replay_npz_local.py`, and
  `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py`.
- Artifact manifest result remains status `ok`, now with `95` hashed artifacts, missing count `0`, and
  `tracking_local_smoke_scripts_hashed=true`. This strengthens hash traceability for the prepared Level-B smoke
  entrypoints without executing the Kit-gated tracking pipeline.
- Hardened the artifact manifest writer:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- The manifest now writes `artifact_manifest.json` and `artifact_manifest.tsv` via temporary files followed by atomic
  replacement, and records `atomic_write_used=true` in machine-readable checks. This gives the master/final-report
  refresh chain the same half-write protection added to the progress audit.
- Artifact manifest remains status `ok` with `95` hashed artifacts, missing count `0`, tracking local smoke scripts
  hashed, and no leftover `.tmp` output files.
- Hardened the final reproduction report writer:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- The final report now writes `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`,
  `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`, and
  `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md` via temporary files followed by atomic
  replacement. It records `atomic_write_used=true` and `does_not_claim_goal_complete=true` in the report JSON checks.
- Hardened the final report requirement audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_report_requirement_audit.py`.
- The requirement audit now checks `summary_atomic_write_used=true`, writes its own JSON/TSV outputs atomically, and
  remains limited to reporting coverage for the 12 explicit final-report requirements rather than paper-level
  training, deployment, Fig. 5/Fig. 6, or video completion.
- Expanded and hardened the verification command coverage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_coverage_audit.py`.
- The bounded lightweight smoke subset now runs `10` final-report commands instead of `5`: progress audit, final-report
  requirement audit, required-artifact absence, evaluation-metrics coverage, ablation coverage, goal traceability,
  goal directive index, goal requirement matrix, final deliverables, and paper formula/code trace. The audit now maps
  all ten outputs and writes its own JSON/TSV outputs atomically with `atomic_write_used=true`.
- This strengthens current repeatability evidence for goal.md indexing/reporting/formula-trace coverage, while still
  intentionally not executing heavy IsaacLab/Kit, ROS 2, full training, TensorRT, Fig. 5/Fig. 6, or real-robot gates.
- Added final-report command syntax audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_syntax_audit.py`.
- The syntax audit reads `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`, extracts every
  unique Python script referenced by the final-report verification commands, compiles them with `py_compile`, and writes
  `/mnt/infini-data/test/BeyondMimic/res/verification_command_syntax/verification_command_syntax_audit.json` plus TSV
  atomically. Current run compiles `109` unique Python scripts with failed count `0`.
- This adds broad static repeatability evidence for the command surface while still not executing heavy commands,
  training, deployment, Fig. 5/Fig. 6, or robot gates.
- Added final-report command script hash manifest:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_script_manifest.py`.
- The manifest reads the final report command list, extracts the same unique Python script set as the syntax audit,
  records each script's relative path, byte size, and SHA256 hash, and writes
  `/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.json`
  plus TSV atomically. It cross-checks its script count against the syntax audit and does not execute the commands.
- Added and executed Level-B local tracking smoke preflight:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_local_smoke_preflight.py`.
- The preflight executes only non-Kit steps: regenerates local non-WandB tracking scripts, checks
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_tracking_local_smoke.sh` with `bash -n`, compiles the generated
  `csv_to_npz_local.py`, `replay_npz_local.py`, `rsl_rl/train_local.py`, and `rsl_rl/cli_args.py`, and validates all
  three debug `motion.npz` fixtures with `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_motion_npz_contract.py`.
  Result: `/mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json` status
  `ok`, `6/6` steps passed, fixture count `3`, and `does_not_launch_kit_or_training=true`.
- Boundary: this is stronger Level-B local readiness evidence, but it still does not execute the official IsaacLab/Kit
  `csv_to_npz.py`, rendered replay, PPO tracking smoke, rollout metrics, or paper tracking result reproduction.
- Added and executed unified clean-room reimplementation test suite:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_reimpl_test_suite.py`.
- The suite runs five pure-Python/NumPy gates in order: core math unit tests, reimplementation package API tests,
  package import/symbol audit, runtime integration audit, and core-test coverage audit. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_test_suite/reimpl_test_suite.json` status `ok`, `5/5` steps passed,
  core math rows `23`, API rows `8`, package symbols `29`, runtime windows `84`, token shape `[84,21,131]`, and
  coverage required count `20`.
- Boundary: this is executable clean-room formula/API/runtime evidence. It still does not execute IsaacLab/Kit
  rollouts, ROS 2 deployment, TensorRT, long PPO/VAE/diffusion training, Fig. 5/Fig. 6 evaluation, or real Unitree G1
  hardware.
- Added and executed unified Level-C debug suite:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_debug_suite.py`.
- The suite executes ten lightweight VAE/diffusion/guidance gates: VAE accumulation, VAE latent probe, diffusion
  equation audit, reverse denoising, paper-state mask/reverse, guidance formula gradients, guided reverse loop,
  guidance scale sweep, receding-horizon decoder, and diffusion-to-VAE action interface. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/debug_suite/level_c_debug_suite.json` status `ok`, `10/10` steps passed,
  reverse MSE `0.11735366650108521 -> 5.253923735449295e-10`, paper-state reverse final MSE
  `5.232890366443532e-10`, guided final MSE `2.377871825333812e-09`, validation/test current-action MSE
  `0.00911132119759807/0.0039371201747736865`, and current decoder index `4`.
- Boundary: this is still a debug mechanics suite. It does not train paper-scale VAE/diffusion policies, produce
  official checkpoints, execute TensorRT deployment, reproduce Fig. 5/Fig. 6, or run Unitree G1 hardware.
- Added and executed unified Level-A released-data/table suite:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_a_released_data_suite.py`.
- The suite reruns six released-data and paper-table gates: paper table values, skill-success table data, released
  panel mapping, released metrics summary, released statistical audit, and paper-vs-reproduction comparison. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_a/released_data_suite/level_a_released_data_suite.json` status `ok`,
  `6/6` steps passed, paper table rows `58`, mismatch rows `0`, released panel rows/failures `15/0`, source CSV count
  `10`, ablation rows `30`, GRF CI rows `12`, and IMU CI rows `11`.
- Boundary: this is Level-A released-data/table audit evidence only. It does not execute paper-scale training,
  closed-loop IsaacLab/Kit rollouts, ROS 2 deployment, TensorRT deployment, Fig. 5/Fig. 6 reproduction, or real Unitree
  G1 hardware.
- Added and executed official `whole_body_tracking` static source contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_source_contract_audit.py`.
- The audit reads the official G1 source, tracking env config, PPO config, command/reward MDP files, and G1 URDF without
  launching Kit. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json`
  status `ok`; it verifies 14 target bodies with `torso_link` anchor, 50 Hz control config, 8 policy observation terms,
  10 critic observation terms, 9 reward terms, 4 events, 4 terminations, PPO key values, G1 action-scale assignment,
  5 actuator groups, complete URDF mesh coverage, and zero uncovered non-fixed URDF joints under the official action
  regexes.
- Boundary: this strengthens Level-B official-code evidence, but it is still static/non-Kit evidence. It does not run
  IsaacLab closed-loop rollouts, PPO training, policy export, TensorRT, ROS 2 deployment, or real Unitree G1 hardware.
- Added and executed official G1 actuator/action-scale numeric audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_action_scale_audit.py`.
- The audit expands the official `robots/g1.py` actuator settings over the local G1 URDF without importing IsaacLab/Kit.
  Result: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
  status `ok` with `29` per-joint rows and group counts `legs=8`, `feet=4`, `waist=2`, `waist_yaw=1`, `arms=14`.
  It writes the per-joint TSV at
  `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.tsv`, verifies all
  non-fixed URDF joints match exactly one official actuator regex, and checks the official action-scale formula
  `0.25 * effort / stiffness`, 10 Hz natural-frequency formula, positive armature/stiffness/action-scale values, and
  no Kit/training launch.
- Boundary: this is a static official-code numeric contract for G1 tracking. It does not execute IsaacLab rollouts,
  train PPO, export deployment models, run ROS 2/TensorRT, or touch Unitree G1 hardware.
- Added and executed official tracking reward formula audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_reward_formula_audit.py`.
- The audit reads official `tracking_env_cfg.py` and `mdp/rewards.py` without importing IsaacLab/Kit, then runs a NumPy
  equivalent of the six motion-imitation exponential error rewards. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/reward_formula_audit/tracking_reward_formula_audit.json` status `ok`,
  with 9 official reward terms, 6 motion `exp(-error/std^2)` terms, 3 regularizers, 30 numeric scan rows, official
  weight/std checks, reward-at-zero and weighted-at-zero checks, monotone non-increasing reward curves, and bounded
  `[0,1]` unweighted rewards. It also writes scan and summary TSVs under
  `/mnt/infini-data/test/BeyondMimic/res/tracking/reward_formula_audit/`.
- Boundary: this is formula-level official-code evidence for the tracking reward, not IsaacLab rollout, PPO training,
  policy evaluation, TensorRT/ROS deployment, or hardware evidence.
- Added and executed official tracking observation/action schema audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_observation_action_schema_audit.py`.
- The audit reads official `tracking_env_cfg.py`, `mdp/observations.py`, `mdp/commands.py`, the G1 flat env target-body
  list, the G1 action-scale audit, and the local debug `motion.npz` fixtures without importing IsaacLab/Kit. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json`
  status `ok`, actor observation dimension `160`, critic observation dimension `286`, action dimension `29`, 8 policy
  terms, 10 critic terms, 6 noisy policy terms, 0 noisy critic terms, 14 target bodies, 3 local fixtures, 50 Hz,
  29-joint fixture alignment, 40-body URDF fixture alignment, Rot6D/source frame-transform checks, and finite/unit-quat
  fixture checks. It also writes the per-term TSV
  `/mnt/infini-data/test/BeyondMimic/res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.tsv`.
- Boundary: this is official-code schema plus local fixture contract evidence. It does not execute IsaacLab rollouts,
  train PPO, evaluate a policy, export TensorRT/ONNX deployment artifacts, run ROS 2, or use real hardware.
- Added and executed official tracking randomization/termination audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_randomization_termination_audit.py`.
- The audit reads official `tracking_env_cfg.py`, `mdp/events.py`, and `mdp/terminations.py` without importing
  IsaacLab/Kit. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json`
  status `ok`, with 4 event terms, 13 event range rows, 3 startup event types, 1 interval push event, 4 termination
  terms, official friction/restitution ranges, joint-default offset range, torso COM ranges, root velocity push ranges,
  anchor/EE/orientation thresholds, EE body names, and strict-greater-than termination boundary probes. It also writes
  `/mnt/infini-data/test/BeyondMimic/res/tracking/randomization_termination_audit/tracking_randomization_events.tsv` and
  `/mnt/infini-data/test/BeyondMimic/res/tracking/randomization_termination_audit/tracking_termination_terms.tsv`.
- Boundary: this is official-code config plus numeric boundary evidence. It does not execute IsaacLab rollouts, train
  PPO, evaluate a policy, export TensorRT/ONNX deployment artifacts, run ROS 2, or use real hardware.
- Added and executed unified Level-B non-Kit tracking suite:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_b_tracking_nonkit_suite.py`.
- The suite reruns 13 non-Kit tracking gates: official source contract, G1 action-scale, reward formula,
  observation/action schema, randomization/termination, motion preprocessing contract, debug `motion.npz` fixture build,
  local preflight, adaptive sampling discrepancy, ONNX export contract, debug motion-policy ONNX export, debug
  motion-policy ONNX reference inference, and MuJoCo/ROS launch contract. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json` status `ok`, `13/13`
  steps passed, target bodies `14`, G1 action-scale rows `29`, reward motion terms `6`, policy/critic dimensions
  `160/286`, randomization event terms `4`, fixture count `3`, local preflight steps `6`, and adaptive-sampling
  discrepancy L1 difference `1.0730253353204173`; the debug ONNX is
  `/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx`.
- Boundary: this is a reproducibility suite for Level-B non-Kit static/schema/fixture evidence. It does not launch
  IsaacLab/Kit, run official `csv_to_npz.py`/rendered replay/PPO rollout, export a trained policy, execute ROS 2, or use
  real hardware.
- Added and executed unified Level-C extended debug suite:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_extended_debug_suite.py`.
- The suite reruns 10 nontraining Level-C gates: state-latent dataset consistency, debug VAE latent artifact,
  VAE motion-split heldout eval, VAE-latent diffusion overfit, VAE-latent heldout eval, VAE-latent heldout multiseed,
  diffusion-to-VAE action smoke, diffusion-to-VAE action multiseed, diffusion-to-VAE action smoothness, and
  small-dataset heldout multiseed. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/extended_debug_suite/level_c_extended_debug_suite.json` status `ok`,
  `10/10` steps passed, state-latent rows `84`, dimensions `99/32/131/29`, VAE debug latent abs mean
  `0.1456600978669161`, VAE heldout test action MSE `0.008726497973125745`, VAE-latent test prediction loss
  `0.007374265934471514`, VAE-latent multiseed test reduction mean `0.8925315111873378`, action multiseed test
  current MSE mean `0.003998468510847306`, action smoothness test penalty `0.01579856790234264`, and small-dataset
  heldout test reduction mean `0.4483171668248292`.
- Boundary: this is a reproducibility suite for Level-C debug evidence. It does not create true VAE rollout latents,
  train paper-scale VAE/diffusion models, produce official checkpoints, reproduce Fig. 5/Fig. 6 rollouts, deploy
  TensorRT, or run hardware.
- Added and executed official deployment controller semantics audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_deployment_controller_semantics_audit.py`.
- The audit reads 14 files from the official `motion_tracking_controller` working copy without running ROS, MuJoCo,
  rosbag, network interfaces, or robot commands. Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/deployment_controller_semantics_audit/tracking_deployment_controller_semantics_audit.json`
  status `ok`, 18/18 checks passed, 14 source files hashed, 500 Hz controller manager, 50 Hz walking controller, 29
  standby joints, motion command dimension `58`, and motion-observation formula `9 + 9 * body_names_count` (`135` for
  14 target bodies). It verifies package/plugin identity, core controller dependencies, Unitree description and
  robot-state-publisher launch references, sim-vs-real controller startup modes, policy/W&B/start-step wiring, ONNX
  `time_step` plus reference-output metadata semantics, motion command/observation aliases, yaw-anchor local-frame
  alignment, real-robot risk/remote-switch documentation, and the host ROS 2 Jazzy/Noble runtime gate.
- Boundary: this is static source-level deployment semantics evidence. It does not build with `colcon`, launch MuJoCo,
  export or execute a trained BeyondMimic ONNX, collect rosbag data, run TensorRT diffusion deployment, or control a real
  Unitree G1.
- Added and executed unified DAgger-to-VAE debug pipeline audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_dagger_vae_pipeline_audit.py`.
- The audit stitches together 9 existing local debug stages: DAgger schema, 3-iteration DAgger aggregation smoke, VAE
  contract, VAE gradient accumulation, VAE checkpoint roundtrip, tiny-VAE nonzero latent artifact, VAE motion-split
  heldout eval, state-latent consistency, and receding-horizon current-action decoding. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json` status
  `ok`, 9/9 stages passed, 288 synthetic teacher queries, 288 aggregated samples, 35 VAE contract rows, VAE parameter
  count `5697117`, effective VAE batch size `30`, checkpoint size `68387282` bytes, 84 state-latent rows, dimensions
  `99/32/131/29`, debug latent abs mean `0.1456600978669161`, VAE motion-split test action MSE
  `0.008726497973125745`, and receding-horizon current-action MSE mean/max
  `0.0001923113866260608/0.0008571597683028913`.
- Boundary: this proves local debug DAgger-to-VAE pipeline consistency only. It does not collect true Isaac DAgger
  rollouts, train a paper VAE checkpoint, evaluate VAE rollout survival/stability, reproduce Fig. 5/Fig. 6, deploy
  TensorRT, or run hardware.
- Added and executed guidance visual deliverables audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_visual_deliverables_audit.py`.
- The audit inventories the existing debug guidance PNG/SVG/PDF/GIF/NPZ/TSV files with size and SHA256 hashes, links the
  five task-level debug primary metrics and 40-row scale sweep, and cross-checks Fig. 5/Fig. 6 plus video requirements.
  Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_visual_deliverables_audit/level_c_guidance_visual_deliverables_audit.json`
  status `ok`, 6 visual files, total visual size `239631` bytes, 5 guidance tasks, 5 improved primary metrics, mean
  primary delta `0.08454003905275727`, 40 scale-sweep rows, 6 Fig. 5/Fig. 6 panels recorded as blocked, and 6
  success/failure video requirements recorded as blocked.
- Boundary: this is a debug qualitative deliverable audit. It does not produce paper Fig. 5/Fig. 6, success/failure
  videos, trained diffusion rollouts, closed-loop sim logs, TensorRT deployment, or hardware results.
- Added and executed debug-only motion-policy ONNX contract fixture:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_motion_policy_onnx_contract_fixture.py`.
- Because the current Python environments lack the `onnx` package and no trained BeyondMimic tracking checkpoint exists,
  the script does not write a real `.onnx`. It materializes the official exporter/C++ consumer contract as JSON/TSV/NPZ:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json`
  status `ok`, with 2 required inputs (`obs`, `time_step`), 7 required outputs, 11 metadata fields, observation
  dimension `160`, action/joint dimension `29`, 14 target bodies, unit body quaternions, and official G1 action-scale
  metadata. The NPZ fixture is
  `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_policy_onnx_contract_fixture/debug_motion_policy_onnx_contract_fixture.npz`
  with SHA256 `a271b026cf85196258ea0bc368288a3ad80a68ef5457020ff74919fc508c2fa9`.
- Boundary: this is an ONNX contract fixture only. It is not a trained BeyondMimic motion policy ONNX, ONNX Runtime
  inference, IsaacLab export, ROS/MuJoCo execution, rollout metric, or deployment evidence.
- Added and executed full run deliverable gap audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/full_run_deliverable_gap_audit.py`.
- The audit checks all current `res/runs/*` directories against the `goal.md` run contract and final experiment
  deliverables. Result:
  `/mnt/infini-data/test/BeyondMimic/res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json`
  status `ok`, 2 run dirs, 2 schema-complete dirs, 0 valid training runs, 2 diagnostic/debug runs, 1 nonempty
  checkpoint dir, 0 nonempty figures dirs, 0 nonempty videos dirs, and 0 runs with training endpoint runtime metrics.
- Boundary: this is a gap audit for run deliverables. It does not create full training logs, paper checkpoints, figures,
  videos, or evaluation metrics.
- Added and executed state-latent training dataset contract audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_latent_training_dataset_contract_audit.py`.
- The audit checks whether the current Level-C state-latent artifacts can be treated as the paper's trainable diffusion
  dataset. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_training_dataset_contract_audit/level_c_state_latent_training_dataset_contract_audit.json`
  status `ok`, 12 contract rows, 3 debug-satisfied rows, 9 missing/debug-only rows, 84 debug NPZ samples, state shape
  `21x99`, latent shape `21x32`, and 0 failed checks.
- Boundary: this is a contract/gap audit. It does not create trained VAE rollout latents, true DAgger teacher/student
  rollouts, live OU-perturbed accepted episodes, paper-scale state-latent training data, or diffusion checkpoints.
- Added and executed bounded debug diffusion training run:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_bounded_debug_diffusion_training_run.py`.
- The run uses `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python`, executes 3 CPU optimizer steps on
  paper-state `99` + tiny-VAE debug latent `32` tokens, and writes a goal-schema run directory:
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000`.
- Outputs include debug checkpoint
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/checkpoint/debug_bounded_diffusion_checkpoint.pt`,
  loss figure
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/figures/debug_training_loss.png`,
  metrics/status files, and audit
  `/mnt/infini-data/test/BeyondMimic/res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json`.
  Result: status `ok`, 3 debug steps, token dim `131`, checkpoint size `305413425` bytes, loss figure size `3239`
  bytes, and 0 failed checks.
- Boundary: this is bounded debug optimization only. The run is `SUCCESS` for its own debug scope but explicitly
  `is_training_run=false` and `paper_level=false`; it is not paper-scale diffusion training, a paper checkpoint,
  rollout evaluation, Fig. 5/Fig. 6 evidence, TensorRT deployment, or hardware evidence.
- Added and executed bounded debug diffusion checkpoint evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_bounded_debug_diffusion_checkpoint_eval.py`.
- The audit reloads the 305 MB bounded debug checkpoint, verifies its SHA256 against the training audit, and evaluates
  fixed-noise clean-trajectory MSE on train/validation/test motion-level splits. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/bounded_debug_diffusion_checkpoint_eval/level_c_bounded_debug_diffusion_checkpoint_eval.json`
  status `ok`, 3 split rows, 28 windows per split, trained checkpoint MSE `1.4378801584243774` train,
  `1.4262079000473022` validation, and `1.4602681398391724` test.
- Boundary: this verifies checkpoint loading and per-split offline MSE accounting only. The debug checkpoint is only
  microscopically different from initialization and far worse than the noisy-identity baseline, so it is not a useful
  trained diffusion model, closed-loop controller, paper metric, Fig. 5/Fig. 6 result, TensorRT artifact, or hardware
  result.
- Added and executed bounded debug diffusion action evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_bounded_debug_diffusion_action_eval.py`.
- The audit routes bounded-checkpoint token predictions through the same train-split action-decoder surrogate used by
  the diffusion-to-VAE action smoke. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/bounded_debug_diffusion_action_eval/level_c_bounded_debug_diffusion_action_eval.json`
  status `ok`, 3 split rows, 28 windows per split, checkpoint current-action MSE `0.5037098564088699` train,
  `0.45596258052197547` validation, and `0.489066042088447` test.
- Boundary: this is offline token-to-action accounting only. The bounded checkpoint performs much worse than the
  noisy-token action baseline and clean-token reconstruction, so it is not a usable policy, closed-loop controller,
  paper action metric, Fig. 5/Fig. 6 evidence, TensorRT artifact, or hardware result.
- Added and executed resource-adjusted tiny diffusion training run:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_training_run.py`.
- The run trains a 143,491-parameter tiny denoiser for 180 epochs on the debug train split and evaluates token/action
  MSE on motion-level train/validation/test splits. Outputs include run directory
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500`,
  checkpoint
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/checkpoint/tiny_resource_adjusted_denoiser.pt`,
  two figures, NPZ, TSV, and audit
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_training_run/level_c_resource_adjusted_tiny_diffusion_training_run.json`.
- Result: status `ok`, train/validation/test token MSE `0.0007233637408933863`/`0.007219147213891228`/`0.006055245326838027`,
  validation/test current-action MSE `0.009416264484913103`/`0.008886663625808453`, checkpoint size `1730542`
  bytes, and 0 failed checks.
- Boundary: this is real resource-adjusted debug training, but it is not the paper Transformer, not true VAE rollout
  data, not paper-scale training, not a closed-loop controller, not a valid full paper training run, not Fig. 5/Fig. 6
  evidence, not TensorRT, and not hardware.
- Added and executed resource-adjusted tiny diffusion multi-seed audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.py`.
- The audit repeats the tiny denoiser train/eval path over 3 seeds for 80 epochs each and writes JSON/TSV/NPZ outputs:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json`.
- Result: status `ok`; validation/test token MSE mean/std
  `0.00685544651330446`/`6.214080227912959e-05` and
  `0.0064945946151577505`/`0.00011200788695217837`; validation/test current-action MSE mean/std
  `0.008245099355464781`/`0.0007448994513621566` and
  `0.009321537230231582`/`0.0005786748631468493`; validation/test token reduction vs noisy mean/std
  `0.8978161670947907`/`0.002111188888480484` and
  `0.9032752972188615`/`0.002777241031187853`; 0 failed checks.
- Boundary: this is resource-adjusted debug multi-seed stability evidence only. It is not paper multi-seed diffusion
  training/evaluation, not the paper Transformer, not true VAE rollout data, not closed-loop rollout, not Fig. 5/Fig. 6
  evidence, not TensorRT, and not hardware.
- Added and executed resource-adjusted tiny diffusion checkpoint reload/eval:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.py`.
- The audit reloads
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/checkpoint/tiny_resource_adjusted_denoiser.pt`,
  verifies its SHA256 against the training JSON, rebuilds the same debug state-latent/action eval inputs, and writes
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json`.
- Result: status `ok`, parameter count `143491`, max absolute token/action MSE delta vs source eval `0.0`/`0.0`,
  and held-out validation/test predictions remain better than noisy baselines.
- Boundary: this proves reloadable debug checkpoint accounting only. It is not the paper diffusion Transformer
  checkpoint, not TensorRT, not closed-loop control, not Fig. 5/Fig. 6, and not hardware.
- Added and executed resource-adjusted tiny diffusion ONNX export/inference:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.py`.
- The audit reloads the same tiny denoiser checkpoint, exports ONNX
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/resource_adjusted_tiny_denoiser_debug.onnx`,
  validates it with `onnx.checker`, loads it with `onnx.reference.ReferenceEvaluator`, and writes
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json`.
- Result: status `ok`, parameter count `143491`, ONNX/PyTorch max absolute output difference
  `1.7881393432617188e-07`, token shape `[1, 21, 131]`, and debug ONNX size recorded in the JSON.
- Boundary: this links debug checkpoint -> ONNX graph -> graph inference. It is not the paper diffusion Transformer
  checkpoint, not TensorRT/asynchronous deployment, not closed-loop guidance/control, not Fig. 5/Fig. 6, and not
  hardware.
- Added and executed resource-adjusted tiny diffusion latency audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_latency_audit.py`.
- The audit times the same validation-window fixture for 100 measured CPU iterations after warmup in PyTorch and
  `onnx.reference.ReferenceEvaluator`, and writes
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.json`.
- Result: status `ok`; PyTorch p95 latency `0.17619896680116653` ms; ONNX reference p95 latency
  `0.5360735580325127` ms; both are under the paper's 20 ms diffusion budget for this tiny debug graph on this host.
- Boundary: this is a resource-adjusted debug latency measurement only. It is not TensorRT, not the paper Transformer
  checkpoint, not the RTX 4060 Mobile Mini PC latency claim, not asynchronous deployment, not closed-loop control, and
  not Fig. 5/Fig. 6.
- Added and executed unified resource-adjusted tiny diffusion suite:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_resource_adjusted_tiny_diffusion_suite.py`.
- The suite reruns 6 resource-adjusted debug gates: tiny denoiser training, multi-seed stability, checkpoint
  reload/eval, ONNX export/inference, latency audit, and video preview. Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_suite/level_c_resource_adjusted_tiny_diffusion_suite.json`
  status `ok`, `6/6` steps passed, parameter count `143491`, validation/test token MSE
  `0.007219147213891228`/`0.006055245326838027`, ONNX/PyTorch max error `1.7881393432617188e-07`, ONNX
  reference p95 latency `0.5516432225704193` ms, and 2 debug previews.
- Boundary: this is a unified resource-adjusted debug suite. It is not the paper Transformer, not true VAE rollout
  data, not paper-scale diffusion training, not an official checkpoint, not TensorRT, not closed-loop Fig. 5/Fig. 6
  rollout, and not hardware evidence.
- Added and executed visual media inventory audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/visual_media_inventory_audit.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/visual_media_inventory/visual_media_inventory_audit.json` status `ok`, 80 local
  visual media rows, kind counts `{"gif": 3, "pdf": 24, "png": 29, "svg": 24}`, and category counts separating 60
  released-data figures from debug guidance/augmentation/tracking/run/tiny-diffusion preview media.
- Boundary: this inventory confirms all current visual media are hashed and nonempty, and explicitly records missing
  paper-required videos for tracking rollouts, Fig. 5, Fig. 6, and real-robot success/failure. It does not create or
  claim closed-loop simulation, paper Fig. 5/Fig. 6, or hardware videos.
- Added and executed bounded official tracking train-entry retry audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_train_entry_retry_audit.py`.
- The audit runs a 90-second timeout-protected attempt of the locally patched official `whole_body_tracking`
  `train_local.py` entry with local debug `motion.npz`, `--task=Tracking-Flat-G1-v0`, `--num_envs=1`,
  `--max_iterations=1`, `--headless`, and CPU device. Logs:
  `/mnt/infini-data/test/BeyondMimic/logs/tracking_official_train_entry_retry/official_train_entry_retry.log` and tail
  `/mnt/infini-data/test/BeyondMimic/logs/tracking_official_train_entry_retry/official_train_entry_retry_tail.log`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json`
  status `ok`, return code `124`, classification `blocked_inotify`, current inotify `8192/128`, and repeated Kit
  `Failed to create change watch` / `errno=28` / `No space left on device` signatures.
- Boundary: this is direct official-entry retry evidence and confirms the blocker persists. It did not reach a PPO
  training endpoint, checkpoint, rollout, video, or paper metric.
- Added and executed failed-run retention for the official train-entry retry:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/record_official_train_entry_failed_run.py`.
- Preserved failed run:
  `/mnt/infini-data/test/BeyondMimic/res/failed_runs/phase1_official_train_entry_retry_inotify_0_20260617_174742`.
  It records run ID, error, command/config, checkpoint absence, last log, GPU status, failure reason, resolution plan,
  and `FAILED` status. Audit:
  `/mnt/infini-data/test/BeyondMimic/res/failed_runs/official_train_entry_failed_run_audit/official_train_entry_failed_run_audit.json`
  status `ok`, GPU rows `48`, and 0 failed checks.
- Boundary: this improves failure retention and reproducibility accounting. It does not resolve the Kit/inotify blocker,
  reach official PPO training success, or produce a checkpoint/video/metric.
- Added resource-adjusted tiny diffusion video preview script:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_video_preview.py`.
- The script uses the tiny denoiser NPZ to create validation/test GIF previews under
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos`
  plus poster PNGs and audit outputs under
  `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview`.
- Boundary: these are offline debug heatmap animations of clean/noisy/predicted state-latent tokens. They are not
  IsaacLab closed-loop rollouts, not robot videos, not paper Fig. 5/Fig. 6, and do not make the tiny run a valid full
  training run.
- Added and executed live inotify usage audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/inotify_live_usage_audit.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json` status `ok`,
  current user live usage `8192/8192` watches with `0` watch headroom, `6` inotify fds, and `122` instance headroom.
  The top live watcher is VS Code Server `bootstrap-fork --type=fileWatcher`, PID `1277124`, with `8164` watches.
- Added project-local VS Code watcher/search excludes at
  `/mnt/infini-data/test/BeyondMimic/.vscode/settings.json` for large generated/source/environment directories
  (`cache`, `download`, `envs`, `logs`, `res`, `tmp`, `reproduction/data`, `reproduction/third_party`,
  `reproduction/generated`, and `__pycache__`).
- Added and executed VS Code watcher exclude audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/vscode_watcher_exclude_audit.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json` status
  `ok`, required exclude missing count `0`; live watcher usage remained saturated immediately after settings write, so
  the VS Code Server/window likely needs reload before this mitigation can release already-open watches.
- Boundary: this identifies a concrete, non-Isaac live inotify consumer and records a workspace-level mitigation. It
  does not kill VS Code Server, modify sysctl, resolve Kit startup, reach official PPO training, or produce a checkpoint,
  rollout, video, or paper metric.
- Added project-local ONNX dependency for debug export in `bm_diffusion`: `onnx 1.22.0` and `protobuf 7.35.1`
  import successfully with `torch 2.5.1+cu121`. The attempted `onnxruntime` install did not complete and `pip freeze`
  was interrupted after blocking in NFS/kernel wait; no ONNX Runtime inference claim is made.
- Added and executed real debug motion-policy ONNX export:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_debug_motion_policy_onnx_export.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json`
  status `ok`; exported ONNX
  `/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx`
  is `23238` bytes with SHA256 `d7ea68b5dab0d2d667b2ab5f3e4bc2b518ff776ce8f5441764ce4e70c80f45fe`.
  The audit runs `onnx.checker`, verifies `obs`/`time_step` inputs, 7 required outputs, and all required metadata keys
  against the official exporter/C++ consumer contract.
- Added and executed debug ONNX reference-inference audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_debug_motion_policy_onnx_inference_audit.py`.
- Inference result:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_inference/tracking_debug_motion_policy_onnx_inference_audit.json`
  status `ok`; `onnx.reference.ReferenceEvaluator` loads the graph and all 7 numeric outputs match the contract fixture
  with max abs error `0.0`.
- Boundary: the ONNX is a zero-weight debug contract/export fixture. It is not produced from a trained BeyondMimic
  tracking checkpoint, not ONNX Runtime/TensorRT deployment evidence, not MuJoCo/ROS execution, not a rollout metric,
  and not a paper-level policy artifact.
- Updated final deliverables audit visual evidence linkage:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_deliverables_audit.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/final_deliverables_audit/final_deliverables_audit.json` remains status `ok` with
  `38` rows and zero missing listed evidence paths. The video/PDF/SVG/PNG rows now cite
  `/mnt/infini-data/test/BeyondMimic/res/visual_media_inventory/visual_media_inventory_audit.json`; the checks verify the
  visual inventory status, media hashes, recorded paper-required video gaps, zero mp4/mov/mkv reproduction videos, and
  PDF/SVG/PNG/GIF count consistency.
- Boundary: this improves final-deliverable traceability only. It does not create trained checkpoints, real rollout
  videos, Fig. 5/Fig. 6 reproduction outputs, TensorRT deployment, or hardware evidence.
- Added and executed paper-architecture public-data VAE/diffusion training:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_vae_diffusion_training.json`
  status `ok`; run
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000`
  completed with all `40` public G1 retargeted LAFAN1 motions, `2200` windows, `46200` tokens, 8-GPU DataParallel,
  paper-sized VAE hidden `[2048, 1024, 512]`, latent dim `32`, paper-sized diffusion Transformer embed `512`, heads
  `8`, layers `6`, denoising steps `20`, and `1000` diffusion epochs. Checkpoint:
  `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/checkpoint/lafan1_paper_arch_vae_diffusion.pt`
  is `302997114` bytes with SHA256 `e62aa14d72646ff27994fd2a18d747421d276660b662962634be045d28920aec`.
- Metrics: VAE validation/test decoded-action MSE `0.02712077833712101`/`0.032622650265693665`; diffusion
  validation/test tau MSE `0.0076224529184401035`/`0.007946033962070942`; VAE parameters `6047354`, diffusion
  parameters `19191023`, elapsed `148.18651772290468` seconds, CUDA peak memory on GPU0 `868.31298828125` MiB.
- Boundary: this is no longer a smoke/debug-only tiny model; it is real paper-architecture training on public retargeted
  motion data. It is still not exact paper reproduction because the official teacher-policy DAgger rollouts, VAE rollout
  state-latent dataset, closed-loop Isaac/robot evaluation, Fig. 5/Fig. 6 rollouts, TensorRT, and hardware evidence are
  not available or remain blocked.
- Added and executed full public-LAFAN1 paper-architecture ONNX/latency audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_onnx_latency_audit.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json`
  status `ok`. It exports the trained public-data VAE decoder to
  `/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_vae_decoder.onnx`
  and the full paper-sized diffusion denoiser to
  `/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx`.
- Metrics: VAE ONNX-vs-Torch max abs error `1.1920928955078125e-07`, diffusion ONNX-vs-Torch max abs error
  `2.251937985420227e-06`; VAE decoder CPU p95 latency `0.3471106290817261` ms, PyTorch diffusion denoiser CPU
  p95 latency `11.761460453271866` ms, ONNX ReferenceEvaluator diffusion p95 latency `56.45706132054329` ms.
  The diffusion ONNX is `76924041` bytes with SHA256
  `66a04543b4a6042bd2a981ebe8f5848c17e481c0972fd1ec74566c3fc09d1d8a`; the VAE decoder ONNX SHA256 is
  `bb0b37621b7231457d74c030bbcf06747af8e4a4673c00783198280711026c6d`.
- Boundary: this advances the full local paper-architecture checkpoint to executable ONNX and host CPU latency evidence.
  It is still not TensorRT, not asynchronous deployment, not official teacher-rollout paper training, not closed-loop
  IsaacLab/robot execution, and not Fig. 5/Fig. 6 rollout evidence.
- Added and executed full public-LAFAN1 paper-architecture offline metrics audit:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_offline_metrics_audit.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json`
  status `ok`; it evaluates the full paper-sized public-data checkpoint over all `2200` windows and writes split-wise VAE,
  diffusion reconstruction, downstream decoded-action, latent/action/trajectory smoothness, and latency-linked metrics.
- Metrics: validation/test diffusion tau MSE `0.007752980571240187`/`0.007869400084018707`; validation/test decoded
  predicted current-action MSE `0.029158219695091248`/`0.03400135412812233`; validation/test VAE KL mean
  `0.4725886881351471`/`0.5659266114234924`; validation/test predicted action second-difference mean norm
  `0.2830184996128082`/`0.2847289443016052`. The audit reuses the ONNX/latency measurements: VAE decoder CPU p95
  `0.3471106290817261` ms, PyTorch diffusion denoiser CPU p95 `11.761460453271866` ms, ONNX ReferenceEvaluator
  diffusion p95 `56.45706132054329` ms.
- Coverage update: `/mnt/infini-data/test/BeyondMimic/res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json`
  now maps `11` Section 12 VAE/diffusion metrics to `public_data_checkpoint` evidence instead of debug-only evidence.
- Boundary: this is offline checkpoint evaluation on public retargeted LAFAN1. It is not closed-loop unconditional/guided
  success, fall/collision statistics, TensorRT deployment, Fig. 5/Fig. 6 rollout evidence, or real robot execution.
- Added and executed full public-LAFAN1 paper-architecture offline guidance evaluation:
  `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py`.
- Result:
  `/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json`
  status `ok`; it applies task-cost gradients to full-checkpoint predicted clean trajectories for joystick, waypoint,
  obstacle avoidance, inpainting, and composed objectives over `5` validation windows and `7` guidance scales, producing
  `175` rows.
- Metrics: all five task best costs improve. Mean best cost deltas are joystick `0.00259857177734375`, waypoint
  `0.0004933252930641174`, obstacle avoidance `215.9341796875`, inpainting `2.0314473658800125e-08`, and composed
  objectives `215.9378173828125`. Best-row primary metrics improve for joystick `5/5`, waypoint `4/5`, obstacle
  avoidance `5/5`, inpainting `5/5`, and composed objectives `5/5` validation windows.
- Coverage update: `/mnt/infini-data/test/BeyondMimic/res/guidance_task_coverage/guidance_task_coverage_audit.json` now maps
  with/without guidance, multiple guidance weights, and quantitative metrics for five Phase 8 tasks to
  `public_data_offline_guidance*` evidence. Success/failure videos remain explicitly `blocked_missing_videos`.
- Boundary: this is offline task-cost guidance on predicted trajectories from the full public-data checkpoint. It is not
  a reverse-denoising controller, not closed-loop IsaacLab/robot rollout, not TensorRT deployment, and not Fig. 5/Fig. 6
  reproduction.

## 2026-06-19 resource-adjusted full fixture tracking task eval

阶段：Level B IsaacLab / whole_body_tracking task execution gate.
状态：完成 resource-adjusted full available fixture task eval; official replay/PPO remains incomplete.
开始时间：2026-06-18 23:34 Asia/Shanghai.
结束时间：2026-06-19 00:13 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime; device `cuda:6`.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_multi_fixture_eval_audit.py`.
官方/重新实现：official `Tracking-Flat-G1-v0` manager stack with generated resource-adjusted G1 USD and local debug fixtures; not official `csv_to_npz.py` output and not official replay/evaluation.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_001351_resource_adjusted_full_fixture_eval.md`.
配置：three local debug motion fixtures (`walk`, `run`, `jump`), one isolated Kit process per fixture, `num_envs=1`, zero-action diagnostic actions, all available `299` motion steps per fixture, generated enriched G1 USD scaffold.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_multi_fixture_eval_audit.py`.
GPU：GPU6 primary IsaacLab runtime context; diagnostic task only, not a formal two-GPU training experiment.
峰值显存：not sampled as a formal GPU experiment; observed diagnostic GPU6 context was below the formal 10GB/training threshold.
平均 GPU-Util：not recorded for this diagnostic gate.
平均功耗：not recorded for this diagnostic gate.
运行时间：first shared-Kit attempt reached walk `299/299` but timed out during run fixture setup after the configured guard; the successful isolated-fixture run completed three fixture processes in about `80.08` seconds each.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/tracking_g1_resource_adjusted_multi_fixture_eval_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/tracking_g1_resource_adjusted_multi_fixture_eval_metrics.json`; per-fixture metric JSON files under the same directory; failed shared-Kit attempt retained under `/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_g1_resource_adjusted_multi_fixture_eval_20260618T160520Z_shared_kit_timeout`.
主要指标：status `ok_resource_adjusted_multi_fixture_task_eval`; `fixture_count=3`; `total_steps=897`; action dim `29`; policy observation dim `160`; critic observation dim `286`; reward terms `9`; termination terms `4`; robot joints `29`; robot bodies `40`; walk/run/jump each reached `299/299`.
与论文一致性：uses the official tracking task manager API and G1 action/observation/reward/termination contracts, but with generated resource-adjusted USD and local debug fixtures; therefore it is task-contract evidence, not paper-level motion-tracking performance.
发现的差异：single shared Kit process can complete the walk fixture but times out when recreating the next fixture environment; isolated Kit processes per fixture avoid this teardown/recreate blocker.
失败与风险：official `csv_to_npz.py`, official `replay_npz.py`, PPO tracking training/evaluation, DAgger rollout logs, teacher policy rollout dataset, trained tracking checkpoint, and Fig. 5/Fig. 6 closed-loop videos remain missing or blocked.
下一阶段：try official motion replay/conversion again using the now-validated resource-adjusted task path as a diagnostic baseline, then attempt a bounded official tracking train/eval only if official asset/conversion blockers are resolved.

Master audit result after adding resource-adjusted full fixture tracking task eval evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted official-CSV conversion and full replay

阶段：Level B official motion preprocessing/replay blocker narrowing.
状态：完成 resource-adjusted official-CSV conversion and full replay gates; official URDF/USD converter path remains incomplete.
开始时间：2026-06-19 00:18 Asia/Shanghai.
结束时间：2026-06-19 00:30 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime; device `cuda:6`.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_csv_conversion_audit.py`; `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_csv_full_replay_audit.py`.
官方/重新实现：official downloaded G1 LAFAN CSV and official interpolation/logging schema, but generated resource-adjusted enriched USD instead of official URDF-converter output.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_003039_resource_adjusted_csv_replay.md`.
配置：input CSV `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv`, frame range `1 180`, input fps `30`, output fps `50`, generated enriched G1 USD, `num_envs=1`, full `299` converted/replayed steps.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_conversion_audit.py`; `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_full_replay_audit.py`.
GPU：GPU6 primary IsaacLab runtime context; diagnostic conversion/replay only, not a formal two-GPU training experiment.
峰值显存：not sampled as a formal GPU experiment.
平均 GPU-Util：not recorded for this diagnostic gate.
平均功耗：not recorded for this diagnostic gate.
运行时间：conversion and full replay completed without stall timeout; wrapper uses log-progress stall detection rather than a short fixed timeout.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/tracking_g1_resource_adjusted_csv_conversion_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/tracking_g1_resource_adjusted_csv_conversion_metrics.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/walk1_subject1_frames_1_180_resource_adjusted_motion_contract.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_full_replay/tracking_g1_resource_adjusted_csv_full_replay_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_full_replay/walk1_subject1_frames_1_180_resource_adjusted_full_replay_metrics.json`.
主要指标：conversion status `ok_resource_adjusted_csv_conversion`, input frames `180`, input columns `36`, output frames `299`, joint shape `[299,29]`, body position shape `[299,40,3]`, max body quaternion norm error `4.768e-07`; full replay status `ok_resource_adjusted_csv_full_replay`, executed steps `299`, motion total steps `299`, max joint position error `0.0`, max root position error `0.0`.
与论文一致性：uses official downloaded motion CSV and the paper-side G1/50Hz preprocessing contract, but not the official successful URDF/USD conversion path; evidence is closer to official replay than debug fixtures but remains resource-adjusted.
发现的差异：official asset conversion path is still bypassed; the generated NPZ is not official `csv_to_npz.py` output and is not a paper-level replay/evaluation artifact.
失败与风险：official `csv_to_npz.py`, official `replay_npz.py`, PPO tracking training/evaluation, DAgger rollout logs, teacher rollout dataset, trained tracking checkpoint, and Fig. 5/Fig. 6 closed-loop videos remain missing or blocked.
下一阶段：try wiring the official-source resource-adjusted NPZ into the official tracking task eval, then retry the official URDF/USD converter path or a bounded official train entry only after preserving the boundary.

Master audit result after adding resource-adjusted official-CSV conversion/replay evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted official-CSV tracking task eval

阶段：Level B official-source motion task-manager evaluation gate.
状态：完成 resource-adjusted official-CSV-derived `Tracking-Flat-G1-v0` full task eval; official replay/PPO remains incomplete.
开始时间：2026-06-19 00:35 Asia/Shanghai.
结束时间：2026-06-19 00:40 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime; device `cuda:6`.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py`.
官方/重新实现：official `Tracking-Flat-G1-v0` manager stack and official downloaded G1 LAFAN CSV-derived motion, but generated resource-adjusted enriched USD instead of official URDF-converter output.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_004056_resource_adjusted_csv_task_eval.md`.
配置：input motion `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/walk1_subject1_frames_1_180_resource_adjusted_motion.npz`, full `299` steps, `num_envs=1`, zero diagnostic action, generated enriched G1 USD, task `Tracking-Flat-G1-v0`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py`.
GPU：GPU6 primary IsaacLab runtime context; diagnostic task eval only, not a formal two-GPU training experiment.
峰值显存：not sampled as a formal GPU experiment.
平均 GPU-Util：not recorded for this diagnostic gate.
平均功耗：not recorded for this diagnostic gate.
运行时间：completed without stall timeout.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_task_eval/tracking_g1_resource_adjusted_csv_task_eval_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_task_eval/tracking_g1_resource_adjusted_csv_task_eval_metrics.json`.
主要指标：status `ok_resource_adjusted_csv_task_eval`; step count `299`; action dim `29`; policy observation dim `160`; critic observation dim `286`; reward terms `9`; termination terms `4`; robot joints `29`; robot bodies `40`; reward mean/min/max `0.02670689582525687`/`-0.010335305705666542`/`0.052452173084020615`; terminated/truncated totals `26`/`12`.
与论文一致性：uses official-source motion and official task-manager surfaces, but it is zero-action diagnostic evidence with generated USD, not trained-policy tracking performance.
发现的差异：termination/truncation counts are diagnostic zero-action behavior and cannot be interpreted as paper success/failure metrics.
失败与风险：official `csv_to_npz.py`, official `replay_npz.py`, PPO tracking training/evaluation, DAgger rollout logs, teacher rollout dataset, trained tracking checkpoint, and Fig. 5/Fig. 6 closed-loop videos remain missing or blocked.
下一阶段：use this task gate to justify a bounded train-entry retry only if official asset/conversion boundaries are clearly documented; otherwise continue official URDF/USD converter diagnosis.

Master audit result after adding resource-adjusted official-CSV task eval evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted RSL-RL train-entry diagnostic

阶段：Level B IsaacLab / whole_body_tracking RSL-RL train-entry wiring gate.
状态：完成 bounded train-entry diagnostic; official PPO tracking training/evaluation remains incomplete.
开始时间：2026-06-19 00:45 Asia/Shanghai.
结束时间：2026-06-19 00:56 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim/RSL-RL runtime; device `cuda:6`.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.py`.
官方/重新实现：official `Tracking-Flat-G1-v0`, official IsaacLab `RslRlVecEnvWrapper`, and official `MotionOnPolicyRunner`; generated resource-adjusted enriched USD and official-CSV-derived resource-adjusted motion.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_005657_resource_adjusted_train_entry_diagnostic.md`.
配置：`num_envs=1`, `num_steps_per_env=4`, one learning iteration, one PPO epoch, one minibatch, no log directory, no checkpoint writing, task `Tracking-Flat-G1-v0`, motion `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/walk1_subject1_frames_1_180_resource_adjusted_motion.npz`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.py`.
GPU：GPU6 primary IsaacLab runtime context; diagnostic train-entry wiring only, not a formal two-GPU training experiment.
峰值显存：not sampled as a formal GPU experiment because this is intentionally a one-env/four-step entry diagnostic.
平均 GPU-Util：not recorded for this diagnostic gate.
平均功耗：not recorded for this diagnostic gate.
运行时间：final successful probe duration about `30.0` seconds.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_train_entry_diagnostic/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_train_entry_diagnostic/tracking_g1_resource_adjusted_train_entry_diagnostic_metrics.json`; raw local log under `/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_resource_adjusted_train_entry_diagnostic/tracking_g1_resource_adjusted_train_entry_diagnostic.log`.
主要指标：status `ok_resource_adjusted_train_entry_diagnostic`; runner `MotionOnPolicyRunner`; training type `rl`; action dim `29`; policy obs dim `160`; privileged obs dim `286`; robot joints `29`; robot bodies `40`; requested learning iterations `1`; configured rollout steps per env `4`; checkpoint written `false`.
与论文一致性：verifies the recovered local environment can enter the official tracking PPO runner stack, but this is only a resource-adjusted wiring diagnostic and not paper-level PPO training/evaluation.
发现的差异：first run reached runner creation but failed because RSL-RL still tried to save code state with `log_dir=None`; the probe now sets `runner.disable_logs=True`. Successful runs still log PhysX GPU kernel launch warnings before the success sentinel.
失败与风险：official URDF/USD converter output, official `csv_to_npz.py`, official `replay_npz.py`, formal PPO tracking training/evaluation, teacher rollout dataset, DAgger logs, trained checkpoint, Fig. 5/Fig. 6 closed-loop videos, and real-robot evidence remain missing or blocked.
下一阶段：decide whether to run a controlled short PPO training/evaluation attempt with GPU telemetry on GPUs 5 and 6, or continue fixing the official URDF/USD conversion path before formal training.

Master audit result after adding resource-adjusted train-entry diagnostic evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 blocked-gate state correction after live headless recovery

阶段：Audit/report consistency and gate-state correction.
状态：完成 blocked-gate 状态修正；live headless gate is no longer the active blocker, official G1 USD conversion/replay remains blocked.
开始时间：2026-06-19 00:59 Asia/Shanghai.
结束时间：2026-06-19 01:06 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis`; no training launched.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/blocked_gate_audit.py`; `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`; `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
官方/重新实现：audit/report update only.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_010644_blocked_gate_state_correction.md`.
配置：reads current `isaaclab_live_gate_probe`, official replay conversion audit, resource-adjusted task/train-entry audits, ROS/hardware/Level C/Fig.5-Fig.6 audits.
执行命令：`python3 reproduction/scripts/blocked_gate_audit.py`; `python3 reproduction/scripts/final_reproduction_report.py`; `python3 reproduction/scripts/reproduction_master_audit.py`.
GPU：no formal GPU experiment. A stale resource-adjusted train-entry diagnostic process that had already written success sentinels was found and terminated; it was not formal training.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.tsv`; refreshed final report and master audit outputs.
主要指标：blocked-gate status counts after cleanup: `blocked=4`, `clear=1`, `clear_with_historical_failure=1`, `clear_with_runtime_warning=1`, `out_of_scope=1`. `isaaclab_kit_inotify` is now `clear_with_historical_failure`; `isaaclab_kit_vulkan_cuda_runtime` remains `clear_with_runtime_warning`; `official_g1_usd_conversion_replay` is the active tracking-side blocked gate.
与论文一致性：improves audit fidelity for the reading report and reproduction boundary; it does not add a new paper-level result.
发现的差异：older generated reports still framed inotify as the active tracking blocker. Current evidence shows AppLauncher live sentinel passes, so official replay recovery should focus on G1 USD conversion/replay and formal training warnings.
失败与风险：official G1 USD conversion/replay, formal PPO tracking training/evaluation, DAgger/teacher rollouts, Level C official checkpoints, Fig.5/Fig.6 paper-level videos, TensorRT deployment, and real robot remain incomplete.
下一阶段：continue official G1 USD conversion/replay recovery or run a controlled short PPO training/evaluation only after documenting PhysX warnings and GPU telemetry requirements.

Master audit result after blocked-gate correction: pending verification rerun; goal_complete=false.

## 2026-06-19 G1 URDF source-equivalence audit

阶段：Level B official-source asset boundary audit for G1 conversion/replay recovery.
状态：完成 G1 URDF source-equivalence audit；official G1 USD conversion/replay remains blocked.
开始时间：2026-06-19 01:08 Asia/Shanghai.
结束时间：2026-06-19 01:21 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis`; no Kit launch, replay, PPO, or training started.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_source_equivalence_audit.py`.
官方/重新实现：official downloaded LAFAN G1 URDF, reproduction-data G1 URDF copy, and official `whole_body_tracking` G1 URDF; source-boundary audit only.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_012123_g1_urdf_source_equivalence.md`.
配置：compares link/joint/inertial/visual/collision summaries, SHA256 identity, non-fixed/action joint set equality, and support link/joint differences.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_source_equivalence_audit.py`.
GPU：no GPU experiment. This audit reads URDF/XML and existing JSON evidence only.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json`; `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.tsv`.
主要指标：status `ok_with_source_differences_recorded`; downloaded official LAFAN G1 URDF and reproduction-data copy are SHA256/structurally identical; official `whole_body_tracking` G1 URDF preserves the same 29 non-fixed/action joints; support-link differences are `d435_link` vs `LL_FOOT`/`LR_FOOT`; support-joint differences are `d435_joint` vs `LL_FOOT_frame`/`LR_FOOT_frame`; common action joint fields match.
与论文一致性：improves traceability of the Unitree G1 asset source used by the tracking setup, but the paper does not publish a numeric URDF source-equivalence metric.
发现的差异：the official downloaded/reproduction-data URDF and the official `whole_body_tracking` URDF are not identical sources; support links/joints and physical bookkeeping differ even though the 29 action joints align.
失败与风险：official URDF/USD converter output, official `csv_to_npz.py`, official `replay_npz.py`, formal PPO tracking training/evaluation, teacher rollout dataset, DAgger logs, trained checkpoint, Fig. 5/Fig. 6 closed-loop videos, and real-robot evidence remain missing or blocked.
下一阶段：use the source-equivalence boundary when deciding whether to continue offline USD scaffold refinement or run a controlled short PPO training/evaluation attempt; do not claim official replay until an official converter/replay artifact passes.

Master audit result after adding G1 URDF source-equivalence evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 official replay_npz entry diagnostic

阶段：Level B official `whole_body_tracking/scripts/replay_npz.py` entry diagnostic.
状态：完成 bounded official-entry diagnostic；official replay remains blocked by the official URDF converter layer-save path.
开始时间：2026-06-19 01:24 Asia/Shanghai.
结束时间：2026-06-19 01:32 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime; device `cuda:6`.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`.
官方/重新实现：unmodified official `reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py`; local fake-WandB artifact and bounded AppLauncher wrapper for diagnostics only.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_013245_official_replay_entry_diagnostic.md`.
配置：fake registry `bm-local/g1_resource_adjusted_motion:latest` points to the existing official-CSV-derived resource-adjusted `motion.npz`; AppLauncher wrapper caps `simulation_app.is_running()` at 299 calls; official worktree is not modified.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`.
GPU：GPU6 IsaacLab runtime context; diagnostic replay-entry gate only, not a formal two-GPU training experiment.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json`; failed log copy `/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic.log`; local raw log under `/mnt/infini-data/test/BeyondMimic/logs/tracking_official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic.log`.
主要指标：status `ok_with_official_replay_npz_entry_blocker`; latest blocker `official_urdf_converter_layer_save_blocked`; AppLauncher constructed `true`; fake artifact download `false`; layer save blocker `true`; empty robot after converter `true`.
与论文一致性：uses the official replay entry script surface, but does not reach official artifact download, motion loading, replay loop, trained-policy evaluation, or paper-level tracking metrics.
发现的差异：the unmodified official replay entry blocks during scene construction because official URDF conversion attempts to save USD layers under `/tmp/IsaacLab/...` and saving is not allowed, producing an unresolved/empty robot prim with no contact sensors.
失败与风险：official G1 USD converter output, official `csv_to_npz.py`, official `replay_npz.py` replay loop, formal PPO tracking training/evaluation, DAgger/teacher rollouts, Fig.5/Fig.6 paper-level videos, TensorRT deployment, and real robot remain incomplete.
下一阶段：continue official URDF/USD converter recovery or use the already labeled resource-adjusted gates for a controlled short PPO diagnostic only if the generated-asset boundary and GPU telemetry requirements are explicitly documented.

Master audit result after adding official replay entry diagnostic evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 G1 URDF ImportConfig variant probe

阶段：Level B official G1 URDF converter recovery boundary probe.
状态：完成 ImportConfig surface and baseline converter probe；official replay remains blocked, and this branch should not be pursued as the main reproduction path.
开始时间：2026-06-19 01:41 Asia/Shanghai.
结束时间：2026-06-19 02:07 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime; device `cuda:6`.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_import_config_variant_probe.py`.
官方/重新实现：official `whole_body_tracking` G1 URDF through Isaac Sim 4.5 URDF importer; diagnostic only, no replay or PPO.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_020706_g1_import_config_probe.md`.
配置：method-surface subprobe enumerates `URDFCreateImportConfig`; baseline converter subprobe writes under project `tmp/` and exits after sentinel to avoid Kit shutdown hangs.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_import_config_variant_probe.py`.
GPU：GPU6 IsaacLab runtime context; converter diagnostic only, not a formal two-GPU training experiment.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_import_config_variant_probe/tracking_g1_urdf_import_config_variant_probe.json`; local raw logs under `/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_urdf_import_config_variant_probe/`.
主要指标：status `ok_with_import_config_surface_recorded_and_variants_blocked`; `has_set_make_instanceable=false`; `has_set_instanceable_usd_path=false`; baseline USD `stage_open_ok=true` but `prim_count=0`, `joint_count=0`, `rigid_body_like_count=0`; current blocker `official_urdf_converter_layer_save_or_vulkan_device_lost_after_import_config_variants`.
与论文一致性：narrows why official replay still cannot start from the official G1 URDF converter path, but it is not a paper metric, not official `motion.npz`, not replay, and not policy evaluation.
发现的差异：the Isaac Sim 4.5 Python `ImportConfig` surface does not expose the instanceable setters available in IsaacLab converter cfg fields, so a Python-level instanceable patch cannot repair this official replay blocker.
失败与风险：official G1 USD converter output, official `csv_to_npz.py`, official `replay_npz.py` replay loop, formal PPO tracking training/evaluation, DAgger/teacher rollouts, Fig.5/Fig.6 paper-level videos, TensorRT deployment, and real robot remain incomplete.
下一阶段：return to the reproduction mainline by running available resource-adjusted/full virtual task or controlled PPO diagnostics, while continuing official converter work only if a new lower-level importer path is identified.

Master audit result after adding G1 ImportConfig probe evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted PPO training harness and GPU preflight

阶段：Level B resource-adjusted PPO training after train-entry smoke.
状态：完成 resource-adjusted PPO training harness and one completed two-GPU training run on selected physical GPUs `[4, 7]`.
开始时间：2026-06-19 02:10 Asia/Shanghai.
结束时间：2026-06-19 03:44 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/RSL-RL runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`.
官方/重新实现：official `Tracking-Flat-G1-v0`, official RSL-RL PPO runner stack, generated resource-adjusted G1 USD, and official-CSV-derived resource-adjusted motion. This is not official asset/replay training.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_021821_resource_adjusted_ppo_harness.md`.
配置：candidate physical GPUs `[4, 5, 6, 7]`, selected physical GPUs `[4, 7]`, `CUDA_VISIBLE_DEVICES=4,7`, torch distributed `world_size=2`, `num_envs_per_rank=512`, total envs `1024`, official PPO rollout length `num_steps_per_env=24`, `max_iterations=100`, seed `20260619`, GPU telemetry path under `res/runs`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py`.
GPU：preflight selected GPUs `[4, 7]` and launched distributed PPO. GPU telemetry CSV was retained at `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_resource_adjusted_ppo_training/resource_adjusted_ppo_20260618_182241_seed20260619/gpu_metrics.csv`.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.json`; worker script `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_worker.py`.
主要指标：status `ok_resource_adjusted_ppo_training_completed`; `attempted_training=true`; `resource_ready=true`; `train_entry_smoke_passed=true`; total envs `1024`; rollout steps per env `24`; iterations `100`; duration `4887.514` seconds; rank 0 total timesteps `2457600`; checkpoint count `3` with `model_0.pt`, `model_50.pt`, and `model_99.pt` retained under ignored local `res/runs`.
与论文一致性：moves beyond smoke wiring to an actual virtual PPO training run through the official task/runner stack, but it remains resource-adjusted, lower-scale, and not official paper-level tracking teacher training.
发现的差异：the run can train through IsaacLab/RSL-RL once the resource-adjusted USD/motion workaround is used, while the official G1 URDF conversion/replay path remains blocked.
失败与风险：no official replay asset, no official paper-scale PPO evaluation, no teacher rollout dataset, no DAgger logs, no Fig.5/Fig.6 closed-loop videos, no TensorRT deployment evidence, and no real robot result were produced.
下一阶段：evaluate `model_99.pt` in the same resource-adjusted task, then decide whether to collect clearly labeled resource-adjusted teacher rollouts while continuing to keep official replay and paper-level tracking gates separate.

Master audit result after adding PPO harness evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted PPO checkpoint evaluation

阶段：Level B resource-adjusted PPO checkpoint evaluation after completed PPO training.
状态：完成 `model_99.pt` checkpoint evaluation through the official `Tracking-Flat-G1-v0` task and RSL-RL inference API.
开始时间：2026-06-19 04:00 Asia/Shanghai.
结束时间：2026-06-19 04:25 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/RSL-RL runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`.
官方/重新实现：official `Tracking-Flat-G1-v0`, official RSL-RL `OnPolicyRunner.load()` / inference policy API, generated resource-adjusted G1 USD, and official-CSV-derived resource-adjusted motion. This is not official asset/replay evaluation.
Git commit：pending at time of progress entry; final commit recorded in `reproduction/docs/progress/20260619_042556_resource_adjusted_ppo_checkpoint_eval.md`.
配置：candidate physical GPUs `[4, 7]`, selected physical GPUs `[4, 7]`, `CUDA_VISIBLE_DEVICES=4,7`, `num_envs=512`, `eval_steps=299`, total env steps `153088`, seed `20260620`, checkpoint `model_99.pt`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`.
GPU：pre-run guard identified and terminated one `/mnt/infini-data/test/wangjc/` process on GPU 4 as authorized by the user; cleanup summary saved at `/mnt/infini-data/test/BeyondMimic/res/gpu_guard/20260619_gpu47_wangjc_kill_summary.json`. Evaluation ran with GPU 4 active and GPU 7 visible; GPU 4 peak memory `54692` MiB and mean utilization `98.2%`.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/tracking_g1_resource_adjusted_ppo_checkpoint_eval.json`; worker script `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py`; local ignored metrics/timeseries under `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_resource_adjusted_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260618_200515_seed20260620`.
主要指标：status `ok_resource_adjusted_ppo_checkpoint_eval_completed`; loaded iteration `99`; reward mean over steps `0.025898515209431035`; anchor position error mean `0.10595783163921091`; body position error mean `0.18350737062859096`; joint position error mean `1.2326450995776965`; done count total `13172`; timeout count total `0`.
与论文一致性：moves from training-only evidence to an actual virtual checkpoint rollout/evaluation using official task and RSL-RL inference APIs, but remains resource-adjusted and not official paper-level tracking evaluation.
发现的差异：the evaluated checkpoint can run through the resource-adjusted task for the full available 299-step motion window; official G1 URDF conversion/replay remains the blocker for official paper-level comparison.
失败与风险：no official replay asset, no official paper-scale PPO benchmark, no teacher rollout dataset, no DAgger logs, no Fig.5/Fig.6 closed-loop videos, no TensorRT deployment evidence, and no real robot result were produced.
下一阶段：use the evaluated checkpoint as a clearly labeled resource-adjusted teacher candidate for rollout dataset collection only if the next audit keeps official-vs-resource-adjusted boundaries explicit.

Master audit result after adding PPO checkpoint-eval evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted teacher rollout dataset gate

阶段：Level B resource-adjusted teacher rollout collection after completed checkpoint evaluation.
状态：完成 fixed-GPU `[4, 7]` rollout dataset collection from local `model_99.pt`; produced two raw dataset shards for downstream local VAE/state-latent experiments.
开始时间：2026-06-19 04:35 Asia/Shanghai.
结束时间：2026-06-19 04:50 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/RSL-RL runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`.
官方/重新实现：official `Tracking-Flat-G1-v0`, official RSL-RL `OnPolicyRunner.load()` / inference policy API, generated resource-adjusted G1 USD, and official-CSV-derived resource-adjusted motion. This is not official DAgger data and not official paper-level trajectory data.
Git commit：pending at time of progress entry; final commit will be recorded in `reproduction/docs/progress/20260619_0458xx_resource_adjusted_teacher_rollout_dataset.md`.
配置：candidate physical GPUs `[4, 7]`, selected physical GPUs `[4, 7]`, `CUDA_VISIBLE_DEVICES=4,7`, torch distributed `world_size=2`, `num_envs_per_rank=512`, total envs `1024`, rollout steps `299`, expected total env steps `306176`, seed `20260621`, checkpoint `model_99.pt`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`.
GPU：pre-run guard terminated one user-authorized `/mnt/infini-data/test/wangjc/` process on GPU 4; guard record saved at `/mnt/infini-data/test/BeyondMimic/res/gpu_guard/20260618_203906_gpu47_wangjc_teacher_rollout_guard.json`. The rollout ran on GPUs 4 and 7 with mean utilization about `91.85%` and `93.24%`; peak memory was about `6775` MiB and `6765` MiB, below the 10GB/card formal-training threshold, so this is reported as a rollout dataset gate rather than a formal high-memory training experiment.
输出文件：summary `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_dataset.json`; worker script `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_worker.py`; raw ignored shards under `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_resource_adjusted_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260618_203906_seed20260621/`.
主要指标：status `ok_resource_adjusted_teacher_rollout_dataset_completed`; shard count `2`; total env steps `306176`; raw compressed dataset size `514367800` bytes; reward means by rank `[0.02579653076827526, 0.025758858770132065]`; done count total `26258`; timeout count total `0`.
与论文一致性：moves from checkpoint evaluation to actual local state/action rollout data generation, which is useful for reading-report evidence and downstream local VAE/state-latent experiments.
发现的差异：the data is generated from a local resource-adjusted teacher candidate and generated asset/motion workaround, not from the paper's official teacher checkpoint, official DAgger pipeline, or official closed-loop diffusion evaluation.
失败与风险：official G1 replay asset, official DAgger rollout logs, paper-scale teacher evaluation, VAE closed-loop evaluation, state-latent diffusion closed-loop evaluation, Fig.5/Fig.6 paper-level videos, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.
下一阶段：derive a small audit/loader contract for these shards, then use them only as clearly labeled local teacher-candidate data for VAE/state-latent experiments after the user confirms the next experimental direction.

Master audit result after adding resource-adjusted teacher rollout dataset evidence: pending verification rerun; goal_complete=false.

## 2026-06-19 current IsaacLab headless AppLauncher gate refresh

阶段：Phase 1 / Level B environment gate refresh.
状态：完成当前 AppLauncher(headless=True) gate 复核；修正旧 `env_import_probe` 硬编码状态，使环境总览与 live/current gate evidence 一致。
开始时间：2026-06-19 04:59 Asia/Shanghai.
结束时间：2026-06-19 05:05 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_current_headless_gate.py`; updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/env_import_probe.py`.
官方/重新实现：IsaacLab `AppLauncher(headless=True)` startup sentinel only; no replay, PPO, DAgger, VAE/diffusion, Fig.5/Fig.6, or robot action.
配置：candidate physical GPUs `[4, 7]`, selected physical GPU `4`, no `CUDA_VISIBLE_DEVICES` hiding, AppLauncher device `cuda:4`, single-GPU renderer/physics Kit args targeting physical GPU 4, timeout `240` seconds.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py`; then `envs/bm_analysis/bin/python reproduction/scripts/env_import_probe.py`.
GPU：GPU4 was free before the gate. GPU6 was occupied by a non-wangjc VLLM process and was not touched. The successful gate activated physical GPU4 and reached `BM_SENTINEL:current_gate:after_app` plus payload with `is_running=true`.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`; current log under `/mnt/infini-data/test/BeyondMimic/logs/setup/isaaclab_current_headless_gate/`; refreshed `/mnt/infini-data/test/BeyondMimic/res/setup/env_probe/env_import_probe.json`.
主要指标：current headless gate status `ok`; `app_launcher_headless_success_sentinel=true`; `payload_is_running=true`; `no_fatal_runtime_error=true`; `isaaclab_live_headless_gate_ok=true` after env probe refresh. CUDA P2P/IOMMU warning is retained but not treated as the active blocker.
与论文一致性：clears the current AppLauncher startup gate needed before replay/task smoke work.
发现的差异：using `CUDA_VISIBLE_DEVICES=4` with `device=cuda:0` reproduces Omniverse/CUDA enumeration mismatch and `activeGpu 0` incompatibility; the passing current gate keeps all GPUs visible and targets physical `cuda:4`.
失败与风险：official G1 conversion/replay remains blocked; current headless startup success does not prove official replay, PPO training, DAgger, VAE/diffusion, Fig.5/Fig.6, TensorRT, or robot behavior.
下一阶段：continue mainline official replay recovery or, if official converter remains blocked, use the already-audited resource-adjusted task stack for the next full-data downstream experiment with explicit boundaries.

Master audit result after current headless gate refresh: pending verification rerun; goal_complete=false.

## 2026-06-19 official replay_npz entry diagnostic on current GPU4 gate

阶段：Level B official `whole_body_tracking` replay entry recovery after current AppLauncher headless gate passed.
状态：完成当前 GPU4 official `replay_npz.py` entry 复测；诊断脚本现在隔离 Kit 子进程、固定物理 `cuda:4`、关闭 Kit 多 GPU，并能在失败时稳定写出 JSON 和 failed-run log。
开始时间：2026-06-19 05:09 Asia/Shanghai.
结束时间：2026-06-19 05:13 Asia/Shanghai.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`; generated probe `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_probe.py`.
官方/重新实现：unmodified official `whole_body_tracking/scripts/replay_npz.py` entrypoint, with a bounded wrapper and local fake-WandB artifact pointing to the existing official-CSV-derived resource-adjusted `motion.npz`. This is a replay-entry diagnostic only, not official `csv_to_npz.py` output and not paper-level replay evidence.
配置：physical GPU `4`, no `CUDA_VISIBLE_DEVICES` masking, AppLauncher device `cuda:4`, Kit args disabling multi-GPU renderer/physics auto-selection, bounded max steps `299`, stall timeout `900` seconds.
执行命令：`envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`; `envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py`.
GPU：GPU4/GPU7 were empty immediately before the diagnostic; global non-target GPU processes were left untouched. The diagnostic launched Kit on physical GPU4.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json`; retained failed log `/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic.log`.
主要指标：status `ok_with_official_replay_npz_entry_blocker`; latest blocker `official_urdf_converter_layer_save_blocked`; process return code `0`; duration `30.029` seconds; markers show `before_runpy=true`, `add_app_launcher_args=true`, `after_real_app_launcher=true`, but `fake_wandb_download=false`, `bounded_loop_complete=false`, `permission_to_save_false=true`, `failed_to_save_layer=true`, and `empty_robot_after_converter=true`.
与论文一致性：confirms the current host can enter the official replay entry surface after the AppLauncher gate, but official G1 conversion/write still blocks before motion loading and replay. This keeps the official-vs-resource-adjusted boundary explicit.
发现的差异：the existing resource-adjusted full replay can run all 299 steps, while the unmodified official replay entry still fails in the official URDF/USD converter layer-save path before fake artifact download.
失败与风险：official `csv_to_npz.py` output, official replay loop execution, formal PPO evaluation, true DAgger rollout logs, VAE/diffusion closed-loop evaluation, Fig.5/Fig.6 paper-level videos, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.
下一阶段：either continue official converter recovery at the USD layer-save/empty-robot boundary or proceed with clearly labeled resource-adjusted downstream experiments for the English report while keeping official replay blocked.

Master audit result after official replay entry refresh: pending verification rerun; goal_complete=false.

## 2026-06-19 official G1 in-memory importer and resource-adjusted rollout VAE

阶段：Level B official importer recovery plus Level C resource-adjusted downstream VAE training.
状态：完成 current GPU4 official G1 in-memory importer probe and full local conditional action VAE training over the existing resource-adjusted teacher rollout dataset.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper; `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` for IsaacLab/Isaac Sim importer probe; `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion` worker runtime for PyTorch VAE training.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py`; `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`.
官方/重新实现：the importer probe uses official Isaac Sim URDF importer and official G1 URDF with `dest_path=""`; the VAE is a local conditional action VAE trained on resource-adjusted teacher rollout shards, not the official BeyondMimic DAgger/VAE pipeline.
配置：official importer probe targets physical GPU4 with AppLauncher/headless settings; VAE training uses `CUDA_VISIBLE_DEVICES=4,7`, two visible CUDA devices, DataParallel, seed `20260624`, latent dim `32`, hidden dim `512`, batch size `16384`, 40 epochs, KL coefficient `1e-4`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_in_memory_gpu4_probe.py`; `envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`.
GPU：GPU4/GPU7 were checked before VAE training; no `/mnt/infini-data/test/wangjc/` process needed to be killed for this run. The VAE training saw two CUDA devices and used DataParallel. Peak recorded memory was below the 10GB/card formal-training threshold because the model is a small conditional action VAE rather than a formal PPO/diffusion training job.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_in_memory_gpu4_probe/tracking_g1_urdf_in_memory_gpu4_probe.json`; `/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_g1_urdf_in_memory_gpu4_probe/tracking_g1_urdf_in_memory_gpu4_probe.log`; `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json`; `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.tsv`; ignored checkpoint/logs under `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_teacher_rollout_vae_training/`.
主要指标：official in-memory importer status `ok_with_vulkan_device_lost_blocker`, return code `-9`, AppLauncher reached, no exported robot stage. VAE training status `ok`; sample count `306176`; train/validation/test split `244940/30618/30618`; validation action MSE `0.0029512199107557535`; test action MSE `0.002976319403387606`; test action MAE `0.04116625525057316`.
与论文一致性：the importer probe directly attacks the official replay prerequisite and records a narrower blocker. The VAE training advances the local teacher-rollout-to-latent-action pipeline, but it is resource-adjusted and not an official BeyondMimic DAgger/VAE checkpoint or closed-loop diffusion result.
失败与风险：official G1 URDF conversion/replay, official `csv_to_npz.py`, paper-scale PPO evaluation, true DAgger rollout logs, official VAE/diffusion checkpoints, closed-loop Fig.5/Fig.6 videos/metrics, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.
下一阶段：refresh artifact manifest, paper-vs-reproduction comparison, final report, completion matrix, verification-command audits, progress audit, and master audit; then commit and push the round.

Master audit result after this entry: ok; goal_complete=false.

## 2026-06-19 resource-adjusted state-latent offline guidance evaluation

阶段：Level C downstream resource-adjusted offline guidance gate.
状态：完成基于本地 resource-adjusted state-latent denoiser 的 validation/test offline guidance 评估，并接入 comparison/final-report/master-audit 证据链。
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper and `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion` PyTorch runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`.
官方/重新实现：local resource-adjusted offline surrogate over the previously trained local denoiser. This is not official BeyondMimic guidance, not IsaacLab closed-loop rollout, and not Fig. 5/Fig. 6 paper-level evidence.
配置：GPU `[4,7]`, `CUDA_VISIBLE_DEVICES=4,7`, seed `20260627`, max windows per split `4096`, batch windows `512`, tasks `velocity_command`, `latent_smoothness`, `latent_magnitude`, `composed`, guidance scales `0,0.0005,0.001,0.002,0.005,0.01`.
执行命令：`envs/bm_analysis/bin/python -m py_compile reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`; `envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`.
GPU：the script checked GPU4/GPU7 and wrote `/mnt/infini-data/test/BeyondMimic/res/gpu_guard/20260619_060505_gpu47_wangjc_state_latent_guidance_guard.json`. Peak recorded memory was small (`328` MiB on GPU4, `4` MiB on GPU7), so this is reported as a quick offline gate rather than a formal large GPU experiment.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_guidance_eval/level_c_resource_adjusted_state_latent_guidance_eval.json`; `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_guidance_eval/level_c_resource_adjusted_state_latent_guidance_eval.tsv`; ignored worker/sample artifacts under `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_state_latent_guidance_eval/`.
主要指标：status `ok`; total selected windows `8192`; aggregate rows `48`; all 4 tasks have nonzero best guidance gradients and best-cost improvement. Mean best cost deltas: velocity command `1.7268666852032766e-07`, latent smoothness `8.558126864954829e-07`, latent magnitude `1.5347613953053951e-06`, composed `1.86315446626395e-07`.
与论文一致性：connects the local teacher-rollout VAE -> state-latent dataset -> denoiser chain to task-cost guidance, which is useful for the English reading report's reproduction narrative.
失败与风险：does not run IsaacLab closed-loop control, does not evaluate success/fall/velocity-tracking paper metrics, does not produce Fig. 5/Fig. 6 videos, and does not validate TensorRT/asynchronous deployment or real robot behavior.
下一阶段：refresh artifact manifest, comparison, final report, completion matrix status audit, verification command audits, progress audit, and master audit; then commit and push.

Master audit result after this entry: pending verification rerun; goal_complete=false.

## 2026-06-19 official replay loop with enriched-USD runtime patch

阶段：Level B official replay mainline recovery.
状态：完成更接近官方入口的 replay gate：官方 `whole_body_tracking/scripts/replay_npz.py` 的 replay loop body 在 runtime asset patch 下跑到 299-step bound。
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper and `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking` IsaacLab/Isaac Sim runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`; official entry `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py`.
官方/重新实现：official replay loop body, with runtime monkeypatch only for dependencies. The official worktree is not modified. The G1 config is patched in memory to use the validated resource-adjusted enriched USD and local fake-WandB points to the official-CSV-derived resource-adjusted `motion.npz`.
配置：physical GPU `4`, no `CUDA_VISIBLE_DEVICES` masking, AppLauncher device `cuda:4`, Kit single-GPU args, bounded `simulation_app.is_running()` wrapper with max calls `299`.
执行命令：`envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`; `envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py`.
GPU：GPU guard `/mnt/infini-data/test/BeyondMimic/res/gpu_guard/20260619_062109_gpu47_wangjc_official_replay_loop_guard.json` recorded no `/mnt/infini-data/test/wangjc/` kills and no skipped non-wangjc target-GPU processes for the successful run.
输出文件：`/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_loop_with_enriched_usd/tracking_official_replay_npz_loop_with_enriched_usd_audit.json`; generated probe `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_loop_with_enriched_usd/tracking_official_replay_npz_loop_with_enriched_usd_probe.py`; raw log retained under `/mnt/infini-data/test/BeyondMimic/logs/tracking_official_replay_npz_loop_with_enriched_usd/`.
主要指标：status `ok_official_replay_loop_with_enriched_usd_patch`; latest blocker `none_official_replay_loop_completed_with_enriched_usd_patch`; AppLauncher constructed; G1 config patched to enriched USD; fake-WandB artifact download seen; official loop sentinels reached calls `1/50/100/150/200/250/299`; official loop complete `299`; process return code `0`.
与论文一致性：this advances the official replay mainline beyond static/local-copy replay evidence by executing the official `replay_npz.py` loop body. It still remains resource-adjusted because the official URDF converter path and official `csv_to_npz.py` output are bypassed.
失败与风险：official G1 URDF/USD converter, official `csv_to_npz.py` output, paper-scale PPO tracking evaluation, true DAgger rollout logs, closed-loop VAE/diffusion guidance, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.
下一阶段：refresh artifact manifest, paper-vs-reproduction comparison, final report, completion matrix status audit, verification command audits, progress audit, and master audit; then commit and push.

Master audit result after this entry: pending verification rerun; goal_complete=false.

## 2026-06-19 resource-adjusted state-latent dataset and diffusion training

阶段：Level C downstream resource-adjusted state-latent/diffusion mainline.
状态：完成 full resource-adjusted state/action-latent dataset construction and full-window denoiser training over all generated windows.
使用环境：`/mnt/infini-data/test/BeyondMimic/envs/bm_analysis` wrapper and `/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion` PyTorch runtime.
使用代码：`/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`; `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`.
官方/重新实现：local resource-adjusted downstream chain from existing teacher rollout shards and local conditional action VAE. This is not official BeyondMimic DAgger/state-latent/diffusion data.
配置：GPU `[4,7]`, `CUDA_VISIBLE_DEVICES=4,7`, sequence length `21`, obs dim `160`, latent dim `32`, token dim `192`; diffusion hidden dim `512`, denoising steps `20`, batch windows `2048`, epochs `30`, seed `20260626`.
执行命令：`envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`; `envs/bm_analysis/bin/python reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`.
GPU：both scripts used GPU4/GPU7 visibility and recorded telemetry. Diffusion training used DataParallel with peak memory around `2216` MiB on GPU4 and `1806` MiB on GPU7; this is below the 10GB formal-PPO threshold because the denoiser is small, so it is reported as a resource-adjusted denoising training gate rather than paper-scale diffusion training.
输出文件：small summaries under `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/` and `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_diffusion_training/`; large latent shards, window index, and denoiser checkpoint under ignored `/mnt/infini-data/test/BeyondMimic/res/runs/`.
主要指标：state/action-latent dataset `306176` samples, `285696` windows, split counts `228558/28569/28569`, weighted posterior reconstruction MSE `0.002923722844570875`. Diffusion training test pred token MSE `0.03726350223379476`, noisy token MSE `0.08264570789677757`, denoising improvement ratio `0.5491175139992032`.
与论文一致性：advances the paper pipeline shape from teacher rollouts to VAE latents to diffusion training, but remains resource-adjusted local evidence and cannot be reported as official BeyondMimic closed-loop or paper-level Fig.5/Fig.6 reproduction.
失败与风险：official G1 converter/replay, official DAgger logs, official VAE/diffusion checkpoints, closed-loop guidance rollout, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.
下一阶段：refresh artifact manifest, comparison, final report, required artifact absence, completion matrix, verification command audits, progress audit, and master audit; then commit and push.

Master audit result after this entry: ok; goal_complete=false.
