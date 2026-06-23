# Progress Update

## Goal

排查当前 MuJoCo 运动控制视频效果差、机器人站不稳的原因。重点验证：最新 clean walk 视频是否真的是纯 teacher/VAE/diffusion/guidance 控制成功，还是依赖 reference anchor 才能看起来稳定。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite/clean_walk_mujoco_control_suite_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_pd_control_demo/clean_lafan1_walk1_subject1_pd_control_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/*summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_clean_walk_model_weight_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_000848_clean_walk_model_weight_sweep.md`

## Commands Run

```bash
MUJOCO_GL=egl BM_CLEAN_SUITE_SWEEP_WEIGHTS=0.20,0.40,0.60,0.80,1.00 \
  mujoco_mp4/.venv/bin/python reproduction/scripts/run_clean_walk_model_weight_sweep.py
```

第一次 sweep 失败是 runner 把 venv Python symlink `.resolve()` 到 `/usr/bin/python3`，导致 `imageio` 缺失。已修复为不 resolve venv symlink，然后重跑成功。

## Results

新增结果目录：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/`
- `/mnt/infini-data/test/BeyondMimic/logs/mujoco/clean_walk_control_suite_sweep/`

聚合文件：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.md`

核心结论：

- `model_target_weight=0.20/0.40/0.60/0.80`：未触发 fall proxy，但 root height 随权重升高下降，root error 增大。
- `model_target_weight=1.00`：纯模型目标不稳定。
- `teacher_policy_action_control`：`fall_proxy_count=14`，root height min `0.4322 m`。
- `diffusion_denoised_latent_action_control`：`fall_proxy_count=14`，root height min `0.4346 m`。
- `guided_latent_action_control`：`fall_proxy_count=10`，root height min `0.4424 m`。
- `vae_reconstructed_action_control`：fall proxy 为 0，但 root height min 也只有 `0.4968 m`，仍然不是正常 paper-level walking。

## Verification

本轮后续运行项目标准验证链：

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

- 纯 learned MuJoCo walking control 仍失败：`model_target_weight=1.0` 下 teacher/diffusion/guided 都触发 unstable。
- 最新 clean walk learned variants 只能写成 reference-anchored diagnostic，不能写成纯 teacher/VAE/diffusion/guidance 成功。
- 根因仍指向 Stage-1 teacher 质量不足、IsaacLab-to-MuJoCo obs/action adapter fidelity 不足，以及下游 VAE/diffusion 继承弱 teacher action distribution。

## Effect on English Reading Report

报告中应把当前 MuJoCo 视频分成两类：

- 可展示资产：reference-anchored clean walk videos，用于说明本地 MuJoCo/G1/PD 渲染链路和诊断流程已经打通。
- 失败证据：pure model target sweep，说明项目尚未达到 paper-level closed-loop learned control。

## Next Step

下一步应优先修 Stage-1 teacher 和官方 deployment contract：

1. 对齐 `whole_body_tracking` / `motion_tracking_controller` 的 ONNX metadata、joint order、default pose、PD gain、action scale。
2. 用官方或更接近官方的 MuJoCo controller 跑 teacher policy，而不是当前 approximate adapter。
3. teacher 稳定后再重新采集 state-action rollout，重训 VAE/diffusion/guidance。

## Git Commit

Pending.
