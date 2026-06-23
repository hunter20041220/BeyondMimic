# Progress Update

## Goal

排查当前 MuJoCo 运动控制视频“机器人站不稳、效果很差”的真实原因，区分 reference motion、MuJoCo 渲染链路、Stage-1 teacher 质量、IsaacLab-to-MuJoCo obs/action adapter、VAE/diffusion/guidance downstream 闭环是否分别有问题。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/exporter.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/w100/*/*_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/*checkpoint_sweep*.json`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_clean_walk_candidate_chain_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/report/failure_analysis.md`
- Generated/updated audit outputs under `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/`, `/mnt/infini-data/test/BeyondMimic/res/final_report/`, `/mnt/infini-data/test/BeyondMimic/res/master_audit/`, `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/`, and `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/`.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/render_clean_walk_mujoco_control_suite.py reproduction/scripts/run_clean_walk_candidate_chain_sweep.py
MUJOCO_GL=egl BM_CLEAN_SUITE_OUT_ROOT=/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_last_action_fix_w100 BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT=1.0 mujoco_mp4/.venv/bin/python reproduction/scripts/render_clean_walk_mujoco_control_suite.py
MUJOCO_GL=egl BM_CLEAN_CANDIDATE_WEIGHT=1.0 mujoco_mp4/.venv/bin/python reproduction/scripts/run_clean_walk_candidate_chain_sweep.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

核心发现：

- `reference_action_control` 在同一段 `lafan1_walk1_subject1` clean walk 上稳定，15 秒内 `fall_proxy_count=0`，说明 reference motion 和 MuJoCo rendering/PD loop 不是主因。
- 原默认 `stage1_multisource` pure target 仍失败：teacher `fall_proxy_count=14`，root height min `0.4322 m`。
- `paper_contract` pure target 仍失败：teacher 不倒但明显蹲低，diffusion 触发 fall proxy。
- `official_importer_export_scaled_ppo` pure target 是当前最佳本地候选：teacher/VAE/diffusion/guided 均 `fall_proxy_count=0`，teacher root height min `0.7001 m`。
- `official_importer_export_full_bundle` 也能通过本地 gate，score 略低于 scaled-PPO。

新增关键结果路径：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_last_action_fix_w100/`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_candidate_chain_sweep/clean_walk_candidate_chain_sweep_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_candidate_chain_sweep/clean_walk_candidate_chain_sweep_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_candidate_chain_sweep/clean_walk_candidate_chain_sweep_summary.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_candidate_chain_sweep/official_importer_export_scaled_ppo_w100/`

## Verification

通过：

- `artifact_manifest.py`: `ok`
- `paper_vs_reproduction_comparison.py`: `ok`
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`
- `verification_command_syntax_audit.py`: `ok`
- `verification_command_script_manifest.py`: `ok`
- `verification_command_coverage_audit.py`: `ok`
- `reproduction_master_audit.py`: `ok`

## Failed / Blocked Items

- 默认 `stage1_multisource` teacher 仍不是高质量 MuJoCo pure-control teacher。
- `paper_contract` 链路仍不适合当前 clean walk pure target 展示。
- scaled-PPO/full-bundle 视频仍是 local MuJoCo/root-assist evidence，不是官方 IsaacLab rollout，不是真实机器人，不是 paper-level Fig.5/Fig.6。
- 当前动作视觉仍偏前倾、偏僵，不能声称达到 BeyondMimic 论文视频质量。

## Effect on English Reading Report

这轮为报告提供了一个更诚实的结论：坏视频不是 MuJoCo 渲染失败或 LAFAN1 reference 不可用，而是 Stage-1 teacher quality、adapter fidelity 和 downstream closed-loop obs contract 的组合问题。报告中可以展示 stable reference-PD baseline、scaled-PPO local MuJoCo pure-target candidate，以及明确说明为什么这些仍低于 paper-level reproduction。

## Next Step

优先使用 `official_importer_export_scaled_ppo` 或 `official_importer_export_full_bundle` 作为当前报告展示链路，同时继续训练/评估更强 Stage-1 teacher。若要进一步改善视频质量，应优先在 IsaacLab 官方 environment 中验证 teacher rollout，再导出 ONNX metadata 并对齐 MuJoCo controller，而不是继续用弱 teacher 训练下游 VAE/diffusion。

## Git Commit

待提交。
