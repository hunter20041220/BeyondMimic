# Progress Update

## Goal

本轮目标是先把 LAFAN1 `jumps1_subject1` 的原始动作放进 MuJoCo，生成可审计的 source/reference baseline，然后再确认训练前公式/参数 gate 仍然阻止 teacher/RL、VAE、diffusion、guidance 长训练。当前不启动新的训练。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/jumps1_subject1.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/motions/lafan1_jumps1_subject1/motion.npz`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/jumps1_subject1_mujoco_baseline_audit.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/official_mp4/scripts/render_official_g1_csv_replay.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/lafan1_jumps1_subject1_mujoco_clean_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_183424_lafan1_jumps1_mujoco_baseline.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py
MUJOCO_GL=osmesa BM_JUMPS1_WINDOW=high_dynamic_52s_67s BM_JUMPS1_WIDTH=640 BM_JUMPS1_HEIGHT=360 mujoco_mp4/.venv/bin/python reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py
MUJOCO_GL=osmesa BM_JUMPS1_WINDOW=stable_dynamic_164s_179s BM_JUMPS1_WIDTH=640 BM_JUMPS1_HEIGHT=360 mujoco_mp4/.venv/bin/python reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py
MUJOCO_GL=osmesa BM_JUMPS1_WINDOW=stable_dynamic_164s_179s BM_JUMPS1_RECENTER_ROOT_XY=1 BM_JUMPS1_WIDTH=640 BM_JUMPS1_HEIGHT=360 mujoco_mp4/.venv/bin/python reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py
python3 reproduction/scripts/lafan1_jumps1_subject1_mujoco_clean_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
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

原始 LAFAN1 文件确认：

- 原始 CSV：`/mnt/infini-data/test/BeyondMimic/download/official/LAFAN1_Retargeting_Dataset/g1/jumps1_subject1.csv`
- 形状：`7334 x 36`
- 内容：root xyz、root quaternion xyzw、29 个 G1 joint positions
- 全长约 `244.47 s`，root z 范围 `0.555741 - 0.912012 m`

本轮生成两个连续 15 秒窗口：

- `high_dynamic_52s_67s`：root z range `0.356271 m`，动作更像跳跃，但 `reference_action_control` 出现 `fall_proxy_count=29` 和 MuJoCo QACC instability warning，因此只保留为 diagnostic。
- `stable_dynamic_164s_179s`：root z range `0.177240 m`，`reference_action_control` 通过，作为当前推荐报告窗口。

推荐窗口输出：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/original_csv_reference_replay/original_csv_reference_replay.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/reference_action_control/reference_action_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/lafan1_jumps1_subject1_mujoco_summary.json`

推荐窗口指标：

- frames：`450`
- duration：`15.0 s`
- reference_action_control：`mj_step=true`
- writes_qpos_each_frame：`false`
- root assist：`true`
- fall_proxy_count：`0`
- mean joint error：`0.151691 rad`
- root height range：`0.672843 - 0.749000 m`
- ffprobe：两个 MP4 均为 `640 x 360`、`450` frames、`30 FPS`

## Verification

通过：

- artifact manifest：`ok`，`1935` artifacts
- paper-vs-reproduction comparison：`ok`
- final reproduction report：`ok`
- completion matrix status audit：`ok`
- verification command syntax audit：`ok`
- verification command script manifest：`ok`
- verification command coverage audit：`ok`
- reproduction master audit：`ok`

## Failed / Blocked Items

- `high_dynamic_52s_67s` 的 `reference_action_control` 不是成功控制 baseline：`fall_proxy_count=29`，日志记录 MuJoCo `QACC` instability warning。
- 当前成功的是 `stable_dynamic_164s_179s` 的 source/reference baseline，不是 teacher/RL、VAE、diffusion 或 guidance 成功。
- `beyondmimic_pretraining_hard_gate_audit.py` 仍为 `blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`。
- `beyondmimic_code_formula_appendix_contract_audit.py` 仍为 `blocked_code_formula_appendix_contract_has_required_fixes_before_training`。
- 因此当前仍不得启动 downstream VAE/diffusion/guidance 长训练，也不得生成“最终成功单脚站立/跳跃模型链”文件夹。

## Effect on English Reading Report

本轮为报告增加了一个更诚实的 LAFAN1 `jumps1_subject1` MuJoCo baseline：

1. 证明原始 Unitree-retargeted LAFAN1 CSV 可以在 MuJoCo G1 mesh 中展示；
2. 区分 kinematic source replay 与 `mj_step` reference_action_control；
3. 记录高动态窗口失败与稳定窗口成功，避免把困难动作控制失败藏掉；
4. 为后续 teacher/RL、VAE、diffusion、guidance 章节提供“先有 source/reference baseline，再谈 learned control”的证据链。

## Next Step

下一步不应直接训练 VAE/diffusion。应先处理训练前 gate：

- 继续修 teacher quality；
- 重新采集包含 raw root/body world-state 的 accepted teacher rollouts；
- 用论文 99-D hybrid state / 163-D emphasis projection 重建 state-latent dataset；
- 修 MuJoCo native obs/action adapter 与 no-root-assist gate；
- 只有这些 gate 过了，再进行 teacher/VAE/diffusion/guidance 训练和视频生成。

## Git Commit

待本轮验证和 staged 文件检查后提交。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
