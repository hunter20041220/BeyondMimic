# Progress Update

## Goal

本轮目标是把 `jumps1_subject1` 的 MuJoCo 基线证据从“散落的视频文件”整理成可审计结果，并继续执行训练前硬门禁：在 teacher/RL、VAE、diffusion、guidance 的公式/数据契约没有确认之前，不启动新的下游长训，也不把旧的前倾/失败视频解释成论文复现。

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `official_mp4/scripts/render_official_g1_csv_replay.py`
- `mujoco_mp4/scripts/mujoco_reference_replay_video.py`
- `mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`

## Files Modified

- `mujoco_mp4/scripts/mujoco_pd_control_video.py`
  - 增加 `reference_action_control` spec。
  - 增加 `BM_MUJOCO_CONTROL_OUTPUT_GROUP`，避免覆盖旧失败目录。
- `reproduction/scripts/jumps1_subject1_mujoco_baseline_audit.py`
  - 新增 `jumps1_subject1` MuJoCo 基线审计。
- `reproduction/scripts/artifact_manifest.py`
  - 纳入 `jumps1_subject1` 审计脚本和结果。
- `reproduction/scripts/reproduction_master_audit.py`
  - 纳入 `jumps1_subject1` 审计 gate。
- `reproduction/scripts/beyondmimic_training_hard_gate_utils.py`
  - 新增训练前硬门禁工具。
- `reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.py`
- `reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py`
- `reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
  - 加入硬门禁，防止继续用 `policy_obs` 160-D state-latent 数据做下游长训。
- `reproduction/scripts/beyondmimic_training_entrypoint_hard_gate_audit.py`
  - 新增训练入口硬门禁审计。

## Commands Run

```bash
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python official_mp4/scripts/render_official_g1_csv_replay.py \
  --csv download/official/LAFAN1_Retargeting_Dataset/g1/jumps1_subject1.csv \
  --motion-name lafan1_jumps1_subject1_original_csv \
  --frames 300 --fps 50 --width 640 --height 360
```

结果：命令返回码显示 EGL 300 帧尝试在 H20 上 abort；`logs/mujoco/jumps1_subject1/original_csv_replay_300f_egl.log` 被创建但为空，因此审计只把它作为失败尝试记录，不把空日志当作完整错误文本。

```bash
MUJOCO_GL=osmesa mujoco_mp4/.venv/bin/python official_mp4/scripts/render_official_g1_csv_replay.py \
  --csv download/official/LAFAN1_Retargeting_Dataset/g1/jumps1_subject1.csv \
  --motion-name lafan1_jumps1_subject1_original_csv_osmesa \
  --frames 300 --fps 50 --width 640 --height 360
```

```bash
MUJOCO_GL=osmesa \
BM_MUJOCO_MOTION_NPZ=/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz/motions/jumps1_subject1/motion.npz \
BM_MUJOCO_MOTION_NAME=lafan1_jumps1_subject1_fk_repaired_npz_osmesa \
BM_MUJOCO_REPLAY_FRAMES=299 \
BM_MUJOCO_WIDTH=640 BM_MUJOCO_HEIGHT=360 \
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_reference_replay_video.py
```

```bash
MUJOCO_GL=osmesa \
BM_MUJOCO_CONTROL_OUTPUT_GROUP=jumps1_subject1_control_videos_osmesa \
BM_MUJOCO_CONTROL_SPECS=reference_action_control \
BM_MUJOCO_MOTION_NPZ=/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz/motions/jumps1_subject1/motion.npz \
BM_MUJOCO_CONTROL_FRAMES=299 \
BM_MUJOCO_WIDTH=640 BM_MUJOCO_HEIGHT=360 \
mujoco_mp4/.venv/bin/python mujoco_mp4/scripts/mujoco_pd_control_video.py
```

```bash
python3 -m py_compile reproduction/scripts/jumps1_subject1_mujoco_baseline_audit.py
python3 reproduction/scripts/jumps1_subject1_mujoco_baseline_audit.py
```

## Results

新增 `jumps1_subject1` 三条 OSMesa 稳定基线：

1. 原始 LAFAN1 36-column CSV kinematic replay
   - MP4: `official_mp4/videos/lafan1_jumps1_subject1_original_csv_osmesa/lafan1_jumps1_subject1_original_csv_osmesa_mujoco_reference_replay.mp4`
   - 300 frames, 10.0 s, 640x360。
   - claim level: reference replay, not policy control。

2. FK-repaired NPZ kinematic replay
   - MP4: `mujoco_mp4/res/reference_replay/lafan1_jumps1_subject1_fk_repaired_npz_osmesa/reference_replay.mp4`
   - 299 frames, 9.9667 s, 640x360。
   - claim level: local reference replay, not policy control。

3. `reference_action_control`
   - MP4: `mujoco_mp4/res/jumps1_subject1_control_videos_osmesa/reference_action_control/reference_action_control.mp4`
   - 299 frames, 9.9667 s, 640x360。
   - `uses_mj_step=true`, `writes_qpos_each_frame=false`, `uses_29_position_actuators=true`。
   - `fall_proxy_count=0`。
   - `joint_error_abs_mean=0.08817889989735798`。
   - `root_position_error_mean_m=0.04231916106179699`。
   - 注意：使用 `root_assist_enabled=true`，所以只能作为 MuJoCo PD reference-action control baseline，不是无辅助 teacher policy。

新增审计：

- `res/audits/jumps1_subject1_mujoco_baseline/jumps1_subject1_mujoco_baseline_audit.json`
- `res/audits/jumps1_subject1_mujoco_baseline/jumps1_subject1_mujoco_baseline_audit.tsv`
- `res/audits/jumps1_subject1_mujoco_baseline/jumps1_subject1_mujoco_baseline_audit.md`

## Verification

本轮已直接通过：

```bash
python3 -m py_compile reproduction/scripts/jumps1_subject1_mujoco_baseline_audit.py
python3 reproduction/scripts/jumps1_subject1_mujoco_baseline_audit.py
```

输出：`ok_jumps1_subject1_mujoco_baseline_audit`，`pass_count=3/3`。

后续还需要刷新全局 verification 命令：

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

- EGL 300-frame render attempt failed by command return code, but tee log is empty；当前 H20 上 MuJoCo 长视频优先走 `MUJOCO_GL=osmesa`。
- `reference_action_control` 虽然没有摔倒，但带 root assist，不能说明无辅助 humanoid balance policy 成功。
- teacher/RL、VAE、diffusion、guidance 仍 blocked：旧 state-latent dataset 来自 `policy_obs` 160-D，不是论文 99-D hybrid state；旧视频不能作为 teacher/VAE/diffusion 成功证据。
- 尚未重新完整审完论文原文 PDF 公式，因为系统缺少 `pdftotext`；后续需要用 Python PDF reader 或 LaTeX source 做逐条公式/参数审查。

## Effect on English Reading Report

这轮可以支持报告中的一段结论：`jumps1_subject1` 的原始 reference motion 和 FK-repaired reference motion 已经能在 MuJoCo 中稳定渲染，PD reference-action baseline 也能用 `mj_step` 跑满 299 帧且不触发 fall proxy。但它不是 teacher policy、不是 VAE/diffusion/guidance，也不是 IsaacLab rendered MP4 或 real-robot evidence。

## Next Step

1. 用 `walk3_subject3` 或 `walk1_subject1` 做同样的 OSMesa baseline，作为更容易观察稳定步态的参考动作。
2. 用 Python PDF reader / LaTeX source 重做 teacher reward、action scale、PD/armature/material、VAE、diffusion、guidance 的逐条公式-代码-参数对照。
3. 重新采集带 raw root/body state 的连续 teacher rollout，再生成论文 99-D hybrid state + latent 数据集。
4. 只有数据源契约和训练入口硬门禁通过后，才继续 teacher/VAE/diffusion/guidance 长训。

## Git Commit

待本轮验证通过后提交。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
