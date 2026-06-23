# Progress Update

## Goal

先暂停新的 teacher/VAE/diffusion 长训练和“成功视频”生成，重新对照 BeyondMimic 论文公式、补充材料参数、官方 `whole_body_tracking` 代码和本地复现链路，判断当前坏视频到底是训练不足还是代码/数据流不符合论文。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`

## Files Modified

- 新增 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_formula_parameter_trace_audit.py`
- 新增本进度文件 `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_041621_formula_parameter_trace_audit.md`

## Commands Run

```bash
ps -eo pid,ppid,stat,etime,cmd | rg -i 'tracking_hub_singleleg|paper_contract_teacher_rollout_vae|resource_adjusted_ppo|torch.distributed|train_lafan|diffusion|isaac|mujoco' || true
python3 -m py_compile reproduction/scripts/beyondmimic_formula_parameter_trace_audit.py
python3 reproduction/scripts/beyondmimic_formula_parameter_trace_audit.py
python3 - <<'PY'
import json
p='res/audits/formula_parameter_trace_audit/beyondmimic_formula_parameter_trace_audit.json'
d=json.load(open(p))
missing=[]
for row in d['rows']:
    for ev in row['evidence']:
        if ':not_found:' in ev or ':missing' in ev:
            missing.append((row['component'], ev))
print('missing_refs', len(missing))
PY
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

- 当前没有发现正在运行的 teacher/VAE/diffusion/Isaac/MuJoCo 训练进程。
- 生成了公式与参数追踪审计：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/formula_parameter_trace_audit/beyondmimic_formula_parameter_trace_audit.json`
  - `/mnt/infini-data/test/BeyondMimic/res/audits/formula_parameter_trace_audit/beyondmimic_formula_parameter_trace_audit.tsv`
  - `/mnt/infini-data/test/BeyondMimic/res/audits/formula_parameter_trace_audit/beyondmimic_formula_parameter_trace_audit.md`
- 审计状态为 `blocked_formula_parameter_trace_has_required_fixes_before_training`。
- 审计矩阵共 16 行，关键结论是：
  - 官方 Stage 1 `whole_body_tracking` tracking 公式、奖励、PPO、PD/action scale/armature 大体对齐论文。
  - 当前旧 VAE 是 `obs+action -> latent`，不符合论文 `E(psi,e_anchor)`。
  - 当前旧 diffusion 是 MLP over `policy_obs+latent`，不符合论文 state-latent Transformer diffusion。
  - 当前 clean-walk/singleleg 视频链包含 reference-anchor blend、latent 插值 guidance 和 MuJoCo root assist，只能作为诊断，不能作为成功复现。
  - 新 paper-contract VAE 输入形式已修正，但源 teacher rollout 的质量 gate 未过，不能继续拿它生成成功视频。

## Verification

- `py_compile` 通过。
- 审计脚本运行通过。
- 生成 JSON/TSV/MD 均为小型文本产物。
- `missing_refs=0`，所有审计证据行均可追溯到本地文件或现有结果。
- 全局验证命令均通过：
  - `artifact_manifest.py`: `ok`, artifacts `1813`
  - `paper_vs_reproduction_comparison.py`: `ok`
  - `final_reproduction_report.py`: `ok`
  - `completion_matrix_status_audit.py`: `ok`, rows `212`
  - `verification_command_syntax_audit.py`: `ok`, failed `0`
  - `verification_command_script_manifest.py`: `ok`, scripts `199`
  - `verification_command_coverage_audit.py`: `ok`, commands `207`
  - `reproduction_master_audit.py`: `ok`

## Failed / Blocked Items

- 当前不得继续把旧 `resource_adjusted` VAE/diffusion 链作为成功链路。
- 当前不得把 `clean_walk`、`final_clean_walk_six`、`hub_singleleg_full_chain` 这类视频说成 paper-level 或真正学到 reference 姿态。
- 继续训练前必须先决定：
  - Stage 1 是严格跟随公开官方代码，还是 patch 论文补充材料里 adaptive kernel size=3 和 ankle default offset。
  - MuJoCo adapter 必须使用官方导出的 joint order、action scale、kp/kd、default pose、normalizer。
  - VAE/diffusion 数据必须来自低 done-rate、连续 motion-time 的 teacher rollout。

## Effect on English Reading Report

这一轮提供了可以写进报告的“失败原因与科研复现审计”证据：项目不是简单训练失败，而是当前本地模型链中存在 paper-contract 缺口。报告中应明确区分：

- 官方 Stage 1 tracking 代码对齐程度较高；
- 本地旧 VAE/diffusion/guidance 是 debug/近似链；
- MuJoCo 视频目前是诊断可视化，不是完整控制复现；
- 下一轮应先修 teacher quality gate 和模型链数据流，再生成单脚站立成功视频。

## Next Step

1. 先修复/固定 Stage 1 训练配置选择：公开代码原样 vs 论文补充参数 patch。
2. 做 checkpoint quality selector：对已有 teacher checkpoints 跑统一 eval，按 done rate、episode length、body error、joint error、motion-time continuity 选 best，不用 last。
3. 若没有合格 teacher，再启动 paper-contract PPO 训练；训练前设定显存/环境数策略，但不能牺牲 IsaacLab 可启动性。
4. teacher 通过后，再采集连续 rollout，重训 paper-contract VAE 和 state-latent Transformer diffusion。
5. 最后再生成无 root assist、无 reference blend 的 MuJoCo action-control 视频。

## Git Commit

本进度文件创建时尚未 commit。
