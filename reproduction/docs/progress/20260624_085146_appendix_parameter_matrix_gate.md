# Progress Update

## Goal

本轮目标是继续执行“先确认 BeyondMimic 论文公式、附录参数、官方代码和本地实现无误，再允许长训练”的硬约束。没有启动新的 teacher/VAE/diffusion 长训练，也没有生成成功视频。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/results.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_appendix_parameter_matrix_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_085146_appendix_parameter_matrix_gate.md`

## Commands Run

```bash
python3 reproduction/scripts/beyondmimic_appendix_parameter_matrix_audit.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 -m py_compile reproduction/scripts/beyondmimic_appendix_parameter_matrix_audit.py reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

日志保存在：

```text
/mnt/infini-data/test/BeyondMimic/logs/verification/20260624_appendix_parameter_matrix_gate_verification.log
/mnt/infini-data/test/BeyondMimic/logs/verification/20260624_appendix_parameter_matrix_gate_standard_verification.log
```

## Results

新增 appendix parameter matrix 审计，把论文/附录和当前代码逐项对齐成 14 行机器可读矩阵。当前结果：

- `pass`: 3
- `partial`: 3
- `blocked`: 8

明确通过的部分：

- Stage-1 policy observation vector 对齐官方 public whole_body_tracking。
- Stage-1 reward table S1 对齐官方 public whole_body_tracking。
- PPO hyperparameters table S4 对齐官方 public whole_body_tracking。

仍阻塞的关键项：

- MuJoCo action ctrlrange 会裁剪/扭曲 paper 中“不按运动学关节限位裁剪”的 normalized PD setpoint 语义。
- native MuJoCo observation adapter 未数值验证到 IsaacLab observation_manager 或 motion_tracking_controller。
- teacher quality 仍 failed，不能从弱 teacher 继续训练 VAE/diffusion。
- 当前 state-latent 数据不是已验证的 fresh hybrid character-yaw-centric 数据集。
- VAE 虽已有 paper-contract 接口，但 source teacher quality 未过。
- diffusion Transformer 有代码契约，但未在 accepted teacher/VAE 数据上完成 full training。
- guidance 仍是 offline/proxy，不是 receding-horizon closed-loop physics result。
- 当前 MuJoCo 视频不能作为 success video。

## Verification

本轮新增审计通过语法检查和单脚本运行。标准 8 个验证脚本最终全部通过：

- `artifact_manifest.py`: ok，artifacts=1872
- `paper_vs_reproduction_comparison.py`: ok
- `final_reproduction_report.py`: ok
- `completion_matrix_status_audit.py`: ok，rows=212
- `verification_command_syntax_audit.py`: ok，failed=0
- `verification_command_script_manifest.py`: ok，scripts=199
- `verification_command_coverage_audit.py`: ok，commands=207
- `reproduction_master_audit.py`: ok，artifact_count=421

中间第一次 master audit 因 pretraining hard gate 行数从 8 增至 9 被旧规则误判失败，已同步更新 master audit 规则并重跑通过。

## Failed / Blocked Items

本轮没有训练失败，因为没有启动训练。当前 blocker 是设计性 hard gate：

- `blocked_appendix_parameter_matrix_has_required_fixes`
- `blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`

这说明当前前倾站姿/动作弱的问题不能靠继续训练下游 VAE/diffusion 解决；必须先修 teacher quality、native adapter 和数据生成契约。

## Effect on English Reading Report

这个矩阵可以作为阅读报告中“我没有把失败视频说成成功复现”的审计证据，也能解释为什么 token-level denoising MSE 不等于机器人闭环控制成功。

## Next Step

下一步应该做 numeric IsaacLab-vs-MuJoCo observation parity probe，并修复 MuJoCo action ctrlrange/action-scale 语义。只有 native adapter 和 teacher quality gate 通过后，才应该重新采集 teacher rollout、训练 paper-contract VAE 和 diffusion。

## Git Commit

本 progress 文件将随本轮 appendix parameter matrix gate 审计一起提交。最终 commit hash 见本轮对用户回复。
