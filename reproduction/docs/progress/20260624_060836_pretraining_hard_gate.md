# Progress Update

## Goal

本轮目标是把“当前 MuJoCo 控制视频为什么仍然差、还能参考哪些动作、下一步能不能直接继续训练 VAE/diffusion”变成可审计结论。重点不是新增长训练，也不是生成新视频，而是在开始下一轮训练前建立 paper/official-code/teacher-quality/MuJoCo-adapter 的硬门控。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_formula_parameter_trace_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json`

## Files Modified

- 新增 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py`
- 更新 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- 更新 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- 更新 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- 新增本 progress 文件 `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_060836_pretraining_hard_gate.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
python3 - <<'PY'
import json
from pathlib import Path
p=Path('res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json')
d=json.loads(p.read_text())
print('status', d['status'])
print('counts', {k:d.get(k) for k in ['row_count','passed_count','blocked_count','partial_count']})
print('permission', d['permission'])
for row in d['rows']:
    print(row['gate'], row['status'], row['passed'])
PY
```

完整标准验证命令将在本 progress 文件加入 manifest 后统一运行。

## Results

新增训练前硬门控审计：

- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.md`

审计状态：

```text
status=blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes
row_count=8
passed_count=1
blocked_count=7
partial_count=1
```

关键结论：

- 官方 Stage-1 motion tracking 公式和参数已经能追溯到 `whole_body_tracking`：G1 action scale、armature、PD stiffness/damping、observation terms、reward terms、termination、PPO config 均已纳入审计。
- 但当前 teacher quality gate 没过。已有 single-leg teacher eval 记录 `local_non_timeout_done_rate=0.2793295359531773`、`reward_mean=0.04114155067647979`，没有达到 downstream rollout dataset 的可信门槛。
- 当前 VAE paper-contract interface 有纠正版本，但不能继续用失败 teacher 的数据做成功声明。
- 当前 state-latent diffusion 仍缺完整 paper-contract Transformer 长训练；已有结果只能作为代码契约或 offline denoising evidence。
- 当前 classifier guidance 仍主要是 offline guidance，不是 receding-horizon MuJoCo closed-loop guidance success。
- 当前 MuJoCo native action adapter 公式接近官方 `theta_sp = theta0 + action_scale * action`，但还没有通过 no-root-assist closed-loop rollout gate。
- 当前 MuJoCo native observation adapter 尚未和 IsaacLab observation manager / deployment controller 做数值逐 slice 对齐。
- 当前 clean-walk 视频能展示，但仍含 root assist / 本地 surrogate / 非 paper-level 边界，不能作为完整复现。

## 可参考复现的动作

当前更适合先做 reference replay 和低动态 teacher-correction 的动作：

- `walk1_subject1`
- `walk1_subject2`
- `walk1_subject5`
- `walk2_subject1`
- `walk2_subject3`
- `walk2_subject4`
- `walk3_subject1`
- `walk3_subject2`
- `walk3_subject3`
- `walk3_subject4`
- `walk3_subject5`
- `walk4_subject1`
- 少量低幅度 `dance1/dance2`
- `hub_squat_video_squat_4`、`hub_squat_video_squat_18`

更难的展示动作可以保留为后续目标，但不应作为当前 teacher 的首轮成功标准：

- `hub_singleleg_video_single_leg_stand_1`
- `hub_swallow_balance_video_swift0322`
- `zenodo_tkd_skill`
- kick / fight / jump / fall-and-get-up / sprint 类 motion

GRF 文件，例如 `GRF_F_AP_PRO_left.csv`、`walk_bag.csv`，是论文 released-data 曲线/力数据/rosbag 导出的分析数据，不是 29/36 维 generalized-coordinate reference motion，不能直接当作 Stage-1 tracking motion 或 MuJoCo 控制输入。

## Verification

本阶段已完成的局部验证：

- `py_compile` 通过。
- `beyondmimic_pretraining_hard_gate_audit.py` 运行成功。
- 输出 JSON/TSV/MD 均生成。
- `artifact_manifest.py` 已加入新 hard-gate 产物和本 progress 文件。
- `final_reproduction_report.py` 已加入 pre-training hard gate 摘要。
- `reproduction_master_audit.py` 已加入 hard gate JSON 检查。

## Failed / Blocked Items

当前 blocked 的 paper-level / control-level 项目：

- Teacher quality gate 未通过，不能用当前 teacher 采集可信 downstream DAgger/VAE/diffusion 数据。
- MuJoCo native observation adapter 未完成 IsaacLab 对齐。
- MuJoCo native action adapter 尚未证明 no-root-assist closed-loop control。
- VAE paper-contract 训练不能从当前 weak teacher 继续做成功声明。
- State-latent diffusion paper-contract Transformer 还没有完成可报告的 full training。
- Guidance 还不是 MuJoCo receding-horizon closed-loop control。
- 现有 clean-walk 视频可作本地诊断/展示，不是 official IsaacLab rendered rollout，不是真实机器人，也不是 Fig.5/Fig.6 paper-level 结果。

## Effect on English Reading Report

这轮内容会直接进入后续中文/英文报告的 failure analysis 和 reproduction boundary：

- 可以清楚说明“reference replay 能看”和“closed-loop controller 能走”是两件不同的事。
- 可以解释为什么 `reference_action_control` 或 clean-walk 展示视频不等于 teacher/VAE/diffusion 已经成功。
- 可以把下一步技术路线收敛为：先修 Stage-1 teacher 和 MuJoCo obs/action adapter，再重新采集 rollout dataset，然后再训练 VAE/diffusion/guidance。

## Next Step

下一步不建议直接跑下游 VAE/diffusion 长训练。建议顺序：

1. 选择低动态连续 walk motion，例如 `walk1_subject1` 的多个连续窗口、`walk1_subject2`、`walk2/walk3/walk4`。
2. 对同一 motion time step 同时采样 IsaacLab observation manager 和 MuJoCo native observation builder，逐 slice 对齐 160-D obs。
3. 保留官方 `theta_sp = theta0 + action_scale * action` 和官方 normalizer，不用 reference anchor/root assist 当作成功证据。
4. 重新评估 teacher checkpoint，要求 done rate、reward、body/joint error 同时过 gate。
5. teacher 过 gate 后，再采集 state-action rollout dataset，训练 paper-contract VAE 和 state-latent diffusion。

## Git Commit

待完整验证通过后提交。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
