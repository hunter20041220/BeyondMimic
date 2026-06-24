# Progress Update

## Goal

继续执行“训练前先审公式、参数、材料、PD/action scale、adapter 合同”的硬门控。本轮不启动新的 teacher/VAE/diffusion 训练，也不生成新的成功视频；只修正 MuJoCo 控制链审计，避免把当前前倾/站不稳视频误判为模型成功。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_native_action_adapter_contract.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_control_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_native_action_adapter_contract.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_control_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml`

Regenerated audit outputs:

- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/mujoco_native_action_adapter_contract.py reproduction/scripts/mujoco_control_contract_audit.py mujoco_mp4/scripts/mujoco_pd_control_video.py reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
python3 reproduction/scripts/mujoco_native_action_adapter_contract.py
python3 reproduction/scripts/mujoco_control_contract_audit.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
python3 reproduction/scripts/beyondmimic_pretraining_hard_gate_audit.py
mujoco_mp4/.venv/bin/python - <<'PY'
import sys, xml.etree.ElementTree as ET
from pathlib import Path
root=Path('/mnt/infini-data/test/BeyondMimic')
sys.path.insert(0, str(root/'mujoco_mp4/scripts'))
from mujoco_pd_control_video import load_action_rows, patch_joints_and_actuators
patch_joints_and_actuators(
    root/'mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml',
    root/'mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_clean_walk_control_suite_pd.xml',
    load_action_rows(),
)
PY
```

## Results

- MuJoCo native action adapter 现在明确分成两层：
  - formula gate: ready，`theta_sp = theta0 + alpha * normalized_action`、joint order、default pose、29D action scale 通过；
  - rollout gate: blocked，因为 MuJoCo ctrlrange 会截断合法 `+-1` normalized action setpoint。
- 具体截断关节：
  - `left_ankle_roll_joint`: raw setpoint `[-0.438577, 0.438577]` rad，MuJoCo ctrlrange `[-0.261800, 0.261800]` rad，最大超出 `0.176777` rad。
  - `right_ankle_roll_joint`: raw setpoint `[-0.438577, 0.438577]` rad，MuJoCo ctrlrange `[-0.261800, 0.261800]` rad，最大超出 `0.176777` rad。
- MuJoCo PD XML 的 nominal floor friction 已从 `0.6` 修到 `1.0`，对齐官方 IsaacLab flat terrain 的 nominal friction。
- 审计仍保留 material randomization blocker：IsaacLab 训练时 robot material 随机化 `static_friction_range=(0.3,1.6)`、`dynamic_friction_range=(0.3,1.2)`、`restitution_range=(0.0,0.5)`，MuJoCo 视频脚本没有复现这个训练随机化。

## Verification

本轮已通过：

- `py_compile` 对 6 个相关脚本通过。
- `mujoco_native_action_adapter_contract.py` 运行通过，状态为 `blocked_native_action_adapter_ctrlrange_rollout_gate_formula_ready`。
- `mujoco_control_contract_audit.py` 运行通过，状态为 `blocked_mujoco_control_semantics_not_native_policy_control`。
- `beyondmimic_model_chain_paper_contract_audit.py` 运行通过，状态为 `blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready`。
- `beyondmimic_code_formula_appendix_contract_audit.py` 运行通过，状态为 `blocked_code_formula_appendix_contract_has_required_fixes_before_training`。
- `beyondmimic_pretraining_hard_gate_audit.py` 运行通过，状态为 `blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`。

## Failed / Blocked Items

- 当前不能把任何 existing MuJoCo teacher/VAE/diffusion video 当作成功视频。
- Native MuJoCo observation adapter 仍未和 IsaacLab observation manager / deployment controller 做数值对齐。
- Native action adapter 仍被 ankle roll ctrlrange 截断阻塞。
- 当前视频脚本仍包含 root-assist / absolute target / IK target 语义，不能证明 PPO/VAE/diffusion 真实闭环控制成功。
- 当前 teacher quality gate 未过，不能从 weak teacher 继续训练下游 VAE/diffusion 成功链。

## Effect on English Reading Report

这轮为报告提供了一个更可信的失败归因：当前视频差不只是“训练不够”，还包含控制部署合同问题。报告里可以写清楚：

1. 论文 action 公式和官方 action scale 已被复核；
2. MuJoCo nominal floor friction 已做基本对齐；
3. 但 ankle roll ctrlrange、native obs adapter、root-assist/absolute-target 视频语义仍阻止 paper-level 或 success-video claim；
4. 因此后续训练必须先保证 teacher quality 和 adapter contract，而不是拿当前视频效果强行包装。

## Next Step

下一步应修 native observation/action rollout adapter：用同一 reset state、motion time step、last action，对比 MuJoCo 160D observation builder 和 IsaacLab observation manager 输出；同时决定 ankle roll ctrlrange 应依据官方 IsaacLab articulation limit、MuJoCo asset physical limit，还是部署 controller metadata 做一致化。

## Git Commit

待本轮标准验证完成后再提交。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
