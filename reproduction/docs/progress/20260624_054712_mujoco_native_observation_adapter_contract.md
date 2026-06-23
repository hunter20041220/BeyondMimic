# Progress Update

## Goal

审计 MuJoCo native PPO/VAE/diffusion 控制失败的关键接口风险：IsaacLab 训练出的 160 维 policy observation 不能靠任意拼接复现，必须逐项匹配官方 `whole_body_tracking` observation manager、RSL-RL empirical normalizer、motion reference alignment 和官方 `motion_tracking_controller` 的 frame-local 语义。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/exporter.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/motion_tracking_controller/include/motion_tracking_controller/MotionObservation.h`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/motion_tracking_controller/src/MotionCommand.cpp`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/motion_tracking_controller/src/MotionOnnxPolicy.cpp`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_ppo_adapter_gap_audit.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

## Files Modified

- 新增 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_native_observation_adapter_contract.py`
- 更新 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py`
- 更新 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- 新增本 progress 文件

## Commands Run

```bash
python3 reproduction/scripts/mujoco_native_observation_adapter_contract.py
python3 reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 -m py_compile reproduction/scripts/mujoco_native_observation_adapter_contract.py reproduction/scripts/beyondmimic_model_chain_paper_contract_audit.py reproduction/scripts/artifact_manifest.py
```

## Results

新增审计输出：

- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.md`

关键结论：

- 官方 policy observation 顺序和切片为 `command[0:58]`、`motion_anchor_pos_b[58:61]`、`motion_anchor_ori_b[61:67]`、`base_lin_vel[67:70]`、`base_ang_vel[70:73]`、`joint_pos[73:102]`、`joint_vel[102:131]`、`actions[131:160]`。
- 官方 G1 PPO 配置启用 `empirical_normalization=True`。
- 当前 best teacher checkpoint 可用项目环境 `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python` 读取，确认 actor shape 为 `[512,160] -> [29,128]`，且 checkpoint 内存在 `obs_norm_state_dict`。
- 当前 MuJoCo native adapter probe 明确仍是 approximate，不可作为 paper-level native PPO rollout 证据。

## Verification

已完成初步验证：

- 新脚本运行成功。
- 主模型链审计运行成功并接入 `mujoco_native_observation_adapter_gate`。
- `py_compile` 通过。

完整标准验证命令会在本轮后续统一运行。

## Failed / Blocked Items

新增 gate 状态为：

```text
blocked_native_mujoco_observation_adapter_not_validated
```

仍 blocked 的硬检查：

- `native_adapter_validated_against_isaaclab_observation_manager=false`
- `native_adapter_validated_against_deployment_controller=false`
- `native_adapter_has_no_root_assist_rollout_success=false`

也就是说，目前还不能把 IsaacLab PPO checkpoint 直接放进 MuJoCo 并声称 native closed-loop policy control 成功。

## Effect on English Reading Report

这为报告中的 failure analysis 和 reproducibility boundary 提供了清楚证据：当前视频效果差不应被简单写成“训练不够”，更严谨的解释是 Stage-1 teacher quality 与 IsaacLab-to-MuJoCo obs/action adapter fidelity 共同阻塞。尤其要说明：维度正确的 160-D observation 不等于语义正确，official normalizer 和 frame alignment 必须保留。

## Next Step

下一步应先实现可数值对齐的 native observation builder，对同一 reset state、motion time step、last action 同时采样 IsaacLab `observation_manager` 和 MuJoCo adapter 输出，逐 slice 比较误差；通过后再跑 no-root-assist native PPO rollout。

## Git Commit

待本轮完整验证后提交。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
