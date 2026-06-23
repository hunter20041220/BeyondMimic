# Progress Update

## Goal

训练前重新审查 BeyondMimic 论文公式、附录参数、官方 `whole_body_tracking` 代码和本地 VAE/diffusion/guidance/MuJoCo 链路，先修确定的公式级错配，再明确哪些项目仍然阻塞长训练。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/guidance/costs.py`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
- `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/guidance/costs.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py reproduction/src/beyondmimic_reimpl/guidance/costs.py reproduction/tests/test_core_math.py
PYTHONPATH=reproduction/src python3 reproduction/tests/test_core_math.py
python3 reproduction/scripts/beyondmimic_code_formula_appendix_contract_audit.py
```

## Results

- `sdf_barrier()` 已修正为论文 S3 relaxed SDF barrier piecewise 公式。
- `test_sdf_barrier_matches_paper_piecewise_formula()` 已加入核心数学测试。
- paper-contract VAE 已修正为 `[2048, 1024, 512]` hidden dims 和 `15` 步梯度累积。
- 新增审计输出：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json`
  - `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.tsv`
  - `/mnt/infini-data/test/BeyondMimic/res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.md`

## Verification

本轮局部验证已通过：

- `py_compile` 通过。
- `reproduction/tests/test_core_math.py` 通过，`23` tests，`0` failed。
- `beyondmimic_code_formula_appendix_contract_audit.py` 生成 `16` 行审计，状态为 `blocked_code_formula_appendix_contract_has_required_fixes_before_training`。

标准全量验证将在接入 manifest/final report/master audit 后运行。

## Failed / Blocked Items

- `state_latent_uses_hybrid_state=false`：当前 state-latent dataset 仍主要是 `policy_obs + latent`，不是论文 hybrid character-yaw-centric state。
- `guidance_closed_loop_receding_horizon=false`：当前 guidance 仍主要是 offline/proxy，不是 MuJoCo receding-horizon closed-loop。
- `mujoco_native_no_root_assist_success=false`：MuJoCo native obs/action/no-root-assist gate 仍未通过。
- Adaptive sampling 仍需确认：论文附录 look-back `u in {0,1,2}`，官方代码默认 `adaptive_kernel_size=1`，需要训练配置显式处理或记录为官方源码差异。

## Effect on English Reading Report

这轮给报告提供了更可信的“复现失败原因”证据：失败不是简单“训练不够”，而是下游数据表征、VAE rollout collection、closed-loop guidance 和 MuJoCo adapter 仍有明确 contract gaps。也能写入报告作为 reproducibility audit 的亮点。

## Next Step

先修 state-latent hybrid representation 与 MuJoCo native observation/action adapter，再做短 probe。长训练、teacher/VAE/diffusion/guidance 视频生成和最终单脚站立成功文件夹仍然禁止。

## Git Commit

待本轮标准验证通过后提交。
