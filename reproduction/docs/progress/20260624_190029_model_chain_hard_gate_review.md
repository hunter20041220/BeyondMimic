# Progress Update

## Goal

重新审查 BeyondMimic teacher/RL、VAE、diffusion、guidance、PD/action scale、armature/material 和 MuJoCo adapter 链条，解释当前单脚站立/走路/跳跃视频为什么退化成前倾站姿，并在继续训练前建立硬门控。

## Files Read

- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/paper/source/tex/results.tex`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
- `res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json`
- `res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
- `res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
- `res/audits/lafan1_jumps1_subject1_mujoco_clean/lafan1_jumps1_subject1_mujoco_clean_audit.json`
- `reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py`
- `reproduction/scripts/lafan1_jumps1_subject1_mujoco_clean_audit.py`

## Files Modified

- `reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py`
- `report/audits/model_chain_hard_gate_review_20260624.md`
- `reproduction/docs/progress/20260624_190029_model_chain_hard_gate_review.md`

## Commands Run

- `find`/`rg`/`sed` commands to inspect paper, official source, current reports, and video summaries.
- No new long training was started.

## Results

- Confirmed official Stage-1 `whole_body_tracking` code matches the main paper contract for observation, reward, action scale, PD gain, armature and PPO configuration.
- Confirmed current project hard gates still block downstream VAE/diffusion/guidance training from current teacher rollout.
- Confirmed MuJoCo native observation adapter is still not validated against IsaacLab, so pure MuJoCo PPO policy rollout remains blocked.
- Fixed a dangerous `jumps1_subject1` default: dynamic root assist is no longer enabled by default, and settle steps default is restored to 40.

## Verification

Verification commands are run after this progress file is written and audit scripts are refreshed.

## Failed / Blocked Items

- Teacher quality is not yet sufficient for downstream rollout dataset collection.
- VAE/diffusion/guidance success videos remain blocked.
- MuJoCo native observation parity is not yet proven.
- Single Leg Balance learned-chain success video does not exist yet.

## Effect on English Reading Report

This update strengthens the report's limitation and methods sections by separating:

- paper/official formula alignment,
- local mechanism implementation,
- actual closed-loop teacher quality,
- MuJoCo adapter trustworthiness,
- and claim boundaries.

## Next Step

Implement and run single-motion teacher quality gates for `jumps1_subject1` and `hub_singleleg_video_single_leg_stand_1`, then build a MuJoCo-vs-IsaacLab observation parity probe before any downstream VAE/diffusion long training.

## Git Commit

Pending after verification.
