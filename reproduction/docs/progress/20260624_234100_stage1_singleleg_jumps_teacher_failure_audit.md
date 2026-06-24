# Progress Update

## Goal

重新审查 BeyondMimic Stage-1 teacher 失败：确认 single-leg 和 jumps1 源动作是否真的包含目标姿态，复核论文/官方代码契约，检查当前 4/7 与 5/6 PPO teacher 训练是否已学到目标姿态，并阻止从弱 teacher 继续生成 VAE/diffusion/guidance 成功视频。

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
- `reproduction/paper/source/tex/method.tex`
- `reproduction/paper/source/root.tex`
- `reproduction/paper/source/tex/results.tex`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`

## Files Modified

- `reproduction/scripts/stage1_singleleg_jumps_motion_teacher_failure_audit.py`
- `reproduction/scripts/stage1_reference_action_scale_feasibility_audit.py`
- `reproduction/docs/progress/20260624_234100_stage1_singleleg_jumps_teacher_failure_audit.md`

## Commands Run

- `ps -eo pid,ppid,stat,etime,cmd | rg ...`
- `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits`
- `find res/runs/... -name 'model_*.pt'`
- `tail -n 120 logs/tracking_hub_singleleg_paper_contract_ppo_training_run_rootxy0_refresh_env16384_20260624_224800/tracking_g1_resource_adjusted_ppo_training_run.log`
- `python3 -m py_compile reproduction/scripts/stage1_singleleg_jumps_motion_teacher_failure_audit.py reproduction/scripts/stage1_reference_action_scale_feasibility_audit.py`
- `python3 reproduction/scripts/stage1_singleleg_jumps_motion_teacher_failure_audit.py`
- `python3 reproduction/scripts/stage1_reference_action_scale_feasibility_audit.py`
- `BM_REFRESH_MOTION_COMMAND_TARGETS=1 ... tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py` for `model_750.pt`
- `BM_REFRESH_MOTION_COMMAND_TARGETS=1 ... tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py` for `model_1000.pt` started after model appeared

## Results

- 4/7 strict single-leg training remains alive; not interrupted.
- 5/6 reset-refresh single-leg training remains alive; not interrupted.
- Latest observed 5/6 training around iteration 871/3000: mean reward about `0.20-0.22`, episode length about `5`, `ee_body_pos` termination remains around `3200`, body error about `0.129`, joint error about `1.16`; this is not a good convergence signal.
- `model_750.pt` quality gate failed: reward `0.040856`, body error `0.110828`, joint error `0.842069`, non-timeout done rate `0.347290`.
- `model_1000.pt` quality gate failed: reward `0.040802`, body error `0.110633`, joint error `0.847404`, non-timeout done rate `0.337624`.
- Single-leg source motion is not empty: right ankle target reaches about `0.372 m`, ankle height difference absolute max about `0.319 m`.
- `jumps1_subject1` source motion is not empty: endpoint z variation reaches roughly `0.55 m` in the full LAFAN1 motion.
- Static action-scale feasibility audit shows high saturation pressure under official `theta_sp = theta0 + action_scale * action`:
  - `hub_singleleg_rootxy0`: all-action p95 `4.661`, legs p95 `1.898`, wrists p95 `9.592`, leg fraction `|a|>1` about `0.328`.
  - `official_short_walk1_subject1`: all-action p95 `9.239`, legs p95 `2.403`, wrists p95 `12.479`.
  - `official_short_jumps1_subject1`: all-action p95 `8.788`, legs p95 `2.518`, wrists p95 `12.659`.
  - `lafan1_jumps1_subject1_full`: all-action p95 `7.149`, legs p95 `2.886`, wrists p95 `12.184`.

## Verification

Passed:

- Python compile for both new audit scripts.
- `stage1_singleleg_jumps_motion_teacher_failure_audit.py` completed and wrote JSON/TSV/MD.
- `stage1_reference_action_scale_feasibility_audit.py` completed and wrote JSON/TSV/MD.
- `model_750.pt` eval completed and failed the screening gate as expected.

Pending this progress note:

- Full artifact/master verification will be run after this progress note is finalized.

## Failed / Blocked Items

- Stage-1 single-leg teacher is still blocked: current checkpoints do not pass quality gate.
- Downstream VAE, diffusion, guidance, and final single-leg success videos remain blocked because the teacher is weak.
- Current failure is not explained by missing target motion. The inspected source motions contain the relevant single-leg and jump endpoint targets.
- The likely local training bottleneck is action-scale/default-pose saturation pressure plus strict endpoint early termination. This requires teacher curriculum or action/termination-contract diagnostics before downstream work.

## Effect on English Reading Report

This gives a precise and honest explanation for the poor videos: the project has not yet obtained a credible Stage-1 tracking teacher for Single Leg Balance or jumps1. It also adds useful evidence that the failure is not just a bad visualization script or an empty motion file.

## Next Step

1. Finish and inspect `model_1000.pt` eval.
2. If it fails, run a focused teacher repair line only after resources are free: relaxed endpoint curriculum or ankle-only endpoint diagnostic followed by strict evaluator.
3. Use `walk1_subject1` or `squat_18` only as easier teacher pipeline probes; they must not replace the required Single Leg Balance and `jumps1_subject1` targets.
4. Do not generate VAE/diffusion/guidance success videos until a teacher checkpoint passes a continuous rollout quality gate.

## Git Commit

Pending. Large checkpoint/video/run artifacts must not be committed.

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
