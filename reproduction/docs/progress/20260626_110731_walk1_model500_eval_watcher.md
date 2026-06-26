# Progress Update

## Goal

继续推进 LAFAN1 `walk1_subject1` 单动作主线。当前 Stage-1 teacher 训练仍在 GPU 5/6 上运行，尚未产生 `model_500.pt`。本轮补充自动 checkpoint watcher，使 `model_500.pt` 一出现就自动在空闲 GPU 4/7 上执行 walk1 质量评估。

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/scripts/eval_walk1_teacher_checkpoint.sh`
- `reproduction/scripts/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.py`
- `res/tracking/stage1_multisource_motion_bundle_robot_joint_order/motions/lafan1_walk1_subject1/motion.npz`

## Files Modified

- `reproduction/scripts/eval_walk1_teacher_checkpoint.sh`
- `reproduction/scripts/watch_and_eval_walk1_teacher_checkpoint.sh`
- `reproduction/docs/progress/20260626_110731_walk1_model500_eval_watcher.md`

## Commands Run

```bash
reproduction/scripts/monitor_walk1_teacher_training.sh lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951
find res/runs/hub_singleleg_paper_contract_ppo_training_lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951 -name 'model_*.pt'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
bash -n reproduction/scripts/watch_and_eval_walk1_teacher_checkpoint.sh
bash -n reproduction/scripts/eval_walk1_teacher_checkpoint.sh
tmux new-session -d -s bm_walk1_eval_watch_model500_20260626_110731 ...
```

## Results

- `walk1` teacher training remains active in tmux session `bm_walk1_teacher_repaired_env40960_iter30000_20260626_104951`.
- GPU 5/6 remain occupied by the training line at about `34 GB/card`.
- Latest observed training progress was around iteration `78/30000`, mean reward about `0.6`, mean episode length about `12-13`. This is still early and not a teacher-quality pass.
- Only `model_0.pt` existed at the time of this update; `model_500.pt` had not appeared.
- Started watcher tmux session `bm_walk1_eval_watch_model500_20260626_110731`.
- Watcher log: `logs/walk1_teacher_eval_watch_lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951_model500.log`.
- Watcher waits for:
  `/mnt/infini-data/test/BeyondMimic/res/runs/hub_singleleg_paper_contract_ppo_training_lafan_walk1_subject1_repaired_env40960_iter30000_20260626_104951/resource_adjusted_ppo_20260626_024951_seed20261051/rank_0/model_500.pt`
- When that checkpoint appears, the watcher will run `eval_walk1_teacher_checkpoint.sh` on GPU `4,7`, leaving GPU 5/6 training undisturbed.

## Verification

- Shell syntax checks passed for both watcher and eval wrapper.
- Watcher tmux session is alive and currently waiting rather than running premature eval.
- GPU 4/7 were confirmed idle when watcher started.
- No downstream VAE/diffusion/guidance run was started because the teacher quality gate has not passed.

## Failed / Blocked Items

- `model_500.pt` is not yet available.
- Teacher quality gate is not yet evaluated for this walk1-only run.
- Teacher rollout dataset, DAgger VAE, state-latent diffusion, and test-time guidance remain blocked until a checkpoint passes quality screening.

## Effect on English Reading Report

This update strengthens the audit trail for the new walk1-only line: it shows that the project is not cherry-picking an old weak teacher or manually selecting a convenient checkpoint, but is waiting for a declared checkpoint and routing it through an explicit quality gate before downstream claims.

## Next Step

Wait for `model_500.pt` to be created and inspect the watcher-triggered eval summary. If it fails, continue training to later checkpoints and evaluate `model_1000.pt`, `model_1500.pt`, etc. Only after teacher quality is acceptable should the project collect walk1 teacher rollout data for VAE training.

## Git Commit

Pending.
