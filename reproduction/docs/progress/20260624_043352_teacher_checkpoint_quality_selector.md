# Progress Update

## Goal

在继续 teacher/VAE/diffusion 训练或生成“成功视频”之前，先把现有 Stage 1 PPO teacher checkpoint 统一做质量选择，避免把弱 teacher、跳变 rollout 或缺少连续性证据的 checkpoint 继续喂给 VAE/diffusion。

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_041621_formula_parameter_trace_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/paper_contract_best_teacher.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json`
- `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/resource_adjusted_ppo_eval_20260619_191356_seed20260671/eval_metrics.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/iter_29999/iter_29999_eval_metrics.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.json`

## Files Modified

- 新增 `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_teacher_checkpoint_quality_selector.py`
- 新增 `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260624_043352_teacher_checkpoint_quality_selector.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/stage1_teacher_checkpoint_quality_selector.py
python3 reproduction/scripts/stage1_teacher_checkpoint_quality_selector.py
python3 - <<'PY'
import json
p='res/tracking/stage1_teacher_checkpoint_quality_selector/stage1_teacher_checkpoint_quality_selector.json'
d=json.load(open(p))
print('status', d['status'])
print('candidate_count', d['candidate_count'])
print('decision_counts', d['decision_counts'])
print('category_counts', d['category_counts'])
print('best', {k:d['best_ranked_candidate'].get(k) for k in ['category','iteration','decision','local_non_timeout_done_rate','error_body_pos_mean','error_joint_pos_mean','checkpoint','blockers']})
print('category_best')
for k,v in sorted(d['category_best_candidates'].items()):
    print(k, v.get('iteration'), v.get('decision'), v.get('local_non_timeout_done_rate'), v.get('error_body_pos_mean'), v.get('error_joint_pos_mean'))
PY
```

## Results

新增 teacher checkpoint 质量选择器，扫描已有 tracking PPO eval / checkpoint sweep / best teacher JSON，只做离线审计，不启动训练或仿真。

新增结果：

- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_teacher_checkpoint_quality_selector/stage1_teacher_checkpoint_quality_selector.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_teacher_checkpoint_quality_selector/stage1_teacher_checkpoint_quality_selector.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_teacher_checkpoint_quality_selector/stage1_teacher_checkpoint_quality_selector.md`

核心结论：

- selector status: `blocked_no_downstream_ready_teacher_checkpoint`
- 去重后候选数: `227`
- `candidate_teacher_usable`: `0`
- `downstream_teacher_ready`: `0`
- 其中 `not_ready_for_vae_diffusion_due_to_quality_gate`: `223`
- `insufficient_evidence_do_not_use`: `4`

各家族最佳候选：

- `other_tracking_eval` iter `299`: done `0.08585`, body error `0.22126`, joint error `1.50753`
- `official_importer_export_paper_contract_4_7_gpu` iter `29500`: done `0.15437`, body error `0.25343`, joint error `1.72818`
- `stage1_multisource_5_6_gpu` iter `29999`: done `0.19414`, body error `1.00950`, joint error `1.67395`
- `fk_repaired_robot_order_diagnostic` iter `999`: done `0.11094`, body error `0.36211`, joint error `2.58015`
- `legacy_scaled_ppo_diagnostic` iter `999`: done `0.99786`, body error `0.59730`, joint error `0.91730`

当前本地最靠前候选仍未过门槛，blockers 包括：

- `done_rate_above_downstream_threshold`
- `error_body_pos_mean_above_threshold`
- `error_joint_pos_mean_above_threshold`
- `error_anchor_pos_mean_above_threshold`

## Verification

- `py_compile` 通过。
- selector 运行通过并生成 JSON/TSV/MD。
- 结论与此前视频现象一致：当前 teacher 质量不足以支撑“纯 learned controller 成功视频”或可靠 VAE/diffusion 数据采集。

## Failed / Blocked Items

- 4/7 paper-contract PPO 的 best checkpoint 不是 last，而是 iter `29500`；但仍未过 downstream readiness。
- 5/6 multi-source PPO 的 best checkpoint 是 iter `29999`；但 done/body/joint error 更差，不能直接进入 VAE/diffusion。
- 当前不得用这些 teacher 采集最终 DAgger/VAE/diffusion 数据，也不得把基于它们的视频写成成功复现。

## Effect on English Reading Report

这一轮给报告提供了一个很重要的负结果：失败不是“视频脚本画得不好”，而是 Stage 1 teacher checkpoint 的 closed-loop tracking 质量没有达到下游生成模型所需门槛。报告里可以诚实说明我们已经训练并评估了多条 teacher 线，但当前公开资源本地 teacher 仍未达到 paper-level，也不适合作为最终 diffusion 数据来源。

## Next Step

1. 继续按论文/官方代码对齐 Stage 1 teacher：优先排查 reward/termination/curriculum/adaptive sampling、motion FK/body order、PD/action scale、default joint offset 和 IsaacLab-to-MuJoCo adapter。
2. 若要再训练，不能盲目拉长；必须先基于 selector 的失败项决定修哪个门槛，例如 body/joint error 和 done rate。
3. 只有 selector 出现 `ready_for_continuous_teacher_rollout_collection` 后，才进入 continuous no-jump teacher rollout -> paper-contract VAE -> state-latent Transformer diffusion。

## Git Commit

本进度文件创建时尚未 commit。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。
