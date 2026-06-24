# 单动作 Teacher 质量硬门控

生成时间：`2026-06-24T11:26:49.354068+00:00`

## 结论

当前 `jumps1_subject1` 和 Short Sequence `Single Leg Balance` 的 teacher 证据**没有通过**进入下游 VAE / diffusion / guidance 训练所需的质量门控。

Reference replay / reference action-control baseline 仍然可以用于展示源动作，但它们不是 learned control。

## 门控阈值

| 指标 | 要求 |
|---|---:|
| 平均 reward | >= 0.1 |
| 平均 body position error | <= 0.25 m |
| 平均 joint position error | <= 1.0 rad |
| non-timeout done rate | <= 0.05 |

## 证据表

| 动作 | 来源 | Reward | Body err | Joint err | Done rate | Teacher 通过 | Claim level |
|---|---|---:|---:|---:|---:|---|---|
| paper_contract_public_bundle_best_teacher | official_importer_export_paper_contract_checkpoint_sweep | 0.0210356 | 0.253434 | 1.72818 | 0.154369 | False | local_best_candidate_teacher_screening_not_paper_level |
| stage1_multisource_best_teacher | stage1_multisource_checkpoint_sweep | 0.0241314 | 1.0095 | 1.67395 | 0.194137 | False | local_multisource_teacher_screening_not_paper_level |
| hub_singleleg_video_single_leg_stand_1 | hub_singleleg_single_motion_teacher_eval | 0.0411416 | 0.166332 | 0.876853 | 0.27933 | False | local_single_motion_teacher_quality_screening |
| lafan1_jumps1_subject1 | robot_order_fk_repaired_single_motion_task_eval | 0.0184674 | 0.198242 | 1.46495 | 0.083612 | False | local_single_motion_task_eval_screening_not_trained_success |
| lafan1_jumps1_subject1 | resource_adjusted_single_motion_task_eval | 0.0279945 | 0.106679 | 1.33381 | 0.0702341 | False | local_single_motion_task_eval_screening_not_trained_success |
| lafan1_jumps1_subject1 | stable_dynamic_reference_action_baseline | n/a | n/a | n/a | n/a | False | local_mujoco_reference_action_baseline_not_learned_control |
| hub_singleleg_video_single_leg_stand_1 | source_singleleg_reference_replay | n/a | n/a | n/a | n/a | False | source_reference_replay_not_learned_control |

## 解释

- `paper_contract_public_bundle_best_teacher`：reward 约 `0.021`，done rate 约 `0.154`，不能作为可靠 teacher。
- `stage1_multisource_best_teacher`：reward 约 `0.024`，body error 约 `1.01 m`，不应继续喂给下游模型。
- `hub_singleleg_video_single_leg_stand_1`：这是一次真实的本地单动作 teacher 尝试，但主要因为 done rate 太高而失败。
- `jumps1_subject1`：reference baseline 有展示价值，但现有 task/eval 指标不能证明 learned tracking 成功。

## 对后续流程的影响

- 当前 teacher 证据不能解锁 VAE / diffusion / guidance 的成功视频生成。
- 下一步应该先做 corrective single-motion Stage-1 teacher training/evaluation，而不是继续长时间训练下游模型。
- 本审计不标记 BeyondMimic 复现目标完成。
