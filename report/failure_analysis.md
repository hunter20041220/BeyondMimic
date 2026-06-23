# 失败分析：为什么当前运动控制视频效果差

## 1. 现象

最新六条 MuJoCo action-control 视频已经保证连续 motion segment，不再是跳变拼接；但机器人仍然不能稳定完成动作，teacher/VAE/diffusion/guided variants 都表现出明显失稳。

## 2. 主要原因排序

### 原因一：当前视频选段/root target 本身不适合作为站立运动展示

本轮新增诊断脚本确认，最新六条视频虽然已经满足 `motion_time_steps` 连续、`done_count=0`，但自动选中的片段来自：

- source motion：`lafan1_walk3_subject4`
- global motion steps：`418177..418474`
- selected root z：min `0.0459` m，mean `0.0512` m，max `0.0544` m
- whole source root z median：`0.7723` m
- selected reward mean：`-0.081968`

也就是说，视频 pipeline 选中了一个接近地面的片段。`reference_action_control` 会直接把这段低 root target 作为 pose replay；后面的 teacher/VAE/diffusion/guided action-control 又共享同一段 root target，并启用了 root assist，因此控制器会围绕一个近地面目标运行。机器人站不稳首先是这个 root target/segment selection 问题，不应该简单归因为“视频渲染不好”。

更重要的是，按旧规则 `length` 优先、再看 `reward_mean`，前 10 个长连续候选大多是近地面或 get-up/fall 片段；按 `>=60` 帧且 root z 正常的门槛，目前候选数为 `0`。这说明当前 teacher rollout 里还没有足够长、稳定、站立高度正常的连续片段可用于 15 秒展示视频。

新增证据：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_diagnosis.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_mujoco_video_failure_candidate_segments.tsv`

### 原因二：Stage 1 teacher 本身很弱

best teacher 指标仍然不好：

- reward mean：`0.0241314`
- body error mean：`1.0095`
- joint error mean：`1.67395`
- non-timeout done rate：`0.194137`

这说明 teacher 还没有学到高质量 motion tracking。后续 VAE/diffusion 学到的是这个弱 teacher 的 action distribution。

### 原因三：rollout 数据带有大量 done/fall 信号

teacher rollout done count：`118220`。这会污染 VAE 和 state-latent diffusion 的训练分布。

### 原因四：离线 MSE 和闭环控制不是一回事

VAE MSE、diffusion MSE 都是离线指标。机器人在 MuJoCo 里会受到接触、动力学、PD 控制、root height、joint limit、action scale 等因素影响。

### 原因五：IsaacLab -> MuJoCo adapter 仍可能有 contract gap

需要继续查：

- joint order
- action scale
- PD gain
- default joint pose
- obs normalization
- last action
- root twist / IMU-like state
- termination semantics
- reference phase 和 motion_time_step

## 3. 当前视频应该怎么表述

正确说法：

```text
These videos are continuous MuJoCo local action-control diagnostics.
They reveal that the current teacher/control chain is still unstable.
```

错误说法：

```text
These videos reproduce BeyondMimic Fig.5/Fig.6.
```

## 4. 结论

下一步不要优先美化视频，也不要继续把当前六条视频当成功结果。正确顺序是：

1. 把旧六条视频标记为 failed diagnostic；
2. 修改 segment selector，加入 root height、reward、stability 门槛；
3. 先生成 root 高度正常的 reference pose replay；
4. 再生成同一片段的 teacher action-control；
5. 如果 root target 正常后 teacher 仍然不稳，再回到 Stage 1 PPO teacher 训练和 MuJoCo adapter。

teacher 稳定之前，VAE/diffusion/guidance 的闭环视频很难好看，也不能作为 paper-level result。

## 5. 2026-06-23 修复进展：quality-gated 短视频

已经新增 quality-gated selector，并生成新的短视频套件：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/`

新 selector 不再按长度优先，而是要求：

- `motion_time_steps` 连续；
- `done_count=0`；
- 单一 source motion；
- root z mean ≥ `0.45 m`；
- root z min ≥ `0.30 m`；
- root z range ≤ `0.18 m`；
- reward mean ≥ `0`。

最终选中：

- source motion：`lafan1_sprint1_subject4`
- motion steps：`286550..286579`
- frames：`30`
- reward mean：`0.0546488`
- root z：min `0.7880` m，mean `0.7894` m，max `0.7905` m

这证明“旧视频被近地面 root target 拉坏”的问题已经被修掉。新的 reference replay 正常站立显示；teacher/VAE/diffusion/guided action-control 在 30 帧短视频里 `fall_proxy_count=0`。

但这不是最终成功：

- 当前 teacher rollout 中仍没有 `>=60` 帧且 root height 正常的稳定连续片段；
- teacher action-control root height 从目标约 `0.789 m` 下滑到最小约 `0.644 m`；
- VAE/diffusion/guided action-control root height 最小约 `0.526-0.542 m`；
- 这些视频仍使用 MuJoCo position actuators + root assist，不是 native MuJoCo PPO obs/action adapter；
- 因此只能写成 short-horizon diagnostic fix，不能写成 paper-level control reproduction。

新增稳定性审计：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_stability_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_stability_audit.json`

## 6. 下一层 blocker：teacher action / action-scale / obs-action adapter

为区分是 MuJoCo PD/root-assist 本身不稳，还是 teacher action-derived targets 有问题，新增了同片段 baseline：

- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/reference_joint_pd_control/reference_joint_pd_control.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_adapter_diagnostic.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_adapter_diagnostic.json`

这个 baseline 在同一个 normal-root segment 上直接用 reference joint qpos 作为 MuJoCo PD target，不是 policy 输出。结果：

- Reference-PD `fall_proxy_count=0`
- Reference-PD root height min/max：`0.7563 / 0.7722 m`
- Reference-PD root position error mean/max：`0.0533 / 0.0735 m`
- Teacher-action root height min/max：`0.6440 / 0.7457 m`
- Teacher-action root position error mean/max：`0.1439 / 0.2230 m`
- Teacher action-derived target 与 reference joint target 的 per-frame mean absolute gap：mean `0.5034 rad`，max `0.5897 rad`

结论：MuJoCo PD/root-assist 对这个短片段能维持正常站立，teacher action-derived targets 明显偏离 reference joint targets。因此下一层优先问题不是 root target，也不是纯视频渲染，而是：

1. teacher policy 仍弱；
2. IsaacLab action 到 MuJoCo PD target 的 scale/default pose/joint order/normalization 可能仍有 contract gap；
3. 当前视频直接回放 teacher rollout actions，不是 native MuJoCo closed-loop obs/action adapter。

下一步如果继续提升视频质量，应优先做 teacher action contract audit 和 native MuJoCo obs/action adapter，而不是继续拉长短视频。

## 7. Action contract audit 结论：不是简单拉长视频能解决

已经新增：

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_action_contract_audit.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.md`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_action_contract_audit.tsv`

该审计把同一个 quality-gated segment 的 teacher action-derived PD target 与 reference joint qpos 做逐关节对比。关键结果：

- per-frame mean abs gap：mean `0.5034 rad`，median `0.5029 rad`，max `0.5897 rad`
- `13` 个关节 mean gap 大于 `0.5 rad`
- `15` 个关节 teacher/reference delta correlation 小于 `0.2`
- `16` 个关节在 sign-flip probe 下 gap 变小，但不是所有关节同时受益，所以不是一个全局符号翻转可以解释
- top gap joints：`left_knee_joint`、`right_hip_yaw_joint`、`right_shoulder_pitch_joint`、`right_shoulder_yaw_joint`、`left_hip_pitch_joint`、`left_shoulder_yaw_joint`、`left_wrist_roll_joint`、`left_hip_roll_joint`

这说明当前不应把重点放在“把坏视频硬拉长到 15 秒”。更合理的解释是：IsaacLab teacher policy 的 action contract 没有被可信地迁移到 MuJoCo。需要继续核对：

1. IsaacLab `ActionTerm` 中 action 到 joint target 的 scale/default pose；
2. G1 joint order 与 MuJoCo actuator order；
3. policy observation 中 reference phase、anchor error、history、normalization 是否与训练时一致；
4. teacher rollout actions 是否来自真实 closed-loop teacher，而不是离线拼接/导出字段误读；
5. MuJoCo PD gain、armature、joint limits 是否与官方 `whole_body_tracking` / `motion_tracking_controller` 一致。

因此，当前结论是：reference 数据和 MuJoCo 短时 PD baseline 已经能正常展示，但 teacher action bridge 仍不可信。VAE、diffusion、guided 视频建立在这个 teacher action 上，所以视频效果差是预期结果，不能被解读为 BeyondMimic 方法本身失败。

## 8. 已修复的 adapter bug：reference frame 没有对齐到 robot anchor

继续排查后，新增 approximate native MuJoCo PPO adapter probe：

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/native_ppo_obs_adapter_probe/native_ppo_obs_adapter_probe.mp4`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/native_ppo_obs_adapter_probe/native_ppo_obs_adapter_probe_summary.json`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/quality_gated_stage1_multisource_native_adapter_comparison.json`

这个 probe 不再 replay 旧 teacher rollout 里保存的 actions，而是：

```text
MuJoCo current state
→ approximate 160-D IsaacLab policy obs
→ PPO actor + obs normalizer
→ action
→ theta_default + action_scale * action
→ MuJoCo PD actuator
→ mj_step
```

第一次实现时发现核心问题：直接使用 motion bundle 的全局 reference body position，导致 `motion_anchor_pos_b_norm` 首帧约 `8.98 m`。这与官方 `MotionCommand.reset()` 不一致。官方 deployment/training 会把 reference 初始 anchor 通过 yaw 和 translation 对齐到当前 robot anchor，然后再计算 anchor-frame observation。

修复方式：

```text
world_to_init_yaw = yaw(robot_anchor_initial) - yaw(reference_anchor_initial)
world_to_init_translation = robot_anchor_initial - R_yaw * reference_anchor_initial
reference_anchor_aligned = world_to_init_translation + R_yaw * reference_anchor
motion_anchor_pos_b = robot_anchor_frame^{-1}(reference_anchor_aligned)
```

修复后：

- `motion_anchor_pos_b_norm` 首帧从约 `8.98 m` 降到约 `0.0163 m`
- Native adapter `fall_proxy_count=0`
- Native adapter root height min/max：`0.6802 / 0.7830 m`
- Native adapter root position error mean/max：`0.0317 / 0.1062 m`
- 对比旧 open-loop teacher-action replay：
  - root height min 提升 `+0.0362 m`
  - root position error mean 降低 `-0.1122 m`

这说明此前视频站不稳的一个核心原因已经被解决：MuJoCo obs adapter 没有把 reference motion frame 对齐到 robot current anchor，policy 看到的是巨大且错误的 anchor error。修正后短片段动作明显更合理。

但这仍不是最终 paper-level 结果：

1. 当前 native adapter 仍是 approximate，不是官方 C++ `motion_tracking_controller` ONNX deployment；
2. reference body index 仍通过初始最近邻从 40-body motion bundle 映射到 14 tracking bodies；
3. 仍启用 root assist；
4. 只验证 30 帧短片段；
5. VAE、diffusion、guided latent 还没有迁移到该 aligned native adapter 做闭环控制。

下一步不是回到 matplotlib 或 open-loop action replay，而是把 aligned reference-frame adapter 变成统一的 MuJoCo rollout runner，并把 teacher、VAE、diffusion/guidance 都接到这个 runner 上。
