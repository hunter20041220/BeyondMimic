# BeyondMimic 模型链硬门控复查报告

生成时间：2026-06-24 19:00:29 CST

## 结论先行

当前不能把 teacher / VAE / diffusion / guidance 的视频失败简单归因于“动作太难”或“视频渲染不好”。从论文公式、官方 `whole_body_tracking` 代码和本工程审计结果交叉看，现阶段更准确的判断是：

1. Stage-1 官方 motion tracking 公式和参数主干是清楚的，也已经接入本工程。
2. 但当前可用于下游的 teacher quality gate 仍未通过。
3. MuJoCo native observation adapter 尚未和 IsaacLab / motion_tracking_controller 做数值等价验证。
4. VAE / diffusion / guidance 代码中有若干 paper-contract 机制实现和离线指标，但不能弥补上游 teacher 数据质量不足。
5. 现在继续用当前 teacher rollout 训练 VAE / diffusion，大概率只会继续学习“前倾站姿/弱动作分布”，而不是学到 reference 中的单脚站立、走路、跳跃姿态。

因此，本阶段正确动作不是马上拉满 GPU 做 30000 iter 下游训练，而是先把 Stage-1 teacher 和 MuJoCo/Isaac 观测动作契约修到可信。只有 teacher 能稳定跟踪单一连续 motion 后，才应该重采集 rollout、重训 VAE、重训 diffusion，再生成单脚站立和 `jumps1_subject1` 的模型链视频。

## 论文原文对应关系

论文 Stage-1 motion tracking 的核心公式是：

```text
q_ref = (p_ref, R_ref, theta_ref)
nu_ref = (v_ref, omega_ref, theta_dot_ref)
FK(q_ref) -> body pose / body twist
```

tracking 目标不是直接播放关节角，而是以 anchor-centered / yaw-aligned / height-preserving 的方式跟踪 body pose 和 velocity。

policy observation：

```text
o = [psi, e_anchor, V_imu, theta - theta0, theta_dot, a_last]
```

action：

```text
theta_sp = theta0 + alpha * a
```

low-level PD：

```text
tau ~= Kp * (theta_sp - theta) - Kd * theta_dot
```

reward：

```text
r_task = sum_s exp(- mean_square_error_s / sigma_s^2)
r = r_task - lambda_l * r_limit - lambda_s * r_smooth - lambda_c * r_contact
```

论文 Stage-2/Stage-3 的核心不是直接把 reference condition 输入 diffusion，而是：

```text
teacher policy rollout -> state-action trajectories
VAE: action -> latent z, decoder(proprioception, z) -> action
state-latent trajectory tau = [s, z, s, z, ...]
diffusion learns p(tau)
guidance uses task cost gradients at inference
decoder outputs current action
action drives PD controller in closed loop
```

这意味着：如果 teacher rollout 本身是失败姿态或不连续 reset 拼接，VAE/diffusion 即使数学公式写对，也会学到失败分布。

## 官方代码已对齐的部分

本轮读取了官方 `whole_body_tracking` 代码：

- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`

确认到：

1. policy obs 包含 command、anchor pos/orientation error、base lin/ang velocity、joint pos、joint vel、last action。
2. reward 包含 anchor pos/orientation、body pos/orientation、body linear/angular velocity、action rate、joint limit、undesired contacts。
3. PPO 配置包含 `num_steps_per_env=24`、`max_iterations=30000`、actor/critic `[512,256,128]`、ELU、empirical normalization。
4. G1 PD/action scale 来自 manufacturer armature、10 Hz natural frequency、damping ratio 2.0，`G1_ACTION_SCALE = 0.25 * effort_limit / stiffness`。
5. termination 包含 anchor z、anchor ori、endpoint z-only body pos 等。

这说明“照论文和官方 Stage-1 代码训练 teacher”这条路线是对的。

## 当前工程已知阻塞点

来自现有审计：

- `res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`
  - 状态：`blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`
  - 结论：允许 Stage-1 teacher corrective retraining，但阻止从当前 teacher 继续下游 VAE/diffusion/guidance 成功视频。

- `res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json`
  - 状态：`blocked_code_formula_appendix_contract_has_required_fixes_before_training`
  - 说明：Stage-1 obs/action/reward/PPO/PD 主干通过，但 adaptive sampling、state-latent source、MuJoCo adapter 等仍有阻塞项。

- `res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
  - 状态：`blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready`
  - 说明：本地模型链不能当成 paper-level 成功链路。

- `res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
  - 状态：`blocked_native_mujoco_observation_adapter_not_validated`
  - 说明：MuJoCo native obs adapter 还不能可信替代 IsaacLab observation_manager。

- `res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
  - 状态：`blocked_mujoco_control_semantics_not_native_policy_control`
  - 说明：当前很多 MuJoCo 视频仍带 root assist / reference control / diagnostic semantics，不能说成 pure native policy control。

## 关于当前视频为什么像前倾站姿

最可能的根因链条是：

```text
teacher checkpoint 质量不足
    -> rollout 状态-动作轨迹分布偏向前倾/保守/失败恢复
    -> VAE 重构的 action 分布也偏向这个姿态
    -> diffusion 学到的 state-latent 先验也偏向这个姿态
    -> guidance 只能在错误先验附近微调，不能凭空生成正确单脚站立或跳跃
```

另外，MuJoCo 端还有单独风险：

```text
IsaacLab obs/action contract
    -> MuJoCo obs builder 未完成数值等价验证
    -> policy 输入不可信
    -> teacher rollout 视频不能证明 policy 学会动作
```

所以“teacher/VAE/diffusion 都没有摔，但都前倾”不是成功，也不是小瑕疵；它说明当前模型链没有学到 reference 姿态。

## 已修复的一个危险默认值

本轮发现：

```text
reproduction/scripts/render_lafan1_jumps1_subject1_mujoco_clean.py
```

中上一次高动态诊断留下了默认值：

```text
BM_JUMPS1_CONTROL_SETTLE_STEPS default = 0
BM_JUMPS1_DYNAMIC_ROOT_ASSIST default = 1
```

这会让默认 `jumps1_subject1` reference-action baseline 进入动态 root-assist stress-test 模式，容易导致高动态片段数值不稳，并污染后续判断。

本轮已改回：

```text
BM_JUMPS1_CONTROL_SETTLE_STEPS default = 40
BM_JUMPS1_DYNAMIC_ROOT_ASSIST default = 0
```

动态 root assist 仍可通过环境变量显式启用，但不再作为默认路径。

## 当前可保留的动作参考

不是所有动作都适合先修模型链。建议动作顺序如下：

1. `walk3_subject3`：LAFAN1 中统计最稳定的 walk 候选之一，适合检查 MuJoCo action scale / PD / camera。
2. `walk4_subject1`：位移更明显，适合做“真的走起来”的展示，但仍是参考/控制基线。
3. `jumps1_subject5`：比 `jumps1_subject1` 全局统计更温和，可作为跳跃类过渡目标。
4. `jumps1_subject1`：保留 `stable_dynamic_164s_179s` 作为当前安全基线；高动态片段只能当 diagnostic。
5. `hub_singleleg_video_single_leg_stand_1`：Short Sequence 的最终目标，但必须等 teacher gate 过后再重训/重采集。

`Dataset_beyondmimic/GRF/` 下的 real walk/run CSV 主要是 GRF 曲线与论文图复现数据，不是完整 G1 control reference motion。

## 训练前硬门控

下一次长训练前必须满足：

1. 单一 motion 的 reference CSV/NPZ 连续且无 reset 拼接。
2. IsaacLab teacher eval 中 non-timeout done rate 低，endpoint z termination 不主导失败。
3. motion time step 连续，不能从不同 env/reset 拼接。
4. action scale、joint order、default pose、PD gain 和官方 G1 配置一致。
5. MuJoCo obs builder 与 IsaacLab observation_manager 至少在同一状态样本上数值对齐。
6. teacher rollout 能在 MuJoCo/IsaacLab 中展示 reference 姿态，而不是前倾站姿。
7. 只有通过上述 gate，才可启动 VAE training。
8. 只有 VAE reconstruction closed-loop 过 gate，才可训练 state-latent diffusion。
9. 只有 diffusion denoising + receding-horizon rollout 过 gate，才可做 guidance 成功视频。

## GPU 使用建议

如果开始新的 Stage-1 teacher corrective retraining：

- 训练目标应先是 single-motion teacher，不是直接 2.5h 全量。
- 用 GPU 5/6 或用户指定空闲 GPU，不能打扰已有 4/7 任务。
- 可以加大 `num_envs` 和 batch，使显存利用率尽量高，但不要改 PPO/reward/action 语义。
- 每 500 iter 保存 checkpoint。
- 每个 checkpoint 必须跑 checkpoint sweep，不能只拿 last。

如果 teacher gate 没过，不建议拉满 GPU 训练 VAE/diffusion，因为下游会学习失败分布。

## 当前声明边界

当前不得声称：

- 完整复现 BeyondMimic。
- teacher policy 已达到 paper-level。
- VAE/diffusion/guidance 已能完成单脚站立。
- MuJoCo 视频是官方 IsaacLab 结果。
- reference replay 是 closed-loop policy control。
- released-data 曲线是重新训练结果。

当前可以声称：

- 已对论文公式、官方 Stage-1 代码和本地模型链做了硬门控审查。
- Stage-1 官方代码主干与论文公式/参数大体一致。
- 当前 teacher/downstream chain 质量不足，继续下游长训前必须先修 teacher 和 MuJoCo obs parity。
- `jumps1_subject1` 有一个稳定 reference-action baseline，但不是 learned control。

## 下一步

推荐下一步只做两件事：

1. 针对 `jumps1_subject1` 和 `hub_singleleg_video_single_leg_stand_1` 建立 single-motion teacher quality gate。
2. 写/跑 MuJoCo-vs-IsaacLab observation parity probe，确认同一状态下 160-D obs 的逐项误差。

这两项过了，再继续 teacher rollout -> VAE -> diffusion -> guidance 视频。
