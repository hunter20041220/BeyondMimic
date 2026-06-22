# BeyondMimic 中文阅读报告

## 摘要

BeyondMimic 研究的是人形机器人如何从“追踪已有动作”走向“根据任务目标生成和执行新动作”。论文不是单独训练一个 motion tracking policy，也不是直接让扩散模型输出机器人动作，而是把 tracking teacher、DAgger-style rollout、conditional VAE、state-latent diffusion 和 test-time guidance 串成一个系统。

本项目没有完整复现 BeyondMimic 的 paper-level 结果。更准确地说，它是在公开资源约束下完成了一套大规模、可审计的 partial reproduction：公开数据图表和表格、官方 tracking 代码审计、IsaacLab/G1 task gate、40 个 public motions 的 replay/task diagnostic、本地 PPO/VAE/diffusion/guidance 链路、本地闭环 proxy rollout，以及明确的不可复现边界。

## 1. 论文核心问题

人形机器人控制难在多个约束同时成立：动态平衡、接触切换、高维关节协调、动作自然性、仿真稳定性和任务目标。motion tracking 可以让机器人模仿参考动作，但它很难直接回答“如何走到某个点”“如何绕开障碍物”“如何补全关键帧”或“如何从一种动作平滑过渡到另一种动作”。

BeyondMimic 的思路是把 tracking 当成基础能力来源，而不是最终目标。先训练一个 tracking teacher，再用 teacher 产生的数据训练 VAE 和 diffusion，最后通过 guidance 让生成结果满足任务目标。这个设计把物理可执行性、动作抽象、轨迹生成和任务优化分给不同模块。

## 2. 方法拆解

我把论文拆成六个模块理解：

1. motion tracking teacher：用 IsaacLab/RSL-RL/PPO 训练 G1 tracking policy。
2. teacher rollout / DAgger-style data：采集 teacher 在闭环环境中访问到的状态和动作。
3. conditional action VAE：把高维动作压缩成低维 latent action。
4. state-latent trajectory dataset：把状态和 latent 组织成时间窗口。
5. latent diffusion：学习可行动作轨迹的先验分布。
6. test-time guidance：用速度、目标点、障碍物、关键帧等任务代价引导扩散采样。

我认为这篇论文最有价值的地方不是“用了 diffusion”这个单点，而是系统组合：强化学习给物理能力，VAE 给可控低维动作空间，diffusion 给轨迹先验，guidance 给任务泛化。

为了复现，我把方法图进一步变成下面这种模块-证据表，而不是直接写一个大脚本：

| 论文模块 | 本地实现/审计对象 | 当前证据边界 |
|---|---|---|
| Motion tracking teacher | IsaacLab/RSL-RL task gates、reward/termination schema、PPO train/eval wrapper | local virtual teacher；done/endpoint 仍未达 paper-level |
| Teacher rollout / DAgger | rollout shard schema、teacher action/obs/latent collection、DAgger sample audit | local teacher rollout；不是官方 DAgger 数据 |
| Conditional VAE | reparameterization、KL、action reconstruction、checkpoint smoke、teacher-rollout VAE training | paper-faithful/local training；不是官方 VAE checkpoint |
| State-latent dataset | state+latent temporal window、split/index、finite/shape checks | 来源于 local teacher；不是官方 state-latent 数据 |
| Diffusion denoiser | DDPM noise schedule、mask、Transformer denoising、held-out denoising metrics | local denoiser；不是官方 diffusion checkpoint |
| Test-time guidance | joystick/waypoint/obstacle/inpainting/transition/composed cost gradients | local proxy closed-loop/offline evidence；不是 Fig.5/Fig.6 paper-level |
| Deployment | ONNX contract、controller semantics、MuJoCo/ROS launch audit | contract-level audit；没有 TensorRT/Mini-PC/real robot |

## 3. 当前复现状态

当前审计状态如下：

- master audit：`ok`，`397/397` 通过。
- artifact manifest：`1564` 个 artifact，missing `0`。
- paper-vs-reproduction：`235` 行。
- exactly comparable：`58`。
- approximately comparable：`19`。
- qualitative-only：`145`。
- not publicly reproducible：`10`。
- requires real robot：`3`。
- completion matrix：complete `74`，partial `132`，blocked `2`，out of scope `1`。
- required artifact absence：`32` 行，debug_only_not_required_artifact: 2, missing_required_artifact: 12, present_but_not_required_artifact: 18。

这些数字说明工程很完整，但不是论文完整复现。它证明当前证据可追溯，也证明还有很多 paper-level artifact 缺失。

从完成度角度看，我会分三层估计：课程阅读报告和答辩材料约 `85-90%` 可用；公开资源工程覆盖度约 `75-80%`；严格 non-robot paper-level reproduction 约 `40-50%`。这个估计的核心原因是：报告和审计材料已经很完整，但 tracking teacher 质量、true DAgger、官方 VAE/diffusion、Fig.5/Fig.6 和 TensorRT 仍没有达到论文级证据。

我在报告中按下面的证据层级来描述结果：

| 证据层级 | 代表内容 | 能否作为论文级结果 |
|---|---|---|
| exact/public | released-data 图表、表格、源码契约、公式 trace | 可以用于公开可复现部分 |
| approximate/resource-adjusted | official-loop body、captured G1 USDA、本地 PPO/eval | 只能说明本地虚拟链路 |
| qualitative/proxy | guidance rollouts、task protocol、可视化 | 可用于分析和答辩展示 |
| missing/non-public | 官方 checkpoint、DAgger logs、Fig.5/Fig.6 logs、TensorRT | 不能声称复现 |
| hardware-only | Unitree G1 deployment | 当前不可做 |

## 4. 已完成内容

第一，公开数据和论文表格/图表复现比较可靠。项目完成了 released-data figure/table reproduction、paper panel map、source coverage、formula/code trace 和 table value audit。这部分最接近 exact reproduction。

第二，官方 tracking 代码做了较完整审计。包括 observation/action schema、reward terms、termination、motion preprocessing、ONNX contract 和 MuJoCo/ROS launch contract。

第三，IsaacLab 和 G1 task gate 已经打通。当前 headless AppLauncher gate 是 `ok`，G1 task construction gate 是 `ok_current_task_env_construction_gate`。task contract 验证了 29 维 action、160 维 policy observation、286 维 critic observation、9 个 reward term、4 个 termination term、29 个关节和 40 个 body。

第四，官方 `csv_to_npz.py` / `replay_npz.py` 的 loop body 已经在 full public motion bundle 上跑通。40 个 public motions 合计 11960 帧/步。captured official-importer-export G1 USDA 路径比早期 scaffold 更可信，但仍不是 unmodified official converter entry。

第五，本地 PPO/VAE/diffusion/guidance 链路已经跑通。它证明了公开资源下可以实现一个 local virtual BeyondMimic-like pipeline，但不能把它说成官方 checkpoint 复现。

## 5. 当前效果

tracking 侧现在的关键结论是：链路能跑，但 teacher 还不够好。最重要的技术发现是 motion 数据语义比 policy 本身还敏感。早期 FK-repaired bundle 修复了 `body_pos_w` 退化问题，但后续发现更隐蔽的 body-order mismatch：motion target 是按 URDF body order 写入，而 IsaacLab runtime `MotionLoader` 按 simulator articulation body order 读取。这会导致 target body 错位、endpoint z error 和大量 termination。

当前主线已经切到 robot-order FK-repaired bundle。robot-order PPO checkpoint eval 状态是 `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_completed`，共评估 `612352` 个 virtual env steps，reward mean 约 `0.02073384587805606`，done count `109170`，anchor/body/joint position error mean 分别约 `0.07790673197711191`、`0.36114187777839774`、`1.5732512252785291`。三 seed eval 共 `1837056` 个 virtual env steps，mean done rate `0.1785340240036232`，reward mean `0.020480790998840676`，body-position error mean `0.3597400628005382`。

最新质量诊断显示，三个 multi-seed eval 都有 step-0 done rate `1.0`，step-0 body-position error 约 `43.29219436645508` 米。去掉 step 0 后 body-position error 降到 `0.2156714241976706`，但 post-step0 done rate 仍约 `0.175777426768736`。reset-command warmup live probe 的结论是 `command_warmup_partially_reduces_reset_endpoint_z_spike`。

随后我做了一个 2048 env x 299 step 的 full checkpoint warmup eval：`ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed`。它把 step-0 done count 从 `2048.0` 降到 `568.0`，把 step-0 body-position error 从 `43.294166564941406` m 降到 `0.2640186548233032` m；但整体 done rate 从 `0.1782798129180602` 升到 `0.22864463576505017`，也就是更差。因此现在不能说 warmup 修好了 teacher，只能说它定位了 reset bootstrap artifact，下一步要查 post-warmup termination / policy-state mismatch。

同 seed follow-up 进一步排除了随机 seed 影响：`ok_robot_order_fk_warmup_seed_matched_phase_diagnostic`。在和 non-warmup baseline 相同的 seed 下，step-0 仍然明显改善，但总 done rate 变差 `0.04325779943561875`，post-step0 done rate 变差 `0.04578210203439598`，`ee_body_pos` termination fraction 增加 `0.04554896530100333`，而 sampling top-bin delta 是 `0.0`。所以现在最可能的问题不是随机采样到坏 motion，而是 command/observation phase consistency：`Run a targeted reset-target refresh variant that recomputes body_pos_relative_w at reset without advancing MotionCommand.time_steps, then only run full PPO after this termination gate improves.`。

随后我又做了 no-advance reset-target refresh，直接验证“不推进 `MotionCommand.time_steps`，只刷新 reset target”这个想法。live probe 状态是 `ok_robot_order_fk_reset_target_refresh_no_advance_live_probe`：endpoint-z done rate 从 `1.0` 降到 `0.2734375`，endpoint-z error mean 从 `0.5298784375190735` 降到 `0.104344442486763`，并且 `time_steps_unchanged_by_refresh = True`。full eval 状态是 `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed`，step-0 done count delta `-1453.0`，但 total done rate 仍从 `0.1782798129180602` 变成 `0.22340745192307693`，post-step0 done-rate delta 是 `0.047659854760906034`。这说明 stale reset target 确实存在，但不是 teacher 弱的全部原因；下一步更应该查 reset state/action distribution、初始 joint velocity mismatch、endpoint 阈值和 `ee_body_pos` termination。

随后这个方向已经被静态 full-trace 诊断量化：`ok_robot_order_fk_reset_state_action_distribution_diagnostic`。它比较 baseline、reset-command warmup、no-advance target-refresh 三组同 seed full eval。target refresh 让 step-0 body-position error 改善 `-43.02953788638115` m，但 step-0 joint-velocity error 增加 `17.829124450683594`，first-five-step action mean 增加 `0.07184725403785702`，post-step0 done-rate delta 是 `0.047659854760906034`，`ee_body_pos` termination fraction delta 是 `0.0478825904055184`。所以现在最具体的判断是：No-advance target refresh removes the stale step-0 body target, but it exposes or creates a large initial joint-velocity/action transient and still worsens post-step0 done rate. The current teacher should not be used as the final DAgger/VAE/diffusion data source.

最新 live probe 继续检查了一个更直接的问题：target refresh 之后，能不能通过 action-history reset、action-offset alignment 或 motion-state rewrite 直接得到可用于 full eval 的修复。结果状态是 `ok_robot_order_fk_reset_state_action_consistency_live_probe`。target refresh alone 的 policy-step done rate 是 `0.28125`，post-step joint-velocity error 是 `14.182840347290039`；action reset 把 joint velocity 降到 `10.899185180664062`，但 done rate 变差到 `0.4765625`；action-offset alignment 把 joint velocity 降到 `10.263128280639648`，但 done rate 变差到 `0.49609375`；motion-state/action-offset candidate 把 joint velocity 降到 `8.305423736572266`，但 done rate 变差到 `0.73828125`。关键检查 `any_variant_improves_done_and_joint_velocity = False`。所以这一轮没有推荐 full eval，也没有重跑 PPO；这不是停在失败审计，而是避免把一个会恶化 termination 的 patch 带进主线训练。

最新 endpoint-group ablation 让下一步 tracking 修复更具体：`ok_endpoint_group_ablation_completed`。在同 seed、2048 env x 299 step 条件下，target-refresh done rate 是 `0.22340745192307693`；只保留 ankle endpoint termination 时 done rate 是 `0.1132420568561873`；只保留 wrist endpoint termination 时 done rate 是 `0.18382727581521738`；全部 endpoint threshold 放宽时 done rate 是 `0.07152912050585285`。诊断记录 dominant endpoint group 是 `wrists`。

随后我测试了一个比移除 endpoint body 更保守的候选：`ok_endpoint_threshold_sweep_completed`。这个 sweep 保留四个官方 endpoint body，只改变 z-only `ee_body_pos` threshold。target-refresh baseline done rate 是 `0.22340745192307693`；最佳 threshold 是 `0.5`，done rate `0.08907621760033445`，相对 baseline 变化 `-0.13433123432274247`；moderate threshold candidates 数量 `3`。这说明可以在 full PPO 前评估 threshold candidate，但它改变 evaluator，所以仍不是 paper-level tracking score。recommended next action 是 `evaluate_threshold_candidate_before_full_ppo`。

这一轮我又用 live wrist/ankle endpoint alignment probe 直接验证了这个判断：`ok_robot_order_fk_wrist_endpoint_alignment_live_probe`。它在真实 IsaacLab task 中分别记录 `body_pos_w`、`body_pos_relative_w` 和 `robot_body_pos_w`，比较 ankles 和 wrists 在 target refresh 前后以及 zero/policy step 后的 z error。结果是：refresh 后 wrist z-error mean `0.12762290239334106` m，ankle `0.06544189155101776` m；refresh wrist done rate `0.15234375`，ankle `0.09765625`；policy-step wrist done rate `0.09375`，ankle `0.06640625`。诊断是 `wrist_endpoint_target_or_body_semantics_remain_primary_done_source`。因此下一步不是再盲目 PPO，而是优先查 wrist endpoint 的 target/body order、wrist FK height 和 `ee_body_pos` body semantics。

最新 full-size source diagnostic 把这个 live probe 扩展到 2048 env x 299 step：`ok_robot_order_fk_wrist_endpoint_source_full_diagnostic`。它按 endpoint body、motion 和 phase bin 统计 z-error 与 termination 来源。总体 done rate 是 `0.21957958821070234`，`ee_body_pos` rate 是 `0.19802009301839466`；pre-step wrist exceed rate mean `0.06626907399665552`，ankle `0.057275880539297656`；post-step wrist `0.06591470265468227`，ankle `0.05699989548494983`。top wrist-heavy motions 包括 `fallAndGetUp1_subject4, dance1_subject2, walk3_subject5`。这个结果把问题从“怀疑 wrist endpoint”推进到“哪些 motion/phase/body 触发最多”，所以它是下一步修 tracking 数据质量的直接依据。

最新 deterministic reset live gate 又从另一个角度验证了这个判断：`ok_robot_order_fk_deterministic_reset_live_probe`。official-refresh policy done rate 是 `0.33203125`，joint velocity error 是 `15.347245216369629`；deterministic reset 把 joint velocity 降到 `11.510969161987305`，但 done rate 变差到 `0.453125`；motion-state reset 的 done rate 也变差到 `0.55078125`。最终 recommended full-eval variant 是 `none`。所以当前最应该修的是 termination/body-target 语义，而不是简单关闭 reset 随机性后直接开 PPO。

Level C 侧的 VAE、state-latent diffusion 和 guidance 能形成完整本地链路，但因为上游 teacher 弱，这些结果只能解释为机制复现和本地 proxy 实验。它们适合写进阅读报告，用来说明我理解并实现了论文 pipeline；但它们不能替代论文 Fig.5/Fig.6 的闭环结果。

当前统一任务协议表覆盖 `6` 个本地 proxy 任务，其中 `4` 个是 multi-seed proxy，`2` 个是 single-seed proxy。最重要的是 `paper_level_reproduced_count = 0`。这说明 joystick、waypoint、obstacle、composed、transition、inpainting 等任务在本地机制层面被覆盖，但还没有达到论文 Fig.5/Fig.6 协议。

## 6. 从公式到代码

这次复现不是只看论文图，而是把公式变成代码 contract。tracking objective 被拆成 anchor、target body、endpoint、action regularization 和 termination 检查；VAE 被拆成 state-conditioned encoder/decoder、reparameterization、KL 和 reconstruction；diffusion 被拆成 state-latent window、timestep noise、denoising objective 和 validation/test split；guidance 被拆成 joystick、waypoint、obstacle、inpainting、transition 等 cost gradient。

这样做的意义是：每个公式都能对应到一个可运行模块或审计表。公式里没有公开的数据或 checkpoint，则明确标成 local proxy，而不是假装已经 paper-level 复现。

本地代码刻意没有重写 IsaacLab 或官方 tracking 仓库，而是只实现适合独立验证的数学和数据契约：finite tensor check、yaw-frame transform、VAE latent math、DDPM-style noise/reverse helper、state-latent windows、DAgger sample schema、guidance cost 和 summary metrics。真正的机器人闭环仿真仍然以官方 whole_body_tracking/IsaacLab 栈为准。

## 7. 主要困难

第一是 IsaacLab/Isaac Sim 环境。真实机器人学习复现不是安装 PyTorch 就结束，Kit、Vulkan、USD save policy、GPU 可见性、AppLauncher 和 extension context 都会影响结果。

第二是机器人资产和 motion preprocessing。G1 URDF/USD、body names、target bodies、endpoint z、FK、`body_pos_w` 和 MotionLoader 格式都直接影响 tracking 结果。一个看似能加载的 motion bundle 仍可能在身体位置或终止条件上出问题。

第三是官方 artifact 缺失。论文最关键的 DAgger rollout、VAE checkpoint、diffusion checkpoint、Fig.5/Fig.6 rollout logs 和 TensorRT deployment artifacts 没有公开。

第四是闭环验证。offline denoising 或 guidance 指标变好，并不等于机器人在 IsaacLab 中稳定完成任务。真正有说服力的结果必须包含 rollout、termination、tracking error、success/fall/collision 指标和视频。

## 8. 还缺什么

除真实机器人外，仍缺：

- 高质量 paper-level tracking teacher。
- true DAgger rollout logs。
- 官方 VAE checkpoint。
- 官方 diffusion Transformer checkpoint。
- Fig.5/Fig.6 严格任务协议下的闭环指标和视频。
- TensorRT engine、Mini-PC latency 和异步部署复现。
- MuJoCo/ROS sim-to-sim 实际运行日志。

因此当前不能声称完整复现 BeyondMimic，也不得声称完整复现 BeyondMimic。

## 9. 个人理解

这次复现让我意识到，机器人学习论文的复现难点不只在算法公式。一个方法能不能复现，取决于环境、资产、训练数据、checkpoint、评测协议和部署细节是否一起公开。BeyondMimic 的方法图很清楚，但真正复现时，每个接口都有可能成为 blocker。

我认为这个项目最有价值的地方，是把证据分层说清楚：哪些是 official-code reproduction，哪些是 released-data reproduction，哪些是 paper-faithful reimplementation，哪些只是 local virtual proxy，哪些根本 not publicly reproducible。这个区分比简单说“复现成功”或“复现失败”更接近科研复现的真实状态。

这也是我对论文更深的一点理解：BeyondMimic 的扩散模型并不是凭空生成“机器人能力”，它依赖 tracking teacher 提供一个物理可执行的行为分布。如果 teacher 的 reset、endpoint、body order 或 termination 有问题，下游 VAE/diffusion 即使公式正确，也是在学习一个有偏的闭环分布。因此当前 wrist endpoint 和 reset-target 诊断不是偏离主线，而是在修复生成式控制链条最前面的数据基础。

## 10. 结论

本项目已经足够支撑一篇有独立思考的课程阅读报告：它不仅总结论文，还实际检查了代码、恢复环境、运行任务、实现公式、生成本地实验，并记录失败边界。但它不是完整 paper-level reproduction。下一步最重要的是修 tracking 数据质量和 termination/done count，得到更可信 teacher，再重做 downstream VAE、diffusion 和 guidance。
