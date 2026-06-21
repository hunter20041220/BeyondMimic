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

## 3. 当前复现状态

当前审计状态如下：

- master audit：`ok`，`345/345` 个 master-audited artifacts 通过。
- artifact manifest：`1415` 个 artifact，missing `0`。
- paper-vs-reproduction：加入 robot-order FK PPO multi-seed eval 行后为 `220` 行。
- exactly comparable：`58`。
- approximately comparable：`19`。
- qualitative-only：`130`。
- not publicly reproducible：`10`。
- requires real robot：`3`。
- completion matrix：complete `74`，partial `123`，blocked `2`，out of scope `1`。
- required artifact absence：`32` 行，debug_only_not_required_artifact: 2, missing_required_artifact: 12, present_but_not_required_artifact: 18。

这些数字说明工程很完整，但不是论文完整复现。它证明当前证据可追溯，也证明还有很多 paper-level artifact 缺失。

## 4. 已完成内容

第一，公开数据和论文表格/图表复现比较可靠。项目完成了 released-data figure/table reproduction、paper panel map、source coverage、formula/code trace 和 table value audit。这部分最接近 exact reproduction。

第二，官方 tracking 代码做了较完整审计。包括 observation/action schema、reward terms、termination、motion preprocessing、ONNX contract 和 MuJoCo/ROS launch contract。

第三，IsaacLab 和 G1 task gate 已经打通。当前 headless AppLauncher gate 是 `ok`，G1 task construction gate 是 `ok_current_task_env_construction_gate`。task contract 验证了 29 维 action、160 维 policy observation、286 维 critic observation、9 个 reward term、4 个 termination term、29 个关节和 40 个 body。

第四，官方 `csv_to_npz.py` / `replay_npz.py` 的 loop body 已经在 full public motion bundle 上跑通。40 个 public motions 合计 11960 帧/步。captured official-importer-export G1 USDA 路径比早期 scaffold 更可信，但仍不是 unmodified official converter entry。

第五，本地 PPO/VAE/diffusion/guidance 链路已经跑通。它证明了公开资源下可以实现一个 local virtual BeyondMimic-like pipeline，但不能把它说成官方 checkpoint 复现。

## 5. 当前效果

tracking 侧现在的关键结论是：链路能跑，而且最近的数据质量修复比之前更接近主线。早期 FK-repaired motion bundle 解决了旧 `body_pos_w` 退化问题，但后来进一步发现一个更隐蔽的问题：motion target 按 URDF body order 写入，而 IsaacLab runtime `MotionLoader` 是按 simulator articulation 的 `robot.body_names` 索引 `body_pos_w`。这会导致目标 body 错位，例如脚踝目标可能被读成上肢 link，进而造成 endpoint z error 和大量 done/termination。

最新修复把 full 40-motion FK bundle 重排成 IsaacLab robot body order。这个修复后的 zero-action full-split task diagnostic 从旧 FK bundle 的 `11958/11960` done/termination 降到 `2166/11960`，mean anchor error 从约 `0.494` 降到 `0.084`，mean body-position error 从约 `0.516` 降到 `0.214`。这说明 motion target 的质量明显改善。

在这个基础上，我已经完成了一轮新的 robot-order FK-repaired PPO。训练使用 GPU 4/7、1000 iteration、4096 个总环境，并生成 21 个 checkpoint。iteration-999 checkpoint eval 使用 2048 个环境、299 步，总计 `612352` 个虚拟环境步：done rate 约 `0.178`，reward mean `0.0207`，anchor-position error mean `0.0779`，body-position error mean `0.3611`，joint-position error mean `1.5733`。它明显强于旧 URDF-order FK PPO checkpoint，但仍不是论文级 tracking teacher。

所以当前 gate 的结论不是“可以直接做最终 downstream”，而是：robot-order FK PPO 是当前最强 local virtual tracking baseline，可以用于报告曲线和视频。我也生成了一段 299 帧的 policy-vs-reference rollout video；这个单环境视频资产记录 target-body error mean `0.1547`、target-body error max `0.2961`、reward mean `0.0244`、done count `44`。它是有用的可视化证据，但仍是 local virtual media，不是论文指标。

随后我补了同一 iteration-999 checkpoint 的三 seed 完整评估：seeds `20260730`、`20260731`、`20260732`，每个 seed 使用 2048 个环境、299 步，总计 `1,837,056` 个虚拟环境步。多 seed 结果比较稳定，但也进一步说明 teacher 还弱：mean done rate `0.1785`，reward mean `0.02048`，anchor-position error mean `0.07762`，body-position error mean `0.35974`，joint-position error mean `1.57722`。因此 multi-seed eval 这个缺口已经补上，但结论不是“可以进入最终 downstream”，而是应该先做 checkpoint sweep、termination/source diagnostic 和更强 PPO 训练，再决定是否重跑 DAgger/VAE/diffusion。

进一步的 tracking-quality diagnostic 把问题定位得更具体：三个 multi-seed eval 都在 step 0 出现 `2048/2048` done，同时 body-position error spike 约 `43.29` m。去掉 step 0 后，body-position error mean 从约 `0.360` 降到约 `0.216`，但 post-step0 done rate 仍约 `0.176`。这说明下一步不应该直接重跑 downstream，而应该先查 reset/target alignment 和 `ee_body_pos` termination，再决定如何重训 PPO。

Level C 侧的 VAE、state-latent diffusion 和 guidance 能形成完整本地链路，但因为上游 teacher 弱，这些结果只能解释为机制复现和本地 proxy 实验。它们适合写进阅读报告，用来说明我理解并实现了论文 pipeline；但它们不能替代论文 Fig.5/Fig.6 的闭环结果。

    当前统一任务协议表覆盖 `6` 个本地 proxy 任务，其中 `4` 个是 multi-seed proxy，`2` 个是 single-seed proxy。最重要的是 `paper_level_reproduced_count = 0`。这说明 joystick、waypoint、obstacle、composed、transition、inpainting 等任务在本地机制层面被覆盖，但还没有达到论文 Fig.5/Fig.6 协议。

## 6. 主要困难

第一是 IsaacLab/Isaac Sim 环境。真实机器人学习复现不是安装 PyTorch 就结束，Kit、Vulkan、USD save policy、GPU 可见性、AppLauncher、EULA/extension context 都会影响结果。

第二是机器人资产和 motion preprocessing。G1 URDF/USD、body names、target bodies、endpoint z、FK、`body_pos_w` 和 MotionLoader 格式都直接影响 tracking 结果。一个看似能加载的 motion bundle 仍可能在身体位置或终止条件上出问题。

第三是官方 artifact 缺失。论文最关键的 DAgger rollout、VAE checkpoint、diffusion checkpoint、Fig.5/Fig.6 rollout logs 和 TensorRT deployment artifacts 没有公开。因此本项目必须用 public LAFAN1/G1 motions、本地 PPO teacher 和 paper-faithful reimplementation 来构建 local virtual pipeline，并明确它和官方 paper-level result 的边界。

第四是闭环验证。offline denoising 或 guidance 指标变好，并不等于机器人在 IsaacLab 中稳定完成任务。真正有说服力的结果必须包含 rollout、termination、tracking error、success/fall/collision 指标和视频。

## 7. 还缺什么

除真实机器人外，仍缺：

- 高质量 paper-level tracking teacher。
- true DAgger rollout logs。
- 官方 VAE checkpoint。
- 官方 diffusion Transformer checkpoint。
- Fig.5/Fig.6 严格任务协议下的闭环指标和视频。
- TensorRT engine、Mini-PC latency 和异步部署复现。
- MuJoCo/ROS sim-to-sim 实际运行日志。

因此当前不得声称完整复现 BeyondMimic。

## 8. 个人理解

这次复现让我意识到，机器人学习论文的复现难点不只在算法公式。一个方法能不能复现，取决于环境、资产、训练数据、checkpoint、评测协议和部署细节是否一起公开。BeyondMimic 的方法图很清楚，但真正复现时，每个接口都有可能成为 blocker。

我认为这个项目最有价值的地方，是把证据分层说清楚：哪些是 official-code reproduction，哪些是 released-data reproduction，哪些是 paper-faithful reimplementation，哪些只是 local virtual proxy，哪些根本 not publicly reproducible。这个区分比简单说“复现成功”或“复现失败”更接近科研复现的真实状态。

## 9. 结论

本项目已经足够支撑一篇有独立思考的课程阅读报告：它不仅总结论文，还实际检查了代码、恢复环境、运行任务、实现公式、生成本地实验，并记录失败边界。但它不是完整 paper-level reproduction。下一步最重要的是先修 robot-order FK PPO 的 reset/target alignment 和 `ee_body_pos` termination 瓶颈，再重跑更强 tracking PPO，并用更可信 teacher 重做 downstream VAE、diffusion 和 guidance。
