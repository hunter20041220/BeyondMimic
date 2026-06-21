# BeyondMimic 中文阅读报告

## 摘要

BeyondMimic 研究的问题是：如何让人形机器人从“模仿已有动作”进一步走向“能够根据任务目标生成和执行新动作”。论文的核心思路不是单纯训练一个 motion tracking policy，也不是直接用扩散模型生成动作，而是把强化学习、VAE、扩散模型和 test-time guidance 组合成一个分层系统。

本复现工程并没有完整复现 BeyondMimic 的 paper-level 结果，尤其没有官方 DAgger rollout、官方 VAE/diffusion checkpoint、论文 Fig.5/Fig.6 的真实闭环评测协议、TensorRT/Mini-PC 部署结果和真实机器人结果。但工程已经完成了大量公开资源范围内的复现和审计：released-data 图表与表格、官方 tracking 代码契约、IsaacLab 环境恢复、G1 tracking task gate、本地 PPO/teacher rollout/VAE/state-latent/diffusion/guidance 链路、局部闭环 guidance proxy，以及用于答辩展示的视频、曲线和表格。

因此，本报告的结论是：这是一个公开资源约束下的可信 partial reproduction，而不是完整 paper-level reproduction。它的价值在于清楚展示了论文方法的模块化结构、哪些部分可以被公开复现、哪些部分只能近似实现，以及 robotics paper 复现中环境、资产、数据和闭环验证的重要性。

## 1. 论文背景

人形机器人控制比普通机器人控制更难，因为它同时涉及高维关节控制、动态平衡、接触切换、全身协调和任务泛化。motion tracking 可以让机器人模仿 motion capture 或 reference motion，但它天然依赖参考动作。对于“走到目标点”“避开障碍物”“补全关键帧动作”“把走路过渡到跑步”这样的任务，仅仅 tracking 一段已有动作是不够的。

BeyondMimic 的有趣之处在于，它没有把 motion tracking 当成最终目标，而是把 tracking teacher 当成行为能力来源。论文先通过强化学习训练一个能稳定追踪动作的 teacher，再收集 teacher rollout，把高维 action 压缩到 VAE latent 中，然后训练 state-latent diffusion model。最后在测试时通过 guidance 把扩散生成的轨迹推向具体任务目标。

这个思路很适合人形机器人，因为它把复杂问题拆成几层：

- 强化学习负责物理可执行性；
- VAE 负责把高维动作压缩成低维 latent；
- diffusion 负责生成可行轨迹；
- guidance 负责注入任务目标。

## 2. 方法拆解

我把论文方法拆成六个模块来理解和复现。

第一是 motion tracking teacher。这个模块通过 IsaacLab/RSL-RL/PPO 训练 G1 humanoid tracking policy。它决定了后续所有数据质量。如果 teacher 很弱，下游 VAE、diffusion 和 guidance 都会继承这个问题。

第二是 teacher rollout 和 DAgger。论文需要从 teacher policy 采集状态、动作和轨迹数据。这个数据不是普通离线 motion clip，而是 teacher 在模拟环境中访问到的分布。它是 tracking 到 generative control 的桥。

第三是 conditional action VAE。VAE 输入当前状态和 teacher action，学习一个低维 latent action representation。解码器之后可以把 state + latent 变回机器人动作。这个模块让 diffusion 不必直接在高维原始 action 空间建模。

第四是 state-latent trajectory dataset。论文不是只生成单步动作，而是把状态和 latent 组成时间窗口，形成轨迹级训练样本。这个模块连接了 VAE 和 diffusion。

第五是 latent diffusion。扩散模型学习如何从噪声中恢复 state-latent trajectory。它提供的是一种 motion prior，也就是“什么样的轨迹看起来像 teacher 数据中的可行动作”。

第六是 classifier/task guidance。测试时根据任务代价函数对扩散采样结果施加梯度引导，例如速度目标、路径目标、障碍物、关键帧补全等。这一步把 motion prior 转换为 task-directed controller。

## 3. 复现工程做了什么

当前工程已经建立了完整的审计体系。它不是只跑了几个脚本，而是把论文、源码、数据、配置、环境、实验结果和失败边界都做成了可追溯产物。

当前核心审计数字如下：

- artifact manifest：1382 个关键 artifact，缺失 0；
- master audit：343/343 通过；
- paper-vs-reproduction：212 行对照；
- exactly comparable：58；
- approximately comparable：19；
- qualitative only：122；
- not publicly reproducible：10；
- requires real robot：3；
- completion matrix：73 complete，122 partial，3 blocked，1 out of scope。

这些数字说明：工程已经覆盖了大量 paper/source/experiment 对照项，但 qualitative-only 和 partial 项仍然很多，不能声称完整复现。

## 4. 数据和替代方案

论文完整训练所需的官方 DAgger rollout、官方 VAE checkpoint、官方 diffusion checkpoint 和 Fig.5/Fig.6 rollout 数据并不公开。因此工程采用了分层替代策略。

对于 Level A 图表和表格，优先使用论文公开 released dataset，尽量做 exact 或 approximate comparison。

对于 tracking，优先使用官方 `whole_body_tracking`、IsaacLab、RSL-RL 和 Unitree G1 资产。由于官方 G1 URDF/USD conversion 在本机存在 save/conversion/runtime 问题，工程逐步建立了多个层级的证据：resource-adjusted scaffold、official-loop body audit、captured official-importer-export USDA、FK-repaired motion bundle，以及 IsaacLab task diagnostic。

对于 VAE/diffusion/guidance，由于官方 Level C 代码和 checkpoint 不完整，工程做了 paper-faithful PyTorch reimplementation，并用 public LAFAN1、local teacher rollout、official-importer-export tracking chain 等数据训练本地模型。

这套替代策略的原则是：可以用公开数据和本地虚拟实验逼近论文机制，但不能把它说成官方复现。

## 5. 当前复现结果

### 5.1 Released-data 和论文审计

released-data 图表、表格、panel map、formula/code trace、paper/source coverage 已经完成较多。这部分是当前最可靠的复现内容，因为它直接基于公开资料，claim level 较高。

### 5.2 IsaacLab 和 tracking

IsaacLab headless AppLauncher gate 已经通过。`Tracking-Flat-G1-v0` 可以在本地创建 G1 task，并验证核心维度：

- action dimension：29；
- policy observation dimension：160；
- critic observation dimension：286；
- reward terms：9；
- termination terms：4；
- robot joints/bodies：29/40。

官方 `csv_to_npz.py` 和 `replay_npz.py` 的 loop body 已经在 full public motion bundle 上跑通 40/40 motions，合计 11960 steps/frames。captured official-importer-export G1 USDA 路径也完成了 full dataset replay/task diagnostic。

本地 PPO training/evaluation 已经跑过多个版本，包括 scaled PPO、多 seed eval 和 policy rollout video。但当前 tracking policy 质量仍弱，reward 低、done count 高，并且 endpoint/body-position/termination 问题仍需修复。因此这部分只能说是 local virtual tracking evidence，不能说是论文 tracking teacher。

### 5.3 VAE、diffusion 和 guidance

工程已经完成：

- conditional action VAE；
- state-latent dataset；
- state-latent denoiser/diffusion training；
- offline guidance；
- task-conditioned closed-loop proxy rollouts；
- joystick、waypoint、obstacle avoidance、composed、transition、inpainting 等任务的本地 proxy；
- guided-vs-unguided matrix；
- success/fall/collision proxy；
- ONNXRuntime CPU/async audit。

这些结果证明论文机制可以在本地公开资源约束下形成一条可运行链路：

```text
tracking policy
-> teacher rollout
-> conditional VAE
-> state-latent dataset
-> denoiser/diffusion
-> guidance
-> IsaacLab local proxy rollout
```

但是这条链路依赖本地弱 teacher 和本地 proxy cost，不是官方 BeyondMimic checkpoint，也不是论文 Fig.5/Fig.6 的严格协议。

### 5.4 可视化和报告材料

工程已经生成了不少可用于报告和 PPT 的材料：

- reference replay videos；
- policy rollout videos；
- VAE closed-loop rollout videos；
- task-conditioned guidance videos；
- contact sheet；
- reward/done curves；
- tracking error plots；
- success/fall/collision proxy tables；
- latent PCA plots。

visual evidence index 记录了 31 个 report-ready videos 和 137 个 report-ready PNG。视频本体不提交 GitHub，但路径、claim level 和限制会进入 summary/manifest。

## 6. 主要困难

第一类困难是环境和 IsaacLab/Kit。机器人学习复现不是只安装 PyTorch。Isaac Sim、IsaacLab、Kit、Vulkan、GPU、watcher、USD save policy 都会影响结果。本工程早期大量工作都花在恢复 headless gate 和定位 USD conversion blocker 上。

第二类困难是机器人资产和 motion preprocessing。G1 URDF/USD、body positions、joint names、target bodies、FK、MotionLoader 输入格式都会影响 task 是否能跑。一个外形正确的 `motion.npz` 不一定物理上正确，body_pos_w 的退化会直接影响 tracking 和 termination。

第三类困难是官方 artifact 缺失。论文关键的 Level C checkpoint、DAgger rollout 和 Fig.5/Fig.6 rollout logs 没有公开。本工程只能做 paper-faithful reimplementation 和 local virtual proxy，不能做严格官方复现。

第四类困难是闭环验证。offline guidance 指标变好不代表机器人在 IsaacLab 中能稳定执行。真正有意义的证据必须包括 rollout、termination、tracking error、视频和失败分析。

## 7. 还没有完成什么

除真实机器人之外，仍未完成：

- unmodified official converter/replay entry 的完全成功；
- 高质量 paper-level tracking teacher；
- 真实 DAgger rollout dataset；
- 官方 conditional VAE checkpoint；
- 官方 state-latent diffusion checkpoint；
- 严格 Fig.5/Fig.6 task protocol 和 success/fall/collision 指标；
- TensorRT/CUDA provider/Mini-PC deployment；
- 论文级 tracking/diffusion ablation。

因此，本工程不能声称完整复现 BeyondMimic。

## 8. 个人理解

我认为 BeyondMimic 的价值不只是“用了 diffusion”，而是给出了一个机器人学习系统的组合方式。它把 motion tracking、latent action、trajectory diffusion 和 guidance 放在一个分层结构中，让每个模块承担不同责任。

从复现角度看，这篇论文也说明了 robotics reproducibility 的困难：论文中的方法图往往很清楚，但真正复现需要环境、资产、数据、checkpoint、训练协议、评测协议和部署协议全部对齐。只要其中一个环节缺失，结果就只能是 partial reproduction。

这次复现最有价值的地方是诚实地区分了不同层级的证据：released-data reproduction、official-code audit、paper-faithful reimplementation、local virtual proxy、not publicly reproducible 和 requires real robot。这样的区分比简单说“复现成功/失败”更接近科研复现的真实情况。

## 9. 结论

本项目已经形成了一个大规模、可审计的 BeyondMimic 公开资源复现工程。它完成了论文阅读、源码审计、环境恢复、tracking task gate、本地 PPO/VAE/diffusion/guidance 链路和可视化报告材料。但它没有完成 paper-level BeyondMimic 复现。

最终报告中应这样表述：本项目复现、审计并分析了 BeyondMimic 的公开可复现部分，明确记录了不可公开复现的边界，并实现了一条 local virtual BeyondMimic-like pipeline。这个结果足以支撑课程阅读报告和答辩，但不能替代官方 closed-loop humanoid control results。
