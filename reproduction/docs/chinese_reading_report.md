# BeyondMimic 中文阅读报告

## 摘要

BeyondMimic 关注的问题是：人形机器人如何从“追踪一段已有动作”走向“根据任务目标生成并执行新的全身动作”。论文的核心思想不是放弃 motion tracking，而是把 tracking 当作物理能力来源。系统先训练一个 tracking teacher，再用 teacher 的闭环数据训练 conditional action VAE，把动作压缩到 latent action，然后训练 state-latent diffusion，最后在测试时用 guidance 把扩散采样引导到 joystick、waypoint、obstacle avoidance、transition、inpainting 等任务目标。

我的复现工程不是完整 paper-level 复现。更准确的说法是：在公开资源约束下，我做了一个大规模、可审计的 partial reproduction。它包括 released-data 图表和表格复现、官方 tracking 代码审计、IsaacLab/G1 task gate、public-motion replay diagnostic、本地 PPO tracking、本地 VAE/diffusion/guidance、proxy closed-loop rollout、可视化材料，以及明确记录哪些东西没有公开、哪些需要真实机器人。

这个项目最大的收获是：机器人论文复现不是简单运行训练脚本。真正困难的是重建训练脚本背后的假设，包括机器人资产、body order、motion preprocessing、reset 语义、termination 逻辑、数据来源和结果边界。当前工程能支撑一篇严肃的阅读报告和一个 simulation-only 的 BeyondMimic-like pipeline，但不能声称完整复现 BeyondMimic。

## 1. 论文为什么重要

人形机器人控制难在多个约束同时成立：机器人要保持平衡、处理接触、协调高维关节、避免跌倒，同时还要完成任务目标。传统 motion tracking 可以让机器人模仿参考动作，但 tracking 本身并不等于通用控制。一个会跟踪“向前走”动作片段的 policy，不一定会自动学会走到任意目标点、绕过障碍物、补全缺失动作片段，或者在不同动作之间平滑过渡。

BeyondMimic 的有趣之处在于，它把 imitation 当成基础能力，而不是最终目标。tracking teacher 负责提供物理可执行行为，VAE 负责把高维动作压缩成可建模的 latent，diffusion 负责学习可行动作轨迹的先验，guidance 负责在测试时注入新任务目标。

我认为这篇论文的关键贡献不是单独“用了 diffusion”，而是把强化学习、VAE、diffusion 和 test-time optimization 放在一个合理的机器人控制系统里。

## 2. 方法拆解

我把论文拆成六个模块来理解。

第一，motion tracking teacher。系统先用 IsaacLab/RSL-RL/PPO 训练 Unitree G1 tracking policy，让机器人追踪目标 body pose、velocity、orientation 和关节运动。这一步是整个方法的物理基础。

第二，teacher rollout / DAgger-style data。teacher 在闭环环境中运行，产生真实访问到的 state-action 分布。这个数据比单纯 reference motion 更重要，因为后续模型需要适应控制过程中实际遇到的状态。

第三，conditional action VAE。机器人动作维度高、耦合强，直接建模很难。VAE 把 state-conditioned action 压缩成低维 latent action，使后续 diffusion 更容易学习。

第四，state-latent trajectory dataset。系统把状态和 latent action 组织成时间窗口，让模型学习一段轨迹，而不是孤立的一步动作。

第五，latent diffusion。denoiser 学习从噪声中恢复可行的 state-latent trajectory，形成一个动作轨迹先验。

第六，test-time guidance。速度、目标点、障碍物、关键帧、transition smoothness 等任务可以写成 cost。guidance 在采样过程中利用这些 cost，让生成轨迹既接近 learned prior，又满足任务目标。

这套设计优雅的地方在于：可行性和任务偏好被分开了。diffusion prior 回答“什么样的动作像真实可行的人形机器人动作”，guidance 回答“这些可行动作里哪一个更符合当前任务”。

## 3. 我的复现原则

我没有把这个项目当成 demo，而是把论文拆成不同层级的可验证证据：

1. 精确或近似精确复现：公开表格数值、released-data 图表、源码接口、公式和配置审计。
2. 本地虚拟复现：IsaacLab/G1 task gate、public motion replay、本地 PPO tracking、本地 VAE/diffusion/guidance、本地视频。
3. proxy 复现：Fig.5/Fig.6 风格任务的本地协议和机制验证。
4. 公开资源下不可复现：官方 teacher checkpoint、真实 DAgger rollout、官方 VAE/diffusion checkpoint、论文视频和部分 deployment artifact。
5. 真实机器人验证：需要 Unitree G1 硬件，当前没有。

这一区分非常重要。local proxy result 可以很有价值，但不能写成 paper result。本项目的目标是诚实说明哪些可以复现，哪些只能近似，哪些因为资料不公开或硬件缺失不能完成。

## 4. 当前工程状态

最新审计基线如下：

- master audit：`ok`，`353/353` 个 master-audited artifacts 通过。
- artifact manifest：`1428` 个关键 artifact 已哈希。
- paper-vs-reproduction：`222` 行。
- comparison type：`58` exactly comparable，`19` approximately comparable，`132` qualitative-only，`10` not publicly reproducible，`3` requires real robot。
- completion matrix：`74` complete，`125` partial，`2` blocked，`1` out of scope。
- required artifact absence audit：`32` 行，明确记录缺少官方 checkpoint、paper-level rollout video、true DAgger logs 和真实机器人结果。

这些数字不能理解成“论文已经复现”。它们说明的是：当前证据链可追溯，而且哪些缺口仍然存在已经被明确记录。

如果按课程阅读报告来估计，当前材料约 `85-90%` 可用；如果按公开资源工程复现覆盖度估计，约 `75-80%`；如果按严格非实机 paper-level reproduction 估计，则只有约 `40-50%`。差异来自 claim boundary：公开数据、源码审计和本地 proxy 已经很完整，但官方 teacher、true DAgger、官方 VAE/diffusion、Fig.5/Fig.6 paper protocol 和 TensorRT 仍没有真正完成。

环境方面，项目分为 analysis、diffusion 和 tracking 三个环境。analysis 用于图表、表格、ONNX 和报告；diffusion 用于 PyTorch CUDA、VAE、denoiser 和 guidance；tracking 用于 Isaac Sim、IsaacLab、RSL-RL 和 official tracking stack。当前 IsaacLab headless AppLauncher gate 已经通过，G1 task construction gate 也已经通过。这说明环境不是主 blocker；主 blocker 已经转移到 tracking teacher 质量。

## 5. 从公式到代码

我把论文公式当成软件接口来实现，而不是只在报告里复述。tracking objective 对应 anchor pose、target body position、velocity、endpoint height、action rate 和 contact/termination 的 reward/termination 检查。VAE objective 对应 state-conditioned action reconstruction、latent mean/logvar、reparameterization、KL regularization、finite-gradient check 和 checkpoint save/load。diffusion objective 对应 state-latent sequence、independent timestep、noisy-token denoising、train/validation/test split 和 denoising-improvement metric。

guidance 部分则被实现为 trajectory cost gradient：joystick velocity、waypoint final distance、obstacle clearance、inpainting keyframe error 和 transition smoothness 都被写成本地 proxy cost。这个过程让我真正理解论文方法：diffusion prior 负责可行性，guidance 负责任务偏好；但如果没有官方 checkpoint 和 paper protocol，本地 cost 只能支持机制验证，不能直接变成 paper-level Fig.5/Fig.6 复现。

## 6. 已完成的复现和审计

最接近 exact reproduction 的部分是 released-data 和源码级审计。项目检查了论文表格、released-data figure、paper panel map、formula/code trace、observation/action schema、reward terms、termination terms、motion preprocessing、ONNX contract 和 MuJoCo/ROS launch surface。这部分能帮助我确认论文和公开代码到底规定了什么。

tracking 侧更复杂。项目恢复了一个可用的 IsaacLab/G1 路径。public motions 通过 official-loop 预处理和 replay body 检查，full public bundle 覆盖 40 个 motions、11960 帧/步。项目还恢复了 captured official-importer-export G1 asset path，这比早期 scaffold 更可信。但它仍然是本地 captured/importer-export 路径，不等于 unmodified official entry 直接训练出论文 teacher。

最关键的 tracking 发现来自 motion 数据质量。早期 FK-repaired motion bundle 修复了一个明显的 `body_pos_w` 退化问题，但仍存在更隐蔽的 body order mismatch：motion target 按 URDF body order 写入，而 IsaacLab runtime `MotionLoader` 按 simulator articulation body order 索引 `body_pos_w`。这个错位会让某些 target body 指向错误 link，导致 endpoint z error 和大量 termination。

把 40-motion FK bundle 重排成 IsaacLab robot body order 后，zero-action task diagnostic 明显改善。旧 FK bundle 的 done/termination 是 `11958/11960`，robot-order bundle 降到 `2166/11960`；mean anchor error 从约 `0.494` 降到 `0.084`；mean body-position error 从约 `0.516` 降到 `0.214`。这说明问题不只是 policy 没训练好，motion target 的数据格式本身就是关键。

## 7. 当前 PPO Tracking 效果

在 robot-order FK-repaired bundle 基础上，我完成了一轮本地 PPO tracking baseline。训练使用 GPU 4 和 7，1000 个 PPO iteration，4096 个总环境，生成 21 个 checkpoint。iteration-999 checkpoint eval 使用 2048 个环境、299 步，总计 `612352` 个虚拟环境步。

当前指标是：reward mean 约 `0.0207`，done rate 约 `0.178`，anchor-position error 约 `0.0779`，body-position error 约 `0.3611`，joint-position error 约 `1.5733`。这明显优于旧 URDF-order FK checkpoint，但仍不是论文级 tracking teacher。

随后我对同一个 checkpoint 做了三 seed 完整评估。三个 seed 分别使用 2048 个环境、299 步，总计 `1,837,056` 个虚拟环境步。多 seed 结果稳定，但仍然偏弱：mean done rate `0.1785`，reward mean `0.02048`，anchor-position error mean `0.07762`，body-position error mean `0.35974`，joint-position error mean `1.57722`。

进一步的 tracking-quality diagnostic 定位了更具体的问题：三个 multi-seed eval 都在 step 0 出现 `2048/2048` done，step-0 body-position error spike 约 `43.29` 米。去掉 step 0 后，body-position error 从约 `0.360` 降到约 `0.216`，但 post-step0 done rate 仍约 `0.176`。这说明下一步不应该直接重跑 downstream chain，而应该先查 reset/target alignment 和 `ee_body_pos` termination source。

因此，目前 tracking 结论是：链路已经跑通，数据质量已经比早期明显改善，PPO baseline 和视频可以进入报告；但 teacher 质量还不足以支撑 paper-level DAgger/VAE/diffusion/guidance。

## 8. VAE、Diffusion 和 Guidance

Level C 侧已经建立了完整本地链路：teacher rollout、conditional VAE、state-latent dataset、denoiser/diffusion、offline guidance、reverse guidance、task-conditioned rollout 和可视化。项目还在 public LAFAN1 / G1 retargeted 数据上跑了 paper-architecture VAE/diffusion、多 seed、symmetry augmentation、ONNX export 和 latency-style audit。

当前 strongest downstream chain 使用 scaled local PPO teacher data：teacher rollout samples 为 `1,224,704`，VAE test action MSE 约 `0.000198`，state-latent windows 超过一百万，denoiser test pred-token MSE 约 `0.013214`，noisy-token MSE 约 `0.067370`，denoising improvement ratio 约 `0.804`。这些数字说明链路可运行，但不能证明官方 BeyondMimic 模型已经复现。

这些结果能说明我理解并实现了论文机制。它们可以写进阅读报告，作为 local virtual BeyondMimic-like pipeline。但它们不能被写成官方 BeyondMimic checkpoint 复现，因为上游 teacher 不是官方 teacher，rollout distribution 不是论文真实 DAgger distribution，VAE/diffusion checkpoint 也是本地训练产物。

## 9. Fig.5 / Fig.6 本地 Proxy Protocol

项目已经把 joystick、waypoint、obstacle avoidance、composed guidance、transition、inpainting-style task 合并到一个本地 proxy protocol 里。这个表适合答辩展示，因为它把论文核心任务统一到一个口径下。

但最重要的字段是：`paper_level_reproduced_count = 0`。这表示本地协议能说明机制跑通，但不能说明论文 Fig.5/Fig.6 已经复现。

当前任务协议可以这样概括：

| 任务 | 当前证据 | claim boundary |
|---|---|---|
| joystick | 5-seed 本地 proxy；速度、reward、tracking-error 指标 | local virtual proxy，不是 paper Fig.5 |
| waypoint | 5-seed 本地 proxy；final distance / success proxy | local virtual proxy，不是 paper Fig.5 |
| obstacle avoidance | 5-seed 本地 proxy；clearance / collision proxy | local virtual proxy，不是 paper Fig.6 |
| composed | 5-seed 本地 proxy；多目标 tradeoff | local virtual proxy，不是 paper Fig.5/6 |
| transition | single-seed walk-to-run diagnostic | local single-seed proxy |
| inpainting | single-seed keyframe/body-target diagnostic | local single-seed proxy |

下一步应该把 proxy protocol 从“机制指标”升级到“任务指标”：joystick velocity tracking error、waypoint final distance/success rate、obstacle minimum clearance/collision count、inpainting keyframe error、transition smoothness/fall rate，以及 guided-vs-unguided improvement。只有这些任务指标变强，才有可能接近论文 Fig.5/Fig.6 的虚拟复现。

## 10. 主要困难

第一，IsaacLab/Isaac Sim 环境比普通深度学习环境复杂得多。Kit startup、Vulkan/EGL、GPU 可见性、extension context、headless AppLauncher 和系统 watcher limit 都会影响实验。

第二，机器人资产和 motion preprocessing 非常敏感。G1 body names、body order、target bodies、endpoint z、FK、`body_pos_w` 和 reset/termination 都会直接影响 tracking 结果。一个 motion bundle 能加载，不代表它的数据语义正确。

第三，论文的最强结果依赖很多未公开 artifact。官方 tracking teacher checkpoint、true DAgger rollout logs、官方 VAE/diffusion checkpoint、paper Fig.5/Fig.6 rollout videos/metrics、TensorRT deployment artifact 和真实机器人 logs 都没有完整公开。

第四，复现过程必须控制 claim boundary。本地结果可以有研究价值，但不能夸大成 official paper-level result。

## 11. 个人理解和反思

读这篇论文之前，我以为复现的主要难点会是 diffusion 模型。真正做下来发现，最难的是 tracking 和数据语义。diffusion/VAE 的代码可以按公式实现，但如果上游 teacher 不强、motion body order 不对、termination 一直触发，那么 downstream 模型再复杂也没有意义。

我也更理解了为什么机器人论文复现困难。论文方法图上的每个箭头都隐藏了大量工程细节：state tensor 的 body order、motion file 的 schema、reset 的时机、termination 的阈值、teacher rollout 的分布、policy checkpoint 的质量。任何一个环节偏了，最终结果就会偏。

BeyondMimic 本身仍然是很有启发的工作。它把 motion tracking、latent action、diffusion prior 和 guidance 组合成一个有清晰层次的系统。我当前的复现说明这个思路可以在公开资源下被部分重建和分析，但完整 paper-level reproducibility 仍需要更多官方 artifact。

## 12. 结论

This project does not fully reproduce BeyondMimic at paper level.

当前项目完成的是一个大规模、可审计、公开资源约束下的 partial reproduction。它复现和审计了论文公开数据、源码契约、IsaacLab/G1 task、local PPO tracking、VAE/diffusion/guidance 和本地 proxy rollouts；同时也明确记录了官方 checkpoint、true DAgger、paper Fig.5/Fig.6、TensorRT deployment 和真实机器人结果仍未完成。这个结论比“完整复现”更诚实，也更能体现本项目的实际研究价值。
