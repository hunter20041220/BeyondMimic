# BeyondMimic 复现项目报告

## 1. 项目起点

本项目从阅读 BeyondMimic 论文开始。最初目标是理解论文提出的完整 humanoid control pipeline，并尽可能基于公开资料复现其关键模块。项目不是从零写一个相似 demo，而是在已有下载资料、官方代码、旧服务器复现快照和本机实验环境基础上逐步恢复、审计、运行和扩展。

原始资料放在 `download/`，旧服务器工作区快照放在 `other/`。复现工程主体放在 `reproduction/`、`res/`、`logs/`、`envs/`、`cache/` 和 `tmp/`。所有新增代码、报告和小型审计产物都进入 GitHub；大型 checkpoint、raw runs、视频和数据集不提交。

## 2. 论文拆解

我把 BeyondMimic 拆成以下模块推进：

1. 论文资料和公开数据审计；
2. Unitree G1 / IsaacLab / whole_body_tracking 环境恢复；
3. motion preprocessing 和 official replay；
4. PPO motion tracking teacher；
5. teacher rollout / DAgger-style dataset；
6. conditional action VAE；
7. state-latent trajectory dataset；
8. latent diffusion / denoiser；
9. guidance tasks；
10. ONNX/TensorRT/deployment audit；
11. 可视化、报告和论文逐项对照。

这样的拆法对应论文方法结构，也方便把“已复现”“部分复现”“本地近似”“公开不可复现”分开记录。

## 3. 公式和代码实现

论文中涉及的核心机制包括：

- teacher policy 的 tracking objective；
- conditional VAE 的 encoder/decoder 和 latent action；
- state-latent trajectory token；
- diffusion denoising objective；
- classifier/task guidance 的 cost gradient；
- trajectory transform 和 mask/schema。

工程中用 `reproduction/src/beyondmimic_reimpl` 做了 paper-faithful reimplementation。这个包不是官方代码替代品，而是用来验证论文公式和机制是否能独立实现。相关 API tests、runtime integration audit 和 core math tests 都已经通过。

在 tracking 侧，优先使用官方 `whole_body_tracking`、IsaacLab 和 RSL-RL，不重新发明 task。对于官方代码无法直接跑通的部分，采用 wrapper、runtime patch、audit script 和明确 claim boundary，而不是直接修改下载目录或冒充官方路径。

## 4. 环境恢复

环境恢复分成三个层级：

- `bm_analysis`：报告、审计、绘图和 ONNXRuntime；
- `bm_diffusion`：PyTorch CUDA、VAE/diffusion/guidance；
- `bm_tracking`：Isaac Sim、IsaacLab、RSL-RL、whole_body_tracking。

当前 IsaacLab AppLauncher headless gate 已经通过，G1 tracking task 可以创建和 reset。最关键的 task contract 也已经验证：29 维 action、160 维 policy observation、286 维 critic observation、9 个 reward term、4 个 termination term、29 个关节和 40 个 body。

这个阶段最大的困难不是 Python 依赖，而是 Isaac Sim/Kit、Vulkan、USD save policy、GPU 可见性和官方 G1 URDF/USD conversion。

## 5. 数据来源和替代策略

论文完整训练需要官方 DAgger rollout、官方 VAE/diffusion checkpoint 和 Fig.5/Fig.6 rollout 数据，但这些在当前公开资料中不可用。因此项目采用了分层替代：

- 公开 released dataset：用于图表和表格复现；
- LAFAN1 / G1 public motions：用于 motion preprocessing、tracking 和 surrogate training；
- captured official-importer-export G1 USDA：用于替代生成 scaffold，提高机器人资产可信度；
- FK-repaired motion bundle：用于修复 body_pos_w 退化问题；
- local PPO teacher：用于生成 teacher rollout；
- local VAE/diffusion：用于验证论文机制；
- local guidance proxy：用于展示 Fig.5/Fig.6-adjacent 行为。

所有这些替代都在报告中明确标注为 local virtual 或 resource-adjusted，不写成官方复现。

## 6. 实验推进过程

第一阶段是盘点和审计。工程生成了 local inventory、source ledger、paper/source coverage、paper panel map、paper formula/code trace、paper table value audit 和 required artifact absence audit。

第二阶段是 released-data 复现。已经完成大量公开数据图表、表格和 panel 对照。这是当前最接近 exact reproduction 的部分。

第三阶段是 IsaacLab 和 tracking gate。先修 import，再修 AppLauncher headless，再修 G1 task construction，最后跑 full public-motion task diagnostic。这个阶段确认了环境和任务能跑。

第四阶段是 official loop 和 importer-export 路径。官方 `csv_to_npz.py` / `replay_npz.py` loop body 已经能在 40 个 public motions 上跑完。captured official-importer-export USDA 路径也完成了 full replay/task diagnostic。

第五阶段是 PPO tracking。工程跑了 300 iteration、1000 iteration、scaled PPO、checkpoint eval、multiseed eval、policy rollout video。结果证明链路可运行，但 policy 质量仍不足，reward 低、done count 高，不能作为论文 teacher。

第六阶段是 downstream。用 local teacher rollout 训练 conditional VAE，构建 state-latent dataset，训练 denoiser/diffusion，然后做 offline guidance 和 closed-loop proxy rollout。

第七阶段是可视化和报告。工程生成了 reference replay video、policy rollout video、guidance rollout videos、contact sheet、reward/done 曲线、tracking error 图、latent PCA 和 success/fall/collision proxy 表。

## 7. 当前成果

当前机器可读审计结果：

- master audit：343/343 通过；
- artifact manifest：1382 个 artifact；
- paper-vs-reproduction：212 行；
- completion matrix：73 complete、122 partial、3 blocked、1 out of scope；
- visual evidence：31 个 report-ready videos、137 个 report-ready PNG；
- 英文阅读报告：约 11795 words。

比较可靠的成果：

- released-data 图表/表格复现；
- 官方 tracking 代码契约审计；
- IsaacLab headless 和 G1 task gate；
- 40-motion official-loop replay/task diagnostic；
- captured official-importer-export G1 USDA task diagnostic；
- local PPO training/eval/video；
- local VAE/state-latent/diffusion/guidance 链路；
- 多种 report-ready 可视化资产。

需要谨慎解释的成果：

- PPO teacher 是 local weak teacher，不是论文 teacher；
- VAE/diffusion checkpoint 是本地训练，不是官方 checkpoint；
- guidance rollout 是 local proxy，不是论文 Fig.5/Fig.6；
- ONNXRuntime audit 是 CPU/async proxy，不是 TensorRT/Mini-PC。

## 8. 当前效果评价

如果按“工程和报告支撑”评价，项目完成度约 70%。因为环境、审计、报告、可视化、公开数据和本地完整链路都已经建立。

如果按“论文非实机 paper-level 结果”评价，完成度约 45%-55%。原因是 motion tracking teacher、DAgger、VAE/diffusion checkpoint、Fig.5/Fig.6 strict protocol 和 TensorRT deployment 仍没有 paper-level 证据。

这两个百分比要分开说。否则容易把大量工程工作误解成完整论文复现。

## 9. 失败和困难

主要失败和困难包括：

- IsaacLab/Kit startup 和 Vulkan runtime 问题；
- USD `permissionToSave=False` 和官方 URDF conversion/save path blocker；
- 官方 G1 full-robot preconverted USD 不完整；
- body_pos_w 退化导致 endpoint/body tracking 评估不可靠；
- PPO teacher 质量弱，done count 高；
- 官方 DAgger/VAE/diffusion/Fig.5/Fig.6 数据缺失；
- TensorRT provider 不可用；
- 大型 run/checkpoint/video 需要严格管理磁盘和 GitHub 边界。

这些失败不是简单报错，而是 robotics reproduction 的核心困难：环境、资产、数据和闭环协议缺一不可。

## 10. 答辩中可以怎么讲

答辩主线建议如下：

第一，介绍论文问题：motion tracking 不等于 versatile control。

第二，介绍论文 pipeline：tracking teacher、DAgger、VAE、state-latent diffusion、guidance、robot deployment。

第三，讲复现策略：公开资料优先、download 只读、代码/结果可审计、严格区分 claim level。

第四，讲环境恢复：IsaacLab/whole_body_tracking/G1 task gate 如何一步步打通。

第五，讲核心实验：released-data、official-loop replay、task diagnostic、PPO、teacher rollout、VAE、diffusion、guidance。

第六，展示视频和图表：reference replay、policy rollout、guidance rollout、tracking error、reward/done、latent PCA。

第七，讲限制：没有官方 checkpoint、没有真实 DAgger、没有 paper-level Fig.5/Fig.6、没有 TensorRT、没有实机。

第八，讲个人思考：这篇论文的真正贡献是系统组合；复现难点说明机器人学习论文需要更完整的开源 artifact。

## 11. 后续目标

下一阶段不建议盲目继续堆训练。优先目标应该是：

1. 修 tracking 数据质量，重点是 FK-repaired motion bundle、body_pos_w、endpoint z error、termination/done count；
2. 在指标合理后重跑更强 tracking PPO；
3. 用更可信 teacher 重做 teacher rollout、VAE、state-latent、diffusion 和 guidance；
4. 统一 joystick、transition、inpainting、waypoint、obstacle、composed 的本地协议表；
5. 最终把英文/中文阅读报告和项目报告整理成答辩材料。

## 11.5 当前新目标基线

根据最新审计，当前工程应该从“环境恢复和局部链路跑通”转入“tracking teacher 质量修复”。最新 master audit 仍然通过，但这不代表论文复现完成。当前最关键的证据是：FK-repaired motion bundle 已经解决了旧 `body_pos_w` 退化问题，FK-repaired PPO 也能完整训练和评估；但 checkpoint eval 的 done count 接近每步终止，说明 teacher 还不能用于可信 DAgger/VAE/diffusion。

因此新的阶段目标应该写成：先修 tracking eval 指标，再重跑强 teacher，再用强 teacher 重做 downstream。英文阅读报告和中文答辩报告则要强调“公开资源约束下的大规模可审计 partial reproduction”，而不是“完整复现 BeyondMimic”。

目前已经统一了 6 个本地 guidance proxy 任务：joystick、waypoint、obstacle avoidance、composed、transition、inpainting。其中前四个有 5-seed proxy 表，后两个是 single-seed proxy。这个统一表可以作为答辩中解释 Fig.5/Fig.6 复现边界的核心材料：我们覆盖了任务类型和本地指标，但没有官方 checkpoint、paper protocol、TensorRT 或实机，所以 claim level 必须保持为 local virtual proxy。

## 12. 结论

这个项目不是完整复现 BeyondMimic paper-level results，而是一个公开资源约束下的系统化复现、审计和分析工程。它已经证明了很多公开可运行部分，也明确记录了不可公开复现的边界。对于课程任务，它的价值在于：不仅理解论文，还实际走过了从论文公式、官方代码、环境恢复、数据替代、模型实现、闭环验证到报告总结的完整科研复现流程。
