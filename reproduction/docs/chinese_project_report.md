# BeyondMimic 复现项目报告

## 1. 项目起点

这个项目从阅读 BeyondMimic 论文开始。最初的目标不是写一个外观类似的 demo，而是把论文拆成可以检查、可以运行、可以失败也能解释的模块。我的基本问题是：在没有官方完整训练 checkpoint、没有真实 DAgger rollout、没有真实机器人硬件的情况下，公开资源到底能支持我复现到什么程度？

因此项目一开始就分成三类材料：

1. 原始下载资料：论文、公开代码、IsaacLab、RSL-RL、Unitree G1 assets、公开数据集和参考仓库，只读保存。
2. 当前复现工程：脚本、配置、审计、报告、小型结果表和可视化索引。
3. 本地大型产物：环境、cache、checkpoint、rollout shards、视频和数据集，不上传 GitHub，只通过 manifest 和报告记录。

这个划分保证了两件事：第一，不污染原始资料；第二，GitHub 可以展示代码、报告和审计链，而不会把几十 GB 的权重和数据塞进去。

## 2. 我如何拆论文

我把 BeyondMimic 拆成十个工作模块：

1. 论文阅读、背景梳理和公开资料盘点。
2. released-data 图表和论文表格复现。
3. 官方 `whole_body_tracking`、IsaacLab、RSL-RL 环境恢复。
4. Unitree G1 资产、motion preprocessing、motion replay 和 task gate。
5. PPO motion tracking teacher。
6. teacher rollout / DAgger-style dataset。
7. conditional action VAE。
8. state-latent trajectory dataset 和 latent diffusion。
9. joystick、waypoint、obstacle、transition、inpainting、composed guidance tasks。
10. ONNX/TensorRT/deployment audit、可视化、英文报告和答辩材料。

这个拆法对应论文 pipeline，也对应我的复现路线。每个模块都有不同 claim level：有些能精确复现，有些只能本地虚拟复现，有些因为资料不公开只能做 proxy，有些需要真实机器人。

## 3. 公式和源码如何对应

论文里的核心机制包括 tracking objective、VAE latent action、state-latent trajectory token、diffusion denoising objective、guidance cost gradient、trajectory mask 和 evaluation protocol。我的做法不是直接凭感觉写一个模型，而是先建立 paper/source audit，再用本地 `beyondmimic_reimpl` 包实现 paper-faithful 版本。

这个实现主要服务于三件事：

1. 验证公式和 tensor shape 是否一致。
2. 检查每个模块是否有 finite output、确定性 seed 和合理接口。
3. 让 VAE、diffusion、guidance 的代码可以独立于 IsaacLab 先跑通，再接回 tracking 数据。

tracking 部分我优先使用官方代码和 IsaacLab/RSL-RL 入口，而不是重写环境。只有当官方路径在本机被 Kit、USD save policy、URDF importer 或 GPU/renderer 问题阻塞时，才通过 wrapper、runtime patch 或 local captured asset 路径继续推进，并在审计中标注偏离。

## 4. 环境恢复过程

项目环境分三层：

1. `bm_analysis`：用于 Python 表格、图表、JSON 审计、ONNX/ONNXRuntime 和报告生成。
2. `bm_diffusion`：用于 PyTorch CUDA、VAE、diffusion、guidance 和相关训练评估。
3. `bm_tracking`：用于 Isaac Sim、IsaacLab、RSL-RL、official `whole_body_tracking` 和 G1 tracking task。

当前环境已经不是“刚能 import”的状态。IsaacLab headless AppLauncher gate 已经通过，G1 task construction gate 也通过。也就是说，当前环境可以启动 headless IsaacLab，创建 G1 tracking task，检查 action/observation/reward/termination contract，并运行本地 tracking/PPO/eval 脚本。

但这不等于论文 tracking teacher 已经复现。环境恢复只是基础设施，真正的瓶颈已经转移到 motion 数据质量、termination 逻辑和 policy 质量。

## 5. 数据来源和替代方案

论文中最关键的数据和 checkpoint 并没有全部公开。比如官方 tracking teacher checkpoint、true DAgger rollout logs、官方 VAE/diffusion checkpoint、paper Fig.5/Fig.6 rollout videos/metrics 和真实机器人 logs 都不可直接获得。

因此我采用了分层替代：

- 用 released dataset 做公开图表和表格复现。
- 用 public LAFAN1 / G1 motions 做 motion preprocessing、tracking 和 VAE/diffusion 公共数据实验。
- 用 captured official-importer-export G1 USDA 做更可信的本地机器人资产路径。
- 用 FK-repaired motion bundle 修复 `body_pos_w` 退化问题。
- 用 robot-order FK repair 修复 URDF body order 与 IsaacLab runtime body order 不一致的问题。
- 用 local PPO teacher 生成本地 teacher rollout。
- 用 local VAE/diffusion/guidance 复现论文机制。

这些替代能支撑 public-resource reproduction，但不能被说成官方 BeyondMimic 结果。

## 6. 当前正式审计状态

当前最新审计基线是：

- master audit：`ok`，`350/350` 个 master-audited artifacts 通过。
- artifact manifest：`1422` 个关键 artifact。
- paper-vs-reproduction：`221` 行。
- exactly comparable：`58`。
- approximately comparable：`19`。
- qualitative-only：`131`。
- not publicly reproducible：`10`。
- requires real robot：`3`。
- completion matrix：complete `74`，partial `124`，blocked `2`，out of scope `1`。
- required artifact absence audit：`32` 行。

这些数字说明项目已经形成可审计证据链，但也说明控制结果仍然有大量 partial/qualitative-only 内容。它不是完整复现，而是大规模 partial reproduction。

## 7. 已完成的主要工作

第一，完成了论文和公开数据层面的基础复现。包括 released-data figure/table reproduction、paper table value audit、paper panel map、source coverage、formula/code trace 和 PDF/source consistency audit。这部分主要证明我认真读了论文，并把公开可比内容和源码对应起来。

第二，完成了官方 tracking 代码契约审计。包括 observation/action schema、reward terms、termination terms、domain randomization、motion preprocessing、ONNX contract、MuJoCo/ROS launch 和 deployment-controller semantics。这部分帮助我理解官方 tracking teacher 应该看到什么、输出什么、如何终止、如何部署。

第三，恢复了 IsaacLab/G1 tracking 基础设施。当前 headless gate 和 task construction gate 均通过，task contract 包含 29 维 action、160 维 policy observation、286 维 critic observation、9 个 reward term、4 个 termination term、29 个关节和 40 个 body。

第四，跑通了 public motion replay/task diagnostic。full public bundle 包含 40 个 motions、11960 帧/步。项目通过 official-loop body 检查和本地 importer-export G1 asset 路径建立了可运行的 G1 tracking 数据链。

第五，实现了本地 VAE/diffusion/guidance 链路。包括 teacher rollout、conditional VAE、state-latent dataset、denoiser/diffusion、offline guidance、reverse guidance、closed-loop proxy rollout、ONNX/async proxy 和 report assets。

第六，建立了统一 local proxy task protocol。它覆盖 joystick、waypoint、obstacle avoidance、composed、transition 和 inpainting-style tasks，用于答辩展示，但明确记录 `paper_level_reproduced_count = 0`。

## 8. Tracking 数据质量修复

tracking 是整个项目最关键的主线。没有强 tracking teacher，后面的 DAgger、VAE、diffusion 和 guidance 都只是弱 teacher 上的机制验证。

早期问题之一是 `body_pos_w` 退化。FK-repaired motion bundle 修复了一部分问题，但后面我发现更隐蔽的问题：bundle 里的 target body position 按 URDF body order 写入，而 IsaacLab runtime `MotionLoader` 按 simulator articulation 的 `robot.body_names` 读取。这会导致目标 body 错位，比如本来应该对应脚踝的目标被读成其他 link，进而出现 endpoint z error 和大量 termination。

我把 full 40-motion FK bundle 重排成 IsaacLab robot body order。这个修复在 zero-action full-split task diagnostic 中效果明显：

- done/termination 从旧 FK bundle 的 `11958/11960` 降到 `2166/11960`。
- mean anchor error 从约 `0.494` 降到 `0.084`。
- mean body-position error 从约 `0.516` 降到 `0.214`。

这个结果说明，复现控制论文时不能只看 policy loss。motion 数据语义错误会直接毁掉训练和评估。

## 9. 当前 PPO Tracking 效果

基于 robot-order FK-repaired full bundle，我完成了一轮本地 PPO baseline。训练设置是 GPU 4/7、1000 PPO iterations、4096 total envs，并生成 21 个 checkpoint。iteration-999 checkpoint eval 使用 2048 个环境、299 步，总计 `612352` virtual env steps。

当前 checkpoint eval 指标：

- reward mean：约 `0.0207`。
- done rate：约 `0.178`。
- anchor-position error mean：约 `0.0779`。
- body-position error mean：约 `0.3611`。
- joint-position error mean：约 `1.5733`。

我还生成了 299 帧 policy-vs-reference rollout video。该视频资产记录 target-body error mean `0.1547`、target-body error max `0.2961`、reward mean `0.0244`、done count `44`。它可以作为答辩中的可视化材料，但必须标注为 local virtual media。

随后我做了同一 checkpoint 的三 seed 完整评估。每个 seed 使用 2048 个环境、299 步，总计 `1,837,056` virtual env steps。多 seed 指标：

- mean done rate：`0.1785`。
- reward mean：`0.02048`。
- anchor-position error mean：`0.07762`。
- body-position error mean：`0.35974`。
- joint-position error mean：`1.57722`。

这个结果说明当前 baseline 稳定，但 teacher 仍弱。它比旧 URDF-order FK PPO 强，但不能作为 paper-level teacher。

## 10. 当前最重要的问题

最新 tracking-quality diagnostic 发现：三个 multi-seed eval 都在 step 0 出现 `2048/2048` done，step-0 body-position error spike 约 `43.29` 米。去掉 step 0 后，body-position error mean 从约 `0.360` 降到约 `0.216`，但 post-step0 done rate 仍约 `0.176`。

这说明问题不只是“训练不够久”。当前最该做的是：

1. 检查 reset 后第一帧 target 和 robot state 是否对齐。
2. 检查 motion id、body_pos_w、target body mapping 和 endpoint z。
3. 检查 `ee_body_pos` termination 的触发来源。
4. 确认 step 0 的 done 是否是 bootstrap/reset artifact。
5. 在这些问题解释清楚后，再跑更强 PPO full run。

如果现在直接用这个 teacher 重做 downstream chain，VAE/diffusion/guidance 可能只是复现了弱 teacher 的缺陷。

## 11. Downstream Chain 状态

当前已经有一套完整 local downstream chain：

1. local tracking checkpoint。
2. teacher rollout dataset。
3. conditional action VAE。
4. state-latent trajectory dataset。
5. denoiser/diffusion。
6. offline guidance。
7. closed-loop guidance rollout。
8. report-ready plots/videos/tables。

这条链路证明我已经理解并实现了 BeyondMimic 的方法结构。但由于上游 teacher 不是官方 teacher，而且当前 strongest teacher 仍不够强，所以这些 downstream 结果只能称为 local virtual/resource-adjusted/paper-faithful reimplementation，不能称为 official paper-level reproduction。

下一步应该等 tracking quality gate 修好后，用更可信 teacher 重新生成 teacher rollout、VAE、state-latent、denoiser 和 guidance 结果。

## 12. Fig.5/Fig.6 Proxy 和任务指标

论文 Fig.5/Fig.6 的重点是 guidance 能让机器人完成新任务。当前项目已经把 joystick、waypoint、obstacle、composed、transition、inpainting 等任务统一成 local proxy protocol，但现在它更像机制验证。

后续要更接近论文，需要补任务指标：

- joystick：velocity tracking error。
- waypoint：final distance 和 success rate。
- obstacle：minimum clearance、collision count、goal success。
- inpainting：keyframe error。
- transition：smoothness、fall rate。
- composed：多目标 tradeoff 和 guided-vs-unguided improvement。

只有这些指标明确显示 guided 优于 base/denoised，并且视频和曲线一致，才能说接近 paper Fig.5/Fig.6 的 simulation reproduction。

## 13. 磁盘和产物管理

项目中大型产物很多，包括 checkpoint、teacher rollout shard、state-latent dataset、视频和 cache。当前磁盘空间紧张，所以必须管理失败和重复产物。

本轮已经把 cleanup audit 改成幂等记录方式：两个 superseded same-seed duplicate directories 已按策略删除或确认不存在，记录的 previously freed bytes 为 `2368915263`。当前 active teacher rollout/state-latent run directories、checkpoint、报告、manifest 和 final report 均保留。

后续清理原则是：删除明显失败、重复、可重建、无报告价值的大目录；保留成功 checkpoint、summary JSON/CSV、关键日志、视频索引、manifest 和当前最佳模型。清理不能破坏复现证据链。

## 14. 当前能在答辩中如何表述

可以安全表述：

- 我完成了 BeyondMimic 的大规模 public-resource partial reproduction。
- 我把论文拆成 tracking teacher、DAgger-style rollout、VAE、state-latent diffusion、guidance 和 deployment audit 等模块。
- 我复现/审计了 released-data、官方代码契约、IsaacLab/G1 task、public-motion replay、本地 PPO、VAE/diffusion/guidance 和 local proxy tasks。
- 我发现并修复了一个关键 motion body-order 问题，使 tracking diagnostic 明显改善。
- 我当前最强 local PPO baseline 已经完成训练、多 seed eval 和视频，但仍不是 paper-level teacher。
- 我明确记录了官方 checkpoint、true DAgger、Fig.5/Fig.6、TensorRT 和真实机器人结果缺失。

不能表述：

- 不能说完整复现 BeyondMimic。
- 不能说复现了官方 teacher/VAE/diffusion checkpoint。
- 不能说复现了论文 Fig.5/Fig.6。
- 不能说完成 TensorRT/Mini-PC deployment。
- 不能把仿真结果说成真实机器人结果。

## 15. 下一阶段路线

下一步目标是先把 tracking teacher 质量从“链路可运行”推进到“可以支撑 downstream 数据”的水平，而不是继续在弱 teacher 上堆更多 VAE/diffusion 结果。

下一步应该回到论文主线，而不是继续堆失败审计：

1. 修 tracking reset/target alignment 和 `ee_body_pos` termination。
2. 在修复后用 GPU 4/7 跑更强 full tracking PPO。
3. 做 checkpoint sweep、多 seed eval、policy video、tracking error/reward/done curves。
4. 如果 teacher 质量提升，再重做 teacher rollout。
5. 重新训练 VAE、state-latent denoiser/diffusion。
6. 重新跑 joystick、waypoint、obstacle、transition、inpainting、composed guidance。
7. 统一 guided-vs-unguided task metrics。
8. 更新英文 reading report、中文报告、PPT 和 GitHub progress。

## 16. 总结

这个项目目前的完成度可以分三层理解：

- 严格 paper-level virtual reproduction，不含真实机器人：约 `35-45%`。
- 可审计工程和 public-resource reproduction 覆盖度：约 `70-80%`。
- 课程 reading report 和答辩材料可用度：约 `85%`。

当前结论必须诚实：This project does not fully reproduce BeyondMimic at paper level. 但它已经形成了一个可审计、可解释、能展示工作量和研究思考的 partial reproduction，并且下一步技术主线非常清楚：先修 tracking 数据和 termination，再重训更强 teacher，再重做 downstream chain。
