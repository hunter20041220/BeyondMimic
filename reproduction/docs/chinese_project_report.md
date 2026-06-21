# BeyondMimic 复现项目报告

## 1. 项目从哪里开始

这个项目从阅读 BeyondMimic 论文开始。我的目标不是做一个外观相似的 demo，而是尽可能把论文拆成可验证模块：哪些能用公开资料精确复现，哪些只能做本地近似，哪些因为 checkpoint、数据或硬件不公开而无法 paper-level 复现。

项目材料分成几类：原始下载资料保持只读，旧服务器工作区作为历史快照保存，当前复现工程放在项目根目录下的代码、结果、日志、环境和报告目录中。GitHub 只提交代码、文档、小型 JSON/CSV/Markdown 审计结果；大型 checkpoint、raw rollout、视频和数据集留在本机并通过 manifest 记录。

## 2. 我如何拆解论文

我把论文拆成 10 个工作模块：

1. 论文阅读和公开数据盘点。
2. released-data 图表和表格复现。
3. 官方 `whole_body_tracking`、IsaacLab、RSL-RL 环境恢复。
4. Unitree G1 资产、motion preprocessing 和 replay。
5. PPO motion tracking teacher。
6. teacher rollout / DAgger-style dataset。
7. conditional action VAE。
8. state-latent trajectory dataset 和 latent diffusion。
9. joystick、waypoint、obstacle、transition、inpainting、composed guidance tasks。
10. ONNX/TensorRT/deployment audit、可视化和报告。

这样的拆法对应论文主线，也方便答辩时说明每一步为什么做、做到什么程度、不能声称什么。

## 3. 公式和源码实现

论文中的核心公式和机制包括 tracking objective、VAE latent action、state-latent token、diffusion denoising objective、guidance cost gradient、trajectory mask 和数据 schema。工程中用本地 `beyondmimic_reimpl` 包实现了 paper-faithful 版本，用来验证公式、shape、finite check 和模块接口。

tracking 部分优先用官方代码，不重新发明环境。遇到官方路径跑不通时，我没有直接修改下载目录，而是通过 wrapper、runtime patch、audit script 和 claim boundary 保留可追溯性。这样做的好处是：即使结果不是 paper-level，也能知道具体偏离在哪里。

## 4. 环境和任务恢复

环境分三层：

- analysis：表格、图、JSON、ONNXRuntime 和报告。
- diffusion：PyTorch CUDA、VAE、diffusion 和 guidance。
- tracking：Isaac Sim、IsaacLab、RSL-RL 和 official tracking task。

当前 IsaacLab headless gate 是 `ok`，G1 task construction gate 是 `ok_current_task_env_construction_gate`。这说明环境已经从“包层可导入”推进到“能启动 headless AppLauncher 并创建 G1 task”。但它不等于 PPO teacher 已经达到论文效果。

## 5. 数据来源和替代方案

论文需要的官方 DAgger rollout、VAE checkpoint、diffusion checkpoint 和 Fig.5/Fig.6 rollout logs 没有公开。因此我采用分层替代：

- released dataset 用于图表和表格复现。
- public LAFAN1 / G1 motions 用于 tracking 和 motion preprocessing。
- captured official-importer-export G1 USDA 用于更可信的本地 G1 资产路径。
- FK-repaired motion bundle 用于修复 `body_pos_w` 退化问题，并进一步用 robot body order 修复 URDF body order 与 IsaacLab runtime body order 不一致的问题。
- local PPO teacher 用于本地 teacher rollout。
- local VAE/diffusion/guidance 用于复现论文机制。

这些替代可以支撑课程报告和本地虚拟链路，但不能写成官方 BeyondMimic 结果。

## 6. 已完成成果

当前正式审计数字：

- master audit：`ok`，`345/345` 个 master-audited artifacts 通过。
- artifact manifest：`1415` 个 artifact。
- paper-vs-reproduction：加入 robot-order FK PPO multi-seed eval 行后为 `220` 行。
- exactly comparable：`58`。
- approximately comparable：`19`。
- qualitative-only：`130`。
- not publicly reproducible：`10`。
- requires real robot：`3`。
- completion matrix：complete `74`，partial `123`，blocked `2`，out of scope `1`。

比较可靠的成果包括 released-data 图表/表格复现、官方 tracking 代码契约审计、IsaacLab task gate、40-motion replay/task diagnostic、local PPO/VAE/diffusion/guidance 链路、统一 local proxy protocol table 和可视化材料。

## 7. 当前效果和问题

目前工程已经证明“链路能跑”，但还没有证明“论文效果复现”。tracking teacher 仍是最关键瓶颈。近期最重要的推进是定位并修复 FK-repaired bundle 的 body-order 问题：原 bundle 按 URDF 顺序写 `body_pos_w`，但 IsaacLab `MotionLoader` 按 runtime `robot.body_names` 索引，导致目标 body 错位和 endpoint z error。

修复后的 robot-order FK bundle 在 full 40-motion zero-action task diagnostic 中明显改善：done/termination 从旧 FK bundle 的 `11958/11960` 降到 `2166/11960`，mean anchor error 从约 `0.494` 降到 `0.084`，mean body-position error 从约 `0.516` 降到 `0.214`。这说明 tracking 数据质量已经比上一版更合理。

随后我基于 robot-order FK-repaired full bundle 完成了一轮新的 PPO baseline。训练使用 GPU 4/7、1000 iteration、4096 个总环境，生成 21 个 checkpoint；iteration-999 checkpoint eval 使用 2048 个环境、299 步，总计 `612352` 个虚拟环境步。当前指标是：done rate 约 `0.178`，reward mean `0.0207`，anchor-position error mean `0.0779`，body-position error mean `0.3611`，joint-position error mean `1.5733`。

这已经是当前最强 local virtual tracking baseline，明显优于旧 URDF-order FK PPO checkpoint。我也生成了一段 299 帧 policy-vs-reference rollout video：target-body error mean `0.1547`，target-body error max `0.2961`，reward mean `0.0244`，done count `44`。但它还不是论文效果。done rate 仍然不低，joint/velocity error 仍然偏高，没有官方 BeyondMimic teacher checkpoint，也没有 paper-level DAgger/Fig.5/Fig.6 结果。因此它适合进入报告和视频展示，但还不适合作为最终 downstream teacher。

为了确认这个结论不是单 seed 偶然现象，我又补了 iteration-999 checkpoint 的三 seed 完整 eval：seeds `20260730`、`20260731`、`20260732`，每个 seed 2048 个环境、299 步，总计 `1,837,056` 个虚拟环境步。多 seed 平均指标为：done rate `0.1785`，reward mean `0.02048`，anchor-position error mean `0.07762`，body-position error mean `0.35974`，joint-position error mean `1.57722`。结论是 tracking baseline 稳定但不够强，后续应先做 checkpoint sweep、termination/source diagnostic 和更长或更合理的 PPO 训练。

统一任务协议表覆盖 `6` 个本地 proxy tasks，其中前几个任务有 multi-seed 证据，transition/inpainting 仍偏单 seed 或 proxy。它适合答辩展示“我如何把论文 Fig.5/Fig.6 拆成本地协议”，但 `paper_level_reproduced_count = 0`，所以不能说复现了 Fig.5/Fig.6。

## 8. 失败产物和存储管理

项目现在保留大型成功 checkpoint、teacher rollout、state-latent shard 和可视化视频在本机，不提交 GitHub。失败运行、临时缓存和可重建中间产物需要定期清理。清理原则是：保留 summary、CSV、JSON、关键日志、manifest 和当前最佳 checkpoint；删除明确失败、临时、重复或可重建的大目录。

这件事对答辩也有意义：它说明这个项目不是只写代码，还包含多 GPU 实验平台管理、artifact boundary、GitHub 版本追溯和科研复现审计。

## 9. 答辩主线

答辩可以这样讲：

1. 先讲论文问题：motion tracking 不等于 versatile humanoid control。
2. 再讲方法：tracking teacher -> DAgger -> VAE -> state-latent diffusion -> guidance -> deployment。
3. 讲复现原则：download 只读、公开资料优先、结果分级、不能过度声称。
4. 讲环境恢复：IsaacLab/headless/G1 task gate。
5. 讲实验链路：released-data、official loop、PPO、teacher rollout、VAE、diffusion、guidance。
6. 展示图和视频：reference replay、policy rollout、guidance rollout、tracking error、reward/done、task proxy table。
7. 讲失败：tracking teacher 弱、done count 高、官方 checkpoint 缺失、TensorRT 和真实机器人不可用。
8. 讲个人思考：机器人论文复现需要代码、资产、数据、checkpoint、协议和部署细节共同开源。

## 10. 下一步计划

下一步应该回到论文主线，而不是继续为失败堆审计：

1. 围绕 robot-order FK PPO 做 checkpoint sweep、更细的 termination/source diagnostic 和更强训练。
2. 继续提高 PPO teacher，必要时调整训练长度、termination/curriculum 或 motion sampling。
3. 用更可信 teacher 重做 teacher rollout、VAE、state-latent、denoiser 和 guidance。
4. 给 joystick、waypoint、obstacle、transition、inpainting、composed 补更真实的任务指标。
5. 把英文阅读报告、中文阅读报告和项目报告整理成最终提交/答辩版本。

## 11. 结论

这个项目当前是一套公开资源约束下的大规模 BeyondMimic partial reproduction。它完成了环境、代码、公开数据、公式实现、本地虚拟实验和报告材料，并且已经把 tracking 数据质量问题推进到 robot-order FK 修复阶段；但它没有完成 paper-level BeyondMimic 全部非实机结果。最诚实、也最有价值的表述是：我复现、审计并分析了公开可复现部分，建立了 local virtual BeyondMimic-like pipeline，并明确指出了官方 checkpoint、DAgger、Fig.5/Fig.6、TensorRT 和真实机器人结果的不可公开复现边界。
