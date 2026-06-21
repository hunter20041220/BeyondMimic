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

更具体地说，源码实现和论文模块的对应关系如下：

| 论文模块 | 本地实现/审计对象 | 当前证据边界 |
|---|---|---|
| Motion tracking teacher | IsaacLab/RSL-RL task gates、reward/termination schema、PPO train/eval wrapper | local virtual teacher；done/endpoint 仍未达 paper-level |
| Teacher rollout / DAgger | rollout shard schema、teacher action/obs/latent collection、DAgger sample audit | local teacher rollout；不是官方 DAgger 数据 |
| Conditional VAE | reparameterization、KL、action reconstruction、checkpoint smoke、teacher-rollout VAE training | paper-faithful/local training；不是官方 VAE checkpoint |
| State-latent dataset | state+latent temporal window、split/index、finite/shape checks | 来源于 local teacher；不是官方 state-latent 数据 |
| Diffusion denoiser | DDPM noise schedule、mask、Transformer denoising、held-out denoising metrics | local denoiser；不是官方 diffusion checkpoint |
| Test-time guidance | joystick/waypoint/obstacle/inpainting/transition/composed cost gradients | local proxy closed-loop/offline evidence；不是 Fig.5/Fig.6 paper-level |
| Deployment | ONNX contract、controller semantics、MuJoCo/ROS launch audit | contract-level audit；没有 TensorRT/Mini-PC/real robot |

答辩时可以把这张表当成“我不是只跑脚本，而是把论文拆成了可验证工程模块”的证据。尤其要强调：本地 `beyondmimic_reimpl` 包只负责独立数学契约，官方 IsaacLab/whole_body_tracking 仍然负责 embodied closed-loop simulation。

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
- FK-repaired motion bundle 用于修复 `body_pos_w` 退化问题。
- local PPO teacher 用于本地 teacher rollout。
- local VAE/diffusion/guidance 用于复现论文机制。

这些替代可以支撑课程报告和本地虚拟链路，但不能写成官方 BeyondMimic 结果。

我在项目里采用的证据分级如下：

| 证据层级 | 代表内容 | 能否作为论文级结果 |
|---|---|---|
| exact/public | released-data 图表、表格、源码契约、公式 trace | 可以用于公开可复现部分 |
| approximate/resource-adjusted | official-loop body、captured G1 USDA、本地 PPO/eval | 只能说明本地虚拟链路 |
| qualitative/proxy | guidance rollouts、task protocol、可视化 | 可用于分析和答辩展示 |
| missing/non-public | 官方 checkpoint、DAgger logs、Fig.5/Fig.6 logs、TensorRT | 不能声称复现 |
| hardware-only | Unitree G1 deployment | 当前不可做 |

## 6. 已完成成果

当前正式审计数字：

- master audit：`391/391` 通过。
- artifact manifest：`1537` 个 artifact。
- paper-vs-reproduction：`232` 行。
- exactly comparable：`58`。
- approximately comparable：`19`。
- qualitative-only：`142`。
- not publicly reproducible：`10`。
- requires real robot：`3`。
- completion matrix：complete `74`，partial `132`，blocked `2`，out of scope `1`。

比较可靠的成果包括 released-data 图表/表格复现、官方 tracking 代码契约审计、IsaacLab task gate、40-motion replay/task diagnostic、local PPO/VAE/diffusion/guidance 链路、统一 local proxy protocol table 和可视化材料。

## 7. 每一步是怎么做出来的

第一步是读论文和做资料盘点。我先把论文方法图拆成 tracking、DAgger、VAE、diffusion、guidance、deployment 几个模块，再把下载资料分成论文、官方代码、公开数据集、IsaacLab/RSL-RL、Unitree G1 assets 和参考仓库。这个阶段的产物是 local inventory、source ledger、paper/source map 和 unresolved details。

第二步是恢复环境。普通 Python 分析环境先跑通，然后恢复 PyTorch/CUDA diffusion 环境，最后恢复最难的 Isaac Sim/IsaacLab tracking 环境。早期遇到 inotify、Vulkan、USD save policy、URDF importer 等问题；后面通过 headless AppLauncher gate、G1 task construction gate 和 official-importer-export G1 USDA 路径把 tracking 基础设施推进到可运行状态。

第三步是先做 released-data 和源码审计，而不是直接训练。这样可以先确认论文公开数值、表格、图、reward、termination、obs/action schema 和 motion preprocessing contract，避免后面训练失败时不知道问题来自论文理解还是环境实现。

第四步是攻 tracking 数据链。最初的 enriched scaffold 和 FK-repaired bundle 只能证明链路，但后面发现 body order 和 `body_pos_w` 语义才是关键。robot-order FK repair 把 motion target 重排到 IsaacLab runtime body order，这是当前 tracking 主线的核心修复。

第五步是跑本地 PPO 和多 seed eval。robot-order PPO checkpoint eval 共 `612352` virtual env steps，reward mean `0.02073384587805606`，done count `109170`；三 seed eval 共 `1837056` virtual env steps，mean done rate `0.1785340240036232`，body-position error mean `0.3597400628005382`。这些结果说明当前 teacher 可以跑，但不够强。

第六步是做下游机制复现。因为官方 VAE/diffusion 和 DAgger 数据不公开，我用 local teacher rollout 训练 conditional VAE、state-latent denoiser 和 guidance proxy，证明 BeyondMimic-like pipeline 可以在公开资源下部分重建。

第七步是统一任务协议和报告。论文 Fig.5/Fig.6 涉及 joystick、waypoint、obstacle avoidance、transition、inpainting 和 composed objectives；我把它们整理成本地统一 protocol table，并明确 `paper_level_reproduced_count = 0`。这样答辩时可以展示“我做了哪些任务形式”，同时不把 local proxy 写成论文结果。

第八步是整理失败和边界。所有 missing checkpoint、失败 run、Vulkan/inotify/URDF/importer 问题、tracking done/termination 异常都保留为审计证据。这样做不是给失败找借口，而是让后续每一步知道应该修哪里：当前最明确的是 wrist endpoint / `ee_body_pos` termination，而不是盲目继续训练。

## 8. 当前效果和问题

目前工程已经证明“链路能跑”，但还没有证明“论文效果复现”。tracking teacher 仍是最关键瓶颈。FK-repaired motion bundle 修复了旧 body position 退化，PPO 也能完整训练和评估；但 eval 中 done/termination 仍然过高，说明 teacher 还不能作为可信 DAgger 数据源。

最新 tracking quality diagnostic 更具体：step-0 done rate 是 `1.0`，step-0 body-position error 约 `43.29219436645508` 米；去掉 step 0 后 body-position error 降到 `0.2156714241976706`，但 post-step0 done rate 仍约 `0.175777426768736`。reset command warmup 的当前结论是 `command_warmup_partially_reduces_reset_endpoint_z_spike`。

最新 full warmup eval 状态是 `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_completed`：step-0 done count 从 `2048.0` 降到 `568.0`，step-0 body-position error 从 `43.294166564941406` m 降到 `0.2640186548233032` m；但整体 done rate 从 `0.1782798129180602` 升到 `0.22864463576505017`。因此下一步不是盲目重训，而是先让 reset/target alignment、endpoint z、post-warmup policy-state distribution 和 `ee_body_pos` termination 变合理。

同 seed phase diagnostic 状态是 `ok_robot_order_fk_warmup_seed_matched_phase_diagnostic`。它说明即便 seed 对齐，warmup 仍使 total done rate 增加 `0.04325779943561875`，post-step0 done rate 增加 `0.04578210203439598`，`ee_body_pos` termination fraction 增加 `0.04554896530100333`，而 sampling top-bin 不变。这让下一步更明确：`Run a targeted reset-target refresh variant that recomputes body_pos_relative_w at reset without advancing MotionCommand.time_steps, then only run full PPO after this termination gate improves.`。

no-advance reset-target refresh 是这一轮最新主线诊断。它不调用 `command_manager.compute()` 去推进 motion phase，而是直接重算 reset 后的 body targets。live probe 状态 `ok_robot_order_fk_reset_target_refresh_no_advance_live_probe`，endpoint-z done rate `1.0` -> `0.2734375`，endpoint-z error mean `0.5298784375190735` -> `0.104344442486763`，`time_steps_unchanged_by_refresh = True`。full eval 状态 `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed`，step-0 done count delta `-1453.0`，但 total done rate `0.1782798129180602` -> `0.22340745192307693`，post-step0 done-rate delta `0.047659854760906034`。所以它把问题缩小了：reset target 陈旧是一个真问题，但 teacher 质量差还包含 reset state/action distribution、初始速度和 `ee_body_pos` termination 的问题。

现在 reset state/action distribution 也已经被具体量化：`ok_robot_order_fk_reset_state_action_distribution_diagnostic`。它说明 target refresh 虽然让 step-0 body-position error 改善 `-43.02953788638115` m，但 step-0 joint-velocity error 增加 `17.829124450683594`，first-five-step action mean 增加 `0.07184725403785702`，post-step0 done-rate delta `0.047659854760906034`，`ee_body_pos` termination fraction delta `0.0478825904055184`。这意味着下一步 full PPO 前要先修 reset-state、last-action observation、initial velocity 和 termination consistency。

最新 reset state/action consistency live probe 状态是 `ok_robot_order_fk_reset_state_action_consistency_live_probe`。它把 target refresh、action reset、action-offset alignment 和 motion-state rewrite 放在同一个 256-env live gate 里比较。target refresh alone 的 policy-step done rate 是 `0.28125`，joint velocity error 是 `14.182840347290039`；action reset 和 action-offset alignment 虽然分别把 joint velocity error 降到 `10.899185180664062` 和 `10.263128280639648`，但 done rate 变差到 `0.4765625` 和 `0.49609375`。motion-state/action-offset candidate 的 joint velocity 最低，是 `8.305423736572266`，但 done rate 最差，是 `0.73828125`。最终 `any_variant_improves_done_and_joint_velocity = False`，所以没有推荐 full eval。这一步在答辩中可以解释为：我不是为了制造成功结果而盲目重训，而是在确认修复不会破坏 termination 之前，不把它推进到正式 PPO。

最新 endpoint-group ablation 可以作为答辩里“下一步为什么要修 wrist endpoint”的直接证据：`ok_endpoint_group_ablation_completed`。同 seed 条件下，target-refresh done rate `0.22340745192307693`，ankles-only `0.1132420568561873`，wrists-only `0.18382727581521738`，all-relaxed `0.07152912050585285`，dominant endpoint group 是 `wrists`。

本轮新增 live wrist/ankle endpoint alignment probe 后，这个结论更具体：`ok_robot_order_fk_wrist_endpoint_alignment_live_probe`。probe 直接记录 `body_pos_w`、`body_pos_relative_w` 和 `robot_body_pos_w` 三组张量。target refresh 后，wrist z-error mean `0.12762290239334106` m，高于 ankle `0.06544189155101776` m；wrist done rate `0.15234375`，也高于 ankle `0.09765625`；policy step 后 wrist done rate `0.09375`，仍高于 ankle `0.06640625`。诊断是 `wrist_endpoint_target_or_body_semantics_remain_primary_done_source`。这说明下一轮 tracking 数据质量修复应该优先查 wrist endpoint target/body order、wrist FK height 和 `ee_body_pos` termination，而不是直接启动新的 downstream。

本轮又补了 deterministic reset live gate：`ok_robot_order_fk_deterministic_reset_live_probe`。它说明 deterministic reset 确实能降低一部分 joint velocity transient，比如 policy joint velocity 从 `15.347245216369629` 降到 `11.510969161987305`，但 done rate 从 `0.33203125` 变差到 `0.453125`；motion-state reset 的 done rate 也达到 `0.55078125`。因此 recommended full-eval variant 是 `none`。这进一步说明当前 blocker 不是“reset 随机性太大”这么简单，而是 body target、endpoint、初始速度、last-action observation 和 termination 的耦合问题。

统一任务协议表覆盖 `6` 个本地 proxy tasks，其中前几个任务有 multi-seed 证据，transition/inpainting 仍偏单 seed 或 proxy。它适合答辩展示“我如何把论文 Fig.5/Fig.6 拆成本地协议”，但 `paper_level_reproduced_count = 0`，所以不能说复现了 Fig.5/Fig.6。

## 9. 失败产物和存储管理

项目现在保留大型成功 checkpoint、teacher rollout、state-latent shard 和可视化视频在本机，不提交 GitHub。失败运行、临时缓存和可重建中间产物需要定期清理。清理原则是：保留 summary、CSV、JSON、关键日志、manifest 和当前最佳 checkpoint；删除明确失败、临时、重复或可重建的大目录。

当前 conservative cleanup audit 记录 `2` 个 deleted-or-previously-deleted bulky candidates，管理的已删除或确认缺席空间约 `4853459410` bytes。项目文件系统当前剩余约 `13.37` GiB / `249856.0` GiB。

本轮还把 debug-only VAE/diffusion smoke 权重纳入清理策略：这些 `.pt` 文件只证明 save/load 或 3-step optimizer plumbing，不能作为论文训练权重；删除它们后保留 JSON/TSV/metrics/figure 摘要，不影响论文复现结论，也能缓解磁盘压力。

当前最大的本地 run 目录是：

| Size | Role |
|---:|---|
| 1.79 GiB | active scaled-PPO teacher rollout shards |
| 428.25 MiB | active scaled-PPO state-latent dataset |
| 289.00 MiB | superseded LAFAN1 symmetry VAE/diffusion seed 20260623 |
| 289.00 MiB | superseded LAFAN1 symmetry VAE/diffusion base seed |
| 289.00 MiB | superseded LAFAN1 symmetry VAE/diffusion seed 20260622 |
| 289.00 MiB | superseded LAFAN1 paper-architecture VAE/diffusion seed 20260618 |
| 289.00 MiB | superseded LAFAN1 paper-architecture VAE/diffusion seed 20260619 |
| 289.00 MiB | superseded LAFAN1 paper-architecture VAE/diffusion base seed |

这轮没有直接删除 active scaled teacher rollout、scaled state-latent dataset 或当前 robot-order PPO checkpoint，因为它们仍可能服务下一轮 downstream 对照。后续如果继续 full training，应该优先处理旧 LAFAN1/debug checkpoints、重复的 superseded PPO 目录和可重建 scratch；删除前必须确认 required-artifact absence audit、report assets 和 final report 不依赖这些 raw files。

这件事对答辩也有意义：它说明这个项目不是只写代码，还包含多 GPU 实验平台管理、artifact boundary、GitHub 版本追溯和科研复现审计。

GitHub 侧的策略是只上传代码、脚本、文档、小型 JSON/CSV/Markdown 审计结果和报告。环境、download、other、cache、raw logs、checkpoint、videos、datasets、large ONNX/engine 等都不上传。每轮有效推进都写 progress Markdown、commit、push，这样可以体现版本追溯和工作量。

## 10. 答辩主线

答辩可以这样讲：

1. 先讲论文问题：motion tracking 不等于 versatile humanoid control。
2. 再讲方法：tracking teacher -> DAgger -> VAE -> state-latent diffusion -> guidance -> deployment。
3. 讲复现原则：download 只读、公开资料优先、结果分级、不能过度声称。
4. 讲环境恢复：IsaacLab/headless/G1 task gate。
5. 讲实验链路：released-data、official loop、PPO、teacher rollout、VAE、diffusion、guidance。
6. 展示图和视频：reference replay、policy rollout、guidance rollout、tracking error、reward/done、task proxy table。
7. 讲失败：tracking teacher 弱、done count 高、官方 checkpoint 缺失、TensorRT 和真实机器人不可用。
8. 讲个人思考：机器人论文复现需要代码、资产、数据、checkpoint、协议和部署细节共同开源。

## 11. 下一阶段计划

下一步应该回到论文主线，而不是继续为失败堆审计：

1. 修 tracking 数据质量，重点是 FK-repaired bundle、endpoint z、body_pos_w、reset、last-action/initial-velocity 和 termination。
2. 先用小 live probe 证明 done rate 和 joint/action transient 同时改善；一旦 smoke/gate 成功，就直接用 GPU 4/7 做 full PPO，而不是长期停在小数据集。
3. 指标合理后，做 multi-seed eval、曲线和 policy video。
4. 用更可信 teacher 重做 teacher rollout、VAE、state-latent、denoiser 和 guidance。
5. 给 joystick、waypoint、obstacle、transition、inpainting、composed 补更真实的任务指标。
6. 把英文阅读报告、中文阅读报告和项目报告整理成最终提交/答辩版本。

## 12. 结论

这个项目当前是一套公开资源约束下的大规模 BeyondMimic partial reproduction。它完成了环境、代码、公开数据、公式实现、本地虚拟实验和报告材料，但没有完成 paper-level BeyondMimic 全部非实机结果。最诚实、也最有价值的表述是：我复现、审计并分析了公开可复现部分，建立了 local virtual BeyondMimic-like pipeline，并明确指出了官方 checkpoint、DAgger、Fig.5/Fig.6、TensorRT 和真实机器人结果的不可公开复现边界。
