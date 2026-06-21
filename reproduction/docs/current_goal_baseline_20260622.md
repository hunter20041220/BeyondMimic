# BeyondMimic 当前新目标基线

## 目标更新原因

旧目标中最关键的环境 blocker 是 IsaacLab / AppLauncher headless gate。当前审计已经显示该 gate 可以到达 success sentinel，G1 task construction gate 也已经通过。因此后续目标不应继续停留在“修环境能否 import / 启动”上，而应回到论文复现主线：

1. 修 tracking 数据质量和 termination / done count。
2. 在指标合理后重新训练更可信的 tracking teacher。
3. 用更可信 teacher 重做 teacher rollout、VAE、state-latent denoiser 和 guidance rollout。
4. 把英文阅读报告、中文阅读报告和中文项目答辩报告整理成课程提交材料。

## 当前真实状态

- Master audit: `370/370` artifacts passed.
- Artifact manifest: `1485` artifacts, `0` missing.
- Paper-vs-reproduction: `227` rows.
- Comparison classes: exactly comparable `58`, approximately comparable `19`, qualitative-only `137`, not publicly reproducible `10`, requires real robot `3`.
- Completion matrix: complete `74`, partial `130`, blocked `2`, out of scope `1`.
- Goal remains incomplete.

## 当前完成度估计

为了避免一个百分比掩盖真实差异，本项目后续按三层进度表述：

- 课程 reading report / 答辩材料：约 `85-90%` 可用。
- 公开资源工程覆盖度：约 `70-75%`。
- 严格 non-robot paper-level reproduction：约 `35-45%`。

原因是：公开图表、源码审计、环境恢复、本地 pipeline 和报告材料已经很完整；但 tracking teacher 质量、true DAgger、官方 VAE/diffusion、Fig.5/Fig.6 任务协议和 TensorRT deployment 仍没有达到 paper-level。

## 目前已经做成的主线

1. 公开资料盘点、source ledger、paper/source map 和审计体系。
2. Released-data 图表/表格复现。
3. 官方 `whole_body_tracking` / IsaacLab / RSL-RL / G1 schema、reward、termination、motion preprocessing、ONNX contract 审计。
4. IsaacLab headless gate 和 G1 task construction gate。
5. 40-motion public G1 bundle 的 official-loop conversion / replay body diagnostics。
6. FK-repaired 和 robot-order FK-repaired motion bundle。
7. 本地 PPO training / eval / multi-seed eval。
8. 本地 teacher rollout、conditional VAE、state-latent denoiser、offline guidance 和 closed-loop proxy guidance。
9. Joystick、waypoint、obstacle、composed、transition、inpainting 的 local proxy protocol table。
10. 英文阅读报告、中文阅读报告和中文项目报告初版。

## 当前最大技术瓶颈

当前瓶颈不是“能不能启动 IsaacLab”，而是 tracking teacher 质量：

- robot-order FK-repaired multi-seed eval 已能跑完，但 done rate 仍高。
- step-0 body target stale 问题已经定位。
- no-advance target refresh 能显著降低 step-0 body error，但会暴露或制造 initial joint-velocity / action transient。
- 最新 reset state/action diagnostic 记录：target refresh 让 step-0 body-position error 改善约 `-43.03 m`，但 step-0 joint-velocity error 增加约 `17.83`，first-five-step action mean 增加约 `0.0718`，post-step0 done-rate 变差约 `0.0477`。

结论：当前 checkpoint 不能作为最终 DAgger / VAE / diffusion teacher。

## 除真实机器人外仍缺什么

1. 高质量 paper-level tracking teacher。
2. true DAgger rollout logs。
3. 官方或 paper-equivalent VAE checkpoint。
4. 官方或 paper-equivalent state-latent diffusion Transformer checkpoint。
5. Fig.5 / Fig.6 严格任务协议下的 closed-loop success / fall / collision / tracking metrics。
6. TensorRT engine、Mini-PC latency 和 asynchronous deployment evidence。
7. MuJoCo / ROS sim-to-sim 实际运行日志。

## 下一阶段建议目标

### Goal A: 报告和答辩材料完成

把三份报告维护为当前主线入口：

- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`

报告重点不是“完整复现论文”，而是：

> I reproduced, audited, and analyzed the publicly reproducible parts of BeyondMimic, rebuilt a local virtual BeyondMimic-like pipeline, and identified the missing artifacts required for paper-level reproduction.

### Goal B: tracking 数据质量修复

优先检查：

- reset state / target refresh / observation consistency
- initial joint velocity
- last-action observation
- endpoint z error
- `ee_body_pos` termination
- body order and `body_pos_w`

只有当 termination / done count 变合理后，才进入更大 PPO。

### Goal C: 更可信 tracking PPO

一旦 smoke / gate 成功，就直接做 full run，而不是反复小规模扩容。使用 GPU 4/7，记录：

- config
- seed
- GPU metrics
- reward/error curves
- done / termination breakdown
- policy video
- multi-seed eval

### Goal D: downstream 重做

使用更可信 teacher 重新跑：

1. teacher rollout
2. conditional VAE
3. state-latent dataset
4. denoiser / diffusion
5. guidance rollout
6. unified task protocol table

## 存储策略

当前 filesystem 剩余空间紧张。保守策略是：

- GitHub 只提交代码、文档、小型 JSON/CSV/Markdown。
- 不提交 `download/`、`other/`、`envs/`、`cache/`、`tmp/`、large logs、raw rollout、checkpoint、video。
- Active scaled teacher rollout、state-latent dataset 和当前 robot-order PPO checkpoint 暂不删除。
- 下一轮 cleanup 优先处理旧 LAFAN1/debug checkpoints、重复 superseded PPO runs 和可重建 scratch。
- 删除前必须保留 summary JSON/CSV/MD，并确保 audit 不红。

## 明确边界

当前不得声称完整复现 BeyondMimic。当前最诚实的项目定位是：

> A public-resource, audit-heavy partial reproduction and local virtual BeyondMimic-like pipeline for reading-report and research-analysis purposes.
