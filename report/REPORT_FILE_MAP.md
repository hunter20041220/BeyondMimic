# Report 文件地图

这个文件是 `report/` 的导航。你不需要一上来读所有文件，按阅读顺序走就行。

| 阅读顺序 | 文件 | 类别 | 内容 | 什么时候看 | 是否需要手改 |
| --- | --- | --- | --- | --- | --- |
| 1 | report/README.md | 入口 | 中文报告包总览，告诉你先读哪些文件、哪些文件只是索引或附录。 | 第一次打开 report 文件夹时先看。 | 可以按你的课程要求继续改。 |
| 2 | report/REPORT_FILE_MAP.md | 地图 | 逐项解释 report 目录下每个核心文件是什么。 | 不知道某个文件用途时看。 | 一般不用，新增文件后可更新。 |
| 3 | report/report_main.md | 主报告 | 中文主报告，按论文方法、数据、Stage 1、VAE、diffusion、guidance、MuJoCo、失败原因组织。 | 写中文终稿和后续英文阅读报告时主要参考。 | 建议你后续改成自己的表达。 |
| 4 | report/module_pipeline.md | 流程 | 模块级流程图说明：motion tracking teacher -> rollout -> VAE -> diffusion -> guidance -> MuJoCo。 | 需要给师兄/老师讲整体流程时看。 | 可直接改成答辩稿。 |
| 5 | report/data_report.md | 数据 | 解释数据从哪里来、2.5h motion bundle 怎么构成、哪些数据不能作为训练动作。 | 回答“数据集是什么、够不够、是不是论文原始数据”时看。 | 建议保留边界声明。 |
| 6 | report/code_snippets.md | 代码复现 | 核心代码入口、关键实现片段和每段代码对应论文流程的说明。 | 写 code reproduction section 时看。 | 可以补更多你想展示的代码。 |
| 7 | report/pseudocode.md | 伪代码 | 把整条复现主线写成中文伪代码，适合放入报告。 | 解释算法流程时看。 | 可以转成论文式 Algorithm。 |
| 8 | report/experiment_results.md | 结果 | 当前指标：teacher 很弱、VAE MSE、diffusion MSE improvement、guidance proxy、MuJoCo 视频状态。 | 写实验结果/当前效果时看。 | 后续新实验完成后需要更新。 |
| 9 | report/failure_analysis.md | 失败分析 | 解释为什么现在视频效果差：teacher 弱、done 高、MuJoCo adapter/root-assist/PD mismatch 等。 | 回答“为什么不能正常动、是不是训练没复现好”时看。 | 建议保留，不要掩盖失败。 |
| 10 | report/next_steps.md | 下一步 | 后续应该先修 Stage 1 teacher，再重采 rollout、重训 VAE/diffusion、最后再做 guidance 视频。 | 规划下一轮实验时看。 | 可作为新 goal 的基础。 |
| 11 | report/paper_vs_project.md | 论文对照 | 论文声称、本项目已做、差距和 claim level 的中文解释。 | 写论文逐项对照时看。 | 建议保留严格措辞。 |
| 12 | report/videos/video_index.md | 视频索引 | 索引本地 363 个视频，说明哪些是 MuJoCo 诊断、哪些不是 paper-level。 | 找视频素材时看。 | 一般不用。 |
| 13 | report/figures/ | 图片 | 流程图、MSE 图、checkpoint sweep 图、视频关键帧和失败 montage。 | 做 PPT 或报告插图时看。 | 图片由脚本生成，必要时重画。 |
| 14 | report/tables/ | 表格 | 中文指标表、模块状态表、数据来源表、论文对照表。 | 写结果表格时看。 | 后续新实验后更新。 |
| 15 | report/report_main.html | HTML | 中文主报告的 HTML 版。 | 浏览器里看报告时用。 | 由脚本生成。 |

## 自动清单类文件

下面这些文件主要是脚本自动生成的索引，不建议当正文读：

- `report/file_inventory.txt`
- `report/file_tree_depth4.txt`
- `report/all_relevant_files.txt`
- `report/disk_usage_depth2.txt`
- `report/disk_usage_depth3.txt`
- `report/logs/log_inventory.csv`
- `report/videos/video_index.csv`

它们的作用是审计和查路径，不是课程报告正文。

## 图片怎么用

- `report/figures/pipeline_overview.png`：总流程图。
- `report/figures/denoising_mse_improvement.png`：diffusion MSE 改善图。
- `report/figures/metric_plots/stage1_checkpoint_sweep.png`：teacher checkpoint sweep 指标图。
- `report/figures/failure_montage.png`：当前失败视频关键帧拼图。
- `report/figures/video_frames/`：六条 MuJoCo 诊断视频的抽帧。

## 最适合写进中文终稿的文件

1. `report/report_main.md`
2. `report/data_report.md`
3. `report/module_pipeline.md`
4. `report/code_snippets.md`
5. `report/pseudocode.md`
6. `report/experiment_results.md`
7. `report/failure_analysis.md`
8. `report/paper_vs_project.md`
9. `report/next_steps.md`
