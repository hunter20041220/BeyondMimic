#!/usr/bin/env python3
"""Rewrite the generated report package into Chinese.

The first report generator intentionally emits an evidence-first technical
package.  This companion step turns the reader-facing Markdown/HTML/CSV files
into Chinese while keeping code identifiers, file paths, formulas, and metric
names stable for later English translation.
"""

from __future__ import annotations

import csv
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
REPORT = ROOT / "report"


def load_json(rel_path: str, default: Any | None = None) -> Any:
    path = ROOT / rel_path
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(rel_path: str, text: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(rel_path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def get(data: Any, *keys: str, default: Any = "") -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value: Any, digits: int = 6) -> str:
    if value == "" or value is None:
        return "未知"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_metrics() -> dict[str, Any]:
    return {
        "motion": load_json(
            "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"
        ),
        "training": load_json(
            "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
            "tracking_stage1_multisource_paper_contract_ppo_training_run.json"
        ),
        "sweep": load_json(
            "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
            "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"
        ),
        "rollout": load_json(
            "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
            "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
        ),
        "vae": load_json(
            "res/level_c/stage1_multisource_teacher_rollout_vae_training/"
            "level_c_stage1_multisource_teacher_rollout_vae_training.json"
        ),
        "state": load_json(
            "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/"
            "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
        ),
        "diffusion": load_json(
            "res/level_c/stage1_multisource_state_latent_diffusion_training/"
            "level_c_stage1_multisource_state_latent_diffusion_training.json"
        ),
        "guidance": load_json(
            "res/level_c/stage1_multisource_state_latent_guidance_eval/"
            "level_c_stage1_multisource_state_latent_guidance_eval.json"
        ),
        "videos": load_json(
            "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/"
            "stage1_multisource_continuous_video_suite_summary.json"
        ),
        "report_summary": load_json("report/report_generation_summary.json"),
        "master": load_json("res/master_audit/reproduction_master_audit.json"),
        "comparison": load_json("res/comparison/paper_vs_reproduction.json"),
    }


def summary_values(metrics: dict[str, Any]) -> dict[str, Any]:
    motion_metrics = get(metrics["motion"], "metrics", default={})
    sweep_metrics = get(metrics["sweep"], "metrics", default={})
    rollout_metrics = get(metrics["rollout"], "aggregate_metrics", default={})
    vae_metrics = get(metrics["vae"], "metrics", default={})
    state_metrics = get(metrics["state"], "metrics", default={})
    diffusion_metrics = get(metrics["diffusion"], "metrics", default={})
    guidance_metrics = get(metrics["guidance"], "metrics", default={})
    vae_worker = get(metrics["vae"], "worker_summary", default={})
    state_worker = get(metrics["state"], "worker_summary", default={})
    diffusion_worker = get(metrics["diffusion"], "worker_summary", default={})
    guidance_worker = get(metrics["guidance"], "worker_summary", default={})
    video_checks = get(metrics["videos"], "checks", default={})
    video_segment = get(metrics["videos"], "selected_segment", default={}) or get(
        metrics["videos"], "selected_continuous_segment", default={}
    )
    return {
        "motion_count": motion_metrics.get("motion_count"),
        "motion_hours": motion_metrics.get("total_duration_hours", motion_metrics.get("total_motion_hours")),
        "total_frames": motion_metrics.get("total_frames", motion_metrics.get("total_motion_frames")),
        "source_counts": motion_metrics.get("source_counts", motion_metrics.get("source_family_counts", {})),
        "best_iteration": sweep_metrics.get("best_iteration"),
        "best_reward_mean": sweep_metrics.get("best_reward_mean"),
        "best_done_rate": sweep_metrics.get("best_local_non_timeout_done_rate"),
        "best_body_error": sweep_metrics.get("best_error_body_pos_mean"),
        "best_joint_error": sweep_metrics.get("best_error_joint_pos_mean"),
        "rollout_steps": rollout_metrics.get("total_env_steps"),
        "rollout_done": rollout_metrics.get("done_count_total"),
        "rollout_shards": rollout_metrics.get("shard_count"),
        "vae_mse": first_value(
            vae_metrics.get("test_action_mse"),
            get(vae_worker, "evaluation", "test", "action_mse", default=None),
        ),
        "vae_abs_error": first_value(
            vae_metrics.get("test_action_abs_error_mean"),
            get(vae_worker, "evaluation", "test", "action_abs_error_mean", default=None),
        ),
        "state_windows": first_value(
            state_metrics.get("window_count"),
            get(state_worker, "dataset", "window_count", default=None),
        ),
        "state_token_dim": first_value(
            state_metrics.get("token_dim"),
            get(state_worker, "dataset", "token_dim", default=None),
        ),
        "diff_noisy": first_value(
            diffusion_metrics.get("test_noisy_token_mse"),
            get(diffusion_worker, "evaluation", "test", "noisy_token_mse", default=None),
        ),
        "diff_pred": first_value(
            diffusion_metrics.get("test_pred_token_mse"),
            get(diffusion_worker, "evaluation", "test", "pred_token_mse", default=None),
        ),
        "diff_improve": first_value(
            diffusion_metrics.get("relative_denoising_improvement"),
            get(diffusion_worker, "evaluation", "test", "denoising_improvement_ratio", default=None),
        ),
        "guidance_windows": first_value(
            guidance_metrics.get("selected_window_count"),
            guidance_metrics.get("total_selected_windows"),
            get(guidance_worker, "metrics", "total_selected_windows", default=None),
        ),
        "guidance_rows": first_value(
            guidance_metrics.get("row_count"),
            get(guidance_worker, "metrics", "row_count", default=None),
        ),
        "video_checks": video_checks,
        "video_segment": video_segment,
        "video_count_indexed": get(metrics["report_summary"], "video_count_indexed"),
        "pdf_status": get(metrics["report_summary"], "pdf_status"),
    }


def file_map_rows() -> list[dict[str, str]]:
    return [
        {
            "阅读顺序": "1",
            "文件": "report/README.md",
            "类别": "入口",
            "内容": "中文报告包总览，告诉你先读哪些文件、哪些文件只是索引或附录。",
            "什么时候看": "第一次打开 report 文件夹时先看。",
            "是否需要手改": "可以按你的课程要求继续改。",
        },
        {
            "阅读顺序": "2",
            "文件": "report/REPORT_FILE_MAP.md",
            "类别": "地图",
            "内容": "逐项解释 report 目录下每个核心文件是什么。",
            "什么时候看": "不知道某个文件用途时看。",
            "是否需要手改": "一般不用，新增文件后可更新。",
        },
        {
            "阅读顺序": "3",
            "文件": "report/report_main.md",
            "类别": "主报告",
            "内容": "中文主报告，按论文方法、数据、Stage 1、VAE、diffusion、guidance、MuJoCo、失败原因组织。",
            "什么时候看": "写中文终稿和后续英文阅读报告时主要参考。",
            "是否需要手改": "建议你后续改成自己的表达。",
        },
        {
            "阅读顺序": "4",
            "文件": "report/module_pipeline.md",
            "类别": "流程",
            "内容": "模块级流程图说明：motion tracking teacher -> rollout -> VAE -> diffusion -> guidance -> MuJoCo。",
            "什么时候看": "需要给师兄/老师讲整体流程时看。",
            "是否需要手改": "可直接改成答辩稿。",
        },
        {
            "阅读顺序": "5",
            "文件": "report/data_report.md",
            "类别": "数据",
            "内容": "解释数据从哪里来、2.5h motion bundle 怎么构成、哪些数据不能作为训练动作。",
            "什么时候看": "回答“数据集是什么、够不够、是不是论文原始数据”时看。",
            "是否需要手改": "建议保留边界声明。",
        },
        {
            "阅读顺序": "6",
            "文件": "report/code_snippets.md",
            "类别": "代码复现",
            "内容": "核心代码入口、关键实现片段和每段代码对应论文流程的说明。",
            "什么时候看": "写 code reproduction section 时看。",
            "是否需要手改": "可以补更多你想展示的代码。",
        },
        {
            "阅读顺序": "7",
            "文件": "report/pseudocode.md",
            "类别": "伪代码",
            "内容": "把整条复现主线写成中文伪代码，适合放入报告。",
            "什么时候看": "解释算法流程时看。",
            "是否需要手改": "可以转成论文式 Algorithm。",
        },
        {
            "阅读顺序": "8",
            "文件": "report/experiment_results.md",
            "类别": "结果",
            "内容": "当前指标：teacher 很弱、VAE MSE、diffusion MSE improvement、guidance proxy、MuJoCo 视频状态。",
            "什么时候看": "写实验结果/当前效果时看。",
            "是否需要手改": "后续新实验完成后需要更新。",
        },
        {
            "阅读顺序": "9",
            "文件": "report/failure_analysis.md",
            "类别": "失败分析",
            "内容": "解释为什么现在视频效果差：teacher 弱、done 高、MuJoCo adapter/root-assist/PD mismatch 等。",
            "什么时候看": "回答“为什么不能正常动、是不是训练没复现好”时看。",
            "是否需要手改": "建议保留，不要掩盖失败。",
        },
        {
            "阅读顺序": "10",
            "文件": "report/next_steps.md",
            "类别": "下一步",
            "内容": "后续应该先修 Stage 1 teacher，再重采 rollout、重训 VAE/diffusion、最后再做 guidance 视频。",
            "什么时候看": "规划下一轮实验时看。",
            "是否需要手改": "可作为新 goal 的基础。",
        },
        {
            "阅读顺序": "11",
            "文件": "report/paper_vs_project.md",
            "类别": "论文对照",
            "内容": "论文声称、本项目已做、差距和 claim level 的中文解释。",
            "什么时候看": "写论文逐项对照时看。",
            "是否需要手改": "建议保留严格措辞。",
        },
        {
            "阅读顺序": "12",
            "文件": "report/videos/video_index.md",
            "类别": "视频索引",
            "内容": "索引本地 363 个视频，说明哪些是 MuJoCo 诊断、哪些不是 paper-level。",
            "什么时候看": "找视频素材时看。",
            "是否需要手改": "一般不用。",
        },
        {
            "阅读顺序": "13",
            "文件": "report/figures/",
            "类别": "图片",
            "内容": "流程图、MSE 图、checkpoint sweep 图、视频关键帧和失败 montage。",
            "什么时候看": "做 PPT 或报告插图时看。",
            "是否需要手改": "图片由脚本生成，必要时重画。",
        },
        {
            "阅读顺序": "14",
            "文件": "report/tables/",
            "类别": "表格",
            "内容": "中文指标表、模块状态表、数据来源表、论文对照表。",
            "什么时候看": "写结果表格时看。",
            "是否需要手改": "后续新实验后更新。",
        },
        {
            "阅读顺序": "15",
            "文件": "report/report_main.html",
            "类别": "HTML",
            "内容": "中文主报告的 HTML 版。",
            "什么时候看": "浏览器里看报告时用。",
            "是否需要手改": "由脚本生成。",
        },
    ]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            vals.append(str(row.get(col, "")).replace("\n", "<br>").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_html(title: str, markdown: str) -> str:
    lines = markdown.splitlines()
    out = [
        "<!doctype html>",
        "<html lang=\"zh-CN\">",
        "<head><meta charset=\"utf-8\"><title>{}</title>".format(html.escape(title)),
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:980px;margin:40px auto;line-height:1.65;padding:0 24px;}code,pre{background:#f6f8fa;}pre{padding:12px;overflow:auto;}table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ddd;padding:6px;}h1,h2,h3{line-height:1.25;}</style>",
        "</head><body>",
    ]
    in_code = False
    for line in lines:
        if line.startswith("```"):
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            out.append(f"<p>• {html.escape(line[2:])}</p>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
        else:
            out.append("")
    if in_code:
        out.append("</code></pre>")
    out.append("</body></html>")
    return "\n".join(out)


def write_readme(vals: dict[str, Any]) -> None:
    text = f"""# BeyondMimic 中文报告包

生成/中文化时间：`{now()}`

这个 `report/` 文件夹是给你先读懂项目、整理中文终稿、后续再翻译成英文阅读报告用的。现在它的定位不是“已经完整复现论文”，而是：

1. 把 BeyondMimic 的论文流程拆清楚；
2. 把当前工程实际做过的代码、数据、训练、评估、视频和失败原因讲清楚；
3. 把“已复现 / 部分复现 / 近似复现 / 不能公开复现 / 需要真实机器人”的边界写清楚；
4. 给后续英文 reading report、PPT 和实验计划提供材料。

## 建议阅读顺序

1. `REPORT_FILE_MAP.md`：先看这个，它解释每个文件是什么。
2. `report_main.md`：中文主报告。
3. `module_pipeline.md`：整条技术流程。
4. `data_report.md`：数据来源和 2.5h motion bundle 边界。
5. `code_snippets.md` + `pseudocode.md`：代码复现和伪代码。
6. `experiment_results.md`：当前指标和效果。
7. `failure_analysis.md`：为什么视频效果差。
8. `next_steps.md`：下一步该做什么。

## 当前最重要结论

- 当前 5/6 卡 multi-source Stage 1 bundle：`{fmt(vals['motion_count'])}` 个 motion，约 `{fmt(vals['motion_hours'], 4)}` 小时。
- best PPO teacher iteration：`{fmt(vals['best_iteration'])}`。
- teacher reward mean：`{fmt(vals['best_reward_mean'])}`，body error mean：`{fmt(vals['best_body_error'])}`。
- VAE test action MSE：`{fmt(vals['vae_mse'])}`。
- diffusion noisy token MSE：`{fmt(vals['diff_noisy'])}`，pred token MSE：`{fmt(vals['diff_pred'])}`，约 `{fmt((vals['diff_improve'] or 0) * 100, 4)}%` denoising improvement。
- MuJoCo 六条 action-control 视频已经保证连续片段，但控制质量仍差，不能说成论文级复现。

## 文件地图

详见：`REPORT_FILE_MAP.md`

## HTML/PDF

- HTML：`report_main.html`
- PDF：当前没有生成，原因：`{vals['pdf_status']}`

## Claim Boundary

当前不得声称完整复现 BeyondMimic。当前视频是本地 MuJoCo 虚拟诊断证据，不是真实机器人结果，也不是官方 Isaac rendered paper videos，更不是 Fig.5/Fig.6 paper-level 结果。
"""
    write_text("report/README.md", text)


def write_main(vals: dict[str, Any]) -> str:
    text = f"""# BeyondMimic 复现工程中文技术报告

中文化时间：`{now()}`

## 0. 一句话总览

这个工程已经搭起了一条接近 BeyondMimic 思路的本地复现链路：

```text
多来源 G1 reference motions
  -> whole_body_tracking / IsaacLab PPO motion tracking teacher
  -> teacher rollout 状态-动作轨迹
  -> conditional VAE 压缩 action 到 latent action
  -> state-latent diffusion denoiser
  -> classifier / task-cost guidance
  -> MuJoCo 中 action-to-PD 闭环控制和视频诊断
```

但是当前还不能说完成论文级复现。最主要原因是 Stage 1 teacher policy 质量仍然弱，MuJoCo action-control 视频不能稳定完成正常动作。也就是说，VAE 和 diffusion 的离线指标有进步，但底层 teacher/control 分布质量不够好，导致闭环视频效果差。

## 1. 论文 BeyondMimic 的核心思路

BeyondMimic 不是简单播放动作动画，而是把 humanoid control 拆成三层：

1. **Motion tracking teacher**：先用强化学习让 G1 机器人跟踪大量人类/动画 reference motion，得到能在物理环境中输出 action 的 teacher policy。
2. **Latent action policy / VAE**：用 teacher rollout 得到状态-动作轨迹，再训练 conditional VAE，把高维 action 压缩为 latent action。
3. **State-latent diffusion + guidance**：在 state + latent 序列上训练 diffusion model，推理时用 joystick、waypoint、inpainting、obstacle cost 等任务代价做 guidance。

论文里真正关键的不是“画骨架”或“直接设置 qpos”，而是：

```text
当前机器人状态 -> policy / VAE / diffusion 输出 action -> PD controller -> 物理仿真 step -> 新状态反馈
```

本项目现在正在沿着这条主线复现，但 teacher 质量还没有达到论文级。

## 2. 本项目当前做到了哪一步

### 2.1 数据和 motion bundle

当前 Stage 1 multi-source bundle：

- motion 数量：`{fmt(vals['motion_count'])}`
- 总时长：`{fmt(vals['motion_hours'], 4)}` 小时
- 总帧数：`{fmt(vals['total_frames'])}`
- 来源统计：

```json
{jdump(vals['source_counts'])}
```

这个时长接近论文提到的约 2.5 小时 motion pool，但不能说它就是作者未公开的 exact curated 2.5h 数据集。当前包含的是本机能审计、能转换、能进入训练合同的公开/本地可用动作。

### 2.2 Stage 1 motion tracking teacher

使用 `HybridRobotics/whole_body_tracking` 相关任务、G1 obs/action 合同和 PPO 训练流程，5/6 卡 multi-source 训练完成后做了 checkpoint sweep。

当前 best checkpoint：

- best iteration：`{fmt(vals['best_iteration'])}`
- reward mean：`{fmt(vals['best_reward_mean'])}`
- local non-timeout done rate：`{fmt(vals['best_done_rate'])}`
- body-position error mean：`{fmt(vals['best_body_error'])}`
- joint-position error mean：`{fmt(vals['best_joint_error'])}`

结论：这条 teacher 可以用于打通后续流程，但质量明显不够，不能当作稳定 teacher policy，更不能当作 BeyondMimic 官方 teacher。

### 2.3 Teacher rollout 数据

用 best teacher 采集了 rollout 数据：

- rollout env steps：`{fmt(vals['rollout_steps'])}`
- shard 数：`{fmt(vals['rollout_shards'])}`
- done count：`{fmt(vals['rollout_done'])}`

这些数据可以用于本地 VAE/diffusion 训练，但它们继承了弱 teacher 的问题，因此是 partial evidence。

### 2.4 Conditional VAE

VAE 的作用是学习：

```text
obs_t, action_t -> encoder -> latent z_t
obs_t, z_t -> decoder -> reconstructed action_t
```

当前结果：

- test action MSE：`{fmt(vals['vae_mse'])}`
- test action absolute error mean：`{fmt(vals['vae_abs_error'])}`

这个指标说明 action reconstruction 离线可用，但不能证明 VAE decoder 放回物理环境后就能稳定控制。

### 2.5 State-latent diffusion

当前 state-latent dataset：

- window count：`{fmt(vals['state_windows'])}`
- token dim：`{fmt(vals['state_token_dim'])}`

Diffusion denoiser 结果：

- noisy token MSE：`{fmt(vals['diff_noisy'])}`
- pred token MSE：`{fmt(vals['diff_pred'])}`
- 相对 denoising improvement：`{fmt((vals['diff_improve'] or 0) * 100, 4)}%`

这是本轮最清楚的正向结果：模型确实学会了一定 token-level denoising。但它还是离线 token 预测，不能直接等同于 humanoid closed-loop control 成功。

### 2.6 Guidance

当前 guidance 是 offline proxy：

- selected windows：`{fmt(vals['guidance_windows'])}`
- rows/tasks：`{fmt(vals['guidance_rows'])}`

它说明 guidance cost 可以对 diffusion sample 产生非零梯度和局部改进，但还不是论文 Fig.5/Fig.6 那种真实闭环任务验证。

### 2.7 MuJoCo 视频

最新六条视频已经改成连续片段，不再是 reset 拼接或硬拉长 offline sample：

```text
reference_action_control.mp4
teacher_policy_action_control.mp4
vae_reconstructed_action_control.mp4
diffusion_denoised_latent_action_control.mp4
guided_latent_action_control.mp4
guided_vs_unguided_action_control.mp4
```

视频检查：

```json
{jdump(vals['video_checks'])}
```

但视频效果仍差，fall proxy 很高，所以它们只能写作“MuJoCo local action-control diagnostic videos”，不能写成 paper-level simulation result。

## 3. 为什么现在效果不好

当前失败不是单纯“视频脚本错了”，而是链路上游质量不足：

1. **teacher policy 弱**：reward 很低，done/fall 高，body/joint error 高。
2. **rollout 数据质量弱**：VAE 和 diffusion 学到的是弱 teacher 的 action distribution。
3. **MuJoCo adapter 仍有 gap**：IsaacLab 训练出的 obs/action/PD/action scale/termination 与 MuJoCo 控制闭环并非天然完全一致。
4. **offline 指标不能保证 closed-loop 成功**：MSE 下降只说明 token denoising 学到了统计结构，不说明机器人不会摔。

所以后续不要盲目继续堆 VAE/diffusion，而要优先把 Stage 1 teacher 修到稳定跟踪。

## 4. 当前结果可以怎么写进报告

可以写：

- 已经完成公开资料、官方 Stage 1 代码、数据来源和复现边界审计；
- 已经构建约 2.49h 的本地 multi-source motion bundle；
- 已经跑通 PPO teacher -> teacher rollout -> VAE -> state-latent dataset -> diffusion -> offline guidance -> MuJoCo diagnostic video 的完整本地链路；
- diffusion denoising 指标从 `{fmt(vals['diff_noisy'])}` 降到 `{fmt(vals['diff_pred'])}`，约 `{fmt((vals['diff_improve'] or 0) * 100, 4)}%` improvement；
- 当前闭环控制视频效果差，说明 teacher/control 质量仍是主要 blocker。

不能写：

- 不能说已经复现 BeyondMimic 完整 paper-level 结果；
- 不能说当前 MuJoCo 视频等于 Fig.5/Fig.6；
- 不能说 VAE/diffusion checkpoint 是官方 checkpoint；
- 不能说仿真结果是真实机器人结果。

## 5. 后续最重要的工作

1. 先选一个 clean single motion，把 Stage 1 teacher 练到稳定；
2. 严查 reward、termination、reset、action scale、PD gain、joint order、obs normalization；
3. teacher 稳定后重新采集 teacher rollout；
4. 用高质量 rollout 重训 VAE；
5. 再构造 state-latent dataset 和 diffusion；
6. 最后再做 receding-horizon closed-loop guidance 和 MuJoCo/Isaac 视频。

## 6. 文件导航

先看 `REPORT_FILE_MAP.md`。它告诉你每个文件是什么、哪些适合写报告、哪些只是自动清单。

## 7. Claim Boundary

当前不得声称完整复现 BeyondMimic。当前 MP4 是本地 MuJoCo 虚拟诊断视频，不是真实机器人结果，不是官方 Isaac rendered paper video，也不是 paper-level Fig.5/Fig.6 结果。
"""
    write_text("report/report_main.md", text)
    write_text("report/report_main.html", make_html("BeyondMimic 复现工程中文技术报告", text))
    return text


def write_file_map() -> None:
    rows = file_map_rows()
    columns = ["阅读顺序", "文件", "类别", "内容", "什么时候看", "是否需要手改"]
    text = "# Report 文件地图\n\n这个文件是 `report/` 的导航。你不需要一上来读所有文件，按阅读顺序走就行。\n\n"
    text += markdown_table(rows, columns)
    text += """

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
"""
    write_text("report/REPORT_FILE_MAP.md", text)
    write_csv("report/report_file_map.csv", rows, columns)


def write_data_docs(vals: dict[str, Any]) -> None:
    text = f"""# 数据来源和处理说明

## 1. 当前数据从哪里来

当前 Stage 1 训练输入不是作者私有完整数据集，而是本机可审计、可处理、可进入 `whole_body_tracking` 合同的数据：

```json
{jdump(vals['source_counts'])}
```

总计：

- motion 数：`{fmt(vals['motion_count'])}`
- 总帧数：`{fmt(vals['total_frames'])}`
- 总时长：`{fmt(vals['motion_hours'], 4)}` 小时

## 2. 为什么说“接近 2.5h”但不能说“就是论文 2.5h”

论文描述使用约 2.5 小时的多样动作训练 tracking teacher，并从中挑选代表性动作做真实机器人验证。但作者没有公开完整 curated motion list、teacher checkpoint、完整 DAgger rollout 或 diffusion training rollout。

本项目的 2.49h bundle 是公开/本地可用数据拼成的 trainable bundle，时长接近论文，但不保证动作选择、清洗规则、retargeting、joint order、质量筛选和作者完全一致。

## 3. Dataset_beyondmimic 的作用

`Dataset_beyondmimic/` 更适合做论文结果分析和 released-data 可视化，例如 GRF、IMU、ablation、rosbag/mcap replay。它不是完整 teacher/diffusion 主训练集。

可以用：

- `ablation/tkd_skill.csv`：36 列 generalized coordinate CSV，可作为 Stage 1 reference motion 或 replay reference。
- GRF/IMU/ablation CSV：用于复画论文曲线、做报告对照。
- MCAP/rosbag：用于 released real-robot state replay / visualization。

不能直接说：

- 所有 GRF CSV 都是可训练 reference motion；
- Zenodo 数据就是完整 2.5h training set；
- Zenodo 数据包含官方 teacher/VAE/diffusion checkpoint。

## 4. 当前处理流程

```text
原始/retargeted motion
  -> 检查列数、DoF、fps、joint order
  -> 生成/验证 G1 generalized coordinate motion
  -> 合并为 Stage 1 multi-source bundle
  -> 进入 whole_body_tracking PPO training
```

## 5. 证据路径

- `res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`
- `res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle_rows.tsv`
- `report/tables/dataset_inventory.csv`
- `report/tables/motion_duration_summary.csv`
"""
    for rel in [
        "report/data_report.md",
        "report/data/dataset_inventory.md",
        "report/data/data_processing_flow.md",
    ]:
        write_text(rel, text)


def write_pipeline_docs(vals: dict[str, Any]) -> None:
    text = f"""# BeyondMimic 复现流程图解

## 主线

```text
Stage 1: reference motion -> PPO motion tracking teacher
Stage 2: teacher rollout -> conditional VAE
Stage 3: state + latent trajectory -> diffusion denoiser
Stage 4: task cost / classifier guidance -> guided latent trajectory
Stage 5: VAE decoder -> action -> MuJoCo PD control -> video and metrics
```

## 每一阶段输入输出

### Stage 1：Motion Tracking Teacher

- 输入：G1 reference motion bundle、IsaacLab task、reward/termination/PPO 配置。
- 输出：PPO checkpoint / policy action。
- 当前状态：best iteration `{fmt(vals['best_iteration'])}`，但 reward 低、error 高。

### Stage 2：Teacher Rollout

- 输入：Stage 1 best teacher。
- 输出：obs/action rollout dataset。
- 当前状态：`{fmt(vals['rollout_steps'])}` env steps，done count `{fmt(vals['rollout_done'])}`。

### Stage 3：Conditional VAE

- 输入：teacher rollout 的 obs/action。
- 输出：encoder、decoder、latent action。
- 当前状态：test action MSE `{fmt(vals['vae_mse'])}`。

### Stage 4：State-Latent Diffusion

- 输入：obs + latent token windows。
- 输出：denoised state-latent sequence。
- 当前状态：MSE `{fmt(vals['diff_noisy'])}` -> `{fmt(vals['diff_pred'])}`。

### Stage 5：Guidance + MuJoCo

- 输入：当前状态、task cost、diffusion future plan、VAE decoder。
- 输出：action-to-PD closed-loop control。
- 当前状态：视频连续但动作差，仍是 failure/diagnostic。

## 关键控制公式

```text
theta_sp = theta_0 + alpha * action
tau ~= Kp * (theta_sp - theta) - Kd * theta_dot
```

注意：真正的运动控制必须输出 action 再进入物理仿真，不能直接把 reference qpos 写进机器人状态冒充 policy。
"""
    for rel in [
        "report/module_pipeline.md",
        "report/pipeline/pipeline_overview.md",
        "report/pipeline/full_pipeline_mermaid.md",
        "report/pipeline/data_flow.md",
        "report/pipeline/stage1_tracking.md",
        "report/pipeline/stage2_vae.md",
        "report/pipeline/stage3_diffusion.md",
        "report/pipeline/stage4_guidance.md",
        "report/pipeline/mujoco_or_isaac_rendering.md",
        "report/pipeline/failure_diagnosis.md",
    ]:
        write_text(rel, text)


def write_code_docs() -> None:
    text = """# 关键代码复现说明

这里不是把所有源码完整贴一遍，而是告诉你报告里应该引用哪些核心代码、每个脚本在论文流程里对应哪一步。

## 1. Stage 1 teacher checkpoint sweep

路径：

```text
reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py
reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py
```

作用：遍历 5/6 卡 multi-source PPO 训练保存的 checkpoint，用同一套 eval 合同筛选 best teacher。它对应论文第一阶段 motion tracking policy 的本地复现入口。

伪代码：

```python
for checkpoint in saved_checkpoints:
    env = make_tracking_env(task=\"Tracking-Flat-G1-v0\")
    policy = load_checkpoint(checkpoint)
    metrics = rollout_eval(env, policy)
    save_metrics(checkpoint, metrics)
select_best_checkpoint(metrics_table)
```

## 2. Teacher rollout dataset

路径：

```text
reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py
```

作用：用 best teacher 在环境里 rollout，保存 obs/action/reward/done/motion_time_steps。这一步对应 VAE 和 diffusion 的数据来源。

## 3. Conditional VAE

路径：

```text
reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py
```

核心逻辑：

```python
mu, logvar = encoder(obs, action)
z = reparameterize(mu, logvar)
action_hat = decoder(obs, z)
loss = mse(action_hat, action) + beta * kl(mu, logvar)
```

这对应论文里把 teacher 高维 action 压缩为 latent action 的模块。

## 4. State-latent dataset

路径：

```text
reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py
```

作用：把 obs 和 VAE latent 拼成 token window：

```python
token_t = concat(obs_t, z_t)
trajectory = [token_t, token_{t+1}, ...]
```

## 5. Diffusion denoiser

路径：

```text
reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py
```

核心逻辑：

```python
clean_tokens = state_latent_window
noise = sample_gaussian()
noisy_tokens = clean_tokens + sigma * noise
pred_tokens = denoiser(noisy_tokens, timestep)
loss = mse(pred_tokens, clean_tokens)
```

## 6. Guidance

路径：

```text
reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py
```

核心逻辑：

```python
for guidance_scale in scales:
    guided_sample = diffusion_sample - scale * grad(task_cost)
    score = evaluate_task_cost(guided_sample)
```

当前这一步仍是 offline proxy，不是 paper Fig.5/Fig.6 闭环控制。

## 7. MuJoCo 连续视频

路径：

```text
reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py
```

核心要求：

```text
必须选连续 motion_time_steps
必须用 action -> PD target -> mujoco step
不能直接设置 reference qpos 冒充控制
不能把 offline 21-step sample 硬拉成 15 秒视频
```

## 8. 报告生成和中文化

路径：

```text
reproduction/scripts/generate_report_package.py
reproduction/scripts/localize_report_to_chinese.py
```

`generate_report_package.py` 生成基础报告包；`localize_report_to_chinese.py` 把报告重写成中文并生成文件地图。
"""
    for rel in [
        "report/code_snippets.md",
        "report/appendix/full_code_snippets.md",
        "report/code_review/key_snippets_tracking.md",
        "report/code_review/key_snippets_vae.md",
        "report/code_review/key_snippets_diffusion.md",
        "report/code_review/key_snippets_guidance.md",
        "report/code_review/key_snippets_mujoco.md",
        "report/code_review/code_inventory.md",
        "report/code_review/key_code_index.md",
    ]:
        write_text(rel, text)


def write_pseudocode() -> None:
    text = """# 中文伪代码：BeyondMimic 本地复现主线

## Algorithm 1：Stage 1 Motion Tracking Teacher

```text
输入：G1 reference motions、IsaacLab tracking task、PPO config
输出：motion tracking teacher policy

1. 读取并验证 reference motion bundle
2. 创建 Tracking-Flat-G1-v0 环境
3. 初始化 PPO actor-critic
4. 对每个训练 iteration:
      obs <- env.reset/step 返回的机器人状态和 reference phase
      action <- policy(obs)
      theta_sp <- theta_0 + alpha * action
      env.step(theta_sp)
      reward <- DeepMimic-style tracking reward + smoothness terms
      done <- termination / timeout / fall
      PPO update
5. 每 500 iteration 保存 checkpoint
6. checkpoint sweep 选择 best teacher
```

## Algorithm 2：Teacher Rollout Collection

```text
输入：best teacher checkpoint
输出：state-action rollout dataset

for each environment shard:
    reset env
    for t in rollout horizon:
        obs_t = current observation
        action_t = teacher(obs_t)
        next_obs, reward, done = env.step(action_t)
        save(obs_t, action_t, reward, done, motion_id, motion_time_step)
```

## Algorithm 3：Conditional VAE

```text
输入：teacher rollout obs/action
输出：encoder E、decoder D

for batch in rollout_dataset:
    mu, logvar = E(obs, action)
    z = mu + eps * exp(0.5 * logvar)
    action_hat = D(obs, z)
    loss = MSE(action_hat, action) + beta * KL(q(z|obs,action) || N(0,I))
    update(E, D)
```

## Algorithm 4：State-Latent Diffusion

```text
输入：obs 序列、VAE latent z 序列
输出：denoiser

token_t = concat(obs_t, z_t)
window = [token_t, ..., token_{t+H}]

for batch in windows:
    sigma = sample_noise_level()
    noisy = window + sigma * noise
    pred = denoiser(noisy, sigma)
    loss = MSE(pred, window)
    update(denoiser)
```

## Algorithm 5：Guided Receding-Horizon Control

```text
输入：当前 MuJoCo state、历史 state-latent、task cost
输出：物理仿真中的连续 action control

while episode not done:
    current_state = read_mujoco_state()
    sample future state-latent trajectory using diffusion
    apply guidance: trajectory <- trajectory - lambda * grad(task_cost)
    z_t = first latent token from guided trajectory
    action_t = VAE_decoder(obs_t, z_t)
    theta_sp = theta_0 + alpha * action_t
    mujoco.step(theta_sp)
    render frame and log metrics
```

当前项目只部分做到 Algorithm 5；视频仍然是诊断失败结果，不是论文级成功控制。
"""
    for rel in ["report/pseudocode.md", "report/code_review/pseudocode_all_stages.md"]:
        write_text(rel, text)


def write_results(vals: dict[str, Any]) -> None:
    metrics_rows = [
        {
            "模块": "数据和 motion bundle",
            "状态": "部分完成",
            "指标": "motion_count / duration",
            "数值": f"{fmt(vals['motion_count'])} motions / {fmt(vals['motion_hours'], 4)} h",
            "证据路径": "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json",
            "说明": "接近论文 2.5h，但不是作者未公开 exact set。",
        },
        {
            "模块": "PPO teacher",
            "状态": "失败/部分完成",
            "指标": "reward / body error / joint error",
            "数值": f"{fmt(vals['best_reward_mean'])} / {fmt(vals['best_body_error'])} / {fmt(vals['best_joint_error'])}",
            "证据路径": "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json",
            "说明": "teacher 很弱，是当前主 blocker。",
        },
        {
            "模块": "Teacher rollout",
            "状态": "部分完成",
            "指标": "env steps / done count",
            "数值": f"{fmt(vals['rollout_steps'])} / {fmt(vals['rollout_done'])}",
            "证据路径": "res/tracking/stage1_multisource_best_teacher_rollout_dataset/tracking_stage1_multisource_best_teacher_rollout_dataset.json",
            "说明": "可用于本地 VAE/diffusion，但质量受 teacher 限制。",
        },
        {
            "模块": "Conditional VAE",
            "状态": "部分完成",
            "指标": "test action MSE",
            "数值": fmt(vals["vae_mse"]),
            "证据路径": "res/level_c/stage1_multisource_teacher_rollout_vae_training/level_c_stage1_multisource_teacher_rollout_vae_training.json",
            "说明": "离线重构可用，不等于闭环成功。",
        },
        {
            "模块": "Diffusion denoiser",
            "状态": "部分完成",
            "指标": "noisy MSE -> pred MSE",
            "数值": f"{fmt(vals['diff_noisy'])} -> {fmt(vals['diff_pred'])}",
            "证据路径": "res/level_c/stage1_multisource_state_latent_diffusion_training/level_c_stage1_multisource_state_latent_diffusion_training.json",
            "说明": f"约 {fmt((vals['diff_improve'] or 0) * 100, 4)}% denoising improvement。",
        },
        {
            "模块": "Guidance",
            "状态": "部分完成",
            "指标": "offline windows",
            "数值": fmt(vals["guidance_windows"]),
            "证据路径": "res/level_c/stage1_multisource_state_latent_guidance_eval/level_c_stage1_multisource_state_latent_guidance_eval.json",
            "说明": "offline proxy，不是 Fig.5/Fig.6。",
        },
        {
            "模块": "MuJoCo videos",
            "状态": "失败/部分完成",
            "指标": "continuous checks",
            "数值": jdump(vals["video_checks"]).replace("\n", " "),
            "证据路径": "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/",
            "说明": "连续但控制差，只能当失败诊断视频。",
        },
    ]
    write_csv("report/tables/metrics_summary.csv", metrics_rows, ["模块", "状态", "指标", "数值", "证据路径", "说明"])
    write_csv("report/experiments/metrics_summary.csv", metrics_rows, ["模块", "状态", "指标", "数值", "证据路径", "说明"])
    text = "# 当前实验结果\n\n" + markdown_table(metrics_rows, ["模块", "状态", "指标", "数值", "证据路径", "说明"])
    text += f"""

## 重点解释

最值得写进报告的正向指标是 diffusion denoising：

```text
noisy token MSE = {fmt(vals['diff_noisy'])}
pred token MSE  = {fmt(vals['diff_pred'])}
relative improvement = {fmt((vals['diff_improve'] or 0) * 100, 4)}%
```

但这个结果只是 token-level denoising，不代表机器人闭环控制成功。当前视频仍然差，说明 teacher/control 是主要短板。
"""
    for rel in [
        "report/experiment_results.md",
        "report/experiments/experiment_inventory.md",
        "report/experiments/tracking_analysis.md",
        "report/experiments/vae_analysis.md",
        "report/experiments/diffusion_analysis.md",
        "report/experiments/guidance_analysis.md",
        "report/experiments/mse_denoising_analysis.md",
        "report/experiments/metrics_summary.md",
    ]:
        write_text(rel, text)


def write_failure_next(vals: dict[str, Any]) -> None:
    failure = f"""# 失败分析：为什么当前运动控制视频效果差

## 1. 现象

最新六条 MuJoCo action-control 视频已经保证连续 motion segment，不再是跳变拼接；但机器人仍然不能稳定完成动作，teacher/VAE/diffusion/guided variants 都表现出明显失稳。

## 2. 主要原因排序

### 原因一：Stage 1 teacher 本身很弱

best teacher 指标：

- reward mean：`{fmt(vals['best_reward_mean'])}`
- body error mean：`{fmt(vals['best_body_error'])}`
- joint error mean：`{fmt(vals['best_joint_error'])}`
- non-timeout done rate：`{fmt(vals['best_done_rate'])}`

这说明 teacher 还没有学到高质量 motion tracking。后续 VAE/diffusion 学到的是这个弱 teacher 的 action distribution。

### 原因二：rollout 数据带有大量 done/fall 信号

teacher rollout done count：`{fmt(vals['rollout_done'])}`。这会污染 VAE 和 state-latent diffusion 的训练分布。

### 原因三：离线 MSE 和闭环控制不是一回事

VAE MSE、diffusion MSE 都是离线指标。机器人在 MuJoCo 里会受到接触、动力学、PD 控制、root height、joint limit、action scale 等因素影响。

### 原因四：IsaacLab -> MuJoCo adapter 仍可能有 contract gap

需要继续查：

- joint order
- action scale
- PD gain
- default joint pose
- obs normalization
- last action
- root twist / IMU-like state
- termination semantics
- reference phase 和 motion_time_step

## 3. 当前视频应该怎么表述

正确说法：

```text
These videos are continuous MuJoCo local action-control diagnostics.
They reveal that the current teacher/control chain is still unstable.
```

错误说法：

```text
These videos reproduce BeyondMimic Fig.5/Fig.6.
```

## 4. 结论

下一步不要优先美化视频，而是先把 Stage 1 teacher 修好。teacher 稳定之前，VAE/diffusion/guidance 的闭环视频很难好看。
"""
    next_steps = """# 下一步建议

## 第一优先级：修 Stage 1 teacher

1. 选一个最干净、最短、最容易跟踪的单一 motion。
2. 确认 reference qpos/qvel/body pose 连续且物理合理。
3. 检查 reward 和 termination，不要让 wrist/endpoint 过早终止支配训练。
4. 检查 action scale、PD gain、joint order、default pose。
5. 跑 single-motion PPO，先追求稳定动作，而不是一开始追求 2.5h 全量。

## 第二优先级：重新采集高质量 teacher rollout

teacher 能稳定后再采集 state-action trajectory。否则 VAE/diffusion 会继续学习失败动作。

## 第三优先级：重训 VAE 和 diffusion

用高质量 rollout 重新训练：

```text
teacher rollout -> VAE -> state-latent dataset -> diffusion denoiser
```

## 第四优先级：重做 closed-loop guidance 视频

先做：

1. reference replay
2. teacher policy rollout
3. VAE decoder rollout
4. diffusion unguided rollout
5. guided rollout
6. guided-vs-unguided comparison

每条视频都必须保证：

- 单一连续 motion 或连续 receding-horizon context；
- action -> PD target -> MuJoCo step；
- 不直接写 qpos 冒充控制；
- 不把 21-step offline sample 硬拉成十几秒视频。
"""
    for rel in ["report/failure_analysis.md", "report/logs/failure_logs/failure_analysis.md"]:
        write_text(rel, failure)
    for rel in ["report/next_steps.md", "report/limitations_and_next_steps.md"]:
        write_text(rel, next_steps)


def write_alignment_docs(vals: dict[str, Any]) -> None:
    rows = [
        {
            "论文模块": "Stage 1 motion tracking",
            "本项目状态": "部分完成但质量弱",
            "证据": "5/6 multi-source PPO checkpoint sweep",
            "差距": "teacher reward 低、done/error 高；不是官方 teacher。",
            "claim level": "local virtual partial",
        },
        {
            "论文模块": "DAgger / teacher rollout",
            "本项目状态": "部分完成",
            "证据": f"{fmt(vals['rollout_steps'])} rollout samples",
            "差距": "不是官方 DAgger 数据，且继承弱 teacher。",
            "claim level": "local weak-teacher dataset",
        },
        {
            "论文模块": "Conditional VAE",
            "本项目状态": "离线复现",
            "证据": f"test action MSE {fmt(vals['vae_mse'])}",
            "差距": "没有 paper-level closed-loop VAE rollout。",
            "claim level": "offline approximate",
        },
        {
            "论文模块": "State-latent diffusion",
            "本项目状态": "离线复现",
            "证据": f"MSE {fmt(vals['diff_noisy'])} -> {fmt(vals['diff_pred'])}",
            "差距": "没有官方 checkpoint / 完整 paper architecture strict eval。",
            "claim level": "offline approximate",
        },
        {
            "论文模块": "Classifier guidance",
            "本项目状态": "offline proxy",
            "证据": f"{fmt(vals['guidance_windows'])} windows",
            "差距": "不是 Fig.5/Fig.6 closed-loop protocol。",
            "claim level": "qualitative/proxy",
        },
        {
            "论文模块": "MuJoCo/Isaac video",
            "本项目状态": "MuJoCo diagnostic",
            "证据": "六条连续 MuJoCo action-control MP4",
            "差距": "控制质量差；H20 Isaac rendered MP4 blocked。",
            "claim level": "failed/diagnostic local virtual",
        },
        {
            "论文模块": "Real robot",
            "本项目状态": "未做",
            "证据": "无硬件",
            "差距": "没有 Unitree G1 实机。",
            "claim level": "requires_real_robot",
        },
    ]
    write_csv("report/tables/paper_project_comparison.csv", rows, ["论文模块", "本项目状态", "证据", "差距", "claim level"])
    write_csv("report/data/paper_project_comparison.csv", rows, ["论文模块", "本项目状态", "证据", "差距", "claim level"])
    text = "# 论文和本项目对照\n\n" + markdown_table(rows, ["论文模块", "本项目状态", "证据", "差距", "claim level"])
    text += "\n\n结论：本项目已经形成了可审计的 local reproduction chain，但 paper-level 完整复现仍未完成。"
    for rel in ["report/paper_alignment.md", "report/paper_vs_project.md", "report/reproduction_status.md"]:
        write_text(rel, text)


def write_appendix_docs(vals: dict[str, Any]) -> None:
    equations = """# 公式和符号说明

## PD target action

```text
theta_sp = theta_0 + alpha * action
tau ~= Kp * (theta_sp - theta) - Kd * theta_dot
```

## VAE loss

```text
L = MSE(action_hat, action) + beta * KL(q(z|obs, action) || N(0, I))
```

## Diffusion denoising

```text
clean_token = concat(obs, latent)
noisy_token = clean_token + sigma * noise
loss = MSE(denoiser(noisy_token, sigma), clean_token)
```

## Guidance

```text
guided_sample = sample - lambda * grad(task_cost(sample))
```
"""
    write_text("report/appendix/equations.md", equations)
    write_text(
        "report/appendix/references.md",
        """# 参考资料

- BeyondMimic paper PDF：`download/papers/BeyondMimic_2508.08241.pdf`
- BeyondMimic paper source：`download/papers/BeyondMimic_2508.08241_source.tar`
- whole_body_tracking：`download/official/whole_body_tracking`
- motion_tracking_controller：`download/official/motion_tracking_controller`
- LAFAN1 retargeted G1：`download/official/LAFAN1_Retargeting_Dataset/g1`
- BeyondMimic released dataset：`Dataset_beyondmimic/`
""",
    )
    write_text(
        "report/appendix/environment_summary.md",
        """# 环境摘要

- `bm_diffusion`：PyTorch/CUDA 可用于 VAE/diffusion 训练。
- `bm_tracking`：IsaacLab/whole_body_tracking 包层可用，但 H20 true Isaac rendered MP4 仍受 Vulkan/Hydra/Kit 渲染栈限制。
- `mujoco_mp4/.venv`：MuJoCo offscreen rendering 可生成本地诊断视频。
""",
    )
    write_text(
        "report/appendix/command_history.md",
        """# 本轮关键命令

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/localize_report_to_chinese.py
```
""",
    )
    write_text(
        "report/appendix/unresolved_details.md",
        """# 当前未解决问题

1. Stage 1 teacher 质量弱。
2. MuJoCo action-control 视频不能稳定完成动作。
3. 官方 DAgger rollout、官方 VAE/diffusion checkpoint 未公开。
4. H20 上 true Isaac rendered MP4 blocked。
5. 没有真实 Unitree G1 硬件验证。
""",
    )


def write_video_docs(vals: dict[str, Any]) -> None:
    video_text = f"""# 视频索引说明

本项目本地索引到 `{fmt(vals['video_count_indexed'])}` 个视频。注意：视频数量多不代表 paper-level 复现完成。

## 最新六条 MuJoCo action-control 诊断视频

路径根目录：

```text
res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/
```

六条视频：

```text
reference_action_control.mp4
teacher_policy_action_control.mp4
vae_reconstructed_action_control.mp4
diffusion_denoised_latent_action_control.mp4
guided_latent_action_control.mp4
guided_vs_unguided_action_control.mp4
```

当前检查：

```json
{jdump(vals['video_checks'])}
```

这些视频是连续片段，但控制质量差。报告中应写为“MuJoCo local diagnostic videos”，不要写成 BeyondMimic paper-level Fig.5/Fig.6。
"""
    for rel in ["report/video_index.md", "report/videos/video_index.md"]:
        write_text(rel, video_text)


def write_csv_tables(vals: dict[str, Any]) -> None:
    dataset_rows = []
    for name, count in (vals["source_counts"] or {}).items():
        dataset_rows.append(
            {
                "数据来源": name,
                "数量": count,
                "用途": "Stage 1 reference motion / 本地训练输入",
                "说明": "可审计公开/本地可用数据；不等于作者私有 exact curated set。",
            }
        )
    write_csv("report/tables/dataset_inventory.csv", dataset_rows, ["数据来源", "数量", "用途", "说明"])
    write_csv("report/data/data_provenance_table.csv", dataset_rows, ["数据来源", "数量", "用途", "说明"])
    module_rows = [
        {"模块": "数据处理", "状态": "部分完成", "说明": "2.49h multi-source bundle 已构建。"},
        {"模块": "Stage 1 PPO teacher", "状态": "失败/部分完成", "说明": "训练完成但 teacher 弱。"},
        {"模块": "Teacher rollout", "状态": "部分完成", "说明": "数据已采集，但质量受 teacher 限制。"},
        {"模块": "VAE", "状态": "部分完成", "说明": "离线 action reconstruction 有结果。"},
        {"模块": "Diffusion", "状态": "部分完成", "说明": "token-level denoising 有改善。"},
        {"模块": "Guidance", "状态": "partial proxy", "说明": "当前是 offline proxy。"},
        {"模块": "MuJoCo 视频", "状态": "失败/诊断", "说明": "连续但控制质量差。"},
    ]
    write_csv("report/tables/module_status.csv", module_rows, ["模块", "状态", "说明"])


def write_misc(vals: dict[str, Any]) -> None:
    write_text(
        "report/executive_summary.md",
        f"""# 执行摘要

本项目已经把 BeyondMimic 的公开可复现部分和本地近似控制链路串起来，但还没有完成 paper-level 复现。当前最明确的正向结果是 diffusion denoising：MSE 从 `{fmt(vals['diff_noisy'])}` 降到 `{fmt(vals['diff_pred'])}`，约 `{fmt((vals['diff_improve'] or 0) * 100, 4)}%` 改善。当前最主要的问题是 Stage 1 teacher 质量弱，导致 MuJoCo action-control 视频效果差。
""",
    )
    write_text(
        "report/logs/log_inventory.md",
        "# 日志索引说明\n\n这个文件原本用于索引日志。中文报告正文不需要逐条阅读日志；需要查失败原因时优先看 `report/failure_analysis.md` 和 `reproduction/PROGRESS.md`。\n",
    )
    write_text(
        "report/logs_summary/log_inventory.md",
        "# 日志摘要\n\n日志用于审计命令、失败原因和路径，不是正文材料。当前主失败仍是 teacher/control quality 和 H20 Isaac rendering stack。\n",
    )


def update_summary() -> None:
    summary_path = ROOT / "report/report_generation_summary.json"
    summary = load_json("report/report_generation_summary.json")
    summary["language"] = "zh-CN"
    summary["localized_at"] = now()
    summary["localization_script"] = "reproduction/scripts/localize_report_to_chinese.py"
    summary["file_map"] = "report/REPORT_FILE_MAP.md"
    summary["claim_boundary_cn"] = (
        "当前不得声称完整复现 BeyondMimic；中文报告包用于阅读、汇报和后续英文翻译。"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    metrics = load_metrics()
    vals = summary_values(metrics)
    write_file_map()
    write_readme(vals)
    write_main(vals)
    write_data_docs(vals)
    write_pipeline_docs(vals)
    write_code_docs()
    write_pseudocode()
    write_results(vals)
    write_failure_next(vals)
    write_alignment_docs(vals)
    write_appendix_docs(vals)
    write_video_docs(vals)
    write_csv_tables(vals)
    write_misc(vals)
    update_summary()
    print(
        json.dumps(
            {
                "status": "ok",
                "language": "zh-CN",
                "report_main": str(REPORT / "report_main.md"),
                "file_map": str(REPORT / "REPORT_FILE_MAP.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
