#!/usr/bin/env python3
"""Generate course-facing BeyondMimic reading and project reports.

The machine audit reports are intentionally exhaustive.  These course-facing
reports are shorter and more readable: they summarize the current evidence,
keep the claim boundary explicit, and avoid flooding the narrative with local
paths.  A small set of canonical paths is kept only where it helps future
maintenance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOCS = ROOT / "reproduction/docs"
FINAL = ROOT / "res/final_report"


def read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def read_first_json(*rels: str) -> dict[str, Any]:
    for rel in rels:
        path = ROOT / rel
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(", ".join(rels))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def current_stats() -> dict[str, Any]:
    master = read_json("res/master_audit/reproduction_master_audit.json")
    manifest = read_json("res/artifact_manifest/artifact_manifest.json")
    comparison = read_json("res/comparison/paper_vs_reproduction.json")
    absence = read_json("res/required_artifact_absence/required_artifact_absence_audit.json")
    fk_gate = read_json("res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.json")
    headless = read_json("res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json")
    task_gate = read_json("res/tracking/g1_current_task_env_construction_gate/tracking_g1_current_task_env_construction_gate.json")

    scaled_eval = read_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    )
    fk_eval = read_json(
        "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval.json"
    )
    split_eval = read_json(
        "res/tracking/g1_official_importer_export_fk_repaired_split_task_eval/"
        "tracking_g1_official_importer_export_fk_repaired_split_task_eval.json"
    )
    protocol = read_first_json(
        "res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json",
        "res/report_assets/unified_local_task_protocol/unified_local_task_protocol_table.json",
    )

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "master_status": master.get("status"),
        "master_artifacts": master.get("artifact_count"),
        "master_pass": master.get("artifact_pass_count"),
        "completion_counts": master.get("completion_matrix_counts", {}),
        "artifact_count": manifest.get("artifact_count"),
        "artifact_missing": manifest.get("missing_count"),
        "comparison_rows": comparison.get("total_rows"),
        "comparison_counts": comparison.get("comparison_type_counts", {}),
        "absence_rows": absence.get("row_count"),
        "absence_counts": absence.get("status_counts", {}),
        "fk_gate_status": fk_gate.get("status"),
        "fk_gate_checks": fk_gate.get("checks", {}),
        "headless_status": headless.get("status"),
        "task_gate_status": task_gate.get("status"),
        "scaled_eval_metrics": scaled_eval.get("run", {}).get("metrics", {}),
        "fk_eval_metrics": fk_eval.get("run", {}).get("metrics", {}),
        "split_eval_aggregate": split_eval.get("aggregate", {}),
        "protocol_metrics": protocol.get("metrics", {}),
        "protocol_counts": protocol.get("claim_level_counts", {}),
    }


def fmt_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def metric_value(metrics: dict[str, Any], key: str, nested: str = "mean") -> Any:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get(nested)
    return value


def reward_mean(metrics: dict[str, Any]) -> Any:
    reward = metrics.get("reward", {})
    if isinstance(reward, dict):
        mean_over_steps = reward.get("mean_over_steps", {})
        if isinstance(mean_over_steps, dict):
            return mean_over_steps.get("mean")
    return None


def done_count(metrics: dict[str, Any]) -> Any:
    return metrics.get("done_count_total")


def total_env_steps(metrics: dict[str, Any]) -> Any:
    return metrics.get("total_env_steps")


def english_report(s: dict[str, Any]) -> str:
    cc = s["comparison_counts"]
    cm = s["completion_counts"]
    ac = s["absence_counts"]
    fk_m = s["fk_eval_metrics"]
    scaled_m = s["scaled_eval_metrics"]
    protocol_m = s["protocol_metrics"]

    return f"""# BeyondMimic Reading Report

## Abstract

BeyondMimic addresses a central tension in humanoid control: motion tracking can produce physically grounded behavior, but tracking a fixed reference clip is not the same as versatile task-directed control. The paper proposes a pipeline that first learns a strong motion-tracking teacher, then distills teacher behavior into a conditional latent action space, trains a state-latent diffusion model, and finally uses test-time guidance to satisfy new objectives.

This report combines paper reading with a public-resource reproduction. The local project does not fully reproduce BeyondMimic at paper-level. Instead, it provides an auditable partial reproduction: released-data figures and tables, official tracking-code audits, IsaacLab task gates, full public-motion replay diagnostics, local PPO/VAE/diffusion/guidance experiments, proxy closed-loop rollouts, and a clear record of what remains non-public or hardware-dependent.

## 1. Why The Paper Matters

Humanoid control is difficult because balance, contacts, high-dimensional joints, embodiment constraints, and task objectives are coupled. A controller that only imitates a library motion may look impressive but remains tied to reference trajectories. BeyondMimic is interesting because it treats imitation as a source of competence rather than the final goal. The tracking teacher supplies physical behavior, the VAE compresses action choices, diffusion models feasible state-latent trajectories, and guidance turns the learned prior toward new tasks.

## 2. Method Summary

I understand the method as six connected modules:

1. Motion tracking teacher: a PPO/RSL-RL/IsaacLab policy learns to track Unitree G1 motions.
2. Teacher rollout and DAgger-style data: the teacher's closed-loop state-action distribution becomes the downstream dataset.
3. Conditional action VAE: high-dimensional actions are compressed into a low-dimensional latent conditioned on robot state.
4. State-latent trajectory dataset: states and VAE latents are organized into temporal windows.
5. Latent diffusion: a denoiser learns the distribution of feasible future trajectories.
6. Test-time guidance: task costs such as velocity, waypoint, obstacle, transition, or inpainting objectives guide diffusion samples.

The elegant part is the division of labor. Reinforcement learning handles physical execution, the VAE gives a compact controllable action interface, diffusion handles sequence generation, and guidance injects task objectives without training a new policy for every task.

## 3. Reproduction Setup

The local project uses three project-local environments: an analysis environment for audits and plots, a diffusion environment with PyTorch CUDA, and a tracking environment for Isaac Sim, IsaacLab, RSL-RL, and the official `whole_body_tracking` stack. Raw downloaded materials are kept read-only, while scripts, reports, small JSON/CSV/Markdown evidence, and GitHub-tracked code live under the reproduction workspace. Large checkpoints, videos, raw rollout shards, and datasets stay local and are summarized through manifests rather than pushed to GitHub.

The current environment state is no longer "import-only". The headless IsaacLab AppLauncher gate is `{s['headless_status']}`, and the G1 task construction gate is `{s['task_gate_status']}`. This means the project can create and reset the local G1 tracking task, but that gate alone is not a paper-level tracking result.

## 4. Current Audit State

The current machine-readable evidence set is internally consistent:

- master audit: `{s['master_status']}`, `{s['master_pass']}/{s['master_artifacts']}` artifacts passing.
- artifact manifest: `{s['artifact_count']}` hashed artifacts, missing `{s['artifact_missing']}`.
- paper-vs-reproduction table: `{s['comparison_rows']}` rows.
- comparison types: exactly comparable `{cc.get('exactly_comparable')}`, approximately comparable `{cc.get('approximately_comparable')}`, qualitative-only `{cc.get('qualitative_only')}`, not publicly reproducible `{cc.get('not_publicly_reproducible')}`, requires real robot `{cc.get('requires_real_robot')}`.
- completion matrix: complete `{cm.get('complete')}`, partial `{cm.get('partial')}`, blocked `{cm.get('blocked')}`, out of scope `{cm.get('out_of_scope')}`.
- required-artifact absence audit: `{s['absence_rows']}` rows, with {fmt_counts(ac)}.

These numbers are useful because they prevent overclaiming. A large number of artifacts and passing audits does not mean the paper is fully reproduced. It means the current evidence is traceable and the remaining gaps are explicitly documented.

## 5. What Has Been Reproduced Or Audited

The strongest exact evidence is in released-data and source-level reproduction. The project checks paper table values, released-data figures, panel mappings, formula/code traces, tracking observation and action schemas, reward and termination contracts, motion preprocessing contracts, ONNX interface contracts, and MuJoCo/ROS launch surfaces. This part is valuable because it tells us what the paper and public code actually specify.

On the tracking side, the project recovered a useful IsaacLab path. The official `csv_to_npz.py` and `replay_npz.py` loop bodies have been exercised over the full public G1 motion bundle with 40 motions and 11960 frames/steps. The captured official-importer-export G1 USDA path is stronger than the earlier generated scaffold because it comes from the Isaac Sim importer, but it is still a captured local asset path rather than a clean unmodified official converter entry.

This is the main official-loop virtual chain in the project: official-loop tracking/PPO eval begins with public G1 motions converted through the official-loop preprocessing body, replayed through the official-loop reference path, loaded into the local IsaacLab tracking task, used for local PPO training/evaluation, then connected to local teacher rollout, VAE, state-latent diffusion, and guidance experiments. I use the phrase "official-loop virtual chain" deliberately. It means the public code path and local simulation chain are substantially exercised, but the result is still virtual and resource-adjusted. It does not mean the unmodified official entrypoint, official teacher checkpoint, official DAgger dataset, or paper deployment stack has been reproduced.

Local PPO training and evaluation have been run on the public-motion bundle. The scaled official-importer-export PPO chain ran through a larger local training/evaluation protocol, and the FK-repaired chain fixed a major `body_pos_w` degeneracy in the motion bundle. However, the tracking teacher is still weak. The FK-repaired checkpoint evaluation completed but recorded reward mean about `{reward_mean(fk_m)}` and done count `{done_count(fk_m)}` over `{total_env_steps(fk_m)}` evaluated environment steps. The latest gate marks `fk_repaired_ppo_eval_done_rate_below_0_1 = {s['fk_gate_checks'].get('fk_repaired_ppo_eval_done_rate_below_0_1')}`. This is the key reason I do not treat the current policy as a paper-level teacher.

For Level C, the project implements a paper-faithful local chain: teacher rollout, conditional VAE, state-latent windows, denoiser/diffusion training, offline guidance, and local proxy closed-loop guidance. This proves that the method can be studied and partially recreated from public resources, but it is not the official BeyondMimic VAE/diffusion checkpoint chain.

## 6. Local Fig. 5 / Fig. 6 Proxy Evidence

    The project has consolidated the local guidance tasks into a unified protocol table. It covers `{protocol_m.get('task_count')}` local proxy tasks with `{protocol_m.get('multiseed_proxy_task_count')}` multi-seed proxy groups and `{protocol_m.get('single_seed_proxy_task_count')}` single-seed proxy groups. The important number is `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`. This means the local protocol is useful for analysis and presentation, but it must not be described as reproducing the paper's Fig. 5 or Fig. 6.

The current protocol is best described as a local virtual BeyondMimic-like pipeline. It covers joystick, waypoint, obstacle avoidance, composed objectives, transition, and inpainting-style proxies. The next scientific step is to make task metrics stronger: velocity error for joystick, final distance and success rate for waypoint, clearance and collision counts for obstacle avoidance, keyframe error for inpainting, transition smoothness and fall rate for transitions, and guided-vs-unguided improvement for each task.

## 7. Limitations

The major missing pieces are not cosmetic. They are the pieces that make the original paper a closed-loop humanoid-control result:

- no official BeyondMimic tracking teacher checkpoint.
- no official motion-policy ONNX export from a reproduced trained teacher.
- no true DAgger rollout logs from a mature teacher/student loop.
- no official conditional VAE checkpoint.
- no official state-latent diffusion Transformer checkpoint.
- no paper-level Fig. 5/Fig. 6 closed-loop task logs, success/failure videos, or metrics.
- no TensorRT engine, Mini-PC latency benchmark, or asynchronous deployment reproduction.
- no real Unitree G1 hardware validation.

The largest current technical blocker, excluding real robot work, is tracking quality. The pipeline runs, but the local teacher terminates too often and does not yet provide the stable rollout distribution needed for convincing DAgger, VAE, diffusion, and guidance reproduction.

This boundary also shapes how I would present the result in class. I would not say "I reproduced BeyondMimic." I would say: this project does not fully reproduce BeyondMimic at paper-level, but it reproduces and audits a large public subset, rebuilds the method as a local virtual pipeline, and identifies the exact missing artifacts needed to close the gap. That is a more useful scientific statement than a vague success claim.

## 8. Personal Reflection

This reproduction changed how I read the paper. At first the method looks like a clean sequence of modules: tracking, VAE, diffusion, guidance. In practice, every module depends on embodied details: robot assets, body names, endpoint heights, reset logic, termination thresholds, observation history, simulation stability, and data provenance. A small coordinate or body-position issue can invalidate a beautiful downstream model.

The most important lesson is that robotics reproducibility is not only about code availability. It needs assets, checkpoints, datasets, evaluation scripts, logs, videos, and deployment details. BeyondMimic is technically compelling, but the public artifact boundary makes exact reproduction impossible at several points. A good reproduction report should therefore avoid a binary "success/failure" story. The honest story is that many public components can be reproduced and analyzed, a local virtual pipeline can be built, and the remaining paper-level claims require non-public artifacts or hardware.

## 9. Conclusion

This project currently supports a strong course reading report and defense: it explains the paper, audits the public code and data, implements the main ideas in a local pipeline, and identifies where paper-level reproduction is blocked. It does not fully reproduce BeyondMimic at paper level. The next research step is to repair tracking quality, train a more reliable teacher, then rerun the downstream VAE, state-latent diffusion, and guidance experiments from that stronger teacher.
"""


def chinese_reading_report(s: dict[str, Any]) -> str:
    cc = s["comparison_counts"]
    cm = s["completion_counts"]
    ac = s["absence_counts"]
    fk_m = s["fk_eval_metrics"]
    protocol_m = s["protocol_metrics"]

    return f"""# BeyondMimic 中文阅读报告

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

- master audit：`{s['master_status']}`，`{s['master_pass']}/{s['master_artifacts']}` 通过。
- artifact manifest：`{s['artifact_count']}` 个 artifact，missing `{s['artifact_missing']}`。
- paper-vs-reproduction：`{s['comparison_rows']}` 行。
- exactly comparable：`{cc.get('exactly_comparable')}`。
- approximately comparable：`{cc.get('approximately_comparable')}`。
- qualitative-only：`{cc.get('qualitative_only')}`。
- not publicly reproducible：`{cc.get('not_publicly_reproducible')}`。
- requires real robot：`{cc.get('requires_real_robot')}`。
- completion matrix：complete `{cm.get('complete')}`，partial `{cm.get('partial')}`，blocked `{cm.get('blocked')}`，out of scope `{cm.get('out_of_scope')}`。
- required artifact absence：`{s['absence_rows']}` 行，{fmt_counts(ac)}。

这些数字说明工程很完整，但不是论文完整复现。它证明当前证据可追溯，也证明还有很多 paper-level artifact 缺失。

## 4. 已完成内容

第一，公开数据和论文表格/图表复现比较可靠。项目完成了 released-data figure/table reproduction、paper panel map、source coverage、formula/code trace 和 table value audit。这部分最接近 exact reproduction。

第二，官方 tracking 代码做了较完整审计。包括 observation/action schema、reward terms、termination、motion preprocessing、ONNX contract 和 MuJoCo/ROS launch contract。

第三，IsaacLab 和 G1 task gate 已经打通。当前 headless AppLauncher gate 是 `{s['headless_status']}`，G1 task construction gate 是 `{s['task_gate_status']}`。task contract 验证了 29 维 action、160 维 policy observation、286 维 critic observation、9 个 reward term、4 个 termination term、29 个关节和 40 个 body。

第四，官方 `csv_to_npz.py` / `replay_npz.py` 的 loop body 已经在 full public motion bundle 上跑通。40 个 public motions 合计 11960 帧/步。captured official-importer-export G1 USDA 路径比早期 scaffold 更可信，但仍不是 unmodified official converter entry。

第五，本地 PPO/VAE/diffusion/guidance 链路已经跑通。它证明了公开资源下可以实现一个 local virtual BeyondMimic-like pipeline，但不能把它说成官方 checkpoint 复现。

## 5. 当前效果

tracking 侧现在的关键结论是：链路能跑，但 teacher 还不够好。FK-repaired motion bundle 已经解决旧 `body_pos_w` 退化问题，并且覆盖 40 个 public motions；但 FK-repaired PPO checkpoint eval 仍然 reward 低、termination 高。当前 FK gate 中 `fk_repaired_ppo_eval_done_rate_below_0_1 = {s['fk_gate_checks'].get('fk_repaired_ppo_eval_done_rate_below_0_1')}`，这说明它还不能作为可信 teacher。

Level C 侧的 VAE、state-latent diffusion 和 guidance 能形成完整本地链路，但因为上游 teacher 弱，这些结果只能解释为机制复现和本地 proxy 实验。它们适合写进阅读报告，用来说明我理解并实现了论文 pipeline；但它们不能替代论文 Fig.5/Fig.6 的闭环结果。

    当前统一任务协议表覆盖 `{protocol_m.get('task_count')}` 个本地 proxy 任务，其中 `{protocol_m.get('multiseed_proxy_task_count')}` 个是 multi-seed proxy，`{protocol_m.get('single_seed_proxy_task_count')}` 个是 single-seed proxy。最重要的是 `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`。这说明 joystick、waypoint、obstacle、composed、transition、inpainting 等任务在本地机制层面被覆盖，但还没有达到论文 Fig.5/Fig.6 协议。

## 6. 主要困难

第一是 IsaacLab/Isaac Sim 环境。真实机器人学习复现不是安装 PyTorch 就结束，Kit、Vulkan、USD save policy、GPU 可见性、AppLauncher 和 extension context 都会影响结果。

第二是机器人资产和 motion preprocessing。G1 URDF/USD、body names、target bodies、endpoint z、FK、`body_pos_w` 和 MotionLoader 格式都直接影响 tracking 结果。一个看似能加载的 motion bundle 仍可能在身体位置或终止条件上出问题。

第三是官方 artifact 缺失。论文最关键的 DAgger rollout、VAE checkpoint、diffusion checkpoint、Fig.5/Fig.6 rollout logs 和 TensorRT deployment artifacts 没有公开。

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

本项目已经足够支撑一篇有独立思考的课程阅读报告：它不仅总结论文，还实际检查了代码、恢复环境、运行任务、实现公式、生成本地实验，并记录失败边界。但它不是完整 paper-level reproduction。下一步最重要的是修 tracking 数据质量和 termination/done count，得到更可信 teacher，再重做 downstream VAE、diffusion 和 guidance。
"""


def chinese_project_report(s: dict[str, Any]) -> str:
    cc = s["comparison_counts"]
    cm = s["completion_counts"]
    protocol_m = s["protocol_metrics"]

    return f"""# BeyondMimic 复现项目报告

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

当前 IsaacLab headless gate 是 `{s['headless_status']}`，G1 task construction gate 是 `{s['task_gate_status']}`。这说明环境已经从“包层可导入”推进到“能启动 headless AppLauncher 并创建 G1 task”。但它不等于 PPO teacher 已经达到论文效果。

## 5. 数据来源和替代方案

论文需要的官方 DAgger rollout、VAE checkpoint、diffusion checkpoint 和 Fig.5/Fig.6 rollout logs 没有公开。因此我采用分层替代：

- released dataset 用于图表和表格复现。
- public LAFAN1 / G1 motions 用于 tracking 和 motion preprocessing。
- captured official-importer-export G1 USDA 用于更可信的本地 G1 资产路径。
- FK-repaired motion bundle 用于修复 `body_pos_w` 退化问题。
- local PPO teacher 用于本地 teacher rollout。
- local VAE/diffusion/guidance 用于复现论文机制。

这些替代可以支撑课程报告和本地虚拟链路，但不能写成官方 BeyondMimic 结果。

## 6. 已完成成果

当前正式审计数字：

- master audit：`{s['master_pass']}/{s['master_artifacts']}` 通过。
- artifact manifest：`{s['artifact_count']}` 个 artifact。
- paper-vs-reproduction：`{s['comparison_rows']}` 行。
- exactly comparable：`{cc.get('exactly_comparable')}`。
- approximately comparable：`{cc.get('approximately_comparable')}`。
- qualitative-only：`{cc.get('qualitative_only')}`。
- not publicly reproducible：`{cc.get('not_publicly_reproducible')}`。
- requires real robot：`{cc.get('requires_real_robot')}`。
- completion matrix：complete `{cm.get('complete')}`，partial `{cm.get('partial')}`，blocked `{cm.get('blocked')}`，out of scope `{cm.get('out_of_scope')}`。

比较可靠的成果包括 released-data 图表/表格复现、官方 tracking 代码契约审计、IsaacLab task gate、40-motion replay/task diagnostic、local PPO/VAE/diffusion/guidance 链路、统一 local proxy protocol table 和可视化材料。

## 7. 当前效果和问题

目前工程已经证明“链路能跑”，但还没有证明“论文效果复现”。tracking teacher 仍是最关键瓶颈。FK-repaired motion bundle 修复了旧 body position 退化，PPO 也能完整训练和评估；但 eval 中 done/termination 仍然过高，说明 teacher 还不能作为可信 DAgger 数据源。

统一任务协议表覆盖 `{protocol_m.get('task_count')}` 个本地 proxy tasks，其中前几个任务有 multi-seed 证据，transition/inpainting 仍偏单 seed 或 proxy。它适合答辩展示“我如何把论文 Fig.5/Fig.6 拆成本地协议”，但 `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`，所以不能说复现了 Fig.5/Fig.6。

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

1. 修 tracking 数据质量，重点是 FK-repaired bundle、endpoint z、body_pos_w、reset 和 termination。
2. 指标合理后，用 GPU 4/7 重跑更强 PPO，并做 multi-seed eval、曲线和视频。
3. 用更可信 teacher 重做 teacher rollout、VAE、state-latent、denoiser 和 guidance。
4. 给 joystick、waypoint、obstacle、transition、inpainting、composed 补更真实的任务指标。
5. 把英文阅读报告、中文阅读报告和项目报告整理成最终提交/答辩版本。

## 11. 结论

这个项目当前是一套公开资源约束下的大规模 BeyondMimic partial reproduction。它完成了环境、代码、公开数据、公式实现、本地虚拟实验和报告材料，但没有完成 paper-level BeyondMimic 全部非实机结果。最诚实、也最有价值的表述是：我复现、审计并分析了公开可复现部分，建立了 local virtual BeyondMimic-like pipeline，并明确指出了官方 checkpoint、DAgger、Fig.5/Fig.6、TensorRT 和真实机器人结果的不可公开复现边界。
"""


def main() -> None:
    stats = current_stats()
    outputs = {
        DOCS / "english_reading_report.md": english_report(stats),
        DOCS / "chinese_reading_report.md": chinese_reading_report(stats),
        DOCS / "chinese_project_report.md": chinese_project_report(stats),
    }
    for path, text in outputs.items():
        write(path, text)
    for name in ["english_reading_report.md", "chinese_reading_report.md", "chinese_project_report.md"]:
        write(FINAL / name, (DOCS / name).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "ok",
                "updated": [str(p) for p in outputs],
                "final_copies": [str(FINAL / name) for name in ["english_reading_report.md", "chinese_reading_report.md", "chinese_project_report.md"]],
                "comparison_rows": stats["comparison_rows"],
                "artifact_count": stats["artifact_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
