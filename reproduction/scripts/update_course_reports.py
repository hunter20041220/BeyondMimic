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
import shutil
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


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def bytes_to_human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def top_run_storage_rows(limit: int = 8) -> list[dict[str, Any]]:
    runs = ROOT / "res/runs"
    rows: list[dict[str, Any]] = []
    if not runs.is_dir():
        return rows
    for child in sorted(p for p in runs.iterdir() if p.is_dir()):
        size = directory_size(child)
        rows.append(
            {
                "path": child.relative_to(ROOT).as_posix(),
                "size_bytes": size,
                "size_human": bytes_to_human(size),
            }
        )
    return sorted(rows, key=lambda row: row["size_bytes"], reverse=True)[:limit]


def friendly_storage_label(rel_path: str) -> str:
    labels = {
        "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset": (
            "active scaled-PPO teacher rollout shards"
        ),
        "res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset": (
            "active scaled-PPO state-latent dataset"
        ),
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_seed_20260623_static_000_20260617_215500": (
            "superseded LAFAN1 symmetry VAE/diffusion seed 20260623"
        ),
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500": (
            "superseded LAFAN1 symmetry VAE/diffusion base seed"
        ),
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_seed_20260622_static_000_20260617_215500": (
            "superseded LAFAN1 symmetry VAE/diffusion seed 20260622"
        ),
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_seed_20260618_static_000_20260617_203000": (
            "superseded LAFAN1 paper-architecture VAE/diffusion seed 20260618"
        ),
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_seed_20260619_static_000_20260617_203000": (
            "superseded LAFAN1 paper-architecture VAE/diffusion seed 20260619"
        ),
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000": (
            "superseded LAFAN1 paper-architecture VAE/diffusion base seed"
        ),
    }
    return labels.get(rel_path, rel_path.split("/")[-1])


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
    robot_order_eval = read_json(
        "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
    )
    robot_order_multiseed = read_json(
        "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval.json"
    )
    robot_order_quality = read_json(
        "res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic/"
        "robot_order_fk_ppo_tracking_quality_diagnostic.json"
    )
    reset_warmup = read_json(
        "res/tracking/robot_order_fk_reset_command_warmup_live_probe/"
        "robot_order_fk_reset_command_warmup_live_probe.json"
    )
    warmup_eval = read_json(
        "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
    )
    warmup_phase = read_json(
        "res/tracking/robot_order_fk_warmup_seed_matched_phase_diagnostic/"
        "robot_order_fk_warmup_seed_matched_phase_diagnostic.json"
    )
    target_refresh_live = read_json(
        "res/tracking/robot_order_fk_reset_target_refresh_no_advance_live_probe/"
        "robot_order_fk_reset_target_refresh_no_advance_live_probe.json"
    )
    target_refresh = read_json(
        "res/tracking/"
        "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json"
    )
    reset_state_action = read_json(
        "res/tracking/robot_order_fk_reset_state_action_distribution_diagnostic/"
        "robot_order_fk_reset_state_action_distribution_diagnostic.json"
    )
    reset_state_action_consistency = read_json(
        "res/tracking/robot_order_fk_reset_state_action_consistency_live_probe/"
        "robot_order_fk_reset_state_action_consistency_live_probe.json"
    )
    wrist_endpoint = read_json(
        "res/tracking/robot_order_fk_wrist_endpoint_alignment_live_probe/"
        "robot_order_fk_wrist_endpoint_alignment_live_probe.json"
    )
    wrist_source = read_json(
        "res/tracking/robot_order_fk_wrist_endpoint_source_full_diagnostic/"
        "robot_order_fk_wrist_endpoint_source_full_diagnostic.json"
    )
    endpoint_group = read_json(
        "res/tracking/"
        "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.json"
    )
    deterministic_reset = read_json(
        "res/tracking/robot_order_fk_deterministic_reset_live_probe/"
        "robot_order_fk_deterministic_reset_live_probe.json"
    )
    protocol = read_first_json(
        "res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json",
        "res/report_assets/unified_local_task_protocol/unified_local_task_protocol_table.json",
    )
    cleanup = read_json("res/storage_cleanup/cleanup_failed_large_artifacts.json")
    absence_counts = absence.get("status_counts") or count_by_key(absence.get("rows", []), "status")
    disk = shutil.disk_usage(ROOT)
    storage_top = top_run_storage_rows()

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
        "absence_counts": absence_counts,
        "fk_gate_status": fk_gate.get("status"),
        "fk_gate_checks": fk_gate.get("checks", {}),
        "headless_status": headless.get("status"),
        "task_gate_status": task_gate.get("status"),
        "scaled_eval_metrics": scaled_eval.get("run", {}).get("metrics", {}),
        "robot_order_eval_metrics": robot_order_eval.get("run", {}).get("metrics", {}),
        "robot_order_eval_status": robot_order_eval.get("status"),
        "robot_order_multiseed_aggregate": robot_order_multiseed.get("aggregate", {}),
        "robot_order_multiseed_metrics": robot_order_multiseed.get("metrics", {}),
        "robot_order_quality_aggregate": robot_order_quality.get("aggregate", {}),
        "reset_warmup_checks": reset_warmup.get("checks", {}),
        "reset_warmup_interpretation": reset_warmup.get("interpretation", {}),
        "warmup_eval_status": warmup_eval.get("status"),
        "warmup_eval_config": warmup_eval.get("config", {}),
        "warmup_eval_metrics": warmup_eval.get("run", {}).get("metrics", {}),
        "warmup_eval_comparison": warmup_eval.get("comparison_to_non_warmup_eval", {}),
        "warmup_eval_interpretation": warmup_eval.get("interpretation", {}),
        "warmup_phase_status": warmup_phase.get("status"),
        "warmup_phase_metrics": warmup_phase.get("metrics", {}),
        "warmup_phase_interpretation": warmup_phase.get("interpretation", {}),
        "target_refresh_live_status": target_refresh_live.get("status"),
        "target_refresh_live_metrics": target_refresh_live.get("metrics", {}),
        "target_refresh_status": target_refresh.get("status"),
        "target_refresh_comparison": target_refresh.get("comparison_to_non_warmup_eval", {}),
        "target_refresh_checks": target_refresh.get("checks", {}),
        "target_refresh_interpretation": target_refresh.get("interpretation", {}),
        "reset_state_action_status": reset_state_action.get("status"),
        "reset_state_action_metrics": reset_state_action.get("metrics", {}),
        "reset_state_action_interpretation": reset_state_action.get("interpretation", {}),
        "reset_state_action_consistency_status": reset_state_action_consistency.get("status"),
        "reset_state_action_consistency_metrics": reset_state_action_consistency.get("metrics", {}),
        "reset_state_action_consistency_checks": reset_state_action_consistency.get("checks", {}),
        "reset_state_action_consistency_interpretation": reset_state_action_consistency.get("interpretation", {}),
        "wrist_endpoint_status": wrist_endpoint.get("status"),
        "wrist_endpoint_metrics": wrist_endpoint.get("metrics", {}),
        "wrist_endpoint_checks": wrist_endpoint.get("checks", {}),
        "wrist_endpoint_interpretation": wrist_endpoint.get("interpretation", {}),
        "wrist_source_status": wrist_source.get("status"),
        "wrist_source_metrics": wrist_source.get("metrics", {}),
        "wrist_source_checks": wrist_source.get("checks", {}),
        "wrist_source_top_motions": wrist_source.get("worker_metrics", {}).get("top_wrist_motions", []),
        "wrist_source_body_rows": wrist_source.get("worker_metrics", {}).get("body_rows", []),
        "endpoint_group_status": endpoint_group.get("status"),
        "endpoint_group_comparison": endpoint_group.get("comparison_to_baselines", {}),
        "endpoint_group_interpretation": endpoint_group.get("interpretation", {}),
        "endpoint_group_checks": endpoint_group.get("checks", {}),
        "deterministic_reset_status": deterministic_reset.get("status"),
        "deterministic_reset_metrics": deterministic_reset.get("metrics", {}),
        "deterministic_reset_checks": deterministic_reset.get("checks", {}),
        "deterministic_reset_interpretation": deterministic_reset.get("interpretation", {}),
        "protocol_metrics": protocol.get("metrics", {}),
        "protocol_counts": protocol.get("claim_level_counts", {}),
        "cleanup_metrics": cleanup.get("metrics", {}),
        "cleanup_checks": cleanup.get("checks", {}),
        "disk_free_gib": round(disk.free / (1024**3), 2),
        "disk_total_gib": round(disk.total / (1024**3), 2),
        "storage_top_runs": storage_top,
    }


def fmt_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def metric_value(metrics: dict[str, Any], key: str, nested: str = "mean") -> Any:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get(nested)
    motion_metrics = metrics.get("motion_metrics")
    if isinstance(motion_metrics, dict):
        value = motion_metrics.get(key)
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


def storage_top_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No `res/runs` storage rows were found."
    lines = ["| Size | Role |", "|---:|---|"]
    for row in rows:
        lines.append(f"| {row['size_human']} | {friendly_storage_label(row['path'])} |")
    return "\n".join(lines)


def module_contract_markdown(language: str = "en") -> str:
    if language == "zh":
        return "\n".join(
            [
                "| 论文模块 | 本地实现/审计对象 | 当前证据边界 |",
                "|---|---|---|",
                "| Motion tracking teacher | IsaacLab/RSL-RL task gates、reward/termination schema、PPO train/eval wrapper | local virtual teacher；done/endpoint 仍未达 paper-level |",
                "| Teacher rollout / DAgger | rollout shard schema、teacher action/obs/latent collection、DAgger sample audit | local teacher rollout；不是官方 DAgger 数据 |",
                "| Conditional VAE | reparameterization、KL、action reconstruction、checkpoint smoke、teacher-rollout VAE training | paper-faithful/local training；不是官方 VAE checkpoint |",
                "| State-latent dataset | state+latent temporal window、split/index、finite/shape checks | 来源于 local teacher；不是官方 state-latent 数据 |",
                "| Diffusion denoiser | DDPM noise schedule、mask、Transformer denoising、held-out denoising metrics | local denoiser；不是官方 diffusion checkpoint |",
                "| Test-time guidance | joystick/waypoint/obstacle/inpainting/transition/composed cost gradients | local proxy closed-loop/offline evidence；不是 Fig.5/Fig.6 paper-level |",
                "| Deployment | ONNX contract、controller semantics、MuJoCo/ROS launch audit | contract-level audit；没有 TensorRT/Mini-PC/real robot |",
            ]
        )
    return "\n".join(
        [
            "| Paper module | Local implementation or audit object | Current evidence boundary |",
            "|---|---|---|",
            "| Motion tracking teacher | IsaacLab/RSL-RL task gates, reward/termination schema, PPO train/eval wrappers | local virtual teacher; done/endpoint quality is still below paper-level |",
            "| Teacher rollout / DAgger | rollout shard schema, teacher obs/action/latent collection, DAgger sample audit | local teacher rollout; not official DAgger data |",
            "| Conditional VAE | reparameterization, KL, action reconstruction, checkpoint smoke, teacher-rollout VAE training | paper-faithful/local training; not official VAE checkpoint |",
            "| State-latent dataset | state+latent temporal windows, split/index, finite/shape checks | derived from local teacher; not official state-latent data |",
            "| Diffusion denoiser | DDPM noise schedule, masks, Transformer denoising, held-out denoising metrics | local denoiser; not official diffusion checkpoint |",
            "| Test-time guidance | joystick, waypoint, obstacle, inpainting, transition, composed cost gradients | local proxy closed-loop/offline evidence; not paper Fig. 5/Fig. 6 level |",
            "| Deployment | ONNX contract, controller semantics, MuJoCo/ROS launch audit | contract-level audit; no TensorRT/Mini-PC/real robot evidence |",
        ]
    )


def evidence_ladder_markdown(language: str = "en") -> str:
    if language == "zh":
        return "\n".join(
            [
                "| 证据层级 | 代表内容 | 能否作为论文级结果 |",
                "|---|---|---|",
                "| exact/public | released-data 图表、表格、源码契约、公式 trace | 可以用于公开可复现部分 |",
                "| approximate/resource-adjusted | official-loop body、captured G1 USDA、本地 PPO/eval | 只能说明本地虚拟链路 |",
                "| qualitative/proxy | guidance rollouts、task protocol、可视化 | 可用于分析和答辩展示 |",
                "| missing/non-public | 官方 checkpoint、DAgger logs、Fig.5/Fig.6 logs、TensorRT | 不能声称复现 |",
                "| hardware-only | Unitree G1 deployment | 当前不可做 |",
            ]
        )
    return "\n".join(
        [
            "| Evidence layer | Representative content | Can it be used as a paper-level result? |",
            "|---|---|---|",
            "| exact/public | released-data plots, tables, source contracts, formula traces | yes, for the public reproducible subset |",
            "| approximate/resource-adjusted | official-loop bodies, captured G1 USDA, local PPO/eval | no; local virtual evidence only |",
            "| qualitative/proxy | guidance rollouts, task protocol, visualizations | no; useful for analysis and presentation |",
            "| missing/non-public | official checkpoints, DAgger logs, Fig. 5/Fig. 6 logs, TensorRT | no claim allowed |",
            "| hardware-only | Unitree G1 deployment | unavailable in the current project |",
        ]
    )


def english_report(s: dict[str, Any]) -> str:
    cc = s["comparison_counts"]
    cm = s["completion_counts"]
    ac = s["absence_counts"]
    robot_m = s["robot_order_eval_metrics"]
    robot_multi = s["robot_order_multiseed_aggregate"]
    robot_quality = s["robot_order_quality_aggregate"]
    scaled_m = s["scaled_eval_metrics"]
    protocol_m = s["protocol_metrics"]
    cleanup_m = s["cleanup_metrics"]
    storage_top = s["storage_top_runs"]
    warmup_i = s["reset_warmup_interpretation"]
    warmup_eval_status = s["warmup_eval_status"]
    warmup_eval_c = s["warmup_eval_comparison"]
    warmup_phase_status = s["warmup_phase_status"]
    warmup_phase_m = s["warmup_phase_metrics"]
    warmup_phase_i = s["warmup_phase_interpretation"]
    target_refresh_live_status = s["target_refresh_live_status"]
    target_refresh_live_m = s["target_refresh_live_metrics"]
    target_refresh_status = s["target_refresh_status"]
    target_refresh_c = s["target_refresh_comparison"]
    target_refresh_checks = s["target_refresh_checks"]
    reset_state_action_status = s["reset_state_action_status"]
    reset_state_action_m = s["reset_state_action_metrics"]
    reset_state_action_i = s["reset_state_action_interpretation"]
    reset_state_action_consistency_status = s["reset_state_action_consistency_status"]
    reset_state_action_consistency_m = s["reset_state_action_consistency_metrics"]
    reset_state_action_consistency_checks = s["reset_state_action_consistency_checks"]
    wrist_endpoint_status = s["wrist_endpoint_status"]
    wrist_endpoint_m = s["wrist_endpoint_metrics"]
    wrist_source_status = s["wrist_source_status"]
    wrist_source_m = s["wrist_source_metrics"]
    wrist_source_top = s["wrist_source_top_motions"]
    endpoint_group_status = s["endpoint_group_status"]
    endpoint_group_c = s["endpoint_group_comparison"]
    endpoint_group_i = s["endpoint_group_interpretation"]
    deterministic_reset_status = s["deterministic_reset_status"]
    deterministic_reset_m = s["deterministic_reset_metrics"]
    deterministic_reset_i = s["deterministic_reset_interpretation"]
    deterministic_reset_status = s["deterministic_reset_status"]
    deterministic_reset_m = s["deterministic_reset_metrics"]
    deterministic_reset_i = s["deterministic_reset_interpretation"]

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

For reproduction, I turned this method diagram into a contract table instead of a single monolithic training script:

{module_contract_markdown("en")}

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

My current progress estimate has three layers. For the course reading report and defense, the material is about `85-90%` ready: the paper is understood, the evidence is organized, and the claim boundary is clear. For public-resource engineering coverage, the project is about `75-80%` complete: most released-data, source-audit, environment, and local virtual components are runnable or audited, while tracking-quality and storage-pressure work remain active. For strict simulation-side paper-level reproduction, excluding the real robot, I would estimate only `40-50%`: the highest-weight closed-loop claims still need a stronger tracking teacher, true DAgger-style data, official-equivalent VAE/diffusion evidence, Fig. 5/Fig. 6 protocol metrics, and TensorRT deployment evidence.

I use the following evidence ladder throughout the report:

{evidence_ladder_markdown("en")}

## 5. What Has Been Reproduced Or Audited

The strongest exact evidence is in released-data and source-level reproduction. The project checks paper table values, released-data figures, panel mappings, formula/code traces, tracking observation and action schemas, reward and termination contracts, motion preprocessing contracts, ONNX interface contracts, and MuJoCo/ROS launch surfaces. This part is valuable because it tells us what the paper and public code actually specify.

On the tracking side, the project recovered a useful IsaacLab path. The official `csv_to_npz.py` and `replay_npz.py` loop bodies have been exercised over the full public G1 motion bundle with 40 motions and 11960 frames/steps. The captured official-importer-export G1 USDA path is stronger than the earlier generated scaffold because it comes from the Isaac Sim importer, but it is still a captured local asset path rather than a clean unmodified official converter entry.

This is the main official-loop virtual chain in the project: official-loop tracking/PPO eval begins with public G1 motions converted through the official-loop preprocessing body, replayed through the official-loop reference path, loaded into the local IsaacLab tracking task, used for local PPO training/evaluation, then connected to local teacher rollout, VAE, state-latent diffusion, and guidance experiments. I use the phrase "official-loop virtual chain" deliberately. It means the public code path and local simulation chain are substantially exercised, but the result is still virtual and resource-adjusted. It does not mean the unmodified official entrypoint, official teacher checkpoint, official DAgger dataset, or paper deployment stack has been reproduced.

Local PPO training and evaluation have been run on the public-motion bundle. The most important tracking-side finding is that motion data semantics matter as much as the policy. An FK-repaired motion bundle first fixed a degenerate `body_pos_w` problem, but later diagnostics found a subtler body-order mismatch: target body positions were written in URDF body order, while IsaacLab's runtime `MotionLoader` indexes `body_pos_w` by simulator articulation body order. The robot-order FK-repaired bundle is therefore the current mainline data path.

The current robot-order PPO checkpoint evaluation completed with status `{s['robot_order_eval_status']}`. It evaluated `{total_env_steps(robot_m)}` virtual environment steps and recorded reward mean about `{reward_mean(robot_m)}`, done count `{done_count(robot_m)}`, anchor-position error mean `{metric_value(robot_m, 'error_anchor_pos')}`, body-position error mean `{metric_value(robot_m, 'error_body_pos')}`, and joint-position error mean `{metric_value(robot_m, 'error_joint_pos')}`. The three-seed eval totals `{s['robot_order_multiseed_metrics'].get('total_env_steps')}` virtual environment steps; its mean done rate is `{metric_value(robot_multi, 'done_rate')}`, reward mean `{metric_value(robot_multi, 'reward_mean')}`, body-position error mean `{metric_value(robot_multi, 'error_body_pos_mean')}`, and joint-position error mean `{metric_value(robot_multi, 'error_joint_pos_mean')}`. This is stable local virtual evidence, but it is not a paper-level teacher.

The latest tracking diagnostic explains why. Every multi-seed eval reports a step-0 done rate of `{metric_value(robot_quality, 'step0_done_rate')}` and a step-0 body-position error around `{metric_value(robot_quality, 'step0_error_body_pos')}` meters. Removing step 0 reduces mean body-position error to `{metric_value(robot_quality, 'mean_error_body_pos_post_step0')}`, but the post-step0 done rate remains around `{metric_value(robot_quality, 'post_step0_done_rate')}`. A reset-command warmup live probe found that command warmup `{warmup_i.get('next_mainline_decision')}`.

I then ran a full 2048-env x 299-step checkpoint evaluation with reset-command warmup: `{warmup_eval_status}`. It reduced the step-0 done count from `{warmup_eval_c.get('old_step0', {}).get('done_count')}` to `{warmup_eval_c.get('warmup_eval_step0', {}).get('done_count')}` and the step-0 body-position error from `{warmup_eval_c.get('old_step0', {}).get('error_body_pos')}` m to `{warmup_eval_c.get('warmup_eval_step0', {}).get('error_body_pos')}` m. However, the total done rate worsened from `{warmup_eval_c.get('old_done_rate')}` to `{warmup_eval_c.get('warmup_eval_done_rate')}`. This is important negative evidence: reset warmup fixes a visible bootstrap artifact, but the checkpoint is still not a usable teacher. The next tracking fix should focus on post-warmup termination/policy-state mismatch before another downstream teacher rollout is collected.

A seed-matched follow-up made this conclusion stronger: `{warmup_phase_status}`. With the same seed as the non-warmup baseline, step-0 done count and body error still improved, but total done rate worsened by `{warmup_phase_m.get('same_seed_done_rate_delta')}`, post-step0 done rate worsened by `{warmup_phase_m.get('same_seed_post_step0_done_rate_delta')}`, and the `ee_body_pos` termination fraction increased by `{warmup_phase_m.get('same_seed_ee_body_pos_termination_fraction_delta')}` while the sampling top-bin delta stayed `{warmup_phase_m.get('same_seed_sampling_top1_bin_post_step0_delta')}`. My current interpretation is: {warmup_phase_i.get('primary_bottleneck')}

The next diagnostic tested that recommendation directly with a no-advance reset-target refresh. The live probe status is `{target_refresh_live_status}`: endpoint-z done rate moved from `{target_refresh_live_m.get('endpoint_done_rate_before')}` to `{target_refresh_live_m.get('endpoint_done_rate_after')}`, endpoint-z error mean moved from `{target_refresh_live_m.get('endpoint_z_error_mean_before')}` to `{target_refresh_live_m.get('endpoint_z_error_mean_after')}`, and `time_steps_unchanged_by_refresh` is `{target_refresh_live_m.get('time_steps_unchanged_by_refresh')}`. The full 2048-env x 299-step eval status is `{target_refresh_status}`. It reduced the step-0 done count by `{target_refresh_c.get('step0_done_count_delta')}` and avoided the command-time advance, but the total done rate still moved from `{target_refresh_c.get('old_done_rate')}` to `{target_refresh_c.get('target_refresh_done_rate')}` and the post-step0 done-rate delta was `{target_refresh_c.get('post_step0_done_rate_delta')}`. This narrows the tracking bottleneck: stale reset targets are real, but they are not sufficient to explain the weak teacher. The next repair should inspect reset state/action distribution, initial joint velocity mismatch, endpoint thresholds, and `ee_body_pos` termination before another large PPO/downstream chain.

A follow-up static trace diagnostic made that bottleneck measurable: `{reset_state_action_status}`. It compares the same-seed baseline, reset-command warmup, and no-advance target-refresh full eval traces. Target refresh reduces the step-0 body-position error by `{reset_state_action_m.get('target_refresh_step0_body_error_delta')}` m, but the step-0 joint-velocity error increases by `{reset_state_action_m.get('target_refresh_step0_joint_vel_delta')}`, the first-five-step action mean increases by `{reset_state_action_m.get('target_refresh_first5_action_abs_mean_delta')}`, the post-step0 done-rate delta is `{reset_state_action_m.get('target_refresh_post_step0_done_rate_delta')}`, and the `ee_body_pos` termination fraction delta is `{reset_state_action_m.get('target_refresh_ee_body_pos_termination_fraction_delta')}`. My current conclusion is: {reset_state_action_i.get('primary_bottleneck')}

The newest live probe goes one step further by asking whether the reset/action mismatch has an easy local repair. Its status is `{reset_state_action_consistency_status}`. It compares target refresh alone with action-history reset, action-offset alignment, and motion-state rewrite variants under both zero actions and the checkpoint policy. Target refresh alone gives a policy-step done rate of `{reset_state_action_consistency_m.get('target_refresh_policy_done_rate')}` with post-step joint-velocity error `{reset_state_action_consistency_m.get('target_refresh_policy_joint_vel_after_step')}`. Action reset lowers that velocity error to `{reset_state_action_consistency_m.get('action_reset_policy_joint_vel_after_step')}` but worsens done rate to `{reset_state_action_consistency_m.get('action_reset_policy_done_rate')}`. Action-offset alignment lowers velocity error to `{reset_state_action_consistency_m.get('action_offset_policy_joint_vel_after_step')}` but worsens done rate to `{reset_state_action_consistency_m.get('action_offset_policy_done_rate')}`. The strongest motion-state/action-offset candidate lowers joint velocity to `{reset_state_action_consistency_m.get('candidate_policy_joint_vel_after_step')}` but worsens done rate to `{reset_state_action_consistency_m.get('candidate_policy_done_rate')}`. The key check is `any_variant_improves_done_and_joint_velocity = {reset_state_action_consistency_checks.get('any_variant_improves_done_and_joint_velocity')}`. Therefore I did not promote this patch to a full eval or a new PPO run. This is a useful negative result: it prevents the project from drifting away from the paper by training on a harmful reset workaround.

The newest endpoint-group ablation makes the next tracking target more concrete. Its status is `{endpoint_group_status}`. Under the same seed and 2048-env x 299-step scope, the no-advance target-refresh eval has done rate `{endpoint_group_c.get('target_refresh_done_rate')}`. Keeping only ankle endpoint termination gives done rate `{endpoint_group_c.get('ankles_only_done_rate')}`; keeping only wrist endpoint termination gives `{endpoint_group_c.get('wrists_only_done_rate')}`; relaxing all endpoint bodies gives `{endpoint_group_c.get('all_endpoint_relaxed_done_rate')}`. The recorded dominant endpoint group is `{endpoint_group_i.get('dominant_endpoint_group')}`.

I then tested that hypothesis directly with a live wrist/ankle endpoint alignment probe: `{wrist_endpoint_status}`. It records `body_pos_w`, `body_pos_relative_w`, and `robot_body_pos_w` for ankles and wrists before/after no-advance target refresh and after one zero/policy step. Target refresh reduces both groups, but wrists remain worse: refresh wrist z-error mean `{wrist_endpoint_m.get('refresh_wrist_rel_z_error_mean')}` m versus ankle `{wrist_endpoint_m.get('refresh_ankle_rel_z_error_mean')}` m, refresh wrist done rate `{wrist_endpoint_m.get('refresh_wrist_done_rate')}` versus ankle `{wrist_endpoint_m.get('refresh_ankle_done_rate')}`, and policy-step wrist done rate `{wrist_endpoint_m.get('policy_step_wrist_done_rate')}` versus ankle `{wrist_endpoint_m.get('policy_step_ankle_done_rate')}`. The diagnosis is `{wrist_endpoint_m.get('diagnosis')}`. My current interpretation is that the next repair should inspect wrist endpoint target/body order, wrist FK height, and `ee_body_pos` body semantics before another full PPO/downstream chain.

The newest full-size source diagnostic scales that live probe to the same 2048-env x 299-step evaluation shape used by the tracking checkpoint audits: `{wrist_source_status}`. It records endpoint z-error by body, motion, and phase bin. The overall done rate is `{wrist_source_m.get('done_rate')}`, and `ee_body_pos` accounts for `{wrist_source_m.get('ee_body_pos_rate')}` of env-steps. The mean pre-step wrist exceed rate is `{wrist_source_m.get('pre_wrist_done_rate', {}).get('mean')}` versus ankle `{wrist_source_m.get('pre_ankle_done_rate', {}).get('mean')}`; post-step wrist is `{wrist_source_m.get('post_wrist_done_rate', {}).get('mean')}` versus ankle `{wrist_source_m.get('post_ankle_done_rate', {}).get('mean')}`. The top wrist-heavy motions include `{', '.join(row.get('motion', '') for row in wrist_source_top[:3])}`. This is useful because it moves the repair target from a vague endpoint suspicion to motion/phase-specific source attribution. It also shows why the next step should repair data/termination semantics before another PPO run.

The latest deterministic reset gate confirms the same conclusion from a different angle. Its status is `{deterministic_reset_status}`. The official-refresh policy done rate is `{deterministic_reset_m.get('official_refresh_policy_done_rate')}` with joint-velocity error `{deterministic_reset_m.get('official_refresh_policy_joint_vel_after_step')}`. Deterministic reset lowers joint velocity to `{deterministic_reset_m.get('deterministic_refresh_policy_joint_vel_after_step')}`, but worsens done rate to `{deterministic_reset_m.get('deterministic_refresh_policy_done_rate')}`. Motion-state reset also fails the joint/done tradeoff, with done rate `{deterministic_reset_m.get('motion_state_policy_done_rate')}`. The recommended full-eval variant is `{deterministic_reset_i.get('recommended_full_eval_variant') or 'none'}`. I therefore interpret the current tracking blocker as a termination/body-target semantics problem rather than a simple reset-randomization problem.

For Level C, the project implements a paper-faithful local chain: teacher rollout, conditional VAE, state-latent windows, denoiser/diffusion training, offline guidance, and local proxy closed-loop guidance. This proves that the method can be studied and partially recreated from public resources, but it is not the official BeyondMimic VAE/diffusion checkpoint chain.

## 6. From Paper Equations To Code

I treated the paper formulas as software contracts. The tracking objective became reward and termination checks over anchor pose, target body positions, endpoint height, action regularization, and contact-like events. The VAE objective became a state-conditioned encoder/decoder with reparameterization, reconstruction error, KL regularization, finite-output checks, and checkpoint save/load tests. The diffusion objective became noisy state-latent sequence prediction with train/validation/test splits and denoising-improvement metrics. The guidance equations became task-cost gradients over sampled trajectories.

This matters for a reading report because it shows independent exploration rather than only summarizing the paper. Implementing the formulas forced me to decide which variables are directly public, which are inferred from source code, and which are local proxies because the paper's exact dataset or checkpoint is not available.

The local code is intentionally modest in scope. It does not try to replace IsaacLab or the official tracking repository. Instead, it implements the mathematical pieces that are safe to reproduce independently: finite tensor validation, yaw-frame transforms, VAE latent math, DDPM-style noise/reverse helpers, state-latent windows, DAgger sample schemas, guidance costs, and summary metrics. The official robotics stack remains the source of truth for embodied simulation.

## 7. Local Fig. 5 / Fig. 6 Proxy Evidence

The project has consolidated the local guidance tasks into a unified protocol table. It covers `{protocol_m.get('task_count')}` local proxy tasks with `{protocol_m.get('multiseed_proxy_task_count')}` multi-seed proxy groups and `{protocol_m.get('single_seed_proxy_task_count')}` single-seed proxy groups. The important number is `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`. This means the local protocol is useful for analysis and presentation, but it must not be described as reproducing the paper's Fig. 5 or Fig. 6.

The current protocol is best described as a local virtual BeyondMimic-like pipeline. It covers joystick, waypoint, obstacle avoidance, composed objectives, transition, and inpainting-style proxies. The next scientific step is to make task metrics stronger: velocity error for joystick, final distance and success rate for waypoint, clearance and collision counts for obstacle avoidance, keyframe error for inpainting, transition smoothness and fall rate for transitions, and guided-vs-unguided improvement for each task.

## 8. Storage And Artifact Management

The project deliberately keeps GitHub lightweight. Large environments, checkpoints, videos, raw rollout shards, datasets, and caches are not committed. The latest conservative cleanup audit is `{cleanup_m.get('deleted_or_previously_deleted_count')}` deleted-or-previously-deleted bulky candidates and `{cleanup_m.get('managed_superseded_bytes_removed_or_absent')}` managed bytes removed or confirmed absent. Current disk free space is about `{s['disk_free_gib']}` GiB of `{s['disk_total_gib']}` GiB on the project filesystem. The policy is conservative: delete failed, duplicate, or rebuildable bulky directories; keep current active run directories and preserve JSON/CSV/Markdown/log evidence.

In this reporting phase I also treat debug-only checkpoints as storage candidates, not scientific results. VAE/diffusion smoke weights can be removed after their JSON/TSV summaries prove save/load or tiny optimizer plumbing. This reduces disk pressure without weakening the paper claim, because those weights were never accepted as official trained checkpoints.

Current largest local run directories are:

{storage_top_markdown(storage_top)}

The two largest active candidates are the scaled-PPO teacher rollout shards and scaled-PPO state-latent dataset. I keep them for now because they are still the strongest available local downstream chain. The next safe cleanup pass should first remove or archive older LAFAN1/debug checkpoints and duplicate superseded PPO directories only after the absence audit and report assets no longer depend on their raw files.

## 9. Limitations

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

## 10. Personal Reflection

This reproduction changed how I read the paper. At first the method looks like a clean sequence of modules: tracking, VAE, diffusion, guidance. In practice, every module depends on embodied details: robot assets, body names, endpoint heights, reset logic, termination thresholds, observation history, simulation stability, and data provenance. A small coordinate or body-position issue can invalidate a beautiful downstream model.

The most important lesson is that robotics reproducibility is not only about code availability. It needs assets, checkpoints, datasets, evaluation scripts, logs, videos, and deployment details. BeyondMimic is technically compelling, but the public artifact boundary makes exact reproduction impossible at several points. A good reproduction report should therefore avoid a binary "success/failure" story. The honest story is that many public components can be reproduced and analyzed, a local virtual pipeline can be built, and the remaining paper-level claims require non-public artifacts or hardware.

For a class reading report, this is also the part I find intellectually interesting: the negative results are not just excuses. The wrist-endpoint and reset-target investigations show how a generative-control idea depends on low-level embodied bookkeeping. A diffusion model can only guide behavior that the teacher distribution makes physically meaningful. If the teacher's reset distribution, endpoint targets, or body order are inconsistent, the downstream model may look mathematically correct while learning from the wrong closed-loop behavior.

## 11. Conclusion

This project currently supports a strong course reading report and defense: it explains the paper, audits the public code and data, implements the main ideas in a local pipeline, and identifies where paper-level reproduction is blocked. It does not fully reproduce BeyondMimic at paper level. The next research step is to repair tracking quality, train a more reliable teacher, then rerun the downstream VAE, state-latent diffusion, and guidance experiments from that stronger teacher.
"""


def chinese_reading_report(s: dict[str, Any]) -> str:
    cc = s["comparison_counts"]
    cm = s["completion_counts"]
    ac = s["absence_counts"]
    robot_m = s["robot_order_eval_metrics"]
    robot_multi = s["robot_order_multiseed_aggregate"]
    robot_quality = s["robot_order_quality_aggregate"]
    protocol_m = s["protocol_metrics"]
    warmup_i = s["reset_warmup_interpretation"]
    warmup_eval_status = s["warmup_eval_status"]
    warmup_eval_c = s["warmup_eval_comparison"]
    warmup_phase_status = s["warmup_phase_status"]
    warmup_phase_m = s["warmup_phase_metrics"]
    warmup_phase_i = s["warmup_phase_interpretation"]
    target_refresh_live_status = s["target_refresh_live_status"]
    target_refresh_live_m = s["target_refresh_live_metrics"]
    target_refresh_status = s["target_refresh_status"]
    target_refresh_c = s["target_refresh_comparison"]
    reset_state_action_status = s["reset_state_action_status"]
    reset_state_action_m = s["reset_state_action_metrics"]
    reset_state_action_consistency_status = s["reset_state_action_consistency_status"]
    reset_state_action_consistency_m = s["reset_state_action_consistency_metrics"]
    reset_state_action_consistency_checks = s["reset_state_action_consistency_checks"]
    reset_state_action_i = s["reset_state_action_interpretation"]
    wrist_endpoint_status = s["wrist_endpoint_status"]
    wrist_endpoint_m = s["wrist_endpoint_metrics"]
    wrist_source_status = s["wrist_source_status"]
    wrist_source_m = s["wrist_source_metrics"]
    wrist_source_top = s["wrist_source_top_motions"]
    endpoint_group_status = s["endpoint_group_status"]
    endpoint_group_c = s["endpoint_group_comparison"]
    endpoint_group_i = s["endpoint_group_interpretation"]
    deterministic_reset_status = s["deterministic_reset_status"]
    deterministic_reset_m = s["deterministic_reset_metrics"]
    deterministic_reset_i = s["deterministic_reset_interpretation"]

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

为了复现，我把方法图进一步变成下面这种模块-证据表，而不是直接写一个大脚本：

{module_contract_markdown("zh")}

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

从完成度角度看，我会分三层估计：课程阅读报告和答辩材料约 `85-90%` 可用；公开资源工程覆盖度约 `75-80%`；严格 non-robot paper-level reproduction 约 `40-50%`。这个估计的核心原因是：报告和审计材料已经很完整，但 tracking teacher 质量、true DAgger、官方 VAE/diffusion、Fig.5/Fig.6 和 TensorRT 仍没有达到论文级证据。

我在报告中按下面的证据层级来描述结果：

{evidence_ladder_markdown("zh")}

## 4. 已完成内容

第一，公开数据和论文表格/图表复现比较可靠。项目完成了 released-data figure/table reproduction、paper panel map、source coverage、formula/code trace 和 table value audit。这部分最接近 exact reproduction。

第二，官方 tracking 代码做了较完整审计。包括 observation/action schema、reward terms、termination、motion preprocessing、ONNX contract 和 MuJoCo/ROS launch contract。

第三，IsaacLab 和 G1 task gate 已经打通。当前 headless AppLauncher gate 是 `{s['headless_status']}`，G1 task construction gate 是 `{s['task_gate_status']}`。task contract 验证了 29 维 action、160 维 policy observation、286 维 critic observation、9 个 reward term、4 个 termination term、29 个关节和 40 个 body。

第四，官方 `csv_to_npz.py` / `replay_npz.py` 的 loop body 已经在 full public motion bundle 上跑通。40 个 public motions 合计 11960 帧/步。captured official-importer-export G1 USDA 路径比早期 scaffold 更可信，但仍不是 unmodified official converter entry。

第五，本地 PPO/VAE/diffusion/guidance 链路已经跑通。它证明了公开资源下可以实现一个 local virtual BeyondMimic-like pipeline，但不能把它说成官方 checkpoint 复现。

## 5. 当前效果

tracking 侧现在的关键结论是：链路能跑，但 teacher 还不够好。最重要的技术发现是 motion 数据语义比 policy 本身还敏感。早期 FK-repaired bundle 修复了 `body_pos_w` 退化问题，但后续发现更隐蔽的 body-order mismatch：motion target 是按 URDF body order 写入，而 IsaacLab runtime `MotionLoader` 按 simulator articulation body order 读取。这会导致 target body 错位、endpoint z error 和大量 termination。

当前主线已经切到 robot-order FK-repaired bundle。robot-order PPO checkpoint eval 状态是 `{s['robot_order_eval_status']}`，共评估 `{total_env_steps(robot_m)}` 个 virtual env steps，reward mean 约 `{reward_mean(robot_m)}`，done count `{done_count(robot_m)}`，anchor/body/joint position error mean 分别约 `{metric_value(robot_m, 'error_anchor_pos')}`、`{metric_value(robot_m, 'error_body_pos')}`、`{metric_value(robot_m, 'error_joint_pos')}`。三 seed eval 共 `{s['robot_order_multiseed_metrics'].get('total_env_steps')}` 个 virtual env steps，mean done rate `{metric_value(robot_multi, 'done_rate')}`，reward mean `{metric_value(robot_multi, 'reward_mean')}`，body-position error mean `{metric_value(robot_multi, 'error_body_pos_mean')}`。

最新质量诊断显示，三个 multi-seed eval 都有 step-0 done rate `{metric_value(robot_quality, 'step0_done_rate')}`，step-0 body-position error 约 `{metric_value(robot_quality, 'step0_error_body_pos')}` 米。去掉 step 0 后 body-position error 降到 `{metric_value(robot_quality, 'mean_error_body_pos_post_step0')}`，但 post-step0 done rate 仍约 `{metric_value(robot_quality, 'post_step0_done_rate')}`。reset-command warmup live probe 的结论是 `{warmup_i.get('next_mainline_decision')}`。

随后我做了一个 2048 env x 299 step 的 full checkpoint warmup eval：`{warmup_eval_status}`。它把 step-0 done count 从 `{warmup_eval_c.get('old_step0', {}).get('done_count')}` 降到 `{warmup_eval_c.get('warmup_eval_step0', {}).get('done_count')}`，把 step-0 body-position error 从 `{warmup_eval_c.get('old_step0', {}).get('error_body_pos')}` m 降到 `{warmup_eval_c.get('warmup_eval_step0', {}).get('error_body_pos')}` m；但整体 done rate 从 `{warmup_eval_c.get('old_done_rate')}` 升到 `{warmup_eval_c.get('warmup_eval_done_rate')}`，也就是更差。因此现在不能说 warmup 修好了 teacher，只能说它定位了 reset bootstrap artifact，下一步要查 post-warmup termination / policy-state mismatch。

同 seed follow-up 进一步排除了随机 seed 影响：`{warmup_phase_status}`。在和 non-warmup baseline 相同的 seed 下，step-0 仍然明显改善，但总 done rate 变差 `{warmup_phase_m.get('same_seed_done_rate_delta')}`，post-step0 done rate 变差 `{warmup_phase_m.get('same_seed_post_step0_done_rate_delta')}`，`ee_body_pos` termination fraction 增加 `{warmup_phase_m.get('same_seed_ee_body_pos_termination_fraction_delta')}`，而 sampling top-bin delta 是 `{warmup_phase_m.get('same_seed_sampling_top1_bin_post_step0_delta')}`。所以现在最可能的问题不是随机采样到坏 motion，而是 command/observation phase consistency：`{warmup_phase_i.get('recommended_next_experiment')}`。

随后我又做了 no-advance reset-target refresh，直接验证“不推进 `MotionCommand.time_steps`，只刷新 reset target”这个想法。live probe 状态是 `{target_refresh_live_status}`：endpoint-z done rate 从 `{target_refresh_live_m.get('endpoint_done_rate_before')}` 降到 `{target_refresh_live_m.get('endpoint_done_rate_after')}`，endpoint-z error mean 从 `{target_refresh_live_m.get('endpoint_z_error_mean_before')}` 降到 `{target_refresh_live_m.get('endpoint_z_error_mean_after')}`，并且 `time_steps_unchanged_by_refresh = {target_refresh_live_m.get('time_steps_unchanged_by_refresh')}`。full eval 状态是 `{target_refresh_status}`，step-0 done count delta `{target_refresh_c.get('step0_done_count_delta')}`，但 total done rate 仍从 `{target_refresh_c.get('old_done_rate')}` 变成 `{target_refresh_c.get('target_refresh_done_rate')}`，post-step0 done-rate delta 是 `{target_refresh_c.get('post_step0_done_rate_delta')}`。这说明 stale reset target 确实存在，但不是 teacher 弱的全部原因；下一步更应该查 reset state/action distribution、初始 joint velocity mismatch、endpoint 阈值和 `ee_body_pos` termination。

随后这个方向已经被静态 full-trace 诊断量化：`{reset_state_action_status}`。它比较 baseline、reset-command warmup、no-advance target-refresh 三组同 seed full eval。target refresh 让 step-0 body-position error 改善 `{reset_state_action_m.get('target_refresh_step0_body_error_delta')}` m，但 step-0 joint-velocity error 增加 `{reset_state_action_m.get('target_refresh_step0_joint_vel_delta')}`，first-five-step action mean 增加 `{reset_state_action_m.get('target_refresh_first5_action_abs_mean_delta')}`，post-step0 done-rate delta 是 `{reset_state_action_m.get('target_refresh_post_step0_done_rate_delta')}`，`ee_body_pos` termination fraction delta 是 `{reset_state_action_m.get('target_refresh_ee_body_pos_termination_fraction_delta')}`。所以现在最具体的判断是：{reset_state_action_i.get('primary_bottleneck')}

最新 live probe 继续检查了一个更直接的问题：target refresh 之后，能不能通过 action-history reset、action-offset alignment 或 motion-state rewrite 直接得到可用于 full eval 的修复。结果状态是 `{reset_state_action_consistency_status}`。target refresh alone 的 policy-step done rate 是 `{reset_state_action_consistency_m.get('target_refresh_policy_done_rate')}`，post-step joint-velocity error 是 `{reset_state_action_consistency_m.get('target_refresh_policy_joint_vel_after_step')}`；action reset 把 joint velocity 降到 `{reset_state_action_consistency_m.get('action_reset_policy_joint_vel_after_step')}`，但 done rate 变差到 `{reset_state_action_consistency_m.get('action_reset_policy_done_rate')}`；action-offset alignment 把 joint velocity 降到 `{reset_state_action_consistency_m.get('action_offset_policy_joint_vel_after_step')}`，但 done rate 变差到 `{reset_state_action_consistency_m.get('action_offset_policy_done_rate')}`；motion-state/action-offset candidate 把 joint velocity 降到 `{reset_state_action_consistency_m.get('candidate_policy_joint_vel_after_step')}`，但 done rate 变差到 `{reset_state_action_consistency_m.get('candidate_policy_done_rate')}`。关键检查 `any_variant_improves_done_and_joint_velocity = {reset_state_action_consistency_checks.get('any_variant_improves_done_and_joint_velocity')}`。所以这一轮没有推荐 full eval，也没有重跑 PPO；这不是停在失败审计，而是避免把一个会恶化 termination 的 patch 带进主线训练。

最新 endpoint-group ablation 让下一步 tracking 修复更具体：`{endpoint_group_status}`。在同 seed、2048 env x 299 step 条件下，target-refresh done rate 是 `{endpoint_group_c.get('target_refresh_done_rate')}`；只保留 ankle endpoint termination 时 done rate 是 `{endpoint_group_c.get('ankles_only_done_rate')}`；只保留 wrist endpoint termination 时 done rate 是 `{endpoint_group_c.get('wrists_only_done_rate')}`；全部 endpoint threshold 放宽时 done rate 是 `{endpoint_group_c.get('all_endpoint_relaxed_done_rate')}`。诊断记录 dominant endpoint group 是 `{endpoint_group_i.get('dominant_endpoint_group')}`。

这一轮我又用 live wrist/ankle endpoint alignment probe 直接验证了这个判断：`{wrist_endpoint_status}`。它在真实 IsaacLab task 中分别记录 `body_pos_w`、`body_pos_relative_w` 和 `robot_body_pos_w`，比较 ankles 和 wrists 在 target refresh 前后以及 zero/policy step 后的 z error。结果是：refresh 后 wrist z-error mean `{wrist_endpoint_m.get('refresh_wrist_rel_z_error_mean')}` m，ankle `{wrist_endpoint_m.get('refresh_ankle_rel_z_error_mean')}` m；refresh wrist done rate `{wrist_endpoint_m.get('refresh_wrist_done_rate')}`，ankle `{wrist_endpoint_m.get('refresh_ankle_done_rate')}`；policy-step wrist done rate `{wrist_endpoint_m.get('policy_step_wrist_done_rate')}`，ankle `{wrist_endpoint_m.get('policy_step_ankle_done_rate')}`。诊断是 `{wrist_endpoint_m.get('diagnosis')}`。因此下一步不是再盲目 PPO，而是优先查 wrist endpoint 的 target/body order、wrist FK height 和 `ee_body_pos` body semantics。

最新 full-size source diagnostic 把这个 live probe 扩展到 2048 env x 299 step：`{wrist_source_status}`。它按 endpoint body、motion 和 phase bin 统计 z-error 与 termination 来源。总体 done rate 是 `{wrist_source_m.get('done_rate')}`，`ee_body_pos` rate 是 `{wrist_source_m.get('ee_body_pos_rate')}`；pre-step wrist exceed rate mean `{wrist_source_m.get('pre_wrist_done_rate', {}).get('mean')}`，ankle `{wrist_source_m.get('pre_ankle_done_rate', {}).get('mean')}`；post-step wrist `{wrist_source_m.get('post_wrist_done_rate', {}).get('mean')}`，ankle `{wrist_source_m.get('post_ankle_done_rate', {}).get('mean')}`。top wrist-heavy motions 包括 `{', '.join(row.get('motion', '') for row in wrist_source_top[:3])}`。这个结果把问题从“怀疑 wrist endpoint”推进到“哪些 motion/phase/body 触发最多”，所以它是下一步修 tracking 数据质量的直接依据。

最新 deterministic reset live gate 又从另一个角度验证了这个判断：`{deterministic_reset_status}`。official-refresh policy done rate 是 `{deterministic_reset_m.get('official_refresh_policy_done_rate')}`，joint velocity error 是 `{deterministic_reset_m.get('official_refresh_policy_joint_vel_after_step')}`；deterministic reset 把 joint velocity 降到 `{deterministic_reset_m.get('deterministic_refresh_policy_joint_vel_after_step')}`，但 done rate 变差到 `{deterministic_reset_m.get('deterministic_refresh_policy_done_rate')}`；motion-state reset 的 done rate 也变差到 `{deterministic_reset_m.get('motion_state_policy_done_rate')}`。最终 recommended full-eval variant 是 `{deterministic_reset_i.get('recommended_full_eval_variant') or 'none'}`。所以当前最应该修的是 termination/body-target 语义，而不是简单关闭 reset 随机性后直接开 PPO。

Level C 侧的 VAE、state-latent diffusion 和 guidance 能形成完整本地链路，但因为上游 teacher 弱，这些结果只能解释为机制复现和本地 proxy 实验。它们适合写进阅读报告，用来说明我理解并实现了论文 pipeline；但它们不能替代论文 Fig.5/Fig.6 的闭环结果。

当前统一任务协议表覆盖 `{protocol_m.get('task_count')}` 个本地 proxy 任务，其中 `{protocol_m.get('multiseed_proxy_task_count')}` 个是 multi-seed proxy，`{protocol_m.get('single_seed_proxy_task_count')}` 个是 single-seed proxy。最重要的是 `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`。这说明 joystick、waypoint、obstacle、composed、transition、inpainting 等任务在本地机制层面被覆盖，但还没有达到论文 Fig.5/Fig.6 协议。

## 6. 从公式到代码

这次复现不是只看论文图，而是把公式变成代码 contract。tracking objective 被拆成 anchor、target body、endpoint、action regularization 和 termination 检查；VAE 被拆成 state-conditioned encoder/decoder、reparameterization、KL 和 reconstruction；diffusion 被拆成 state-latent window、timestep noise、denoising objective 和 validation/test split；guidance 被拆成 joystick、waypoint、obstacle、inpainting、transition 等 cost gradient。

这样做的意义是：每个公式都能对应到一个可运行模块或审计表。公式里没有公开的数据或 checkpoint，则明确标成 local proxy，而不是假装已经 paper-level 复现。

本地代码刻意没有重写 IsaacLab 或官方 tracking 仓库，而是只实现适合独立验证的数学和数据契约：finite tensor check、yaw-frame transform、VAE latent math、DDPM-style noise/reverse helper、state-latent windows、DAgger sample schema、guidance cost 和 summary metrics。真正的机器人闭环仿真仍然以官方 whole_body_tracking/IsaacLab 栈为准。

## 7. 主要困难

第一是 IsaacLab/Isaac Sim 环境。真实机器人学习复现不是安装 PyTorch 就结束，Kit、Vulkan、USD save policy、GPU 可见性、AppLauncher 和 extension context 都会影响结果。

第二是机器人资产和 motion preprocessing。G1 URDF/USD、body names、target bodies、endpoint z、FK、`body_pos_w` 和 MotionLoader 格式都直接影响 tracking 结果。一个看似能加载的 motion bundle 仍可能在身体位置或终止条件上出问题。

第三是官方 artifact 缺失。论文最关键的 DAgger rollout、VAE checkpoint、diffusion checkpoint、Fig.5/Fig.6 rollout logs 和 TensorRT deployment artifacts 没有公开。

第四是闭环验证。offline denoising 或 guidance 指标变好，并不等于机器人在 IsaacLab 中稳定完成任务。真正有说服力的结果必须包含 rollout、termination、tracking error、success/fall/collision 指标和视频。

## 8. 还缺什么

除真实机器人外，仍缺：

- 高质量 paper-level tracking teacher。
- true DAgger rollout logs。
- 官方 VAE checkpoint。
- 官方 diffusion Transformer checkpoint。
- Fig.5/Fig.6 严格任务协议下的闭环指标和视频。
- TensorRT engine、Mini-PC latency 和异步部署复现。
- MuJoCo/ROS sim-to-sim 实际运行日志。

因此当前不能声称完整复现 BeyondMimic，也不得声称完整复现 BeyondMimic。

## 9. 个人理解

这次复现让我意识到，机器人学习论文的复现难点不只在算法公式。一个方法能不能复现，取决于环境、资产、训练数据、checkpoint、评测协议和部署细节是否一起公开。BeyondMimic 的方法图很清楚，但真正复现时，每个接口都有可能成为 blocker。

我认为这个项目最有价值的地方，是把证据分层说清楚：哪些是 official-code reproduction，哪些是 released-data reproduction，哪些是 paper-faithful reimplementation，哪些只是 local virtual proxy，哪些根本 not publicly reproducible。这个区分比简单说“复现成功”或“复现失败”更接近科研复现的真实状态。

这也是我对论文更深的一点理解：BeyondMimic 的扩散模型并不是凭空生成“机器人能力”，它依赖 tracking teacher 提供一个物理可执行的行为分布。如果 teacher 的 reset、endpoint、body order 或 termination 有问题，下游 VAE/diffusion 即使公式正确，也是在学习一个有偏的闭环分布。因此当前 wrist endpoint 和 reset-target 诊断不是偏离主线，而是在修复生成式控制链条最前面的数据基础。

## 10. 结论

本项目已经足够支撑一篇有独立思考的课程阅读报告：它不仅总结论文，还实际检查了代码、恢复环境、运行任务、实现公式、生成本地实验，并记录失败边界。但它不是完整 paper-level reproduction。下一步最重要的是修 tracking 数据质量和 termination/done count，得到更可信 teacher，再重做 downstream VAE、diffusion 和 guidance。
"""


def chinese_project_report(s: dict[str, Any]) -> str:
    cc = s["comparison_counts"]
    cm = s["completion_counts"]
    protocol_m = s["protocol_metrics"]
    robot_m = s["robot_order_eval_metrics"]
    robot_multi = s["robot_order_multiseed_aggregate"]
    robot_quality = s["robot_order_quality_aggregate"]
    cleanup_m = s["cleanup_metrics"]
    storage_top = s["storage_top_runs"]
    warmup_i = s["reset_warmup_interpretation"]
    warmup_eval_status = s["warmup_eval_status"]
    warmup_eval_c = s["warmup_eval_comparison"]
    warmup_phase_status = s["warmup_phase_status"]
    warmup_phase_m = s["warmup_phase_metrics"]
    warmup_phase_i = s["warmup_phase_interpretation"]
    target_refresh_live_status = s["target_refresh_live_status"]
    target_refresh_live_m = s["target_refresh_live_metrics"]
    target_refresh_status = s["target_refresh_status"]
    target_refresh_c = s["target_refresh_comparison"]
    reset_state_action_status = s["reset_state_action_status"]
    reset_state_action_m = s["reset_state_action_metrics"]
    reset_state_action_i = s["reset_state_action_interpretation"]
    reset_state_action_consistency_status = s["reset_state_action_consistency_status"]
    reset_state_action_consistency_m = s["reset_state_action_consistency_metrics"]
    reset_state_action_consistency_checks = s["reset_state_action_consistency_checks"]
    wrist_endpoint_status = s["wrist_endpoint_status"]
    wrist_endpoint_m = s["wrist_endpoint_metrics"]
    wrist_source_status = s["wrist_source_status"]
    wrist_source_m = s["wrist_source_metrics"]
    wrist_source_top = s["wrist_source_top_motions"]
    endpoint_group_status = s["endpoint_group_status"]
    endpoint_group_c = s["endpoint_group_comparison"]
    endpoint_group_i = s["endpoint_group_interpretation"]
    deterministic_reset_status = s["deterministic_reset_status"]
    deterministic_reset_m = s["deterministic_reset_metrics"]
    deterministic_reset_i = s["deterministic_reset_interpretation"]

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

更具体地说，源码实现和论文模块的对应关系如下：

{module_contract_markdown("zh")}

答辩时可以把这张表当成“我不是只跑脚本，而是把论文拆成了可验证工程模块”的证据。尤其要强调：本地 `beyondmimic_reimpl` 包只负责独立数学契约，官方 IsaacLab/whole_body_tracking 仍然负责 embodied closed-loop simulation。

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

我在项目里采用的证据分级如下：

{evidence_ladder_markdown("zh")}

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

## 7. 每一步是怎么做出来的

第一步是读论文和做资料盘点。我先把论文方法图拆成 tracking、DAgger、VAE、diffusion、guidance、deployment 几个模块，再把下载资料分成论文、官方代码、公开数据集、IsaacLab/RSL-RL、Unitree G1 assets 和参考仓库。这个阶段的产物是 local inventory、source ledger、paper/source map 和 unresolved details。

第二步是恢复环境。普通 Python 分析环境先跑通，然后恢复 PyTorch/CUDA diffusion 环境，最后恢复最难的 Isaac Sim/IsaacLab tracking 环境。早期遇到 inotify、Vulkan、USD save policy、URDF importer 等问题；后面通过 headless AppLauncher gate、G1 task construction gate 和 official-importer-export G1 USDA 路径把 tracking 基础设施推进到可运行状态。

第三步是先做 released-data 和源码审计，而不是直接训练。这样可以先确认论文公开数值、表格、图、reward、termination、obs/action schema 和 motion preprocessing contract，避免后面训练失败时不知道问题来自论文理解还是环境实现。

第四步是攻 tracking 数据链。最初的 enriched scaffold 和 FK-repaired bundle 只能证明链路，但后面发现 body order 和 `body_pos_w` 语义才是关键。robot-order FK repair 把 motion target 重排到 IsaacLab runtime body order，这是当前 tracking 主线的核心修复。

第五步是跑本地 PPO 和多 seed eval。robot-order PPO checkpoint eval 共 `{total_env_steps(robot_m)}` virtual env steps，reward mean `{reward_mean(robot_m)}`，done count `{done_count(robot_m)}`；三 seed eval 共 `{s['robot_order_multiseed_metrics'].get('total_env_steps')}` virtual env steps，mean done rate `{metric_value(robot_multi, 'done_rate')}`，body-position error mean `{metric_value(robot_multi, 'error_body_pos_mean')}`。这些结果说明当前 teacher 可以跑，但不够强。

第六步是做下游机制复现。因为官方 VAE/diffusion 和 DAgger 数据不公开，我用 local teacher rollout 训练 conditional VAE、state-latent denoiser 和 guidance proxy，证明 BeyondMimic-like pipeline 可以在公开资源下部分重建。

第七步是统一任务协议和报告。论文 Fig.5/Fig.6 涉及 joystick、waypoint、obstacle avoidance、transition、inpainting 和 composed objectives；我把它们整理成本地统一 protocol table，并明确 `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`。这样答辩时可以展示“我做了哪些任务形式”，同时不把 local proxy 写成论文结果。

第八步是整理失败和边界。所有 missing checkpoint、失败 run、Vulkan/inotify/URDF/importer 问题、tracking done/termination 异常都保留为审计证据。这样做不是给失败找借口，而是让后续每一步知道应该修哪里：当前最明确的是 wrist endpoint / `ee_body_pos` termination，而不是盲目继续训练。

## 8. 当前效果和问题

目前工程已经证明“链路能跑”，但还没有证明“论文效果复现”。tracking teacher 仍是最关键瓶颈。FK-repaired motion bundle 修复了旧 body position 退化，PPO 也能完整训练和评估；但 eval 中 done/termination 仍然过高，说明 teacher 还不能作为可信 DAgger 数据源。

最新 tracking quality diagnostic 更具体：step-0 done rate 是 `{metric_value(robot_quality, 'step0_done_rate')}`，step-0 body-position error 约 `{metric_value(robot_quality, 'step0_error_body_pos')}` 米；去掉 step 0 后 body-position error 降到 `{metric_value(robot_quality, 'mean_error_body_pos_post_step0')}`，但 post-step0 done rate 仍约 `{metric_value(robot_quality, 'post_step0_done_rate')}`。reset command warmup 的当前结论是 `{warmup_i.get('next_mainline_decision')}`。

最新 full warmup eval 状态是 `{warmup_eval_status}`：step-0 done count 从 `{warmup_eval_c.get('old_step0', {}).get('done_count')}` 降到 `{warmup_eval_c.get('warmup_eval_step0', {}).get('done_count')}`，step-0 body-position error 从 `{warmup_eval_c.get('old_step0', {}).get('error_body_pos')}` m 降到 `{warmup_eval_c.get('warmup_eval_step0', {}).get('error_body_pos')}` m；但整体 done rate 从 `{warmup_eval_c.get('old_done_rate')}` 升到 `{warmup_eval_c.get('warmup_eval_done_rate')}`。因此下一步不是盲目重训，而是先让 reset/target alignment、endpoint z、post-warmup policy-state distribution 和 `ee_body_pos` termination 变合理。

同 seed phase diagnostic 状态是 `{warmup_phase_status}`。它说明即便 seed 对齐，warmup 仍使 total done rate 增加 `{warmup_phase_m.get('same_seed_done_rate_delta')}`，post-step0 done rate 增加 `{warmup_phase_m.get('same_seed_post_step0_done_rate_delta')}`，`ee_body_pos` termination fraction 增加 `{warmup_phase_m.get('same_seed_ee_body_pos_termination_fraction_delta')}`，而 sampling top-bin 不变。这让下一步更明确：`{warmup_phase_i.get('recommended_next_experiment')}`。

no-advance reset-target refresh 是这一轮最新主线诊断。它不调用 `command_manager.compute()` 去推进 motion phase，而是直接重算 reset 后的 body targets。live probe 状态 `{target_refresh_live_status}`，endpoint-z done rate `{target_refresh_live_m.get('endpoint_done_rate_before')}` -> `{target_refresh_live_m.get('endpoint_done_rate_after')}`，endpoint-z error mean `{target_refresh_live_m.get('endpoint_z_error_mean_before')}` -> `{target_refresh_live_m.get('endpoint_z_error_mean_after')}`，`time_steps_unchanged_by_refresh = {target_refresh_live_m.get('time_steps_unchanged_by_refresh')}`。full eval 状态 `{target_refresh_status}`，step-0 done count delta `{target_refresh_c.get('step0_done_count_delta')}`，但 total done rate `{target_refresh_c.get('old_done_rate')}` -> `{target_refresh_c.get('target_refresh_done_rate')}`，post-step0 done-rate delta `{target_refresh_c.get('post_step0_done_rate_delta')}`。所以它把问题缩小了：reset target 陈旧是一个真问题，但 teacher 质量差还包含 reset state/action distribution、初始速度和 `ee_body_pos` termination 的问题。

现在 reset state/action distribution 也已经被具体量化：`{reset_state_action_status}`。它说明 target refresh 虽然让 step-0 body-position error 改善 `{reset_state_action_m.get('target_refresh_step0_body_error_delta')}` m，但 step-0 joint-velocity error 增加 `{reset_state_action_m.get('target_refresh_step0_joint_vel_delta')}`，first-five-step action mean 增加 `{reset_state_action_m.get('target_refresh_first5_action_abs_mean_delta')}`，post-step0 done-rate delta `{reset_state_action_m.get('target_refresh_post_step0_done_rate_delta')}`，`ee_body_pos` termination fraction delta `{reset_state_action_m.get('target_refresh_ee_body_pos_termination_fraction_delta')}`。这意味着下一步 full PPO 前要先修 reset-state、last-action observation、initial velocity 和 termination consistency。

最新 reset state/action consistency live probe 状态是 `{reset_state_action_consistency_status}`。它把 target refresh、action reset、action-offset alignment 和 motion-state rewrite 放在同一个 256-env live gate 里比较。target refresh alone 的 policy-step done rate 是 `{reset_state_action_consistency_m.get('target_refresh_policy_done_rate')}`，joint velocity error 是 `{reset_state_action_consistency_m.get('target_refresh_policy_joint_vel_after_step')}`；action reset 和 action-offset alignment 虽然分别把 joint velocity error 降到 `{reset_state_action_consistency_m.get('action_reset_policy_joint_vel_after_step')}` 和 `{reset_state_action_consistency_m.get('action_offset_policy_joint_vel_after_step')}`，但 done rate 变差到 `{reset_state_action_consistency_m.get('action_reset_policy_done_rate')}` 和 `{reset_state_action_consistency_m.get('action_offset_policy_done_rate')}`。motion-state/action-offset candidate 的 joint velocity 最低，是 `{reset_state_action_consistency_m.get('candidate_policy_joint_vel_after_step')}`，但 done rate 最差，是 `{reset_state_action_consistency_m.get('candidate_policy_done_rate')}`。最终 `any_variant_improves_done_and_joint_velocity = {reset_state_action_consistency_checks.get('any_variant_improves_done_and_joint_velocity')}`，所以没有推荐 full eval。这一步在答辩中可以解释为：我不是为了制造成功结果而盲目重训，而是在确认修复不会破坏 termination 之前，不把它推进到正式 PPO。

最新 endpoint-group ablation 可以作为答辩里“下一步为什么要修 wrist endpoint”的直接证据：`{endpoint_group_status}`。同 seed 条件下，target-refresh done rate `{endpoint_group_c.get('target_refresh_done_rate')}`，ankles-only `{endpoint_group_c.get('ankles_only_done_rate')}`，wrists-only `{endpoint_group_c.get('wrists_only_done_rate')}`，all-relaxed `{endpoint_group_c.get('all_endpoint_relaxed_done_rate')}`，dominant endpoint group 是 `{endpoint_group_i.get('dominant_endpoint_group')}`。

本轮新增 live wrist/ankle endpoint alignment probe 后，这个结论更具体：`{wrist_endpoint_status}`。probe 直接记录 `body_pos_w`、`body_pos_relative_w` 和 `robot_body_pos_w` 三组张量。target refresh 后，wrist z-error mean `{wrist_endpoint_m.get('refresh_wrist_rel_z_error_mean')}` m，高于 ankle `{wrist_endpoint_m.get('refresh_ankle_rel_z_error_mean')}` m；wrist done rate `{wrist_endpoint_m.get('refresh_wrist_done_rate')}`，也高于 ankle `{wrist_endpoint_m.get('refresh_ankle_done_rate')}`；policy step 后 wrist done rate `{wrist_endpoint_m.get('policy_step_wrist_done_rate')}`，仍高于 ankle `{wrist_endpoint_m.get('policy_step_ankle_done_rate')}`。诊断是 `{wrist_endpoint_m.get('diagnosis')}`。这说明下一轮 tracking 数据质量修复应该优先查 wrist endpoint target/body order、wrist FK height 和 `ee_body_pos` termination，而不是直接启动新的 downstream。

随后我把这个 live 结论扩成 full-size source diagnostic：`{wrist_source_status}`。它不是 smoke，而是 2048 env x 299 step 的完整诊断，统计 motion、phase bin 和 endpoint body 对 done/`ee_body_pos` 的贡献。总体 done rate `{wrist_source_m.get('done_rate')}`，`ee_body_pos` rate `{wrist_source_m.get('ee_body_pos_rate')}`；pre-step wrist exceed rate mean `{wrist_source_m.get('pre_wrist_done_rate', {}).get('mean')}`，ankle `{wrist_source_m.get('pre_ankle_done_rate', {}).get('mean')}`；post-step wrist `{wrist_source_m.get('post_wrist_done_rate', {}).get('mean')}`，ankle `{wrist_source_m.get('post_ankle_done_rate', {}).get('mean')}`。top wrist-heavy motions 包括 `{', '.join(row.get('motion', '') for row in wrist_source_top[:3])}`。这一步的意义是：它把下一轮修复从“再训练试试看”变成“先针对具体 motion/phase/body 修数据语义和 termination”。

本轮又补了 deterministic reset live gate：`{deterministic_reset_status}`。它说明 deterministic reset 确实能降低一部分 joint velocity transient，比如 policy joint velocity 从 `{deterministic_reset_m.get('official_refresh_policy_joint_vel_after_step')}` 降到 `{deterministic_reset_m.get('deterministic_refresh_policy_joint_vel_after_step')}`，但 done rate 从 `{deterministic_reset_m.get('official_refresh_policy_done_rate')}` 变差到 `{deterministic_reset_m.get('deterministic_refresh_policy_done_rate')}`；motion-state reset 的 done rate 也达到 `{deterministic_reset_m.get('motion_state_policy_done_rate')}`。因此 recommended full-eval variant 是 `{deterministic_reset_i.get('recommended_full_eval_variant') or 'none'}`。这进一步说明当前 blocker 不是“reset 随机性太大”这么简单，而是 body target、endpoint、初始速度、last-action observation 和 termination 的耦合问题。

统一任务协议表覆盖 `{protocol_m.get('task_count')}` 个本地 proxy tasks，其中前几个任务有 multi-seed 证据，transition/inpainting 仍偏单 seed 或 proxy。它适合答辩展示“我如何把论文 Fig.5/Fig.6 拆成本地协议”，但 `paper_level_reproduced_count = {protocol_m.get('paper_level_reproduced_count')}`，所以不能说复现了 Fig.5/Fig.6。

## 9. 失败产物和存储管理

项目现在保留大型成功 checkpoint、teacher rollout、state-latent shard 和可视化视频在本机，不提交 GitHub。失败运行、临时缓存和可重建中间产物需要定期清理。清理原则是：保留 summary、CSV、JSON、关键日志、manifest 和当前最佳 checkpoint；删除明确失败、临时、重复或可重建的大目录。

当前 conservative cleanup audit 记录 `{cleanup_m.get('deleted_or_previously_deleted_count')}` 个 deleted-or-previously-deleted bulky candidates，管理的已删除或确认缺席空间约 `{cleanup_m.get('managed_superseded_bytes_removed_or_absent')}` bytes。项目文件系统当前剩余约 `{s['disk_free_gib']}` GiB / `{s['disk_total_gib']}` GiB。

本轮还把 debug-only VAE/diffusion smoke 权重纳入清理策略：这些 `.pt` 文件只证明 save/load 或 3-step optimizer plumbing，不能作为论文训练权重；删除它们后保留 JSON/TSV/metrics/figure 摘要，不影响论文复现结论，也能缓解磁盘压力。

当前最大的本地 run 目录是：

{storage_top_markdown(storage_top)}

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
