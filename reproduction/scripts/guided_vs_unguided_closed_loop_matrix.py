#!/usr/bin/env python3
"""Summarize local guided-vs-unguided rollout evidence for reporting.

This script aggregates existing local virtual/resource-adjusted closed-loop
guidance artifacts. It does not run IsaacLab, start training, or promote the
local rollouts to paper-level Fig. 5/Fig. 6 evidence.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/report_assets/guided_vs_unguided_closed_loop_matrix"

RECEDING_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
    "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json"
)
ACTION_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
    "level_c_official_csv_loop_action_guidance_rollout_eval.json"
)
TASK_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
    "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json"
)
MULTISEED_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
    "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json"
)
FULL_BUNDLE_TASK_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
    "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def metric(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return float(value) if isinstance(value, int | float) else None


def delta(guided: float | None, baseline: float | None) -> float | None:
    if guided is None or baseline is None:
        return None
    return guided - baseline


def ratio(guided: float | None, baseline: float | None) -> float | None:
    if guided is None or baseline in (None, 0):
        return None
    return guided / baseline


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def row_from_variant(
    *,
    experiment: str,
    task: str,
    seed_group: str,
    source_json: Path,
    baseline_variant: str,
    guided_variant: str,
    baseline: dict[str, Any],
    guided: dict[str, Any],
    rollout_steps: int | None,
    video_path: str | None,
    claim_level: str,
    comparison_type: str,
    notes: str,
) -> dict[str, Any]:
    baseline_reward = metric(baseline, "reward_mean")
    guided_reward = metric(guided, "reward_mean")
    baseline_error = metric(baseline, "target_body_error_mean")
    guided_error = metric(guided, "target_body_error_mean")
    baseline_done = metric(baseline, "done_count_total")
    guided_done = metric(guided, "done_count_total")
    baseline_teacher_mse = metric(baseline, "guided_teacher_action_mse_mean")
    guided_teacher_mse = metric(guided, "guided_teacher_action_mse_mean")
    baseline_base_mse = metric(baseline, "guided_base_action_mse_mean")
    guided_base_mse = metric(guided, "guided_base_action_mse_mean")
    cost_delta = metric(guided, "guidance_cost_delta_mean")
    grad_norm = metric(guided, "guidance_grad_norm_mean")
    return {
        "experiment": experiment,
        "task": task,
        "seed_group": seed_group,
        "baseline_variant": baseline_variant,
        "guided_variant": guided_variant,
        "rollout_steps": rollout_steps,
        "comparison_type": comparison_type,
        "claim_level": claim_level,
        "baseline_reward_mean": baseline_reward,
        "guided_reward_mean": guided_reward,
        "reward_delta_guided_minus_baseline": delta(guided_reward, baseline_reward),
        "baseline_target_body_error_mean": baseline_error,
        "guided_target_body_error_mean": guided_error,
        "target_body_error_delta_guided_minus_baseline": delta(guided_error, baseline_error),
        "target_body_error_ratio_guided_over_baseline": ratio(guided_error, baseline_error),
        "baseline_done_count_total": baseline_done,
        "guided_done_count_total": guided_done,
        "done_count_delta_guided_minus_baseline": delta(guided_done, baseline_done),
        "baseline_teacher_action_mse_mean": baseline_teacher_mse,
        "guided_teacher_action_mse_mean": guided_teacher_mse,
        "teacher_action_mse_delta_guided_minus_baseline": delta(guided_teacher_mse, baseline_teacher_mse),
        "baseline_base_action_mse_mean": baseline_base_mse,
        "guided_base_action_mse_mean": guided_base_mse,
        "base_action_mse_delta_guided_minus_baseline": delta(guided_base_mse, baseline_base_mse),
        "guidance_cost_delta_mean": cost_delta,
        "guidance_grad_norm_mean": grad_norm,
        "source_json": rel(source_json),
        "video_path": rel(video_path),
        "notes": notes,
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    receding = load_json(RECEDING_JSON)
    rec_metrics = receding["metrics"]["variant_metrics"]
    rows.append(
        row_from_variant(
            experiment="official_csv_loop_receding_latent_guidance_rollout",
            task="composed_proxy",
            seed_group="single_seed",
            source_json=RECEDING_JSON,
            baseline_variant="vae_base",
            guided_variant="receding_latent_guided",
            baseline=rec_metrics["vae_base"],
            guided=rec_metrics["receding_latent_guided"],
            rollout_steps=int(receding["metrics"]["rollout_steps"]),
            video_path=receding["outputs"]["assets"].get("mp4"),
            claim_level="local_virtual_receding_horizon_state_latent_guidance_rollout",
            comparison_type="qualitative_only",
            notes="Local resource-adjusted closed-loop rollout; not official Fig. 5/Fig. 6 evidence.",
        )
    )
    rows.append(
        row_from_variant(
            experiment="official_csv_loop_receding_latent_guidance_rollout",
            task="composed_proxy",
            seed_group="single_seed",
            source_json=RECEDING_JSON,
            baseline_variant="denoised_latent",
            guided_variant="receding_latent_guided",
            baseline=rec_metrics["denoised_latent"],
            guided=rec_metrics["receding_latent_guided"],
            rollout_steps=int(receding["metrics"]["rollout_steps"]),
            video_path=receding["outputs"]["assets"].get("mp4"),
            claim_level="local_virtual_receding_horizon_state_latent_guidance_rollout",
            comparison_type="qualitative_only",
            notes="Compares local receding guidance to the unguided denoised-latent rollout.",
        )
    )

    action = load_json(ACTION_JSON)
    action_metrics = action["metrics"]["variant_metrics"]
    rows.append(
        row_from_variant(
            experiment="official_csv_loop_action_guidance_rollout",
            task="teacher_consistency_proxy",
            seed_group="single_seed",
            source_json=ACTION_JSON,
            baseline_variant="vae_base",
            guided_variant="action_guided",
            baseline=action_metrics["vae_base"],
            guided=action_metrics["action_guided"],
            rollout_steps=int(action["metrics"]["rollout_steps"]),
            video_path=action["outputs"]["assets"].get("mp4"),
            claim_level="local_virtual_action_space_teacher_consistency_guidance_rollout",
            comparison_type="qualitative_only",
            notes="Action-space guidance sanity bridge, not latent diffusion or paper-level guidance.",
        )
    )

    task_summary = load_json(TASK_JSON)
    for source_row in task_summary.get("rows", []):
        baseline = {
            "reward_mean": source_row.get("denoised_reward_mean"),
            "target_body_error_mean": source_row.get("denoised_target_body_error_mean"),
        }
        guided = {
            "reward_mean": source_row.get("guided_reward_mean"),
            "target_body_error_mean": source_row.get("guided_target_body_error_mean"),
            "done_count_total": source_row.get("guided_done_count_total"),
            "guided_teacher_action_mse_mean": source_row.get("guided_teacher_action_mse_mean"),
            "guided_base_action_mse_mean": source_row.get("guided_base_action_mse_mean"),
            "guidance_cost_delta_mean": source_row.get("guidance_cost_delta_mean"),
            "guidance_grad_norm_mean": source_row.get("guidance_grad_norm_mean"),
        }
        rows.append(
            row_from_variant(
                experiment="official_csv_loop_task_conditioned_latent_guidance_rollout",
                task=str(source_row.get("task", "")),
                seed_group="seed_group_0_existing",
                source_json=Path(source_row.get("summary_json", TASK_JSON)),
                baseline_variant="denoised_latent",
                guided_variant="task_conditioned_guided",
                baseline=baseline,
                guided=guided,
                rollout_steps=int(source_row.get("rollout_steps", 0) or 0),
                video_path=source_row.get("mp4"),
                claim_level=str(source_row.get("claim_level", "local_virtual_task_conditioned_guidance_rollout")),
                comparison_type="qualitative_only",
                notes="Single-seed local task-conditioned rollout over public official-CSV loop motion.",
            )
        )

    multiseed = load_json(MULTISEED_JSON)
    for source_row in multiseed.get("rows", []):
        baseline = {
            "reward_mean": source_row.get("denoised_reward_mean"),
            "target_body_error_mean": source_row.get("denoised_target_body_error_mean"),
            "done_count_total": source_row.get("denoised_done_count_total"),
        }
        guided = {
            "reward_mean": source_row.get("guided_reward_mean"),
            "target_body_error_mean": source_row.get("guided_target_body_error_mean"),
            "done_count_total": source_row.get("guided_done_count_total"),
            "guided_teacher_action_mse_mean": source_row.get("guided_teacher_action_mse_mean"),
            "guided_base_action_mse_mean": source_row.get("guided_base_action_mse_mean"),
            "guidance_cost_delta_mean": source_row.get("guidance_cost_delta_mean"),
            "guidance_grad_norm_mean": source_row.get("guidance_grad_norm_mean"),
        }
        rows.append(
            row_from_variant(
                experiment="official_csv_loop_task_conditioned_latent_guidance_multiseed",
                task=str(source_row.get("task", "")),
                seed_group=str(source_row.get("seed_group", "")),
                source_json=Path(source_row.get("summary_json", MULTISEED_JSON)),
                baseline_variant="denoised_latent",
                guided_variant="task_conditioned_guided",
                baseline=baseline,
                guided=guided,
                rollout_steps=int(source_row.get("rollout_steps", 0) or 0),
                video_path=source_row.get("mp4"),
                claim_level=str(source_row.get("claim_level", "local_virtual_task_conditioned_guidance_multiseed")),
                comparison_type="approximately_comparable",
                notes="Three local virtual seed groups for report-level guided-vs-unguided trend analysis only.",
            )
        )

    full_bundle_task = load_json(FULL_BUNDLE_TASK_JSON)
    for source_row in full_bundle_task.get("rows", []):
        baseline = {
            "reward_mean": source_row.get("denoised_reward_mean"),
            "target_body_error_mean": source_row.get("denoised_target_body_error_mean"),
            "done_count_total": source_row.get("denoised_done_count_total"),
        }
        guided = {
            "reward_mean": source_row.get("guided_reward_mean"),
            "target_body_error_mean": source_row.get("guided_target_body_error_mean"),
            "done_count_total": source_row.get("guided_done_count_total"),
            "guided_teacher_action_mse_mean": source_row.get("guided_teacher_action_mse_mean"),
            "guided_base_action_mse_mean": source_row.get("guided_base_action_mse_mean"),
            "guidance_cost_delta_mean": source_row.get("guidance_cost_delta_mean"),
            "guidance_grad_norm_mean": source_row.get("guidance_grad_norm_mean"),
        }
        rows.append(
            row_from_variant(
                experiment="official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout",
                task=str(source_row.get("task", "")),
                seed_group="full_bundle_seed_group_0",
                source_json=Path(source_row.get("summary_json", FULL_BUNDLE_TASK_JSON)),
                baseline_variant="denoised_latent",
                guided_variant="task_conditioned_guided",
                baseline=baseline,
                guided=guided,
                rollout_steps=int(source_row.get("rollout_steps", 0) or 0),
                video_path=source_row.get("mp4"),
                claim_level=str(
                    source_row.get(
                        "claim_level",
                        "local_virtual_full_bundle_task_conditioned_latent_guidance_rollout",
                    )
                ),
                comparison_type="qualitative_only",
                notes="Full-bundle local virtual task-conditioned rollout over the 40-motion official CSV-loop public bundle.",
            )
        )

    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["experiment"] in {
            "official_csv_loop_task_conditioned_latent_guidance_multiseed",
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout",
        }:
            groups[(row["experiment"], row["task"])].append(row)

    aggregate: list[dict[str, Any]] = []
    metrics = [
        "reward_delta_guided_minus_baseline",
        "target_body_error_delta_guided_minus_baseline",
        "target_body_error_ratio_guided_over_baseline",
        "guidance_cost_delta_mean",
        "guidance_grad_norm_mean",
        "guided_done_count_total",
    ]
    for (experiment, task), group_rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "experiment": experiment,
            "task": task,
            "seed_group_count": len({row["seed_group"] for row in group_rows}),
            "row_count": len(group_rows),
            "claim_level": (
                "local_virtual_full_bundle_task_conditioned_guidance_aggregate"
                if experiment == "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout"
                else "local_virtual_task_conditioned_guidance_multiseed_aggregate"
            ),
            "comparison_type": "approximately_comparable",
        }
        for key in metrics:
            values = [row[key] for row in group_rows if isinstance(row.get(key), int | float)]
            if values:
                out[f"{key}_mean"] = mean(values)
                out[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
                out[f"{key}_min"] = min(values)
                out[f"{key}_max"] = max(values)
        aggregate.append(out)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def make_plots(rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> list[str]:
    outputs: list[str] = []

    multi = [
        row
        for row in rows
        if row["experiment"]
        in {
            "official_csv_loop_task_conditioned_latent_guidance_multiseed",
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout",
        }
    ]
    if multi:
        tasks = sorted({row["task"] for row in multi})
        reward_means = [
            mean(
                row["reward_delta_guided_minus_baseline"]
                for row in multi
                if row["task"] == task and isinstance(row["reward_delta_guided_minus_baseline"], int | float)
            )
            for task in tasks
        ]
        error_means = [
            mean(
                row["target_body_error_delta_guided_minus_baseline"]
                for row in multi
                if row["task"] == task and isinstance(row["target_body_error_delta_guided_minus_baseline"], int | float)
            )
            for task in tasks
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        axes[0].bar(tasks, reward_means, color="#477ca8")
        axes[0].axhline(0, color="#333333", linewidth=0.8)
        axes[0].set_title("Guided reward delta")
        axes[0].set_ylabel("guided - denoised")
        axes[0].tick_params(axis="x", rotation=25)
        axes[1].bar(tasks, error_means, color="#b15f3f")
        axes[1].axhline(0, color="#333333", linewidth=0.8)
        axes[1].set_title("Tracking-error delta")
        axes[1].set_ylabel("guided - denoised")
        axes[1].tick_params(axis="x", rotation=25)
        path = OUT / "task_conditioned_multiseed_guided_deltas.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))

    if aggregate:
        tasks = [row["task"] for row in aggregate]
        cost = [row.get("guidance_cost_delta_mean_mean", 0.0) for row in aggregate]
        grad = [row.get("guidance_grad_norm_mean_mean", 0.0) for row in aggregate]
        fig, ax1 = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
        ax2 = ax1.twinx()
        ax1.bar(tasks, cost, color="#4d8f6f", label="cost delta")
        ax2.plot(tasks, grad, color="#7d4e9a", marker="o", label="grad norm")
        ax1.set_ylabel("mean cost delta")
        ax2.set_ylabel("mean guidance grad norm")
        ax1.set_title("Task-conditioned guidance signal strength")
        ax1.tick_params(axis="x", rotation=25)
        path = OUT / "task_conditioned_guidance_signal_strength.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))

    return outputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    aggregate = aggregate_rows(rows)
    plot_paths = make_plots(rows, aggregate)

    video_rows = [row for row in rows if row["video_path"]]
    checks = {
        "has_rows": len(rows) >= 10,
        "has_multiseed_rows": any(row["experiment"].endswith("multiseed") for row in rows),
        "has_four_task_aggregate": {row["task"] for row in aggregate} == {
            "composed",
            "joystick",
            "obstacle_avoidance",
            "waypoint",
        },
        "all_video_paths_exist_when_recorded": all((ROOT / row["video_path"]).is_file() for row in video_rows),
        "all_rows_have_claim_level": all(bool(row["claim_level"]) for row in rows),
        "all_rows_keep_local_or_limited_claim": all(
            "local_virtual" in row["claim_level"] or row["comparison_type"] in {"qualitative_only", "approximately_comparable"}
            for row in rows
        ),
        "does_not_claim_paper_level_fig5_fig6": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    status = "ok" if all(checks.values()) else "needs_review"

    csv_path = OUT / "guided_vs_unguided_closed_loop_matrix.csv"
    aggregate_csv_path = OUT / "guided_vs_unguided_closed_loop_aggregate.csv"
    json_path = OUT / "guided_vs_unguided_closed_loop_matrix.json"
    md_path = OUT / "guided_vs_unguided_closed_loop_matrix.md"
    write_csv(csv_path, rows)
    write_csv(aggregate_csv_path, aggregate)

    summary = {
        "status": status,
        "experiment_type": "guided_vs_unguided_closed_loop_matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_level": "local_virtual_guided_vs_unguided_closed_loop_report_matrix",
        "scope": (
            "Aggregates existing local virtual/resource-adjusted closed-loop guidance rollouts and videos for the "
            "English reading report. It does not run new training or claim official BeyondMimic Fig. 5/Fig. 6 results."
        ),
        "metrics": {
            "row_count": len(rows),
            "aggregate_row_count": len(aggregate),
            "video_row_count": len(video_rows),
            "task_count": len({row["task"] for row in rows if row["task"]}),
            "multiseed_row_count": sum(
                row["experiment"] == "official_csv_loop_task_conditioned_latent_guidance_multiseed" for row in rows
            ),
            "full_bundle_task_conditioned_row_count": sum(
                row["experiment"] == "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout"
                for row in rows
            ),
            "plot_count": len(plot_paths),
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "not_paper_level",
            "why_not_complete": (
                "These rows summarize local virtual rollouts under an enriched/resource-adjusted official-CSV-loop "
                "chain. They are useful report evidence, but they are not official checkpoints, not unpatched official "
                "rollouts, not paper Fig. 5/Fig. 6 videos, and not real-robot results."
            ),
        },
        "inputs": {
            "receding_latent_guidance_json": rel(RECEDING_JSON),
            "action_guidance_json": rel(ACTION_JSON),
            "task_conditioned_json": rel(TASK_JSON),
            "task_conditioned_multiseed_json": rel(MULTISEED_JSON),
            "full_bundle_task_conditioned_json": rel(FULL_BUNDLE_TASK_JSON),
        },
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "aggregate_csv": str(aggregate_csv_path),
            "md": str(md_path),
            "plots": plot_paths,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Guided vs Unguided Closed-Loop Matrix",
        "",
        "This report-facing matrix aggregates existing local virtual guidance rollouts. It is not a paper-level "
        "Fig. 5/Fig. 6 reproduction and it is not real-robot evidence.",
        "",
        f"- Status: `{status}`",
        f"- Matrix rows: `{len(rows)}`",
        f"- Multiseed rows: `{summary['metrics']['multiseed_row_count']}`",
        f"- Video-linked rows: `{len(video_rows)}`",
        f"- Aggregate task rows: `{len(aggregate)}`",
        "",
        "## Multiseed Task Aggregate",
        "",
        "| Task | Seeds | Reward Delta Mean | Error Delta Mean | Cost Delta Mean | Grad Norm Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| `{row['task']}` | `{row['seed_group_count']}` | "
            f"`{fmt(row.get('reward_delta_guided_minus_baseline_mean'))}` | "
            f"`{fmt(row.get('target_body_error_delta_guided_minus_baseline_mean'))}` | "
            f"`{fmt(row.get('guidance_cost_delta_mean_mean'))}` | "
            f"`{fmt(row.get('guidance_grad_norm_mean_mean'))}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Rows tagged `approximately_comparable` or `qualitative_only` are local virtual/resource-adjusted evidence. "
            "They should support the reading report's reproduction discussion, but they must not be described as "
            "official BeyondMimic checkpoints, unpatched official replay, paper Fig. 5/Fig. 6 metrics, or robot results.",
            "",
            "## Files",
            "",
            f"- Matrix CSV: `{rel(csv_path)}`",
            f"- Aggregate CSV: `{rel(aggregate_csv_path)}`",
            f"- JSON: `{rel(json_path)}`",
        ]
    )
    for path in plot_paths:
        lines.append(f"- Plot: `{rel(path)}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "rows": len(rows), "aggregate": len(aggregate)}, sort_keys=True))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
