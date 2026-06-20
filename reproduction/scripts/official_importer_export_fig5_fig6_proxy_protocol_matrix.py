#!/usr/bin/env python3
"""Map current importer-export proxy evidence onto BeyondMimic Fig. 5/6 panels."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix"
FIG56_JSON = ROOT / "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"
BOUNDARY_JSON = (
    ROOT
    / "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
    "local_proxy_success_boundary.json"
)
VIDEO_INDEX_JSON = (
    ROOT
    / "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/"
    "importer_export_guidance_video_index.json"
)
FULL_SPLIT_GUIDANCE_JSON = (
    ROOT / "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
)
CHECKPOINT_VIS_JSON = (
    ROOT / "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json"
)
INPAINTING_PROXY_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
    "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
)
LATENT_PROJECTION_JSON = (
    ROOT
    / "res/report_assets/official_importer_export_full_bundle_latent_projection/"
    "official_importer_export_full_bundle_latent_projection_assets.json"
)


PANEL_LOCAL_MAP: dict[tuple[str, str], dict[str, Any]] = {
    ("Figure 5", "A"): {
        "local_proxy_tasks": ["joystick"],
        "offline_guidance_tasks": ["joystick"],
        "current_virtual_status": "supporting_local_proxy_closed_loop_not_panel_protocol",
        "available_virtual_next_validation": (
            "Render a local denoising-trajectory visualization under a fixed joystick command and pair it with "
            "a paper-style velocity-tracking rollout metric in IsaacLab."
        ),
    },
    ("Figure 5", "B"): {
        "local_proxy_tasks": [],
        "offline_guidance_tasks": [],
        "current_virtual_status": "training_chain_available_no_transition_panel_protocol",
        "available_virtual_next_validation": (
            "Build a local walking-to-running transition protocol from teacher/VAE latents and evaluate the "
            "transition in closed-loop simulation."
        ),
    },
    ("Figure 5", "C"): {
        "local_proxy_tasks": ["joystick"],
        "offline_guidance_tasks": ["joystick"],
        "current_virtual_status": "supporting_local_proxy_closed_loop_not_paper_success_protocol",
        "available_virtual_next_validation": (
            "Run a joystick-command IsaacLab protocol with velocity-tracking, survival, and disturbance-recovery "
            "metrics over multiple seeds."
        ),
    },
    ("Figure 5", "D"): {
        "local_proxy_tasks": [],
        "offline_guidance_tasks": [],
        "current_virtual_status": "local_pca_latent_projection_proxy_no_tsne_panel_protocol",
        "available_virtual_next_validation": (
            "Upgrade the local PCA latent projection into a t-SNE or UMAP visualization and then build a "
            "closed-loop walking-to-running transition protocol from teacher/VAE latents."
        ),
    },
    ("Figure 6", "A"): {
        "local_proxy_tasks": ["inpainting"],
        "offline_guidance_tasks": ["inpainting"],
        "current_virtual_status": "single_importer_export_keyframe_inpainting_proxy_closed_loop_diagnostic",
        "available_virtual_next_validation": (
            "Turn the single diagnostic future-keyframe/root-path proxy into a multi-seed paper-style keyframe "
            "protocol with explicit fall, transition smoothness, and success/failure thresholds."
        ),
    },
    ("Figure 6", "B"): {
        "local_proxy_tasks": ["waypoint", "obstacle_avoidance", "composed"],
        "offline_guidance_tasks": ["waypoint", "obstacle_avoidance", "composed_objectives"],
        "current_virtual_status": "supporting_local_proxy_closed_loop_but_real_world_panel_not_reproduced",
        "available_virtual_next_validation": (
            "Run a simulated waypoint plus SDF obstacle protocol with explicit collision, clearance, reach-target, "
            "fall, and timeout metrics. The exact paper panel still needs real robot or mocap evidence."
        ),
    },
}


CSV_FIELDS = [
    "figure",
    "panel",
    "paper_label",
    "paper_claim",
    "paper_source",
    "current_virtual_status",
    "comparison_type",
    "local_proxy_tasks",
    "offline_guidance_tasks",
    "closed_loop_rollout_rows",
    "closed_loop_video_rows",
    "closed_loop_seed_groups",
    "completion_rate_299_mean",
    "local_proxy_pass_rate_mean",
    "guidance_signal_positive_rate_mean",
    "action_changed_rate_mean",
    "offline_best_cost_delta_mean",
    "offline_positive_best_cost_delta_fraction_mean",
    "inpainting_proxy_status",
    "inpainting_guided_keyframe_error_mean",
    "inpainting_denoised_keyframe_error_mean",
    "inpainting_guided_keyframe_delta_vs_denoised",
    "latent_projection_status",
    "latent_projection_total_samples",
    "latent_projection_top2_variance",
    "latent_projection_walk_run_trace_rows",
    "debug_visualization_tasks",
    "available_virtual_next_validation",
    "remaining_blockers",
    "claim_level",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def collect_offline_rows(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["task"], row["mode"]): row for row in data.get("rows", [])}


def collect_debug_visual_tasks(data: dict[str, Any]) -> set[str]:
    outputs = data.get("outputs", {}).get("by_task", {})
    return set(outputs.keys())


def inpainting_proxy_payload(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval":
        return {}
    rows = data.get("rows", [])
    if not rows:
        return {}
    row = rows[0]
    return {
        "status": data.get("status"),
        "row": row,
        "mp4": data.get("outputs", {}).get("mp4", ""),
        "guided_keyframe_error_mean": row.get("guided_keyframe_error_mean"),
        "denoised_keyframe_error_mean": row.get("denoised_keyframe_error_mean"),
        "guided_keyframe_delta_vs_denoised": row.get("guided_keyframe_error_delta_vs_denoised"),
    }


def latent_projection_payload(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "ok_official_importer_export_full_bundle_latent_projection_report_assets":
        return {}
    metrics = data.get("metrics", {})
    return {
        "status": data.get("status"),
        "total_latent_samples": metrics.get("total_latent_samples"),
        "pca_explained_variance_ratio_top2": metrics.get("pca_explained_variance_ratio_top2"),
        "walk_run_trace_rows": metrics.get("walk_run_trace_rows"),
        "assets": data.get("assets", {}),
    }


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fig56 = load_json(FIG56_JSON)
    boundary = load_json(BOUNDARY_JSON)
    video_index = load_json(VIDEO_INDEX_JSON)
    full_split = load_json(FULL_SPLIT_GUIDANCE_JSON)
    checkpoint_vis = load_json(CHECKPOINT_VIS_JSON)
    inpainting_proxy = inpainting_proxy_payload(load_json(INPAINTING_PROXY_JSON))
    latent_projection = latent_projection_payload(load_json(LATENT_PROJECTION_JSON))

    aggregate_by_task = {row["task"]: row for row in boundary["aggregate"]}
    boundary_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in boundary["rows"]:
        boundary_rows_by_task.setdefault(row["task"], []).append(row)
    video_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in video_index["rows"]:
        video_rows_by_task.setdefault(row["task"], []).append(row)
    offline_rows = collect_offline_rows(full_split)
    debug_visual_tasks = collect_debug_visual_tasks(checkpoint_vis)

    matrix_rows: list[dict[str, Any]] = []
    for panel in fig56["rows"]:
        key = (panel["figure"], panel["panel"])
        mapping = PANEL_LOCAL_MAP[key]
        local_proxy_tasks = mapping["local_proxy_tasks"]
        offline_tasks = mapping["offline_guidance_tasks"]
        aggregate_rows = [aggregate_by_task[task] for task in local_proxy_tasks if task in aggregate_by_task]
        local_boundary_rows = [
            row for task in local_proxy_tasks for row in boundary_rows_by_task.get(task, [])
        ]
        local_video_rows = [row for task in local_proxy_tasks for row in video_rows_by_task.get(task, [])]
        inpainting_proxy_rows = [inpainting_proxy["row"]] if key == ("Figure 6", "A") and inpainting_proxy else []
        inpainting_video_rows = [inpainting_proxy["mp4"]] if inpainting_proxy_rows and inpainting_proxy.get("mp4") else []
        offline_task_rows = [
            offline_rows[(task, mode)]
            for task in offline_tasks
            for mode in ("offline", "reverse")
            if (task, mode) in offline_rows
        ]
        visual_tasks = sorted(
            task for task in [*offline_tasks, *local_proxy_tasks] if task in debug_visual_tasks
        )
        closed_loop_seed_groups = sorted({row.get("seed_group", "") for row in local_boundary_rows if row})
        if inpainting_proxy_rows:
            closed_loop_seed_groups.append("inpainting_single_seed")
        comparison_type = "requires_real_robot" if key == ("Figure 6", "B") else "qualitative_only"
        aggregate_completion = avg([float(row["completion_rate_299"]) for row in aggregate_rows])
        if aggregate_completion is None and inpainting_proxy_rows:
            aggregate_completion = 1.0 if inpainting_proxy_rows[0].get("rollout_steps") == 299 else 0.0
        matrix_rows.append(
            {
                "figure": panel["figure"],
                "panel": panel["panel"],
                "paper_label": panel["label"],
                "paper_claim": panel["claim"],
                "paper_source": panel["source_evidence"],
                "current_virtual_status": mapping["current_virtual_status"],
                "comparison_type": comparison_type,
                "local_proxy_tasks": ",".join(local_proxy_tasks),
                "offline_guidance_tasks": ",".join(offline_tasks),
                "closed_loop_rollout_rows": len(local_boundary_rows) + len(inpainting_proxy_rows),
                "closed_loop_video_rows": len(local_video_rows) + len(inpainting_video_rows),
                "closed_loop_seed_groups": ",".join(closed_loop_seed_groups),
                "completion_rate_299_mean": aggregate_completion,
                "local_proxy_pass_rate_mean": avg(
                    [float(row["local_proxy_pass_rate"]) for row in aggregate_rows]
                ),
                "guidance_signal_positive_rate_mean": avg(
                    [float(row["guidance_signal_positive_rate"]) for row in aggregate_rows]
                ),
                "action_changed_rate_mean": avg([float(row["action_changed_rate"]) for row in aggregate_rows]),
                "offline_best_cost_delta_mean": avg(
                    [float(row["mean_best_cost_delta"]) for row in offline_task_rows]
                ),
                "offline_positive_best_cost_delta_fraction_mean": avg(
                    [float(row["positive_best_cost_delta_fraction"]) for row in offline_task_rows]
                ),
                "inpainting_proxy_status": inpainting_proxy.get("status", "") if key == ("Figure 6", "A") else "",
                "inpainting_guided_keyframe_error_mean": (
                    inpainting_proxy.get("guided_keyframe_error_mean", "") if key == ("Figure 6", "A") else ""
                ),
                "inpainting_denoised_keyframe_error_mean": (
                    inpainting_proxy.get("denoised_keyframe_error_mean", "") if key == ("Figure 6", "A") else ""
                ),
                "inpainting_guided_keyframe_delta_vs_denoised": (
                    inpainting_proxy.get("guided_keyframe_delta_vs_denoised", "")
                    if key == ("Figure 6", "A")
                    else ""
                ),
                "latent_projection_status": (
                    latent_projection.get("status", "") if key == ("Figure 5", "D") else ""
                ),
                "latent_projection_total_samples": (
                    latent_projection.get("total_latent_samples", "") if key == ("Figure 5", "D") else ""
                ),
                "latent_projection_top2_variance": (
                    latent_projection.get("pca_explained_variance_ratio_top2", "")
                    if key == ("Figure 5", "D")
                    else ""
                ),
                "latent_projection_walk_run_trace_rows": (
                    latent_projection.get("walk_run_trace_rows", "") if key == ("Figure 5", "D") else ""
                ),
                "debug_visualization_tasks": ",".join(visual_tasks),
                "available_virtual_next_validation": mapping["available_virtual_next_validation"],
                "remaining_blockers": "; ".join(panel["blocking_dependencies"]),
                "claim_level": "local_proxy_protocol_matrix_not_paper_fig5_fig6",
                "paper_panel_status": panel["status"],
                "source_debug_evidence": ",".join(panel.get("debug_evidence_present", [])),
                "representative_video_paths": [row["mp4"] for row in local_video_rows[:3]]
                + inpainting_video_rows[:1],
            }
        )
    source_summary = {
        "fig5_fig6_feasibility_status": fig56["status"],
        "boundary_status": boundary["status"],
        "video_index_status": video_index["status"],
        "full_split_guidance_status": full_split["status"],
        "checkpoint_visualization_status": checkpoint_vis["status"],
        "inpainting_proxy_status": inpainting_proxy.get("status", ""),
        "inpainting_proxy_metrics": {
            "guided_keyframe_error_mean": inpainting_proxy.get("guided_keyframe_error_mean"),
            "denoised_keyframe_error_mean": inpainting_proxy.get("denoised_keyframe_error_mean"),
            "guided_keyframe_delta_vs_denoised": inpainting_proxy.get("guided_keyframe_delta_vs_denoised"),
        },
        "latent_projection_status": latent_projection.get("status", ""),
        "latent_projection_metrics": {
            "total_latent_samples": latent_projection.get("total_latent_samples"),
            "pca_explained_variance_ratio_top2": latent_projection.get("pca_explained_variance_ratio_top2"),
            "walk_run_trace_rows": latent_projection.get("walk_run_trace_rows"),
        },
        "boundary_metrics": boundary["metrics"],
        "video_index_metrics": video_index["metrics"],
    }
    return matrix_rows, source_summary


def plot_rows(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['figure'].replace('Figure ', 'F')}{row['panel']}" for row in rows]
    proxy_rates = [row["local_proxy_pass_rate_mean"] or 0.0 for row in rows]
    offline_rates = [row["offline_positive_best_cost_delta_fraction_mean"] or 0.0 for row in rows]
    closed_loop_rows = [row["closed_loop_rollout_rows"] for row in rows]
    x = list(range(len(rows)))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].bar(
        [v - width / 2 for v in x],
        proxy_rates,
        width=width,
        label="local proxy pass",
        color="#3f6f9d",
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[0].bar(
        [v + width / 2 for v in x],
        offline_rates,
        width=width,
        label="offline cost positive",
        color="#3e8d68",
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Evidence rate")
    axes[0].set_title("Panel-aligned local proxy rates")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        closed_loop_rows,
        color="#c98337",
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Importer-export rollout rows")
    axes[1].set_title("Closed-loop proxy coverage by panel")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("BeyondMimic Fig. 5/6 protocol matrix: local virtual evidence only")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Official-importer-export Fig. 5/Fig. 6 proxy protocol matrix",
        "",
        "This matrix maps the current local official-importer-export guidance evidence onto the six paper panels.",
        "It is a report-facing planning aid, not a paper-level reproduction claim.",
        "",
        "| Panel | Paper claim | Current local status | Local proxy tasks | Rollout rows | Proxy pass rate | Next virtual validation |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        panel = f"{row['figure']} {row['panel']}"
        proxy_rate = row["local_proxy_pass_rate_mean"]
        proxy_text = "" if proxy_rate is None else f"{proxy_rate:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    panel,
                    str(row["paper_claim"]).replace("|", "/"),
                    row["current_virtual_status"],
                    row["local_proxy_tasks"],
                    str(row["closed_loop_rollout_rows"]),
                    proxy_text,
                    row["available_virtual_next_validation"].replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Claim boundary: every row is local virtual evidence or a blocked/proxy mapping. None of the rows is an official BeyondMimic Fig. 5/Fig. 6 success, fall, collision, TensorRT, or real-robot result.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, source_summary = build_rows()
    json_path = OUT / "fig5_fig6_proxy_protocol_matrix.json"
    csv_path = OUT / "fig5_fig6_proxy_protocol_matrix.csv"
    md_path = OUT / "fig5_fig6_proxy_protocol_matrix.md"
    png_path = OUT / "fig5_fig6_proxy_protocol_rates.png"
    readme_path = OUT / "README.md"
    plot_rows(rows, png_path)
    write_csv(csv_path, rows, CSV_FIELDS)
    metrics = {
        "panel_count": len(rows),
        "closed_loop_proxy_panel_count": sum(1 for row in rows if row["closed_loop_rollout_rows"] > 0),
        "offline_or_debug_panel_count": sum(
            1
            for row in rows
            if row["offline_guidance_tasks"] or row["debug_visualization_tasks"]
        ),
        "importer_export_closed_loop_rollout_rows_referenced": sum(
            int(row["closed_loop_rollout_rows"]) for row in rows
        ),
        "importer_export_closed_loop_video_rows_referenced": sum(int(row["closed_loop_video_rows"]) for row in rows),
        "inpainting_proxy_rollout_rows_referenced": sum(
            1 for row in rows if row["figure"] == "Figure 6" and row["panel"] == "A" and row["closed_loop_rollout_rows"]
        ),
        "latent_projection_proxy_panel_count": sum(
            1
            for row in rows
            if row["figure"] == "Figure 5"
            and row["panel"] == "D"
            and row["latent_projection_status"]
            == "ok_official_importer_export_full_bundle_latent_projection_report_assets"
        ),
        "paper_level_reproduced_panel_count": 0,
        "requires_real_robot_panel_count": sum(1 for row in rows if row["comparison_type"] == "requires_real_robot"),
    }
    write_markdown(md_path, {"rows": rows})
    checks = {
        "source_fig5_fig6_status_ok": source_summary["fig5_fig6_feasibility_status"] == "ok",
        "source_boundary_status_ok": source_summary["boundary_status"]
        == "ok_official_importer_export_full_bundle_task_conditioned_guidance_success_boundary",
        "source_video_index_status_ok": source_summary["video_index_status"]
        == "ok_official_importer_export_full_bundle_guidance_video_contact_sheet",
        "source_inpainting_proxy_status_ok": source_summary["inpainting_proxy_status"]
        == "ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval",
        "source_latent_projection_status_ok": source_summary["latent_projection_status"]
        == "ok_official_importer_export_full_bundle_latent_projection_report_assets",
        "all_six_paper_panels_mapped": len(rows) == 6
        and {(row["figure"], row["panel"]) for row in rows}
        == {
            ("Figure 5", "A"),
            ("Figure 5", "B"),
            ("Figure 5", "C"),
            ("Figure 5", "D"),
            ("Figure 6", "A"),
            ("Figure 6", "B"),
        },
        "has_importer_export_proxy_closed_loop_rows": metrics[
            "importer_export_closed_loop_rollout_rows_referenced"
        ]
        >= 12,
        "has_inpainting_offline_or_debug_evidence": any(
            row["figure"] == "Figure 6" and row["panel"] == "A" and "inpainting" in row["offline_guidance_tasks"]
            for row in rows
        ),
        "has_inpainting_importer_export_proxy_closed_loop": any(
            row["figure"] == "Figure 6"
            and row["panel"] == "A"
            and row["inpainting_proxy_status"]
            == "ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
            and int(row["closed_loop_rollout_rows"]) >= 1
            for row in rows
        ),
        "records_inpainting_guided_delta": any(
            row["figure"] == "Figure 6"
            and row["panel"] == "A"
            and row["inpainting_guided_keyframe_delta_vs_denoised"] != ""
            for row in rows
        ),
        "has_fig5d_latent_projection_proxy": any(
            row["figure"] == "Figure 5"
            and row["panel"] == "D"
            and row["latent_projection_status"]
            == "ok_official_importer_export_full_bundle_latent_projection_report_assets"
            and row["latent_projection_walk_run_trace_rows"] != ""
            for row in rows
        ),
        "all_rows_not_paper_level": all(
            row["claim_level"] == "local_proxy_protocol_matrix_not_paper_fig5_fig6" for row in rows
        ),
        "does_not_claim_goal_complete": True,
        "assets_written": all(
            path.is_file() and path.stat().st_size > 0 for path in [csv_path, md_path, png_path]
        ),
    }
    payload = {
        "status": "ok_official_importer_export_fig5_fig6_proxy_protocol_matrix"
        if all(checks.values())
        else "failed_official_importer_export_fig5_fig6_proxy_protocol_matrix",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "fig5_fig6_proxy_protocol_matrix",
        "scope": "Map current local virtual importer-export guidance evidence to BeyondMimic Fig. 5/6 panels.",
        "source_summary": source_summary,
        "metrics": metrics,
        "checks": checks,
        "rows": rows,
        "assets": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "plot_png": str(png_path),
            "readme": str(readme_path),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_proxy_protocol_matrix_not_paper_fig5_fig6",
            "why_not_paper_level": (
                "The matrix reuses local PPO/VAE/denoiser checkpoints, local proxy objectives, offline guidance "
                "evidence, and local IsaacLab rollouts. It does not provide official BeyondMimic checkpoints, the "
                "paper Fig. 5/Fig. 6 task protocol, TensorRT deployment, mocap/real-world context, or real robot "
                "evidence. The Fig. 5D row uses a local PCA projection proxy, not the paper t-SNE panel."
            ),
            "no_hardware_next_steps": [
                "paper-style joystick velocity/recovery metric gate in IsaacLab",
                "multi-seed paper-style keyframe/inpainting protocol with success/failure thresholds",
                "simulated waypoint plus SDF obstacle gate with collision/clearance metrics",
                "local latent t-SNE/UMAP visualization from teacher-rollout VAE latents",
            ],
        },
    }
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    readme_path.write_text(
        "\n".join(
            [
                "# Fig. 5/Fig. 6 proxy protocol matrix",
                "",
                "This folder maps current local official-importer-export guidance evidence to the six BeyondMimic Fig. 5/Fig. 6 paper panels.",
                "",
                "It is intentionally conservative: the matrix records local proxy coverage and remaining virtual validation steps, not official paper-level success rates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "json": str(json_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
