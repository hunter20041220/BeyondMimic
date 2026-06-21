#!/usr/bin/env python3
"""Diagnose reward and termination components for scaled PPO checkpoint evals."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
FINAL_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
)
BEST_CONFIRM_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval/"
    "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def mean_value(payload: dict[str, Any], key: str) -> float | None:
    item = payload.get(key, {})
    if isinstance(item, dict) and "mean" in item:
        return float(item["mean"])
    return None


def extract_component_rows(label: str, audit: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = audit["run"]["metrics"]
    episode = metrics.get("episode_log_metrics", {})
    reward_rows: list[dict[str, Any]] = []
    termination_rows: list[dict[str, Any]] = []
    for key, payload in sorted(episode.items()):
        value = mean_value(episode, key)
        if value is None:
            continue
        if key.startswith("Episode_Reward/"):
            reward_rows.append(
                {
                    "checkpoint_label": label,
                    "component": key.removeprefix("Episode_Reward/"),
                    "mean_value": value,
                    "abs_mean_value": abs(value),
                    "loaded_iteration": metrics.get("loaded_iteration"),
                    "reward_mean": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
                    "done_count_total": metrics.get("done_count_total"),
                    "total_env_steps": metrics.get("total_env_steps"),
                }
            )
        elif key.startswith("Episode_Termination/"):
            termination_rows.append(
                {
                    "checkpoint_label": label,
                    "component": key.removeprefix("Episode_Termination/"),
                    "mean_count_per_step": value,
                    "fraction_of_envs_per_step": value / float(metrics.get("num_envs", 1)),
                    "loaded_iteration": metrics.get("loaded_iteration"),
                    "done_count_total": metrics.get("done_count_total"),
                    "total_env_steps": metrics.get("total_env_steps"),
                }
            )
    return reward_rows, termination_rows


def extract_motion_rows(label: str, audit: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = audit["run"]["metrics"]
    motion = metrics.get("motion_metrics", {})
    rows: list[dict[str, Any]] = []
    for key, payload in sorted(motion.items()):
        if not isinstance(payload, dict) or "mean" not in payload:
            continue
        rows.append(
            {
                "checkpoint_label": label,
                "metric": key,
                "mean_value": payload["mean"],
                "max_value": payload.get("max"),
                "min_value": payload.get("min"),
                "loaded_iteration": metrics.get("loaded_iteration"),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final_audit = load_json(FINAL_EVAL_JSON)
    confirmation = load_json(BEST_CONFIRM_JSON)
    best_eval_path = Path(confirmation["outputs"]["json"])
    best_audit = load_json(best_eval_path)

    reward_rows: list[dict[str, Any]] = []
    termination_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    for label, audit in [("best_iter_300", best_audit), ("final_iter_999", final_audit)]:
        rewards, terms = extract_component_rows(label, audit)
        reward_rows.extend(rewards)
        termination_rows.extend(terms)
        motion_rows.extend(extract_motion_rows(label, audit))

    reward_rows.sort(key=lambda row: (row["checkpoint_label"], -row["abs_mean_value"]))
    termination_rows.sort(key=lambda row: (row["checkpoint_label"], -row["fraction_of_envs_per_step"]))
    motion_rows.sort(key=lambda row: (row["checkpoint_label"], row["metric"]))

    reward_csv = OUT / "reward_components.csv"
    termination_csv = OUT / "termination_components.csv"
    motion_csv = OUT / "motion_error_components.csv"
    write_csv(reward_csv, reward_rows, list(reward_rows[0].keys()))
    write_csv(termination_csv, termination_rows, list(termination_rows[0].keys()))
    write_csv(motion_csv, motion_rows, list(motion_rows[0].keys()))

    reward_df = pd.DataFrame(reward_rows)
    term_df = pd.DataFrame(termination_rows)
    motion_df = pd.DataFrame(motion_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    top_rewards = reward_df.groupby("component")["abs_mean_value"].max().sort_values(ascending=False).head(9).index
    fig, ax = plt.subplots(figsize=(11, 6.5))
    pivot = reward_df[reward_df["component"].isin(top_rewards)].pivot(
        index="component", columns="checkpoint_label", values="mean_value"
    )
    pivot.loc[top_rewards].plot(kind="barh", ax=ax)
    ax.set_title("Scaled PPO reward component means")
    ax.set_xlabel("Mean logged reward component")
    fig.tight_layout()
    reward_png = OUT / "reward_component_means.png"
    fig.savefig(reward_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    term_pivot = term_df.pivot(index="component", columns="checkpoint_label", values="fraction_of_envs_per_step")
    term_pivot.plot(kind="bar", ax=ax, color=["#2563eb", "#ea580c"])
    ax.set_title("Termination component rate per evaluation step")
    ax.set_ylabel("Mean count / num_envs")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    termination_png = OUT / "termination_component_rates.png"
    fig.savefig(termination_png, dpi=180)
    plt.close(fig)

    key_motion = [
        "error_anchor_pos",
        "error_body_pos",
        "error_joint_pos",
        "error_anchor_lin_vel",
        "error_body_lin_vel",
        "error_joint_vel",
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    motion_pivot = motion_df[motion_df["metric"].isin(key_motion)].pivot(
        index="metric", columns="checkpoint_label", values="mean_value"
    )
    motion_pivot.loc[key_motion].plot(kind="barh", ax=ax)
    ax.set_title("Motion error means for best-vs-final full-size eval")
    ax.set_xlabel("Mean motion error")
    fig.tight_layout()
    motion_png = OUT / "motion_error_means.png"
    fig.savefig(motion_png, dpi=180)
    plt.close(fig)

    final_term = term_df[term_df["checkpoint_label"] == "final_iter_999"].sort_values(
        "fraction_of_envs_per_step", ascending=False
    )
    best_term = term_df[term_df["checkpoint_label"] == "best_iter_300"].sort_values(
        "fraction_of_envs_per_step", ascending=False
    )
    dominant_final = final_term.iloc[0].to_dict()
    dominant_best = best_term.iloc[0].to_dict()
    diagnostic = {
        "status": "ok_official_importer_export_scaled_ppo_reward_termination_diagnostic",
        "experiment_type": "official_importer_export_scaled_ppo_reward_termination_diagnostic",
        "source_eval_jsons": {
            "best_iter_300": str(best_eval_path),
            "final_iter_999": str(FINAL_EVAL_JSON),
            "confirmation_json": str(BEST_CONFIRM_JSON),
        },
        "metrics": {
            "checkpoint_count": 2,
            "reward_component_count": len(reward_rows),
            "termination_component_count": len(termination_rows),
            "motion_metric_count": len(motion_rows),
            "dominant_final_termination_component": dominant_final["component"],
            "dominant_final_termination_fraction": dominant_final["fraction_of_envs_per_step"],
            "dominant_best_termination_component": dominant_best["component"],
            "dominant_best_termination_fraction": dominant_best["fraction_of_envs_per_step"],
        },
        "checks": {
            "dominant_final_is_ee_body_pos": dominant_final["component"] == "ee_body_pos",
            "dominant_best_is_ee_body_pos": dominant_best["component"] == "ee_body_pos",
            "dominant_final_fraction_gt_0_99": dominant_final["fraction_of_envs_per_step"] > 0.99,
            "dominant_best_fraction_gt_0_99": dominant_best["fraction_of_envs_per_step"] > 0.99,
            "reward_csv_exists": reward_csv.is_file(),
            "termination_csv_exists": termination_csv.is_file(),
            "motion_csv_exists": motion_csv.is_file(),
            "png_assets_exist": all(path.is_file() for path in [reward_png, termination_png, motion_png]),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "assets": {
            "json": str(OUT / "reward_termination_diagnostic.json"),
            "reward_components_csv": str(reward_csv),
            "termination_components_csv": str(termination_csv),
            "motion_error_components_csv": str(motion_csv),
            "reward_components_png": str(reward_png),
            "termination_components_png": str(termination_png),
            "motion_errors_png": str(motion_png),
            "readme": str(OUT / "README.md"),
        },
        "interpretation": {
            "claim_level": "local_virtual_scaled_ppo_reward_termination_diagnostic_not_paper_level",
            "goal_complete": False,
            "main_finding": (
                "The ee_body_pos termination component dominates both evaluated local checkpoints, explaining the "
                "near-total non-timeout done counts. This points to tracking/termination configuration or teacher "
                "quality as the next mainline diagnosis target."
            ),
            "why_not_paper_level": (
                "This analyzes local logged reward and termination components from local PPO checkpoint evals. It is "
                "not an official BeyondMimic metric, checkpoint, Fig. 5/Fig. 6 protocol, or real-robot result."
            ),
        },
    }
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Scaled PPO Reward / Termination Diagnostic",
                "",
                diagnostic["interpretation"]["main_finding"],
                "",
                f"Dominant final termination: `{dominant_final['component']}` at "
                f"`{dominant_final['fraction_of_envs_per_step']}` of envs per step.",
                f"Dominant best-checkpoint termination: `{dominant_best['component']}` at "
                f"`{dominant_best['fraction_of_envs_per_step']}` of envs per step.",
                "",
                "Claim level: local virtual diagnostic only. This is not a paper-level BeyondMimic result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(OUT / "reward_termination_diagnostic.json", diagnostic)
    print(
        json.dumps(
            {
                "status": diagnostic["status"],
                "dominant_final": diagnostic["metrics"]["dominant_final_termination_component"],
                "dominant_final_fraction": diagnostic["metrics"]["dominant_final_termination_fraction"],
                "dominant_best": diagnostic["metrics"]["dominant_best_termination_component"],
                "dominant_best_fraction": diagnostic["metrics"]["dominant_best_termination_fraction"],
                "json": diagnostic["assets"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
