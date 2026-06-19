#!/usr/bin/env python3
"""Create report-ready assets for the official-loop teacher rollout dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
    "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
)
OUT = ROOT / "res/report_assets/official_csv_loop_teacher_rollout_dataset"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shard_paths(summary: dict[str, Any]) -> list[Path]:
    paths = [Path(path) for path in summary["run"]["shard_npz_paths"]]
    if not paths:
        raise ValueError("No teacher rollout shards recorded")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(ROLLOUT_JSON)
    paths = shard_paths(summary)

    per_shard_rows: list[dict[str, Any]] = []
    reward_step_sum: np.ndarray | None = None
    done_step_sum: np.ndarray | None = None
    timeout_step_sum: np.ndarray | None = None
    sample_count_by_step: np.ndarray | None = None
    action_abs_sum: np.ndarray | None = None
    action_abs_max: np.ndarray | None = None
    action_sq_sum: np.ndarray | None = None
    action_sample_count = 0
    motion_step_counts: dict[int, int] = {}

    for path in paths:
        data = np.load(path)
        rewards = np.asarray(data["rewards"], dtype=np.float64)
        dones = np.asarray(data["dones"], dtype=bool)
        timeouts = np.asarray(data["timeouts"], dtype=bool)
        actions = np.asarray(data["actions"], dtype=np.float64)
        motion_time_steps = np.asarray(data["motion_time_steps"], dtype=np.int64)
        rank = int(np.asarray(data["rank"]).reshape(-1)[0])
        seed = int(np.asarray(data["seed"]).reshape(-1)[0])

        per_shard_rows.append(
            {
                "rank": rank,
                "seed": seed,
                "steps": rewards.shape[0],
                "envs": rewards.shape[1],
                "total_env_steps": int(rewards.size),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "done_count": int(dones.sum()),
                "timeout_count": int(timeouts.sum()),
                "action_abs_mean": float(np.abs(actions).mean()),
                "action_abs_max": float(np.abs(actions).max()),
                "unique_motion_steps": int(np.unique(motion_time_steps).size),
            }
        )

        if reward_step_sum is None:
            reward_step_sum = rewards.sum(axis=1)
            done_step_sum = dones.sum(axis=1).astype(np.float64)
            timeout_step_sum = timeouts.sum(axis=1).astype(np.float64)
            sample_count_by_step = np.full(rewards.shape[0], rewards.shape[1], dtype=np.float64)
            action_abs_sum = np.abs(actions).sum(axis=(0, 1))
            action_abs_max = np.abs(actions).max(axis=(0, 1))
            action_sq_sum = np.square(actions).sum(axis=(0, 1))
        else:
            reward_step_sum += rewards.sum(axis=1)
            done_step_sum += dones.sum(axis=1).astype(np.float64)
            timeout_step_sum += timeouts.sum(axis=1).astype(np.float64)
            sample_count_by_step += rewards.shape[1]
            action_abs_sum += np.abs(actions).sum(axis=(0, 1))
            action_abs_max = np.maximum(action_abs_max, np.abs(actions).max(axis=(0, 1)))
            action_sq_sum += np.square(actions).sum(axis=(0, 1))
        action_sample_count += actions.shape[0] * actions.shape[1]

        unique, counts = np.unique(motion_time_steps, return_counts=True)
        for key, count in zip(unique.tolist(), counts.tolist()):
            motion_step_counts[int(key)] = motion_step_counts.get(int(key), 0) + int(count)

    assert reward_step_sum is not None
    assert done_step_sum is not None
    assert timeout_step_sum is not None
    assert sample_count_by_step is not None
    assert action_abs_sum is not None
    assert action_abs_max is not None
    assert action_sq_sum is not None

    step = np.arange(reward_step_sum.shape[0])
    reward_step_mean = reward_step_sum / sample_count_by_step
    action_abs_mean = action_abs_sum / action_sample_count
    action_rms = np.sqrt(action_sq_sum / action_sample_count)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(step, reward_step_mean, color="#2563eb", label="mean reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Official-loop teacher rollout reward trace")
    axes[0].legend(loc="upper right")
    axes[1].bar(step, done_step_sum, width=1.0, color="#dc2626", label="done count")
    axes[1].bar(step, timeout_step_sum, width=1.0, color="#64748b", alpha=0.65, label="timeout count")
    axes[1].set_xlabel("Rollout step")
    axes[1].set_ylabel("Env count")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    reward_done_png = OUT / "teacher_rollout_reward_done_timeseries.png"
    fig.savefig(reward_done_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(action_abs_mean.shape[0])
    ax.bar(x - 0.2, action_abs_mean, width=0.4, color="#10b981", label="mean |action|")
    ax.bar(x + 0.2, action_rms, width=0.4, color="#f59e0b", label="RMS action")
    ax.plot(x, action_abs_max, color="#111827", linewidth=1.2, label="max |action|")
    ax.set_xlabel("Action dimension")
    ax.set_ylabel("Magnitude")
    ax.set_title("Teacher policy action distribution over full rollout shards")
    ax.legend(loc="upper right")
    fig.tight_layout()
    action_png = OUT / "teacher_rollout_action_distribution.png"
    fig.savefig(action_png, dpi=180)
    plt.close(fig)

    motion_steps = np.array(sorted(motion_step_counts), dtype=np.int64)
    motion_counts = np.array([motion_step_counts[int(idx)] for idx in motion_steps], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(motion_steps, motion_counts, color="#7c3aed")
    ax.fill_between(motion_steps, motion_counts, color="#c4b5fd", alpha=0.35)
    ax.set_xlabel("Motion time step")
    ax.set_ylabel("Samples")
    ax.set_title("Coverage of official-loop motion steps in teacher rollout dataset")
    fig.tight_layout()
    coverage_png = OUT / "teacher_rollout_motion_step_coverage.png"
    fig.savefig(coverage_png, dpi=180)
    plt.close(fig)

    shard_csv = OUT / "teacher_rollout_shard_summary.csv"
    write_csv(
        shard_csv,
        per_shard_rows,
        [
            "rank",
            "seed",
            "steps",
            "envs",
            "total_env_steps",
            "reward_mean",
            "reward_std",
            "done_count",
            "timeout_count",
            "action_abs_mean",
            "action_abs_max",
            "unique_motion_steps",
        ],
    )
    action_csv = OUT / "teacher_rollout_action_summary.csv"
    action_rows = [
        {
            "action_dim": idx,
            "mean_abs_action": float(action_abs_mean[idx]),
            "rms_action": float(action_rms[idx]),
            "max_abs_action": float(action_abs_max[idx]),
        }
        for idx in range(action_abs_mean.shape[0])
    ]
    write_csv(action_csv, action_rows, ["action_dim", "mean_abs_action", "rms_action", "max_abs_action"])

    aggregate = {
        "status": "ok",
        "source_rollout_json": str(ROLLOUT_JSON),
        "claim_level": "local_virtual_official_loop_teacher_rollout_report_asset",
        "limitation": (
            "Summarizes local virtual teacher rollout shards from the enriched-USD official-loop chain. "
            "This is not the official BeyondMimic DAgger dataset, not closed-loop guided diffusion, and not real robot evidence."
        ),
        "metrics": {
            "shard_count": len(paths),
            "total_env_steps": int(sum(row["total_env_steps"] for row in per_shard_rows)),
            "rollout_steps": int(reward_step_sum.shape[0]),
            "total_envs": int(sample_count_by_step[0]),
            "reward_mean_over_steps": float(reward_step_mean.mean()),
            "done_count_total": int(done_step_sum.sum()),
            "timeout_count_total": int(timeout_step_sum.sum()),
            "action_dim": int(action_abs_mean.shape[0]),
            "motion_step_coverage_count": int(motion_steps.size),
        },
        "assets": {
            "reward_done_timeseries_png": str(reward_done_png),
            "action_distribution_png": str(action_png),
            "motion_step_coverage_png": str(coverage_png),
            "shard_summary_csv": str(shard_csv),
            "action_summary_csv": str(action_csv),
            "summary_md": str(OUT / "README.md"),
        },
        "checks": {
            "source_rollout_status_ok": summary["status"] == "ok_official_csv_loop_teacher_rollout_dataset_completed",
            "two_shards_loaded": len(paths) == 2,
            "total_env_steps_match_source": int(sum(row["total_env_steps"] for row in per_shard_rows))
            == int(summary["aggregate_metrics"]["total_env_steps"]),
            "action_dim_29": int(action_abs_mean.shape[0]) == 29,
            "rollout_steps_299": int(reward_step_sum.shape[0]) == 299,
            "png_assets_exist": reward_done_png.is_file() and action_png.is_file() and coverage_png.is_file(),
            "csv_assets_exist": shard_csv.is_file() and action_csv.is_file(),
            "does_not_claim_official_dagger": True,
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_real_robot": True,
        },
    }
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Loop Teacher Rollout Dataset Assets",
                "",
                "These plots and tables summarize the full local virtual teacher rollout shards collected from the",
                "official csv-loop PPO checkpoint. They are intended for the English reading report and PPT.",
                "",
                "## Source",
                "",
                f"- Rollout audit: `{ROLLOUT_JSON}`",
                f"- Status: `{summary['status']}`",
                f"- Total env steps: `{aggregate['metrics']['total_env_steps']}`",
                f"- Shards: `{aggregate['metrics']['shard_count']}`",
                "",
                "## Assets",
                "",
                f"- `{reward_done_png}`",
                f"- `{action_png}`",
                f"- `{coverage_png}`",
                f"- `{shard_csv}`",
                f"- `{action_csv}`",
                "",
                "## Claim Level",
                "",
                "local_virtual_official_loop_teacher_rollout_report_asset. This is not the official BeyondMimic",
                "DAgger dataset, not paper-level closed-loop guided diffusion, and not real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    aggregate["assets"]["summary_md"] = str(readme)
    aggregate["checks"]["summary_md_exists"] = readme.is_file()

    asset_json = OUT / "official_csv_loop_teacher_rollout_report_assets.json"
    asset_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "assets": aggregate["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
