#!/usr/bin/env python3
"""Audit the source-level meaning of the scaled PPO ee_body_pos termination blocker."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit"
TRACKING_CFG = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
)
G1_CFG = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"
)
TERMINATIONS_SRC = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/terminations.py"
)
REWARD_DIAGNOSTIC = (
    ROOT
    / "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/"
    "reward_termination_diagnostic.json"
)
MOTION_CSV = (
    ROOT
    / "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/"
    "motion_error_components.csv"
)
TERMINATION_CSV = (
    ROOT
    / "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/"
    "termination_components.csv"
)
MOTION_NPZ = ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle.npz"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def extract_ee_body_pos_block(text: str) -> str:
    match = re.search(r"ee_body_pos\s*=\s*DoneTerm\((.*?)\n    \)", text, flags=re.S)
    if not match:
        return ""
    return match.group(0)


def extract_body_names(block: str) -> list[str]:
    names_block = re.search(r'"body_names":\s*\[(.*?)\]', block, flags=re.S)
    if not names_block:
        return []
    return re.findall(r'"([^"]+)"', names_block.group(1))


def extract_python_body_names_assignment(text: str) -> list[str]:
    match = re.search(r"self\.commands\.motion\.body_names\s*=\s*\[(.*?)\]", text, flags=re.S)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def extract_threshold(block: str) -> float | None:
    match = re.search(r'"threshold":\s*([0-9.]+)', block)
    if not match:
        return None
    return float(match.group(1))


def extract_motion_metric(rows: list[dict[str, str]], label: str, metric: str) -> float | None:
    for row in rows:
        if row.get("checkpoint_label") == label and row.get("metric") == metric:
            return float(row["mean_value"])
    return None


def extract_termination_fraction(rows: list[dict[str, str]], label: str, component: str) -> float | None:
    for row in rows:
        if row.get("checkpoint_label") == label and row.get("component") == component:
            return float(row["fraction_of_envs_per_step"])
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tracking_text = TRACKING_CFG.read_text(encoding="utf-8")
    g1_text = G1_CFG.read_text(encoding="utf-8")
    term_text = TERMINATIONS_SRC.read_text(encoding="utf-8")
    diagnostic = load_json(REWARD_DIAGNOSTIC)
    motion_rows = read_csv(MOTION_CSV)
    termination_rows = read_csv(TERMINATION_CSV)
    motion_data = np.load(MOTION_NPZ, allow_pickle=False)

    block = extract_ee_body_pos_block(tracking_text)
    body_names = extract_body_names(block)
    threshold = extract_threshold(block)
    uses_z_only = "bad_motion_body_pos_z_only" in block
    z_only_source_match = "return torch.any(error > threshold, dim=-1)" in term_text and "[:, body_indexes, -1]" in term_text
    g1_command_body_names = extract_python_body_names_assignment(g1_text)

    labels = ["best_iter_300", "final_iter_999"]
    rows: list[dict[str, Any]] = []
    for label in labels:
        rows.append(
            {
                "checkpoint_label": label,
                "ee_body_pos_fraction": extract_termination_fraction(termination_rows, label, "ee_body_pos"),
                "anchor_ori_fraction": extract_termination_fraction(termination_rows, label, "anchor_ori"),
                "anchor_pos_fraction": extract_termination_fraction(termination_rows, label, "anchor_pos"),
                "error_body_pos_mean": extract_motion_metric(motion_rows, label, "error_body_pos"),
                "error_anchor_pos_mean": extract_motion_metric(motion_rows, label, "error_anchor_pos"),
                "error_joint_pos_mean": extract_motion_metric(motion_rows, label, "error_joint_pos"),
                "ee_body_pos_threshold_m": threshold,
                "threshold_metric_note": "threshold is z-only on four configured distal bodies; error_body_pos_mean is logged xyz/all-body context, not a direct threshold comparison",
            }
        )

    evidence_csv = OUT / "ee_body_pos_source_evidence.csv"
    write_csv(
        evidence_csv,
        rows,
        [
            "checkpoint_label",
            "ee_body_pos_fraction",
            "anchor_ori_fraction",
            "anchor_pos_fraction",
            "error_body_pos_mean",
            "error_anchor_pos_mean",
            "error_joint_pos_mean",
            "ee_body_pos_threshold_m",
            "threshold_metric_note",
        ],
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = range(len(rows))
    ax.bar([x - 0.18 for x in xs], [row["ee_body_pos_fraction"] for row in rows], width=0.36, label="ee_body_pos")
    ax.bar([x + 0.18 for x in xs], [row["anchor_ori_fraction"] for row in rows], width=0.36, label="anchor_ori")
    ax.set_xticks(list(xs), [row["checkpoint_label"] for row in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Termination fraction per env-step")
    ax.set_title("Scaled PPO termination source audit")
    ax.legend()
    fig.tight_layout()
    termination_png = OUT / "ee_body_pos_termination_fraction.png"
    fig.savefig(termination_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar([row["checkpoint_label"] for row in rows], [row["error_body_pos_mean"] for row in rows], color="#2563eb")
    if threshold is not None:
        ax.axhline(threshold, color="#dc2626", linestyle="--", label="ee_body_pos z-only threshold")
        ax.legend()
    ax.set_ylabel("Mean logged body-position error")
    ax.set_title("Logged body-position error context")
    fig.tight_layout()
    motion_png = OUT / "ee_body_pos_motion_error_context.png"
    fig.savefig(motion_png, dpi=180)
    plt.close(fig)

    report_md = OUT / "ee_body_pos_termination_source_audit.md"
    report_md.write_text(
        "\n".join(
            [
                "# ee_body_pos Termination Source Audit",
                "",
                "This audit links the local scaled PPO termination diagnostic back to the official tracking source.",
                "",
                f"- Official termination function: `bad_motion_body_pos_z_only` = `{uses_z_only}`.",
                f"- Official threshold: `{threshold}` meters on z-only distal body tracking.",
                f"- Configured termination body names: `{body_names}`.",
                f"- G1 command body-name count: `{len(g1_command_body_names)}`.",
                f"- Motion bundle shape: body_pos_w `{list(motion_data['body_pos_w'].shape)}`, joint_pos `{list(motion_data['joint_pos'].shape)}`.",
                f"- Best checkpoint ee_body_pos fraction: `{rows[0]['ee_body_pos_fraction']}`.",
                f"- Final checkpoint ee_body_pos fraction: `{rows[1]['ee_body_pos_fraction']}`.",
                "",
                "Interpretation: the weak local scaled PPO teacher is not merely a checkpoint-selection issue; both",
                "full-size evaluated checkpoints terminate almost entirely through the official z-only endpoint body",
                "position gate. The next debugging step should inspect retargeted endpoint z trajectories, policy",
                "stability around the four distal links, and whether the public-data/importer-export setup needs",
                "additional warm start, curriculum, or termination scheduling before downstream DAgger/VAE/diffusion",
                "rollouts are trustworthy.",
                "",
                "Claim level: local virtual source-linked diagnostic only. This is not a paper-level BeyondMimic",
                "closed-loop result, not an official checkpoint, and not real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    audit = {
        "status": "ok_official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit",
        "experiment_type": "source_linked_scaled_ppo_termination_blocker_audit",
        "source_files": {
            "tracking_env_cfg": str(TRACKING_CFG),
            "g1_flat_env_cfg": str(G1_CFG),
            "terminations": str(TERMINATIONS_SRC),
        },
        "source_config": {
            "ee_body_pos_block": block,
            "termination_function": "bad_motion_body_pos_z_only" if uses_z_only else "unknown",
            "threshold_m": threshold,
            "termination_body_names": body_names,
            "g1_command_body_name_count": len(g1_command_body_names),
            "g1_command_body_names": g1_command_body_names,
        },
        "motion_bundle": {
            "path": str(MOTION_NPZ),
            "joint_pos_shape": list(motion_data["joint_pos"].shape),
            "body_pos_w_shape": list(motion_data["body_pos_w"].shape),
            "fps": int(motion_data["fps"][0]),
        },
        "local_eval_evidence": {
            "source_diagnostic": str(REWARD_DIAGNOSTIC),
            "rows": rows,
            "diagnostic_metrics": diagnostic["metrics"],
        },
        "checks": {
            "source_files_exist": all(path.is_file() for path in [TRACKING_CFG, G1_CFG, TERMINATIONS_SRC]),
            "ee_body_pos_uses_z_only_function": uses_z_only,
            "z_only_source_indexes_last_coordinate": z_only_source_match,
            "threshold_is_0_25_m": threshold == 0.25,
            "termination_body_names_are_four_distal_links": body_names
            == [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "motion_bundle_shape_matches_full_public_bundle": list(motion_data["body_pos_w"].shape) == [11960, 40, 3],
            "dominant_termination_is_ee_body_pos_both_checkpoints": (
                diagnostic["checks"]["dominant_best_is_ee_body_pos"]
                and diagnostic["checks"]["dominant_final_is_ee_body_pos"]
            ),
            "ee_body_pos_fraction_gt_0_99_both_checkpoints": (
                rows[0]["ee_body_pos_fraction"] is not None
                and rows[1]["ee_body_pos_fraction"] is not None
                and rows[0]["ee_body_pos_fraction"] > 0.99
                and rows[1]["ee_body_pos_fraction"] > 0.99
            ),
            "assets_exist": all(path.is_file() for path in [evidence_csv, termination_png, motion_png, report_md]),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "assets": {
            "json": str(OUT / "ee_body_pos_termination_source_audit.json"),
            "csv": str(evidence_csv),
            "termination_fraction_png": str(termination_png),
            "motion_error_context_png": str(motion_png),
            "markdown": str(report_md),
            "readme": str(OUT / "README.md"),
        },
        "interpretation": {
            "claim_level": "local_virtual_source_linked_termination_diagnostic_not_paper_level",
            "goal_complete": False,
            "main_finding": (
                "The official tracking config terminates ee_body_pos when any of four distal bodies exceeds a "
                "0.25 m z-only error threshold. The local scaled PPO best and final checkpoints both trip this "
                "gate for more than 99% of env-steps, making endpoint body tracking the next mainline blocker."
            ),
            "next_debug_targets": [
                "retargeted ankle/wrist z trajectories in the official-importer-export motion bundle",
                "distal-link tracking stability during early PPO rollout",
                "termination scheduling/curriculum before collecting teacher rollouts",
                "asset/body-index consistency for the four distal termination links",
            ],
            "why_not_paper_level": (
                "This audit links local termination logs to official source configuration. It does not provide an "
                "official BeyondMimic checkpoint, a paper-level tracking metric, Fig. 5/Fig. 6 guided diffusion "
                "rollout, TensorRT deployment result, or real robot validation."
            ),
        },
    }
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Scaled PPO ee_body_pos Termination Source Audit",
                "",
                audit["interpretation"]["main_finding"],
                "",
                f"Threshold: `{threshold}` m.",
                f"Termination body names: `{body_names}`.",
                "",
                "Claim level: local virtual source-linked diagnostic only. This is not a paper-level BeyondMimic result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(OUT / "ee_body_pos_termination_source_audit.json", audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "threshold_m": threshold,
                "body_names": body_names,
                "best_ee_body_pos_fraction": rows[0]["ee_body_pos_fraction"],
                "final_ee_body_pos_fraction": rows[1]["ee_body_pos_fraction"],
                "json": audit["assets"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
