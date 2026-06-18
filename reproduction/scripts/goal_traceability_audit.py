#!/usr/bin/env python3
"""Trace goal.md sections to current BeyondMimic reproduction evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
GOAL = ROOT / "goal.md"
OUT = ROOT / "res/goal_traceability"


TRACE_ROWS: list[dict[str, Any]] = [
    {
        "goal_section": "2 Fixed runtime context",
        "line_range": "goal.md:29-107",
        "requirement": "Use fixed project roots, keep generated files under /mnt/infini-data/test/BeyondMimic, do not assume real robot.",
        "status": "covered",
        "evidence": [
            "reproduction/docs/completion_matrix.md",
            "res/blocked_gates/blocked_gate_audit.json",
            "reproduction/docs/known_limitations.md",
        ],
        "notes": "Generated artifacts are under project root; Unitree G1 hardware is out of scope and explicitly blocked.",
    },
    {
        "goal_section": "3 Mission",
        "line_range": "goal.md:108-149",
        "requirement": "Reproduce released figures, official tracking, VAE/DAgger, state-latent dataset, diffusion, guidance, ablations, comparison, and final report.",
        "status": "partial",
        "evidence": [
            "res/final_report/final_reproduction_report.json",
            "res/master_audit/reproduction_master_audit.json",
            "reproduction/docs/completion_matrix.md",
        ],
        "notes": "Level A complete for released-data scope; Level B/C paper-level training, deployment, and Fig.5/Fig.6 remain partial/blocked.",
    },
    {
        "goal_section": "4 Operating principles",
        "line_range": "goal.md:150-261",
        "requirement": "Local-first inventory, read-only download area, evidence priority, discrepancy reporting, no hidden proxy claims.",
        "status": "covered",
        "evidence": [
            "reproduction/docs/local_inventory.tsv",
            "reproduction/docs/source_ledger.md",
            "reproduction/docs/discrepancy_report.md",
            "res/level_c/official_artifact_audit/level_c_official_artifact_audit.json",
        ],
        "notes": "Official/reference artifact boundaries and paper/source/code discrepancies are recorded.",
    },
    {
        "goal_section": "5 Environment-first",
        "line_range": "goal.md:262-507",
        "requirement": "System audit, project-local envs/caches, dependency locks, smoke tests before training.",
        "status": "partial",
        "evidence": [
            "logs/setup/system_audit.txt",
            "reproduction/docs/environment.md",
            "res/blocked_gates/blocked_gate_audit.json",
            "res/master_audit/reproduction_master_audit.json",
        ],
        "notes": "Analysis/tracking Python smoke and non-Kit audits pass; full IsaacLab/Kit smoke is blocked by inotify.",
    },
    {
        "goal_section": "6 GPU resource management",
        "line_range": "goal.md:508-651",
        "requirement": "Use GPUs responsibly, monitor training, avoid fake utilization and respect smoke gates.",
        "status": "partial",
        "evidence": ["logs/setup/system_audit.txt", "res/blocked_gates/blocked_gate_audit.json"],
        "notes": "GPU/system audit exists and no BeyondMimic long training process is running; no full training/GPU metrics because smoke gates are blocked.",
    },
    {
        "goal_section": "7 Project structure",
        "line_range": "goal.md:652-720",
        "requirement": "Maintain project structure under download/envs/cache/tmp/reproduction/logs/res.",
        "status": "covered",
        "evidence": ["reproduction/docs/completion_matrix.md", "reproduction/README.md"],
        "notes": "Current generated work is under project root; some future directories remain empty because corresponding training stages are blocked.",
    },
    {
        "goal_section": "8 Level A",
        "line_range": "goal.md:721-746",
        "requirement": "Released-data reproduction for IMU, errors, ablations, adaptive sampling, GRF.",
        "status": "covered",
        "evidence": ["res/released_figures/released_figure_summary.tsv", "reproduction/docs/paper_panel_map.tsv"],
        "notes": "Released-data scope is complete with processed data and PDF/SVG/PNG outputs.",
    },
    {
        "goal_section": "8 Level B",
        "line_range": "goal.md:747-775",
        "requirement": "Official-code motion preprocessing, replay, PPO tracking, adaptive sampling, evaluation, ONNX export, MuJoCo sim-to-sim.",
        "status": "blocked",
        "evidence": [
            "res/tracking/smoke_config_audit/tracking_config_audit.json",
            "res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.json",
            "res/blocked_gates/blocked_gate_audit.json",
        ],
        "notes": "Static/source audits pass; live Kit preprocessing/replay/PPO/evaluation blocked by inotify; ROS 2 Jazzy/Noble unavailable.",
    },
    {
        "goal_section": "8 Level C",
        "line_range": "goal.md:776-799",
        "requirement": "Paper-faithful VAE/DAgger/state-latent/diffusion/guidance reimplementation if official code is absent.",
        "status": "partial",
        "evidence": [
            "reproduction/docs/level_c_diffusion_plan.md",
            "res/level_c/official_artifact_audit/level_c_official_artifact_audit.json",
            "res/final_report/final_reproduction_report.json",
        ],
        "notes": "Paper parameters and debug mechanics are implemented/audited; true DAgger, trainable state-latent data, checkpoints, and rollout metrics remain missing.",
    },
    {
        "goal_section": "8 Level D",
        "line_range": "goal.md:800-815",
        "requirement": "Do not claim real robot reproduction without hardware and safety setup.",
        "status": "out_of_scope",
        "evidence": ["res/blocked_gates/blocked_gate_audit.json", "reproduction/docs/known_limitations.md"],
        "notes": "No Unitree G1 hardware is available; real launch is not executed.",
    },
    {
        "goal_section": "9 Hard constraints",
        "line_range": "goal.md:816-920",
        "requirement": "Avoid result-driven method changes, hidden proxies, fake data/results, and preserve failures.",
        "status": "covered",
        "evidence": [
            "res/paper_table_values/paper_table_value_audit.json",
            "res/final_report/final_reproduction_report.json",
            "reproduction/docs/unresolved_details.md",
        ],
        "notes": "Debug-only and blocked statuses are explicit; no long training or fabricated metrics are claimed.",
    },
    {
        "goal_section": "10.1-10.7 Motion tracking spec",
        "line_range": "goal.md:921-1147",
        "requirement": "Motion tracking frequency, target bodies, rewards, observation/action, randomization, termination, adaptive sampling.",
        "status": "partial",
        "evidence": [
            "res/tracking/smoke_config_audit/tracking_config_audit.json",
            "res/paper_table_values/paper_table_value_audit.json",
            "res/tracking/adaptive_sampling_probe/adaptive_sampling_probe.tsv",
        ],
        "notes": "Static values and official-code config are audited; live rollout and adaptive-sampling paper/code look-back discrepancy remain unresolved.",
    },
    {
        "goal_section": "10.8 Conditional VAE and DAgger",
        "line_range": "goal.md:1148-1190",
        "requirement": "Conditional VAE with real DAgger rollout and teacher query.",
        "status": "partial",
        "evidence": [
            "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
            "res/paper_table_values/paper_table_value_audit.json",
            "reproduction/docs/unresolved_details.md",
        ],
        "notes": "VAE dimensions/loss/accumulation match in debug probe; true DAgger rollout and checkpoint remain missing.",
    },
    {
        "goal_section": "10.9 Diffusion dataset",
        "line_range": "goal.md:1191-1231",
        "requirement": "State-latent tau dataset without reference conditioning, OU perturbation, stability rejection, symmetry, provenance, split.",
        "status": "partial",
        "evidence": [
            "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json",
            "res/level_c/fixture_split_manifest/fixture_split_manifest_summary.json",
            "res/level_c/augmentation_probe/level_c_augmentation_probe.json",
        ],
        "notes": "Debug fixture/provenance/OU/symmetry probes exist; trained VAE rollout dataset and live rejection gate are missing.",
    },
    {
        "goal_section": "10.10 State representation",
        "line_range": "goal.md:1232-1250",
        "requirement": "Hybrid yaw-centric representation, emphasis projection c=6, pseudoinverse reconstruction.",
        "status": "partial",
        "evidence": ["res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"],
        "notes": "Candidate fixture passes invariance and pseudoinverse checks; paper-exact full dataset path remains missing.",
    },
    {
        "goal_section": "10.11 Diffusion Transformer",
        "line_range": "goal.md:1251-1290",
        "requirement": "Transformer denoiser, independent timesteps, EMA, clean target, 25 Hz controller.",
        "status": "partial",
        "evidence": [
            "res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.json",
            "res/level_c/training_schedule_probe/level_c_training_schedule_probe.json",
            "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        ],
        "notes": "Paper architecture/schedule/timestep mechanics are probed; full training/checkpoint/deployment is missing.",
    },
    {
        "goal_section": "10.12 Test-time guidance",
        "line_range": "goal.md:1291-1321",
        "requirement": "Joystick, waypoint, SDF, inpainting, composed guidance and validation-only scale selection.",
        "status": "partial",
        "evidence": [
            "res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
            "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
            "res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json",
        ],
        "notes": "Formula/autograd/debug-loop checks exist; paper-exact validation/test protocol and trained rollouts are missing.",
    },
    {
        "goal_section": "11 Execution phases",
        "line_range": "goal.md:1322-1549",
        "requirement": "Run phases 0-10 through final comparison.",
        "status": "partial",
        "evidence": ["reproduction/PROGRESS.md", "res/final_report/final_reproduction_report.json"],
        "notes": "Phases 0-2 mostly complete; later training/evaluation phases are partial/blocked.",
    },
    {
        "goal_section": "12 Evaluation",
        "line_range": "goal.md:1550-1634",
        "requirement": "Report motion tracking, adaptive sampling, VAE, diffusion/guidance, and statistics metrics.",
        "status": "partial",
        "evidence": ["res/released_figures/released_figure_summary.tsv", "res/level_c", "res/final_report/final_reproduction_report.json"],
        "notes": "Released-data metrics and debug probe metrics exist; full multi-seed training/evaluation metrics are missing.",
    },
    {
        "goal_section": "13 Paper comparison",
        "line_range": "goal.md:1635-1701",
        "requirement": "Create paper-vs-reproduction comparison CSV/MD with comparison types.",
        "status": "covered",
        "evidence": [
            "res/comparison/paper_vs_reproduction.csv",
            "res/comparison/paper_vs_reproduction.md",
            "res/comparison/paper_vs_reproduction.json",
            "res/final_report/final_reproduction_report.json",
        ],
        "notes": "Comparison CSV/MD/JSON exists for current evidence and marks exact, approximate, qualitative, not-publicly-reproducible, and hardware-required rows explicitly.",
    },
    {
        "goal_section": "14 Coding requirements",
        "line_range": "goal.md:1702-1746",
        "requirement": "Type hints, docstrings, configs, tests, math unit tests.",
        "status": "partial",
        "evidence": ["reproduction/scripts", "res/master_audit/reproduction_master_audit.json"],
        "notes": "Audit/probe scripts are typed and compile; full reimplementation package/tests are not complete.",
    },
    {
        "goal_section": "15 Result and run management",
        "line_range": "goal.md:1747-1790",
        "requirement": "Run IDs, run directories, resolved configs, logs, metrics, checkpoints, videos.",
        "status": "partial",
        "evidence": ["reproduction/PROGRESS.md", "res/final_report/final_reproduction_report.json"],
        "notes": "Audit/probe outputs are organized; full training run directories/checkpoints/videos are absent because training phases are blocked/incomplete.",
    },
    {
        "goal_section": "16 Final deliverables",
        "line_range": "goal.md:1791-1868",
        "requirement": "Environment locks, code, experiments, docs, final report with official/reimpl/resource boundaries.",
        "status": "partial",
        "evidence": ["reproduction/docs/final_reproduction_report.md", "res/final_report/final_reproduction_report.json"],
        "notes": "Consolidated report exists; full checkpoints, videos, multi-seed statistics, and trained reimplementation artifacts remain missing.",
    },
    {
        "goal_section": "17 Mandatory progress report",
        "line_range": "goal.md:1869-1906",
        "requirement": "Update PROGRESS.md with detailed stage status.",
        "status": "covered",
        "evidence": ["reproduction/PROGRESS.md"],
        "notes": "Progress report is maintained with commands, outputs, metrics, risks, and next-stage notes.",
    },
    {
        "goal_section": "18 Initial action",
        "line_range": "goal.md:1907-1950",
        "requirement": "Start with audit/environment, then continue released figures, tracking, VAE, diffusion, guidance, ablations, comparison, final report.",
        "status": "partial",
        "evidence": ["res/master_audit/reproduction_master_audit.json", "res/final_report/final_reproduction_report.json"],
        "notes": "Audit/environment/released-data/final evidence reports are complete; full training/evaluation phases are still partial/blocked.",
    },
]


def heading_lines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(GOAL.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            rows.append({"line": i, "level": level, "heading": line.strip()})
    return rows


def evidence_exists(paths: list[str]) -> bool:
    for p in paths:
        if not (ROOT / p).exists():
            return False
    return True


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["goal_section", "line_range", "requirement", "status", "evidence", "evidence_exists", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "goal_section": row["goal_section"],
                    "line_range": row["line_range"],
                    "requirement": row["requirement"],
                    "status": row["status"],
                    "evidence": "; ".join(str(ROOT / p) for p in row["evidence"]),
                    "evidence_exists": row["evidence_exists"],
                    "notes": row["notes"],
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    missing_evidence: list[dict[str, Any]] = []
    for row in TRACE_ROWS:
        item = dict(row)
        item["evidence_exists"] = evidence_exists(item["evidence"])
        rows.append(item)
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        if not item["evidence_exists"]:
            missing_evidence.append(item)

    headings = heading_lines()
    json_path = OUT / "goal_traceability_audit.json"
    tsv_path = OUT / "goal_traceability_audit.tsv"
    summary: dict[str, Any] = {
        "status": "ok" if not missing_evidence else "failed",
        "experiment_type": "audit",
        "scope": "traceability from goal.md sections to current reproduction evidence and blockers",
        "goal_line_count": len(GOAL.read_text(encoding="utf-8").splitlines()),
        "heading_count": len(headings),
        "headings": headings,
        "trace_row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_rows": missing_evidence,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Traceability exists for the goal sections, but many goal requirements remain partial, blocked, "
                "or out of scope because live Kit tracking, true DAgger, trained Level C checkpoints, paper-level "
                "Fig. 5/Fig. 6 results, and real robot execution are not available."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
