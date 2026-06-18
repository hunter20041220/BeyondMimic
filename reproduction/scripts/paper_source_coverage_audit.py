#!/usr/bin/env python3
"""Audit coverage of paper LaTeX figures, tables, and key method claims."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PAPER = ROOT / "reproduction/paper/source"
OUT = ROOT / "res/paper_source_coverage"


EXPECTED_FIGURES = {
    "fig:teaser": {
        "paper_item": "Figure 1",
        "coverage_status": "paper/source indexed",
        "evidence": "reproduction/paper/source/figures/Fig1.pdf",
        "notes": "Framework overview figure; no separate numeric reproduction target.",
    },
    "fig:more_motion": {
        "paper_item": "Figure 2",
        "coverage_status": "blocked/no released panel data",
        "evidence": "res/blocked_gates/blocked_gate_audit.json",
        "notes": "Real-hardware motion-gallery claim requires videos/logs/hardware evidence not released locally.",
    },
    "fig:agile": {
        "paper_item": "Figure 3",
        "coverage_status": "partial released-data reproduced",
        "evidence": "res/released_figures/imu_orientation_accel_angular_velocity/imu_orientation_accel_angular_velocity.pdf",
        "notes": "Panel B IMU curves reproduced from released data; panel A visual forest demonstration is not regenerated.",
    },
    "fig:natural": {
        "paper_item": "Figure 4",
        "coverage_status": "partial released-data reproduced",
        "evidence": "res/released_figures/grf_walk_human_reference/grf_walk_human_reference.pdf",
        "notes": "Panel C GRF components reproduced from released data; video/user-study/interference panels are not reproduced from local data.",
    },
    "fig:versatile_loco": {
        "paper_item": "Figure 5",
        "coverage_status": "blocked; debug mechanisms audited",
        "evidence": "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
        "notes": "Paper-level reproduction blocked by missing trained VAE/diffusion checkpoints and closed-loop task logs.",
    },
    "fig:versatile_inpaint_sdf": {
        "paper_item": "Figure 6",
        "coverage_status": "blocked; debug mechanisms audited",
        "evidence": "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
        "notes": "Paper-level reproduction blocked by missing inpainting/SDF rollout data, checkpoints, and real/mocap evidence.",
    },
    "fig:pipline": {
        "paper_item": "Figure 7",
        "coverage_status": "paper/source indexed; component evidence audited",
        "evidence": "reproduction/docs/paper_parameter_map.md",
        "notes": "Conceptual pipeline; component coverage is split across tracking audits and Level C debug probes.",
    },
    "fig:ablation": {
        "paper_item": "Figure 8",
        "coverage_status": "partial released-data reproduced",
        "evidence": "res/released_figures/released_figure_summary.tsv",
        "notes": "Released ablation/adaptive-sampling panels reproduced or processed; source/code adaptive look-back discrepancy remains.",
    },
    "fig:tracking_pipeline": {
        "paper_item": "Figure S1",
        "coverage_status": "official-code audited",
        "evidence": "res/tracking/smoke_config_audit/tracking_config_audit.json",
        "notes": "Actor/critic observations, target bodies, rewards, PPO, and assets audited; architecture PDF itself is not regenerated.",
    },
    "fig:extra-ablation": {
        "paper_item": "Figure S2",
        "coverage_status": "released-data reproduced",
        "evidence": "res/released_figures/ablation_pd_gain/ablation_pd_gain_global.pdf",
        "notes": "PD gain sensitivity reproduced from released data.",
    },
}


EXPECTED_TABLES = {
    "tab:rewardterms": {
        "paper_item": "Reward terms table",
        "coverage_status": "official-code audited",
        "evidence": "res/tracking/smoke_config_audit/tracking_config_audit.json",
        "notes": "Reward weights and target-body tracking terms audited against official code.",
    },
    "tab:domain_rand": {
        "paper_item": "Domain randomization table",
        "coverage_status": "official-code audited",
        "evidence": "res/tracking/smoke_config_audit/tracking_config_audit.json",
        "notes": "Friction, restitution, joint/default/COM, and velocity perturbation ranges audited.",
    },
    "tab:actuator_inertia": {
        "paper_item": "Actuator reflected inertia table",
        "coverage_status": "paper/source indexed; code effect audited",
        "evidence": "reproduction/docs/paper_parameter_map.md",
        "notes": "Numeric table indexed from paper source; official G1 armature/action-scale code audited separately.",
    },
    "tab:joint_armature": {
        "paper_item": "Joint-axis inertia table",
        "coverage_status": "paper/source indexed; code effect audited",
        "evidence": "reproduction/docs/paper_parameter_map.md",
        "notes": "Numeric table indexed from paper source; official G1 armature/action-scale code audited separately.",
    },
    "tab:ppo_hyperparameters": {
        "paper_item": "Motion tracking hyperparameters",
        "coverage_status": "paper/source and official-code verified",
        "evidence": "res/tracking/smoke_config_audit/tracking_config_audit.tsv",
        "notes": "PPO architecture/training values matched to official code.",
    },
    "tab:vae_hyperparameters": {
        "paper_item": "VAE hyperparameters",
        "coverage_status": "paper/source verified; debug probes only",
        "evidence": "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
        "notes": "Paper dimensions/LR/KL/gradient accumulation exercised in synthetic mechanics probes; no official checkpoint.",
    },
    "tab:diffusion_hyperparameters": {
        "paper_item": "Diffusion policy hyperparameters",
        "coverage_status": "paper/source verified; debug probes only",
        "evidence": "res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.json",
        "notes": "Paper Transformer dimensions and schedule probed; no full training/checkpoint.",
    },
    "tab:skill_success": {
        "paper_item": "Motion segments tested in sim and real",
        "coverage_status": "paper/source indexed; local LAFAN data audited; not reproduced",
        "evidence": "res/paper_skill_success_table_audit/skill_success_table_data_audit.json",
        "notes": (
            "Table claims sim/real execution coverage; local machine has no real robot and live Kit is blocked. "
            "Data audit checks listed LAFAN names and Real intervals against local G1 CSVs and records mismatches."
        ),
    },
}


METHOD_CLAIMS = [
    {
        "claim_id": "conditional_vae_elbo",
        "source": "reproduction/paper/source/tex/method.tex:151-170",
        "coverage_status": "debug/mechanics only",
        "evidence": "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
        "notes": "Conditional VAE dimensions/loss/accumulation are executable on synthetic batches; true DAgger rollout missing.",
    },
    {
        "claim_id": "state_latent_tau_no_reference_conditioning",
        "source": "reproduction/paper/source/tex/method.tex:140-176",
        "coverage_status": "debug fixture only",
        "evidence": "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json",
        "notes": "State-window fixture and provenance exist; trained VAE latents and teacher rollouts missing.",
    },
    {
        "claim_id": "individual_state_latent_denoising_steps",
        "source": "reproduction/paper/source/tex/method.tex:174-196",
        "coverage_status": "debug/mechanics only",
        "evidence": "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        "notes": "Independent state/latent timestep schedules exercised; paper-exact deployed masks remain unresolved.",
    },
    {
        "claim_id": "reverse_denoising_update",
        "source": "reproduction/paper/source/tex/method.tex:197-206",
        "coverage_status": "debug/mechanics only",
        "evidence": "res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json",
        "notes": "Oracle reverse loop checks mechanics; trained network and exact sampler coefficients missing.",
    },
    {
        "claim_id": "classifier_guidance_score",
        "source": "reproduction/paper/source/tex/method.tex:211-226",
        "coverage_status": "debug/mechanics only",
        "evidence": "res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        "notes": "Autograd costs and candidate guided reverse-loop are checked; rollout evaluation and scale protocol missing.",
    },
    {
        "claim_id": "hybrid_yaw_character_frame",
        "source": "reproduction/paper/source/root.tex:480-535",
        "coverage_status": "debug fixture only",
        "evidence": "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json",
        "notes": "Candidate representation passes invariance/pseudoinverse checks; full paper-faithful dataset still absent.",
    },
    {
        "claim_id": "ou_perturbation_and_rejection",
        "source": "reproduction/paper/source/root.tex:536-546",
        "coverage_status": "partial debug probe",
        "evidence": "res/level_c/augmentation_probe/level_c_augmentation_probe.json",
        "notes": "OU parameters and temporal correlation checked; VAE rollout perturbation and 5s rejection gate missing.",
    },
    {
        "claim_id": "joystick_waypoint_sdf_costs",
        "source": "reproduction/paper/source/root.tex:548-586",
        "coverage_status": "debug/mechanics only",
        "evidence": "res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        "notes": "Joystick, waypoint, SDF, composed cost gradients checked; task rollouts and Fig.5/6 data missing.",
    },
    {
        "claim_id": "sagittal_symmetry_tensorrt_async",
        "source": "reproduction/paper/source/root.tex:588-594",
        "coverage_status": "partial debug / blocked deployment",
        "evidence": "res/blocked_gates/blocked_gate_audit.json",
        "notes": "Symmetry candidate checked; TensorRT/asynchronous RTX4060 deployment cannot be reproduced from local artifacts.",
    },
]


def clean_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def line_number(full_text: str, start: int) -> int:
    return full_text.count("\n", 0, start) + 1


def extract_envs(tex_path: Path, env: str) -> list[dict]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", re.S)
    rows = []
    for match in pattern.finditer(text):
        block = match.group(0)
        label_m = re.search(r"\\label\{([^}]+)\}", block)
        caption_m = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|$)", block, re.S)
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", block)
        rows.append(
            {
                "env": env,
                "source_file": str(tex_path.relative_to(ROOT)),
                "line": line_number(text, match.start()),
                "label": label_m.group(1) if label_m else "",
                "caption_excerpt": clean_latex(caption_m.group(1))[:500] if caption_m else "",
                "graphics": "; ".join(graphics),
            }
        )
    return rows


def evidence_exists(evidence: str) -> bool:
    if evidence.startswith("reproduction/paper/source/"):
        return (ROOT / evidence).exists()
    return (ROOT / evidence).exists()


def status_bucket(status: str) -> str:
    lowered = status.lower()
    if "blocked" in lowered or "not reproduced" in lowered:
        return "blocked_or_unreproduced"
    if "debug" in lowered:
        return "debug_only"
    if status.startswith("partial") or "component evidence" in status:
        return "partial"
    if (
        status.startswith("released-data reproduced")
        or status == "official-code audited"
        or "official-code verified" in status
    ):
        return "strong"
    return "indexed"


def coverage_rows(parsed_rows: list[dict]) -> list[dict]:
    rows = []
    for item in parsed_rows:
        label = item["label"]
        mapping = EXPECTED_FIGURES.get(label) if item["env"] == "figure" else EXPECTED_TABLES.get(label)
        if mapping is None:
            mapping = {
                "paper_item": f"unmapped {item['env']}",
                "coverage_status": "unmapped",
                "evidence": "",
                "notes": "No manual mapping exists for this parsed LaTeX item.",
            }
        rows.append(
            {
                "kind": item["env"],
                "paper_item": mapping["paper_item"],
                "label": label,
                "source": f"{item['source_file']}:{item['line']}",
                "caption_excerpt": item["caption_excerpt"],
                "coverage_status": mapping["coverage_status"],
                "status_bucket": status_bucket(mapping["coverage_status"]),
                "evidence": str(ROOT / mapping["evidence"]) if mapping["evidence"] else "",
                "evidence_exists": evidence_exists(mapping["evidence"]) if mapping["evidence"] else False,
                "notes": mapping["notes"],
                "graphics": item["graphics"],
            }
        )
    for claim in METHOD_CLAIMS:
        rows.append(
            {
                "kind": "method_claim",
                "paper_item": claim["claim_id"],
                "label": claim["claim_id"],
                "source": claim["source"],
                "caption_excerpt": "",
                "coverage_status": claim["coverage_status"],
                "status_bucket": status_bucket(claim["coverage_status"]),
                "evidence": str(ROOT / claim["evidence"]),
                "evidence_exists": evidence_exists(claim["evidence"]),
                "notes": claim["notes"],
                "graphics": "",
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "kind",
        "paper_item",
        "label",
        "source",
        "coverage_status",
        "status_bucket",
        "evidence",
        "evidence_exists",
        "notes",
        "caption_excerpt",
        "graphics",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parsed = extract_envs(PAPER / "root.tex", "figure") + extract_envs(PAPER / "root.tex", "table")
    rows = coverage_rows(parsed)
    labels = {row["label"] for row in rows if row["kind"] in {"figure", "table"}}
    expected_labels = set(EXPECTED_FIGURES) | set(EXPECTED_TABLES)
    missing_expected = sorted(expected_labels - labels)
    unmapped = [row for row in rows if row["coverage_status"] == "unmapped"]
    missing_evidence = [row for row in rows if not row["evidence_exists"]]
    bucket_counts = {}
    for row in rows:
        bucket_counts[row["status_bucket"]] = bucket_counts.get(row["status_bucket"], 0) + 1

    json_path = OUT / "paper_source_coverage_audit.json"
    tsv_path = OUT / "paper_source_coverage_audit.tsv"
    summary = {
        "status": "ok" if not missing_expected and not unmapped and not missing_evidence else "failed",
        "experiment_type": "audit",
        "scope": "LaTeX source coverage for BeyondMimic figures, tables, and key method/formula claims",
        "counts": {
            "parsed_figures": sum(1 for row in rows if row["kind"] == "figure"),
            "parsed_tables": sum(1 for row in rows if row["kind"] == "table"),
            "method_claims": sum(1 for row in rows if row["kind"] == "method_claim"),
            "total_rows": len(rows),
            "missing_expected_labels": len(missing_expected),
            "unmapped_rows": len(unmapped),
            "missing_evidence_rows": len(missing_evidence),
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "missing_expected_labels": missing_expected,
        "unmapped_rows": unmapped,
        "missing_evidence_rows": missing_evidence,
        "rows": rows,
        "interpretation": {
            "paper_source_index_complete": not missing_expected and not unmapped,
            "goal_complete": False,
            "why_not_complete": (
                "Paper-source items are now indexed against current evidence, but several rows are blocked, "
                "debug-only, or paper/source-only rather than paper-level reproduced."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
