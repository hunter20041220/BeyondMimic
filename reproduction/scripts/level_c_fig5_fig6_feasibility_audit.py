#!/usr/bin/env python3
"""Audit Fig. 5 / Fig. 6 reproduction feasibility from local artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PAPER = ROOT / "reproduction/paper/source"
RELEASED = ROOT / "reproduction/data/Dataset_beyondmimic"
OUT = ROOT / "res/level_c/fig5_fig6_feasibility_audit"


FIGURES = [
    {
        "figure": "Figure 5",
        "label": "fig:versatile_loco",
        "source_evidence": "reproduction/paper/source/root.tex:223-234",
        "pdf": PAPER / "figures/Fig5.pdf",
        "panels": [
            {
                "panel": "A",
                "claim": "Diffusion process visualization under joystick right-turn command",
                "requires": [
                    "trained state-latent diffusion checkpoint",
                    "reverse denoising samples over a joystick command",
                    "paper rendering pipeline or logged denoising trajectory",
                ],
                "debug_evidence": [
                    "guidance_formula_probe",
                    "guided_reverse_loop_probe",
                    "guidance_scale_sweep_probe",
                ],
            },
            {
                "panel": "B",
                "claim": "Transition from walking to running through latent diffusion",
                "requires": [
                    "trained VAE checkpoint",
                    "trained diffusion checkpoint",
                    "state-latent rollout logs",
                    "latent embedding/visualization data",
                ],
                "debug_evidence": [
                    "vae_accumulation_probe",
                    "receding_horizon_decoder_probe",
                ],
            },
            {
                "panel": "C",
                "claim": "Joystick teleoperation and recovery from disturbance",
                "requires": [
                    "trained diffusion controller",
                    "closed-loop simulation or hardware rollout",
                    "joystick command log",
                    "state/action/velocity tracking log",
                    "disturbance/recovery log or video",
                ],
                "debug_evidence": [
                    "guidance_formula_probe",
                    "guided_reverse_loop_probe",
                    "guidance_scale_sweep_probe",
                ],
            },
            {
                "panel": "D",
                "claim": "t-SNE latent visualization from walking to running",
                "requires": [
                    "trained VAE encoder",
                    "state/action rollout dataset",
                    "latent samples across skills",
                    "t-SNE embedding procedure",
                ],
                "debug_evidence": [
                    "vae_accumulation_probe",
                ],
            },
        ],
    },
    {
        "figure": "Figure 6",
        "label": "fig:versatile_inpaint_sdf",
        "source_evidence": "reproduction/paper/source/root.tex:237-243",
        "pdf": PAPER / "figures/Fig6.pdf",
        "panels": [
            {
                "panel": "A",
                "claim": "Motion inpainting with future cartwheel keyframes and multi-round task switching",
                "requires": [
                    "trained diffusion checkpoint",
                    "trained VAE decoder",
                    "future keyframe conditioning schedule",
                    "closed-loop rollout for cartwheel transitions",
                    "video/state logs for long-horizon execution",
                ],
                "debug_evidence": [
                    "timestep_mask_probe",
                    "reverse_denoising_probe",
                    "receding_horizon_decoder_probe",
                ],
            },
            {
                "panel": "B",
                "claim": "Real-world obstacle avoidance using waypoint plus SDF costs",
                "requires": [
                    "trained diffusion checkpoint",
                    "guided reverse loop with waypoint and SDF costs",
                    "scene SDF or obstacle geometry",
                    "mocap/localization logs",
                    "closed-loop simulation or hardware rollout",
                ],
                "debug_evidence": [
                    "guidance_formula_probe",
                    "guided_reverse_loop_probe",
                    "guidance_scale_sweep_probe",
                ],
            },
        ],
    },
]


PROBE_PATHS = {
    "official_artifact_audit": ROOT / "res/level_c/official_artifact_audit/level_c_official_artifact_audit.json",
    "guidance_formula_probe": ROOT / "res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
    "guided_reverse_loop_probe": ROOT / "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
    "guidance_scale_sweep_probe": ROOT / "res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json",
    "timestep_mask_probe": ROOT / "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
    "reverse_denoising_probe": ROOT / "res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json",
    "vae_accumulation_probe": ROOT / "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
    "receding_horizon_decoder_probe": ROOT / "res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json",
}


def released_matches() -> list[str]:
    if not RELEASED.exists():
        return []
    keywords = [
        "fig5",
        "fig6",
        "versatile",
        "joystick",
        "inpaint",
        "sdf",
        "obstacle",
        "waypoint",
        "cartwheel",
        "latent",
        "diffusion",
        "vae",
    ]
    matches: list[str] = []
    for path in RELEASED.rglob("*"):
        if path.is_file():
            rel = path.relative_to(RELEASED).as_posix()
            if any(keyword in rel.lower() for keyword in keywords):
                matches.append(rel)
    return sorted(matches)


def load_artifact_audit() -> dict[str, Any]:
    path = PROBE_PATHS["official_artifact_audit"]
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "figure",
        "panel",
        "label",
        "claim",
        "status",
        "blocking_dependencies",
        "debug_evidence_present",
        "source_evidence",
        "paper_pdf_exists",
        "released_data_matches",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["blocking_dependencies"] = "; ".join(row["blocking_dependencies"])
            out["debug_evidence_present"] = "; ".join(row["debug_evidence_present"])
            out["released_data_matches"] = "; ".join(row["released_data_matches"])
            writer.writerow({key: out[key] for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact_audit()
    released = released_matches()
    official_code_found = bool(artifact.get("conclusion", {}).get("official_beyondmimic_vae_diffusion_code_found"))
    official_checkpoint_found = bool(artifact.get("conclusion", {}).get("official_beyondmimic_checkpoint_or_engine_found"))
    probe_exists = {name: path.exists() for name, path in PROBE_PATHS.items()}

    rows: list[dict[str, Any]] = []
    for fig in FIGURES:
        for panel in fig["panels"]:
            debug_present = [name for name in panel["debug_evidence"] if probe_exists.get(name, False)]
            blockers = list(panel["requires"])
            if not official_code_found:
                blockers.append("official BeyondMimic VAE/diffusion implementation absent locally")
            if not official_checkpoint_found:
                blockers.append("official trained Level C checkpoint / TensorRT engine absent locally")
            if not released:
                blockers.append("released dataset contains no Fig.5/Fig.6 joystick/inpainting/SDF/latent data files")
            rows.append(
                {
                    "figure": fig["figure"],
                    "panel": panel["panel"],
                    "label": fig["label"],
                    "claim": panel["claim"],
                    "status": "blocked_for_paper_reproduction_debug_mechanics_only",
                    "blocking_dependencies": blockers,
                    "debug_evidence_present": debug_present,
                    "source_evidence": fig["source_evidence"],
                    "paper_pdf_exists": fig["pdf"].exists(),
                    "released_data_matches": released,
                }
            )

    json_path = OUT / "level_c_fig5_fig6_feasibility_audit.json"
    tsv_path = OUT / "level_c_fig5_fig6_feasibility_audit.tsv"
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "audit",
        "scope": "Figure 5 / Figure 6 paper-reproduction feasibility from local released data and Level C artifacts",
        "figures": [
            {
                "figure": fig["figure"],
                "label": fig["label"],
                "pdf": str(fig["pdf"]),
                "pdf_exists": fig["pdf"].exists(),
                "source_evidence": fig["source_evidence"],
            }
            for fig in FIGURES
        ],
        "released_data_matches": released,
        "official_artifact_conclusion": artifact.get("conclusion", {}),
        "probe_exists": probe_exists,
        "rows": rows,
        "counts": {
            "panels_audited": len(rows),
            "paper_pdf_figures_present": sum(1 for fig in FIGURES if fig["pdf"].exists()),
            "released_data_match_count": len(released),
            "panels_blocked_for_paper_reproduction": sum(
                1 for row in rows if row["status"] == "blocked_for_paper_reproduction_debug_mechanics_only"
            ),
        },
        "conclusion": {
            "fig5_fig6_paper_reproduction_possible_from_current_local_artifacts": False,
            "debug_mechanism_evidence_available": True,
            "reason": (
                "Paper Figure 5/6 reproduction requires trained VAE/diffusion checkpoints, state-latent rollout logs, "
                "closed-loop joystick/inpainting/obstacle execution, and/or real-world/mocap evidence. Current local "
                "artifacts provide paper figures as PDFs and debug mechanism probes, but no released Fig.5/Fig.6 data "
                "or official Level C checkpoint/engine."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
