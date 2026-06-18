#!/usr/bin/env python3
"""Audit goal.md Phase 9 ablation coverage."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/ablation_coverage"


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def row(group: str, item: str, levels: str, status: str, evidence: list[str], detail: str) -> dict[str, Any]:
    evidence_exists = [exists(path) for path in evidence]
    return {
        "group": group,
        "item": item,
        "levels": levels,
        "status": status,
        "evidence": evidence,
        "evidence_exists": evidence_exists,
        "all_evidence_exists": all(evidence_exists),
        "detail": detail,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        row("motion_tracking", "orientation representation", "Rot6D / quaternion / axis-angle", "released_data_reproduced", ["res/released_figures/ablation_orientation_representation/ablation_orientation_representation_local.pdf", "res/released_figures/ablation_orientation_representation/ablation_orientation_representation_global.pdf", "res/comparison/paper_vs_reproduction.json"], "Released local/global orientation-representation ablation panels reproduced; no new live PPO rerun."),
        row("motion_tracking", "observation history", "history 1 / 4 / 8 / 25", "released_data_reproduced", ["res/released_figures/ablation_observation_history/ablation_observation_history_local.pdf", "res/released_figures/ablation_observation_history/ablation_observation_history_global.pdf"], "Released observation-history ablation panels reproduced."),
        row("motion_tracking", "armature", "x0 / x0.1 / original / x10", "released_data_reproduced", ["res/released_figures/ablation_armature/ablation_armature_local.pdf", "res/released_figures/ablation_armature/ablation_armature_global.pdf", "res/paper_table_values/paper_table_value_audit.json"], "Released armature ablation panels plus paper/source armature table audit exist."),
        row("motion_tracking", "delay", "0 / 2 / 5 / 10 ms", "released_data_reproduced", ["res/released_figures/ablation_latency/ablation_latency_local.pdf", "res/released_figures/ablation_latency/ablation_latency_global.pdf"], "Released latency ablation panels reproduced."),
        row("motion_tracking", "adaptive sampling", "on / off", "released_and_code_audited", ["res/released_figures/adaptive_sampling/adaptive_sampling_w_iter_matrix.pdf", "res/released_figures/adaptive_sampling/adaptive_sampling_wo_iter_matrix.pdf", "res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json"], "Released adaptive-sampling panels reproduced; paper/code kernel-size discrepancy remains recorded."),
        row("motion_tracking", "PD natural frequency", "paper extra ablation", "released_data_reproduced", ["res/released_figures/ablation_pd_gain/ablation_pd_gain_local.pdf", "res/released_figures/ablation_pd_gain/ablation_pd_gain_global.pdf"], "Released PD-gain sensitivity panels reproduced."),
        row("diffusion", "direct state-action diffusion", "direct state-action diffusion", "debug_ablation_only", ["res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json", "res/results_claims_audit/results_claims_audit.json", "res/comparison/paper_vs_reproduction.json"], "Offline debug direct-state action branch is compared with the existing state-latent action pipe; paper direct-vs-latent cartwheel success still requires trained checkpoints and rollout logs."),
        row("diffusion", "latent diffusion", "latent diffusion", "debug_mechanics_only", ["res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json", "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json", "res/results_claims_audit/results_claims_audit.json"], "Latent trajectory mechanics and debug held-out probes exist; no trained latent diffusion checkpoint or paper success metric."),
        row("diffusion", "without OU perturbation", "without OU perturbation", "debug_or_protocol_only", ["res/level_c/augmentation_probe/level_c_augmentation_probe.json", "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json"], "OU perturbation formula/protocol is checked; no trainable ablation without OU."),
        row("diffusion", "without symmetry augmentation", "without symmetry augmentation", "public_data_trained_comparison", ["res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json", "res/level_c/augmentation_probe/level_c_augmentation_probe.json", "res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_dataset_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json", "res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json"], "Candidate sagittal symmetry mechanics checked, applied to the full public LAFAN1 paper-architecture windows, trained with the paper-sized VAE/diffusion architecture, and compared against the original public-data training run; closed-loop paper success/failure ablation remains unavailable."),
        row("diffusion", "without emphasis projection", "without emphasis projection", "debug_or_formula_only", ["res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json", "res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json"], "Emphasis projection and pseudoinverse formula audited; no trainable ablation without it."),
        row("diffusion", "history sensitivity", "history sensitivity", "debug_or_config_only", ["res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json", "res/config/resolved_reproduction_config.json"], "History length is configured and mask mechanics checked; no trained history sweep."),
        row("diffusion", "horizon sensitivity", "horizon sensitivity", "debug_or_config_only", ["res/config/resolved_reproduction_config.json", "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json"], "Horizon/config and architecture probes exist; no trained horizon sweep."),
        row("diffusion", "denoising-step sensitivity", "denoising-step sensitivity", "debug_or_config_only", ["res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json", "res/level_c/training_schedule_probe/level_c_training_schedule_probe.json"], "Denoising-step schedule and K=20 are audited; no trained K sweep."),
        row("diffusion", "guidance-scale sensitivity", "guidance-scale sensitivity", "public_data_checkpoint_reverse_and_offline_sweep", ["res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json", "res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json", "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json", "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json", "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"], "Full-split symmetry-augmented public-data checkpoint sweeps five reverse-denoising guidance scales and seven offline guidance scales over validation+test windows; task metrics and result table are also linked, but no closed-loop paper scene success/failure protocol."),
    ]
    missing = [r for r in rows if not r["all_evidence_exists"]]
    status_counts = Counter(r["status"] for r in rows)
    group_counts = Counter(r["group"] for r in rows)
    summary = {
        "status": "ok" if not missing and group_counts["motion_tracking"] == 6 and group_counts["diffusion"] == 9 else "failed",
        "experiment_type": "ablation_coverage_audit",
        "scope": "goal.md Phase 9 ablation coverage with explicit evidence/gap status",
        "row_count": len(rows),
        "group_counts": dict(sorted(group_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_rows": missing,
        "rows": rows,
        "checks": {
            "all_15_goal_ablation_items_mapped": len(rows) == 15,
            "motion_tracking_six_items_mapped": group_counts["motion_tracking"] == 6,
            "diffusion_nine_items_mapped": group_counts["diffusion"] == 9,
            "all_evidence_paths_exist": not missing,
            "released_tracking_ablation_panels_present": all(
                r["status"].startswith("released") for r in rows if r["group"] == "motion_tracking"
            ),
            "diffusion_training_ablations_not_overclaimed": all(
                r["status"] != "released_data_reproduced" for r in rows if r["group"] == "diffusion"
            ),
            "guidance_scale_sensitivity_links_full_split_metrics": any(
                r["group"] == "diffusion"
                and r["item"] == "guidance-scale sensitivity"
                and "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json"
                in r["evidence"]
                and "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
                in r["evidence"]
                and "no closed-loop" in r["detail"]
                for r in rows
            ),
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The Phase 9 ablation list is explicitly mapped, but most diffusion ablations remain debug-only or "
                "blocked because trained direct/latent checkpoints, rollout metrics, and Fig.5/Fig.6 task logs are absent."
            ),
        },
        "outputs": {
            "json": str(OUT / "ablation_coverage_audit.json"),
            "tsv": str(OUT / "ablation_coverage_audit.tsv"),
        },
    }
    (OUT / "ablation_coverage_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "ablation_coverage_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["group", "item", "levels", "status", "evidence", "all_evidence_exists", "detail"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "group": r["group"],
                    "item": r["item"],
                    "levels": r["levels"],
                    "status": r["status"],
                    "evidence": ";".join(r["evidence"]),
                    "all_evidence_exists": r["all_evidence_exists"],
                    "detail": r["detail"],
                }
            )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
