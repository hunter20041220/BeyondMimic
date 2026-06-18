#!/usr/bin/env python3
"""Audit Level C diffusion dataset collection protocol against current evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/dataset_collection_protocol_audit"
PAPER = ROOT / "reproduction/paper/source/root.tex"
GOAL = ROOT / "goal.md"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def line_for(path: Path, pattern: str) -> int | None:
    regex = re.compile(pattern)
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if regex.search(line):
            return idx
    return None


def source_checks() -> dict[str, bool]:
    paper = read_text(PAPER)
    goal = read_text(GOAL)
    return {
        "paper_has_diffusion_dataset_collection_section": "Diffusion Dataset Collection" in paper,
        "paper_has_ou_equation": r"\eta_{t+1}" in paper and r"\varepsilon_t" in paper,
        "paper_has_theta_0_8": r"\theta = 0.8" in paper,
        "paper_has_mu_0": r"\mu = 0" in paper,
        "paper_has_dt_1_0": r"\Delta t = 1.0" in paper,
        "paper_has_sigma_0_1": r"\sigma = 0.1" in paper,
        "paper_has_action_perturbation": r"a_t \leftarrow a_t + \eta_t" in paper,
        "paper_has_approximately_100_coverage": "approximately 100 times" in paper,
        "paper_has_2_5_second_rollout": "2.5 seconds" in paper,
        "paper_has_5_second_stability_check": "5 seconds to verify stability" in paper,
        "paper_has_episode_rejection": "episode is rejected" in paper,
        "paper_has_sagittal_symmetry": "sagittal-symmetric" in paper,
        "goal_has_ou_constants": all(token in goal for token in ["theta = 0.8", "mu = 0", "dt = 1.0", "sigma = 0.1"]),
        "goal_has_rollout_windows": "recorded window = 2.5 s" in goal and "stability verification = 5 s" in goal,
        "goal_has_phase6_manifest_fields": all(
            token in goal
            for token in [
                "source motion",
                "teacher/student policy",
                "start timestep",
                "end timestep",
                "state frame",
                "latent",
                "augmentation",
                "accept/reject",
                "split",
            ]
        ),
    }


def evidence_summary() -> dict[str, Any]:
    augmentation = load_json("res/level_c/augmentation_probe/level_c_augmentation_probe.json")
    small_split = load_json("res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json")
    fixture_split = load_json("res/level_c/fixture_split_manifest/fixture_split_manifest_summary.json")
    rollout_manifest = load_json(
        "res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json"
    )
    paper_windows = load_json("res/level_c/paper_state_windows/level_c_paper_state_windows.json")
    official_artifacts = load_json("res/level_c/official_artifact_audit/level_c_official_artifact_audit.json")

    return {
        "augmentation_probe": {
            "path": str(ROOT / "res/level_c/augmentation_probe/level_c_augmentation_probe.json"),
            "status": augmentation["status"],
            "checks": augmentation["checks"],
            "ou": augmentation.get("ou", {}),
            "symmetry": augmentation.get("symmetry", {}),
        },
        "small_dataset_split_manifest": {
            "path": str(
                ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"
            ),
            "status": small_split["status"],
            "counts": small_split["counts"],
            "checks": small_split["checks"],
        },
        "fixture_split_manifest": {
            "path": str(ROOT / "res/level_c/fixture_split_manifest/fixture_split_manifest_summary.json"),
            "status": fixture_split["status"],
            "counts": fixture_split["counts"],
            "checks": fixture_split["checks"],
        },
        "rollout_rejection_manifest_probe": {
            "path": str(
                ROOT
                / "res/level_c/rollout_rejection_manifest_probe/"
                / "level_c_rollout_rejection_manifest_probe.json"
            ),
            "status": rollout_manifest["status"],
            "settings": rollout_manifest["settings"],
            "metrics": rollout_manifest["metrics"],
            "checks": rollout_manifest["checks"],
        },
        "paper_state_windows": {
            "path": str(ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"),
            "status": paper_windows["status"],
            "counts": paper_windows["counts"],
            "checks": paper_windows["checks"],
            "settings": paper_windows["settings"],
        },
        "official_artifacts": {
            "path": str(ROOT / "res/level_c/official_artifact_audit/level_c_official_artifact_audit.json"),
            "status": official_artifacts["status"],
            "conclusion": official_artifacts["conclusion"],
        },
    }


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["category", "key", "value"])
        for key, value in sorted(summary["paper_protocol"].items()):
            writer.writerow(["paper_protocol", key, value])
        for key, value in sorted(summary["current_evidence_checks"].items()):
            writer.writerow(["current_evidence_checks", key, value])
        for key, value in sorted(summary["missing_paper_requirements"].items()):
            writer.writerow(["missing_paper_requirements", key, value])
        for key, value in sorted(summary["checks"].items()):
            writer.writerow(["checks", key, value])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = source_checks()
    evidence = evidence_summary()

    augmentation_checks = evidence["augmentation_probe"]["checks"]
    small_checks = evidence["small_dataset_split_manifest"]["checks"]
    fixture_checks = evidence["fixture_split_manifest"]["checks"]
    rollout_checks = evidence["rollout_rejection_manifest_probe"]["checks"]
    paper_windows_checks = evidence["paper_state_windows"]["checks"]
    official_conclusion = evidence["official_artifacts"]["conclusion"]

    current_evidence_checks = {
        "ou_probe_status_ok": evidence["augmentation_probe"]["status"] == "ok",
        "ou_probe_temporal_correlation": augmentation_checks["ou_more_temporally_correlated_than_iid"],
        "sagittal_double_mirror_candidate": augmentation_checks["symmetry_double_mirror_exact"]
        and augmentation_checks["velocity_symmetry_double_mirror_exact"],
        "small_dataset_manifest_status_ok": evidence["small_dataset_split_manifest"]["status"] == "ok",
        "small_dataset_manifest_has_required_fields": small_checks["all_samples_have_required_fields"],
        "small_dataset_manifest_no_motion_cross_split": small_checks["no_motion_crosses_splits"],
        "small_dataset_manifest_latent_missing_marked": small_checks["latent_marked_missing"],
        "small_dataset_manifest_accept_reject_debug_marked": small_checks["accept_reject_marked_debug"],
        "fixture_manifest_guard_gap_present": fixture_checks["guard_gap_samples_present"],
        "fixture_manifest_no_near_gap_leakage": fixture_checks["no_cross_split_near_gap"],
        "rollout_manifest_status_ok": evidence["rollout_rejection_manifest_probe"]["status"] == "ok",
        "rollout_manifest_windows_match_paper": rollout_checks["all_windows_match_2_5s_and_5s"],
        "rollout_manifest_coverage_100x_debug": rollout_checks["coverage_repeats_per_valid_start_100"]
        and rollout_checks["recorded_coverage_reaches_100_for_nonzero_frames"],
        "rollout_manifest_accept_reject_debug_marked": rollout_checks["accept_reject_field_present_debug_only"]
        and rollout_checks["failure_signal_marked_missing"],
        "rollout_manifest_no_true_vae_claim": rollout_checks["does_not_claim_true_vae_rollout"],
        "paper_state_windows_available": evidence["paper_state_windows"]["status"] == "ok"
        and paper_windows_checks["all_checks_pass"],
        "official_level_c_rollout_artifacts_absent": not official_conclusion[
            "official_beyondmimic_vae_diffusion_code_found"
        ]
        and not official_conclusion["official_beyondmimic_checkpoint_or_engine_found"],
    }

    missing_paper_requirements = {
        "missing_true_vae_rollout_with_action_noise": True,
        "missing_original_state_and_latent_recording_from_trained_vae": True,
        "missing_live_approximately_100x_sample_coverage_measurement": True,
        "missing_live_2_5s_policy_rollout_logs": True,
        "missing_live_5s_stability_verification_logs": True,
        "missing_real_episode_rejection_decisions": True,
        "missing_paper_exact_sagittal_augmented_dataset": True,
        "missing_trainable_state_latent_dataset": True,
    }

    paper_protocol = {
        "ou_theta": 0.8,
        "ou_mu": 0.0,
        "ou_dt": 1.0,
        "ou_sigma": 0.1,
        "recorded_rollout_seconds": 2.5,
        "stability_verification_seconds": 5.0,
        "approx_sample_coverage": 100,
        "history": evidence["paper_state_windows"]["settings"]["history"],
        "horizon": evidence["paper_state_windows"]["settings"]["horizon"],
        "sequence_length": evidence["paper_state_windows"]["settings"]["sequence_length"],
        "paper_state_dim": evidence["paper_state_windows"]["settings"]["paper_state_dim"],
        "latent_dim": 32,
    }

    checks = {
        "paper_protocol_source_indexed": all(source.values()),
        "ou_and_symmetry_debug_evidence_ok": current_evidence_checks["ou_probe_status_ok"]
        and current_evidence_checks["ou_probe_temporal_correlation"]
        and current_evidence_checks["sagittal_double_mirror_candidate"],
        "manifest_debug_evidence_ok": current_evidence_checks["small_dataset_manifest_status_ok"]
        and current_evidence_checks["small_dataset_manifest_has_required_fields"]
        and current_evidence_checks["small_dataset_manifest_no_motion_cross_split"],
        "rollout_manifest_debug_evidence_ok": current_evidence_checks["rollout_manifest_status_ok"]
        and current_evidence_checks["rollout_manifest_windows_match_paper"]
        and current_evidence_checks["rollout_manifest_coverage_100x_debug"]
        and current_evidence_checks["rollout_manifest_accept_reject_debug_marked"]
        and current_evidence_checks["rollout_manifest_no_true_vae_claim"],
        "debug_boundaries_mark_missing_latent_and_accept_reject": current_evidence_checks[
            "small_dataset_manifest_latent_missing_marked"
        ]
        and current_evidence_checks["small_dataset_manifest_accept_reject_debug_marked"]
        and current_evidence_checks["rollout_manifest_accept_reject_debug_marked"],
        "paper_state_windows_available": current_evidence_checks["paper_state_windows_available"],
        "official_level_c_artifacts_absent_recorded": current_evidence_checks["official_level_c_rollout_artifacts_absent"],
        "all_missing_paper_requirements_explicit": all(missing_paper_requirements.values()),
        "does_not_claim_true_dataset_collection": True,
    }

    json_path = OUT / "level_c_dataset_collection_protocol_audit.json"
    tsv_path = OUT / "level_c_dataset_collection_protocol_audit.tsv"
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "protocol_audit",
        "scope": "paper diffusion dataset collection protocol versus current debug-only evidence",
        "paper_source": {
            "root_tex": str(PAPER),
            "goal": str(GOAL),
            "line_refs": {
                "diffusion_dataset_collection": line_for(PAPER, r"Diffusion Dataset Collection"),
                "ou_equation": line_for(PAPER, r"eta_\{t\+1\}|\\eta_\{t\+1\}"),
                "rollout_rejection": line_for(PAPER, r"2\.5 seconds"),
                "sagittal_symmetry": line_for(PAPER, r"sagittal-symmetric"),
                "goal_ou_constants": line_for(GOAL, r"theta = 0\.8"),
                "goal_rollout_window": line_for(GOAL, r"recorded window = 2\.5 s"),
                "goal_phase6_manifest_fields": line_for(GOAL, r"teacher/student policy"),
            },
            "checks": source,
        },
        "paper_protocol": paper_protocol,
        "evidence": evidence,
        "current_evidence_checks": current_evidence_checks,
        "missing_paper_requirements": missing_paper_requirements,
        "metrics": {
            "debug_motion_count": evidence["small_dataset_split_manifest"]["counts"]["motions_total"],
            "debug_sample_count": evidence["small_dataset_split_manifest"]["counts"]["samples_total"],
            "rollout_manifest_episode_rows": evidence["rollout_rejection_manifest_probe"]["metrics"][
                "episode_manifest_rows"
            ],
            "rollout_manifest_valid_start_count": evidence["rollout_rejection_manifest_probe"]["metrics"][
                "valid_start_count_total"
            ],
            "rollout_manifest_recorded_coverage_min_nonzero": evidence["rollout_rejection_manifest_probe"][
                "metrics"
            ]["recorded_coverage_min_nonzero"],
            "paper_state_window_count": evidence["paper_state_windows"]["counts"]["window_count"],
            "paper_state_token_count": evidence["paper_state_windows"]["counts"]["sample_count"],
            "missing_paper_requirement_count": len(missing_paper_requirements),
        },
        "checks": checks,
        "interpretation": {
            "status": "debug_protocol_indexed_but_true_collection_missing",
            "summary": (
                "The paper/goal dataset-collection protocol is indexed and partially exercised by debug OU, symmetry, "
                "state-window, rollout-window, and provenance manifests. Current artifacts still do not contain trained "
                "VAE rollouts, true latents, live 2.5s/5s rollout logs, real episode rejection, or a trainable "
                "state-latent dataset."
            ),
            "not_a_replacement_for": [
                "true VAE policy rollout",
                "teacher/student DAgger collection",
                "state-latent diffusion training dataset",
                "accept/reject decisions from live stability verification",
                "paper Fig. 5/Fig. 6 evaluation",
            ],
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    (OUT / "run.log").write_text(
        "kind=level_c_dataset_collection_protocol_audit\n"
        f"status={summary['status']}\n"
        f"missing_paper_requirement_count={len(missing_paper_requirements)}\n",
        encoding="utf-8",
    )
    print(json_path)


if __name__ == "__main__":
    main()
