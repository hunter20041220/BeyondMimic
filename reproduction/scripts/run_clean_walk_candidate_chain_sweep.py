#!/usr/bin/env python3
"""Compare local teacher/VAE/diffusion chains on the same clean walk.

The clean-walk presentation videos can be made readable with reference anchoring,
but the user-requested target is normal-looking learned-control videos.  This
script keeps the motion/window fixed and runs several existing local chains with
``model_target_weight=1.0`` first, then records whether any chain can produce a
stable pure learned-target walk.

Claim boundary: all candidates are local reproduction artifacts.  None is an
official BeyondMimic checkpoint or paper-level rollout.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
SCRIPT = ROOT / "reproduction/scripts/render_clean_walk_mujoco_control_suite.py"
OUT_ROOT = ROOT / "res/visualization/clean_walk_candidate_chain_sweep"
LOG_ROOT = ROOT / "logs/mujoco/clean_walk_candidate_chain_sweep"
PYTHON = Path(os.environ.get("BM_MUJOCO_PYTHON", str(ROOT / "mujoco_mp4/.venv/bin/python"))).expanduser()


@dataclass(frozen=True)
class Candidate:
    name: str
    teacher_json: Path
    vae_ckpt: Path
    denoiser_ckpt: Path
    note: str


CANDIDATES = [
    Candidate(
        name="stage1_multisource",
        teacher_json=ROOT
        / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json",
        vae_ckpt=ROOT
        / "res/runs/level_c_stage1_multisource_teacher_rollout_vae_training/"
        "resource_adjusted_teacher_rollout_vae_20260623_135755_seed20260855/resource_adjusted_teacher_rollout_action_vae.pt",
        denoiser_ckpt=ROOT
        / "res/runs/level_c_stage1_multisource_state_latent_diffusion_training/"
        "resource_adjusted_state_latent_diffusion_20260623_140110_seed20260857/resource_adjusted_state_latent_denoiser.pt",
        note="Current GPUs 5/6 multi-source teacher chain.",
    ),
    Candidate(
        name="paper_contract",
        teacher_json=ROOT
        / "res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/paper_contract_best_teacher.json",
        vae_ckpt=ROOT
        / "res/runs/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training/"
        "resource_adjusted_teacher_rollout_vae_20260623_062423_seed20260805/resource_adjusted_teacher_rollout_action_vae.pt",
        denoiser_ckpt=ROOT
        / "res/runs/level_c_official_importer_export_paper_contract_state_latent_diffusion_training/"
        "resource_adjusted_state_latent_diffusion_20260623_062635_seed20260807/resource_adjusted_state_latent_denoiser.pt",
        note="Official-importer-export paper-contract teacher chain.",
    ),
    Candidate(
        name="official_importer_export_full_bundle",
        teacher_json=ROOT
        / "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json",
        vae_ckpt=ROOT
        / "res/runs/level_c_official_importer_export_full_bundle_teacher_rollout_vae_training/"
        "resource_adjusted_teacher_rollout_vae_20260620_163252_seed20260683/resource_adjusted_teacher_rollout_action_vae.pt",
        denoiser_ckpt=ROOT
        / "res/runs/level_c_official_importer_export_full_bundle_state_latent_diffusion_training/"
        "resource_adjusted_state_latent_diffusion_20260620_174302_seed20260687/resource_adjusted_state_latent_denoiser.pt",
        note="Official-importer-export full public bundle teacher chain.",
    ),
    Candidate(
        name="official_importer_export_scaled_ppo",
        teacher_json=ROOT
        / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/"
        "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json",
        vae_ckpt=ROOT
        / "res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
        "resource_adjusted_teacher_rollout_vae_20260621_141139_seed20260701/resource_adjusted_teacher_rollout_action_vae.pt",
        denoiser_ckpt=ROOT
        / "res/runs/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training/"
        "resource_adjusted_state_latent_diffusion_20260621_142629_seed20260703/resource_adjusted_state_latent_denoiser.pt",
        note="Official-importer-export scaled PPO chain with largest local rollout dataset.",
    ),
]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "path": str(path), "error": repr(exc)}


def variant_metrics(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, metrics in summary.get("variant_metrics", {}).items():
        if not isinstance(metrics, dict):
            continue
        out[name] = {
            key: metrics.get(key)
            for key in [
                "fall_proxy_count",
                "root_height_min",
                "root_height_mean",
                "root_position_error_mean_m",
                "root_position_error_max_m",
                "final_target_gap_to_reference_abs_mean",
                "model_target_gap_to_reference_abs_mean",
            ]
            if key in metrics
        }
    return out


def score_candidate(metrics: dict[str, dict[str, Any]]) -> float:
    names = [
        "teacher_policy_action_control",
        "vae_reconstructed_action_control",
        "diffusion_denoised_latent_action_control",
        "guided_latent_action_control",
    ]
    score = 0.0
    for name in names:
        m = metrics.get(name, {})
        fall = float(m.get("fall_proxy_count") or 0.0)
        hmin = float(m.get("root_height_min") or 0.0)
        root_err = float(m.get("root_position_error_mean_m") or 1.0)
        score += fall * 10.0
        score += max(0.0, 0.60 - hmin) * 20.0
        score += root_err
    return score


def run_candidate(candidate: Candidate, weight: float) -> dict[str, Any]:
    label = f"{candidate.name}_w{int(round(weight * 100)):03d}"
    out_dir = OUT_ROOT / label
    log_path = LOG_ROOT / f"{label}.log"
    env = os.environ.copy()
    env.update(
        {
            "BM_ROOT": str(ROOT),
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "BM_CLEAN_SUITE_OUT_ROOT": str(out_dir),
            "BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT": f"{weight:.6f}",
            "BM_CLEAN_SUITE_BEST_TEACHER_JSON": str(candidate.teacher_json),
            "BM_CLEAN_SUITE_VAE_CKPT": str(candidate.vae_ckpt),
            "BM_CLEAN_SUITE_DENOISER_CKPT": str(candidate.denoiser_ckpt),
            "BM_CLEAN_WALK_SECONDS": env.get("BM_CLEAN_WALK_SECONDS", "15.0"),
            "BM_CLEAN_WALK_START_INDEX": env.get("BM_CLEAN_WALK_START_INDEX", "0"),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([str(PYTHON), str(SCRIPT)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    summary_path = out_dir / "clean_walk_mujoco_control_suite_summary.json"
    summary = load_json(summary_path)
    metrics = variant_metrics(summary)
    return {
        "candidate": candidate.name,
        "weight": weight,
        "returncode": proc.returncode,
        "status": summary.get("status", "missing_summary"),
        "note": candidate.note,
        "teacher_json": str(candidate.teacher_json),
        "vae_ckpt": str(candidate.vae_ckpt),
        "denoiser_ckpt": str(candidate.denoiser_ckpt),
        "output_root": str(out_dir),
        "summary_json": str(summary_path),
        "log_path": str(log_path),
        "variant_metrics": metrics,
        "score_lower_is_better": score_candidate(metrics),
    }


def write_tables(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = OUT_ROOT / "clean_walk_candidate_chain_sweep_summary.csv"
    md_path = OUT_ROOT / "clean_walk_candidate_chain_sweep_summary.md"
    fields = [
        "candidate",
        "weight",
        "variant",
        "status",
        "score_lower_is_better",
        "fall_proxy_count",
        "root_height_min",
        "root_height_mean",
        "root_position_error_mean_m",
        "final_target_gap_to_reference_abs_mean",
        "summary_json",
        "log_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for variant, metrics in result["variant_metrics"].items():
                writer.writerow(
                    {
                        "candidate": result["candidate"],
                        "weight": result["weight"],
                        "variant": variant,
                        "status": result["status"],
                        "score_lower_is_better": result["score_lower_is_better"],
                        "fall_proxy_count": metrics.get("fall_proxy_count"),
                        "root_height_min": metrics.get("root_height_min"),
                        "root_height_mean": metrics.get("root_height_mean"),
                        "root_position_error_mean_m": metrics.get("root_position_error_mean_m"),
                        "final_target_gap_to_reference_abs_mean": metrics.get(
                            "final_target_gap_to_reference_abs_mean"
                        ),
                        "summary_json": result["summary_json"],
                        "log_path": result["log_path"],
                    }
                )
    lines = [
        "# Clean Walk Candidate Chain Sweep",
        "",
        "## 结论",
        "",
        summary["diagnosis"],
        "",
        "| candidate | status | score | teacher fall | teacher hmin | output |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in sorted(results, key=lambda item: item["score_lower_is_better"]):
        teacher = result["variant_metrics"].get("teacher_policy_action_control", {})
        lines.append(
            f"| `{result['candidate']}` | `{result['status']}` | "
            f"`{result['score_lower_is_better']:.4f}` | `{teacher.get('fall_proxy_count')}` | "
            f"`{teacher.get('root_height_min')}` | `{result['output_root']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "这是本地 MuJoCo 候选链路诊断，不是官方 BeyondMimic checkpoint，不是真实机器人，不是 paper-level Fig.5/Fig.6。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    weight = float(os.environ.get("BM_CLEAN_CANDIDATE_WEIGHT", "1.0"))
    requested = set(
        item.strip() for item in os.environ.get("BM_CLEAN_CANDIDATES", "").split(",") if item.strip()
    )
    candidates = [c for c in CANDIDATES if not requested or c.name in requested]
    results = []
    for candidate in candidates:
        print(f"[candidate] {candidate.name} weight={weight:.2f}", flush=True)
        result = run_candidate(candidate, weight)
        print(
            json.dumps(
                {
                    "candidate": result["candidate"],
                    "status": result["status"],
                    "returncode": result["returncode"],
                    "score": result["score_lower_is_better"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        results.append(result)
    best = min(results, key=lambda item: item["score_lower_is_better"]) if results else None
    pure_successes = [
        result
        for result in results
        if result["status"] == "ok"
        and all(
            (m.get("fall_proxy_count") or 0) == 0 and (m.get("root_height_min") or 0.0) >= 0.55
            for name, m in result["variant_metrics"].items()
            if name in {
                "teacher_policy_action_control",
                "vae_reconstructed_action_control",
                "diffusion_denoised_latent_action_control",
                "guided_latent_action_control",
            }
        )
    ]
    diagnosis = (
        f"Best local candidate at pure model target weight {weight:.2f}: {best['candidate']} "
        f"(score={best['score_lower_is_better']:.4f}). "
        if best
        else "No candidates were run. "
    )
    if pure_successes:
        diagnosis += "At least one candidate met the local pure-target fall/height gate, but this remains MuJoCo/root-assist evidence."
    else:
        diagnosis += "No candidate met the local pure-target fall/height gate for all learned variants."
    summary = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "clean_walk_candidate_chain_sweep",
        "claim_level": "Local MuJoCo candidate teacher/VAE/diffusion chain sweep; not paper-level BeyondMimic.",
        "weight": weight,
        "diagnosis": diagnosis,
        "best_candidate": best,
        "pure_success_candidates": [item["candidate"] for item in pure_successes],
        "results": results,
        "outputs": {
            "summary_json": str(OUT_ROOT / "clean_walk_candidate_chain_sweep_summary.json"),
            "summary_csv": str(OUT_ROOT / "clean_walk_candidate_chain_sweep_summary.csv"),
            "summary_md": str(OUT_ROOT / "clean_walk_candidate_chain_sweep_summary.md"),
            "log_root": str(LOG_ROOT),
        },
    }
    write_json(OUT_ROOT / "clean_walk_candidate_chain_sweep_summary.json", summary)
    write_tables(results, summary)
    print(json.dumps({"status": summary["status"], "summary": summary["outputs"]["summary_json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
