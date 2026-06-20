#!/usr/bin/env python3
"""Run one importer-export keyframe/inpainting proxy guidance rollout.

This launches the existing official-importer-export task-conditioned runner in a
fresh process with ``BM_TASK_CONDITIONED_TASKS=inpainting`` so import-time task
selection is clean. The result is a local virtual future-keyframe/root-path
proxy for Fig. 6A planning, not the paper cartwheel keyframe protocol.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
BASE_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.py"
)
SUMMARY_ROOT = ROOT / "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
VIS_ROOT = ROOT / "res/visualization/official_importer_export_full_bundle_inpainting_guidance_rollout"
LOG_ROOT = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
FAILED_ROOT = ROOT / "res/failed_runs/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
SUMMARY_JSON = SUMMARY_ROOT / "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
SUMMARY_TSV = SUMMARY_ROOT / "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.tsv"
UNDERLYING_JSON = SUMMARY_ROOT / "underlying_task_conditioned_inpainting.json"
UNDERLYING_TSV = SUMMARY_ROOT / "underlying_task_conditioned_inpainting.tsv"
OFFLINE_GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/"
    "level_c_official_importer_export_full_bundle_state_latent_guidance_eval.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "status",
        "rollout_steps",
        "selected_physical_gpu",
        "guided_keyframe_error_mean",
        "denoised_keyframe_error_mean",
        "guided_keyframe_error_delta_vs_denoised",
        "guidance_cost_delta_mean",
        "capture_npz",
        "mp4",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def target_path(step_count: int, keyframe_indices: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, step_count, dtype=np.float64)[keyframe_indices]
    return np.stack([0.30 * t, 0.10 * np.sin(2.0 * np.pi * t)], axis=-1)


def keyframe_error(npz: np.lib.npyio.NpzFile, variant: str) -> dict[str, Any]:
    key = f"{variant}_robot_body_pos_w"
    if key not in npz:
        return {"available": False, "reason": f"missing {key}"}
    body_pos = np.asarray(npz[key], dtype=np.float64)
    if body_pos.ndim != 3 or body_pos.shape[1] < 1 or body_pos.shape[2] < 2:
        return {"available": False, "reason": f"unexpected shape {body_pos.shape}"}
    root_xy = body_pos[:, 0, :2]
    rel_xy = root_xy - root_xy[0:1]
    key_count = min(5, rel_xy.shape[0])
    keyframe_indices = np.linspace(0, rel_xy.shape[0] - 1, key_count).round().astype(int)
    target_xy = target_path(rel_xy.shape[0], keyframe_indices)
    errors = np.linalg.norm(rel_xy[keyframe_indices] - target_xy, axis=-1)
    return {
        "available": True,
        "variant": variant,
        "step_count": int(rel_xy.shape[0]),
        "keyframe_indices": keyframe_indices.tolist(),
        "target_xy": target_xy.tolist(),
        "keyframe_error_mean": float(errors.mean()),
        "keyframe_error_max": float(errors.max()),
        "keyframe_error_by_index": errors.tolist(),
    }


def collect_proxy_metrics(row: dict[str, Any], task_summary: dict[str, Any]) -> dict[str, Any]:
    assets = task_summary.get("outputs", {}).get("assets", {})
    capture_npz = Path(task_summary.get("outputs", {}).get("capture_npz", ""))
    metrics: dict[str, Any] = {
        "capture_npz": str(capture_npz) if str(capture_npz) else "",
        "capture_npz_exists": capture_npz.is_file() if str(capture_npz) else False,
        "mp4": assets.get("mp4", row.get("mp4", "")),
        "mp4_exists": Path(assets.get("mp4", row.get("mp4", ""))).is_file()
        if assets.get("mp4", row.get("mp4", ""))
        else False,
        "task_guidance": task_summary.get("metrics", {}).get("task_guidance", {}),
    }
    if metrics["capture_npz_exists"]:
        with np.load(capture_npz) as npz:
            variant_metrics = {
                variant: keyframe_error(npz, variant)
                for variant in ["teacher", "vae_base", "denoised_latent", "receding_latent_guided"]
            }
        metrics["keyframe_proxy"] = variant_metrics
        guided = variant_metrics["receding_latent_guided"]
        denoised = variant_metrics["denoised_latent"]
        if guided.get("available") and denoised.get("available"):
            metrics["guided_keyframe_error_delta_vs_denoised"] = (
                guided["keyframe_error_mean"] - denoised["keyframe_error_mean"]
            )
    return metrics


def offline_has_inpainting_scale() -> bool:
    data = load_json(OFFLINE_GUIDANCE_JSON)
    tasks = data.get("worker_summary", {}).get("task_summaries", {})
    return "inpainting" in tasks


def run_underlying() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "BM_TASK_CONDITIONED_TASKS": "inpainting",
            "BM_TASK_CONDITIONED_TASK_SEEDS_JSON": json.dumps({"inpainting": 20260821}),
            "BM_TASK_CONDITIONED_OUT_ROOT": str(VIS_ROOT),
            "BM_TASK_CONDITIONED_SUMMARY_ROOT": str(SUMMARY_ROOT / "underlying_tasks"),
            "BM_TASK_CONDITIONED_SUMMARY_JSON": str(UNDERLYING_JSON),
            "BM_TASK_CONDITIONED_SUMMARY_TSV": str(UNDERLYING_TSV),
            "BM_TASK_CONDITIONED_LOG_ROOT": str(LOG_ROOT / "underlying_tasks"),
            "BM_TASK_CONDITIONED_FAILED_ROOT": str(FAILED_ROOT / "underlying_tasks"),
            "BM_TASK_CONDITIONED_RUN_ROOT": str(RUN_ROOT / "underlying_tasks"),
        }
    )
    for path in [SUMMARY_ROOT, VIS_ROOT, LOG_ROOT, FAILED_ROOT, RUN_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [str(TRACKING_PY), str(BASE_SCRIPT)],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_failure(proc: subprocess.CompletedProcess[str]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    failure_dir = FAILED_ROOT / f"inpainting_guidance_{stamp}"
    failure_dir.mkdir(parents=True, exist_ok=True)
    (failure_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (failure_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    write_json(
        failure_dir / "status.json",
        {
            "status": "FAILED",
            "command": [str(TRACKING_PY), str(BASE_SCRIPT)],
            "returncode": proc.returncode,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "failure_reason": "underlying importer-export inpainting task-conditioned rollout did not exit cleanly",
            "does_not_claim_paper_level_result": True,
        },
    )


def build_summary(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if proc.returncode != 0:
        write_failure(proc)
    underlying = load_json(UNDERLYING_JSON)
    rows = underlying.get("rows", [])
    row = rows[0] if rows else {}
    task_summary = load_json(Path(row.get("summary_json", ""))) if row else {}
    proxy_metrics = collect_proxy_metrics(row, task_summary) if row else {}
    underlying_ok = (
        proc.returncode == 0
        and underlying.get("status") == "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
        and row.get("task") == "inpainting"
        and row.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
    )
    guided_keyframe = proxy_metrics.get("keyframe_proxy", {}).get("receding_latent_guided", {})
    denoised_keyframe = proxy_metrics.get("keyframe_proxy", {}).get("denoised_latent", {})
    result_row = {
        "task": "inpainting",
        "status": "ok" if underlying_ok else "failed",
        "rollout_steps": row.get("rollout_steps"),
        "selected_physical_gpu": row.get("selected_physical_gpu"),
        "guided_keyframe_error_mean": guided_keyframe.get("keyframe_error_mean"),
        "denoised_keyframe_error_mean": denoised_keyframe.get("keyframe_error_mean"),
        "guided_keyframe_error_delta_vs_denoised": proxy_metrics.get(
            "guided_keyframe_error_delta_vs_denoised"
        ),
        "guidance_cost_delta_mean": row.get("guidance_cost_delta_mean"),
        "capture_npz": proxy_metrics.get("capture_npz", ""),
        "mp4": proxy_metrics.get("mp4", ""),
    }
    checks = {
        "subprocess_returncode_zero": proc.returncode == 0,
        "underlying_status_ok": underlying.get("status")
        == "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval",
        "single_inpainting_task_attempted": len(rows) == 1 and row.get("task") == "inpainting",
        "task_row_status_ok": row.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
        "rollout_299_steps": row.get("rollout_steps") == 299,
        "capture_npz_exists": proxy_metrics.get("capture_npz_exists") is True,
        "mp4_exists": proxy_metrics.get("mp4_exists") is True,
        "keyframe_proxy_metrics_recorded": bool(guided_keyframe.get("available"))
        and bool(denoised_keyframe.get("available")),
        "uses_official_importer_export_usd": underlying.get("checks", {}).get("uses_official_importer_export_usd")
        is True,
        "uses_full_public_motion_bundle": underlying.get("checks", {}).get("uses_full_public_motion_bundle")
        is True,
        "used_fallback_guidance_scale": not offline_has_inpainting_scale(),
        "does_not_claim_fig6a_paper_protocol": True,
        "does_not_claim_official_checkpoint": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    return {
        "status": "ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
        if all(checks.values())
        else "failed_official_importer_export_full_bundle_inpainting_guidance_rollout_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval",
        "scope": (
            "Runs one local official-importer-export IsaacLab closed-loop future-keyframe/root-path inpainting "
            "proxy rollout. This is a virtual Fig. 6A planning gate only, not the paper cartwheel keyframe "
            "protocol and not official BeyondMimic evidence."
        ),
        "command": [str(TRACKING_PY), str(BASE_SCRIPT)],
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "rows": [result_row],
        "underlying_summary": underlying,
        "proxy_metrics": proxy_metrics,
        "checks": checks,
        "outputs": {
            "json": str(SUMMARY_JSON),
            "tsv": str(SUMMARY_TSV),
            "underlying_json": str(UNDERLYING_JSON),
            "underlying_tsv": str(UNDERLYING_TSV),
            "visualization_root": str(VIS_ROOT),
            "capture_npz": proxy_metrics.get("capture_npz", ""),
            "mp4": proxy_metrics.get("mp4", ""),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_official_importer_export_inpainting_keyframe_proxy",
            "paper_level_status": "qualitative_proxy_only",
            "why_not_paper_level": (
                "The run uses local PPO/VAE/denoiser checkpoints, a synthetic root-path future-keyframe cost, "
                "and a fallback guidance scale because the importer-export offline guidance sweep does not include "
                "an inpainting task. It is not the official BeyondMimic Fig. 6A cartwheel inpainting protocol, not "
                "an official checkpoint, not TensorRT deployment, and not real-robot evidence."
            ),
        },
    }


def main() -> None:
    proc = run_underlying()
    summary = build_summary(proc)
    write_json(SUMMARY_JSON, summary)
    write_tsv(SUMMARY_TSV, summary["rows"])
    print(json.dumps({"status": summary["status"], "json": str(SUMMARY_JSON)}, sort_keys=True))
    if summary["status"] != "ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
