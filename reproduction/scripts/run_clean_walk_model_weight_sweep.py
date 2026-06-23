#!/usr/bin/env python3
"""Run a clean-walk MuJoCo model-target-weight sweep.

This diagnostic answers a narrow question: are the current teacher/VAE/diffusion
targets stable by themselves, or are the readable videos mainly held together by
the reference anchor?  It repeatedly calls
``render_clean_walk_mujoco_control_suite.py`` on the same continuous LAFAN1 walk
window while increasing ``BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT``.

Claim boundary: this is a local MuJoCo diagnostic.  It does not make any
paper-level BeyondMimic claim.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
SCRIPT = ROOT / "reproduction/scripts/render_clean_walk_mujoco_control_suite.py"
OUT_ROOT = ROOT / "res/visualization/clean_walk_mujoco_control_suite_sweep"
LOG_ROOT = ROOT / "logs/mujoco/clean_walk_control_suite_sweep"
# Do not resolve the venv python symlink: resolving it can collapse back to
# /usr/bin/python3 and lose the virtualenv packages.
PYTHON = Path(os.environ.get("BM_MUJOCO_PYTHON", str(ROOT / "mujoco_mp4/.venv/bin/python"))).expanduser()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_weights() -> list[float]:
    raw = os.environ.get("BM_CLEAN_SUITE_SWEEP_WEIGHTS", "0.20,0.40,0.60,0.80,1.00")
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not weights:
        raise ValueError("BM_CLEAN_SUITE_SWEEP_WEIGHTS did not contain any weights")
    return weights


def label_weight(weight: float) -> str:
    return f"w{int(round(weight * 100)):03d}"


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing_summary", "summary_json": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable_summary", "summary_json": str(path), "error": repr(exc)}


def compact_variant_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("variant_metrics", {})
    variants: dict[str, Any] = {}
    for name, value in sorted(metrics.items()):
        if not isinstance(value, dict):
            continue
        variants[name] = {
            key: value.get(key)
            for key in [
                "fall_proxy_count",
                "root_height_min",
                "root_height_mean",
                "root_height_max",
                "root_position_error_mean_m",
                "root_position_error_max_m",
                "model_target_gap_to_reference_abs_mean",
                "final_target_gap_to_reference_abs_mean",
                "joint_error_abs_mean",
                "joint_error_to_reference_abs_mean",
            ]
            if key in value
        }
    return variants


def run_weight(weight: float) -> dict[str, Any]:
    label = label_weight(weight)
    output_dir = OUT_ROOT / label
    log_path = LOG_ROOT / f"{label}.log"
    env = os.environ.copy()
    env.update(
        {
            "BM_ROOT": str(ROOT),
            "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
            "BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT": f"{weight:.6f}",
            "BM_CLEAN_SUITE_OUT_ROOT": str(output_dir),
            "BM_CLEAN_WALK_SECONDS": env.get("BM_CLEAN_WALK_SECONDS", "15.0"),
            "BM_CLEAN_WALK_START_INDEX": env.get("BM_CLEAN_WALK_START_INDEX", "0"),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(SCRIPT)]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    summary_path = output_dir / "clean_walk_mujoco_control_suite_summary.json"
    summary = load_summary(summary_path)
    variants = compact_variant_metrics(summary)
    primary = {k: v for k, v in variants.items() if k != "guided_vs_unguided_action_control"}
    unstable = {
        name: metrics
        for name, metrics in primary.items()
        if (metrics.get("fall_proxy_count") or 0) > 0 or (metrics.get("root_height_min") or 0.0) < 0.45
    }
    return {
        "weight": weight,
        "label": label,
        "returncode": proc.returncode,
        "status": summary.get("status", "missing_or_failed"),
        "output_root": str(output_dir),
        "summary_json": str(summary_path),
        "log_path": str(log_path),
        "all_primary_zero_fall": bool(primary)
        and all((metrics.get("fall_proxy_count") or 0) == 0 for metrics in primary.values()),
        "all_primary_root_height_above_0p45": bool(primary)
        and all((metrics.get("root_height_min") or 0.0) >= 0.45 for metrics in primary.values()),
        "unstable_variants": unstable,
        "variant_metrics": variants,
    }


def write_tables(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = OUT_ROOT / "clean_walk_model_weight_sweep_summary.csv"
    md_path = OUT_ROOT / "clean_walk_model_weight_sweep_summary.md"
    fieldnames = [
        "weight",
        "variant",
        "suite_status",
        "returncode",
        "fall_proxy_count",
        "root_height_min",
        "root_height_mean",
        "root_position_error_mean_m",
        "model_target_gap_to_reference_abs_mean",
        "final_target_gap_to_reference_abs_mean",
        "video_path",
        "summary_json",
        "log_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in results:
            summary_path = Path(item["summary_json"])
            payload = load_summary(summary_path)
            variants_out = payload.get("variants", {})
            for variant, metrics in item["variant_metrics"].items():
                writer.writerow(
                    {
                        "weight": item["weight"],
                        "variant": variant,
                        "suite_status": item["status"],
                        "returncode": item["returncode"],
                        "fall_proxy_count": metrics.get("fall_proxy_count"),
                        "root_height_min": metrics.get("root_height_min"),
                        "root_height_mean": metrics.get("root_height_mean"),
                        "root_position_error_mean_m": metrics.get("root_position_error_mean_m"),
                        "model_target_gap_to_reference_abs_mean": metrics.get(
                            "model_target_gap_to_reference_abs_mean"
                        ),
                        "final_target_gap_to_reference_abs_mean": metrics.get(
                            "final_target_gap_to_reference_abs_mean"
                        ),
                        "video_path": variants_out.get(variant, {}).get("mp4", ""),
                        "summary_json": item["summary_json"],
                        "log_path": item["log_path"],
                    }
                )

    lines = [
        "# Clean Walk Model-Target-Weight Sweep",
        "",
        "## 结论",
        "",
        summary["diagnosis"],
        "",
        "## Claim Boundary",
        "",
        "这是本地 MuJoCo 稳定性诊断。它不证明 BeyondMimic paper-level closed-loop rollout，也不是真实机器人结果。",
        "",
        "## Sweep Results",
        "",
        "| weight | status | primary zero fall | root height > 0.45 | unstable variants | output |",
        "|---:|---|---:|---:|---|---|",
    ]
    for item in results:
        unstable = ", ".join(sorted(item["unstable_variants"].keys())) or "none"
        lines.append(
            f"| {item['weight']:.2f} | {item['status']} | {item['all_primary_zero_fall']} | "
            f"{item['all_primary_root_height_above_0p45']} | {unstable} | `{item['output_root']}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON: `{OUT_ROOT / 'clean_walk_model_weight_sweep_summary.json'}`",
            f"- CSV: `{csv_path}`",
            f"- Logs: `{LOG_ROOT}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def make_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    stable_weights = [
        item["weight"]
        for item in results
        if item["returncode"] == 0
        and item["all_primary_zero_fall"]
        and item["all_primary_root_height_above_0p45"]
    ]
    pure = next((item for item in results if abs(item["weight"] - 1.0) < 1e-9), None)
    pure_ok = bool(
        pure
        and pure["returncode"] == 0
        and pure["all_primary_zero_fall"]
        and pure["all_primary_root_height_above_0p45"]
    )
    if pure_ok:
        diagnosis = (
            "Pure model-target control stayed above the fall proxy in this assisted clean-walk diagnostic, "
            "but this still uses root assist and an approximate MuJoCo adapter."
        )
    else:
        diagnosis = (
            "The readable clean-walk videos are reference-anchored diagnostics. Pure or high-weight "
            "model-target control is not yet a credible Stage-1/VAE/diffusion success signal; the "
            "dominant blockers remain weak teacher quality and IsaacLab-to-MuJoCo obs/action contract fidelity."
        )
    return {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "clean_walk_model_target_weight_sweep",
        "claim_level": "Local MuJoCo stability sweep for reference-anchor vs model-target contribution; not paper-level BeyondMimic.",
        "script": str(Path(__file__).resolve()),
        "render_script": str(SCRIPT),
        "weights": [item["weight"] for item in results],
        "stable_weights_by_fall_proxy": stable_weights,
        "pure_model_weight_1p0_ok": pure_ok,
        "diagnosis": diagnosis,
        "results": results,
        "outputs": {
            "summary_json": str(OUT_ROOT / "clean_walk_model_weight_sweep_summary.json"),
            "summary_csv": str(OUT_ROOT / "clean_walk_model_weight_sweep_summary.csv"),
            "summary_md": str(OUT_ROOT / "clean_walk_model_weight_sweep_summary.md"),
            "log_root": str(LOG_ROOT),
        },
        "limitations": [
            "The sweep uses root assist and position actuators; it is a diagnostic bridge, not official IsaacLab physics.",
            "The reference-action baseline is a controlled reference target, not a learned policy.",
            "The learned variants are only credible as pure learned control when model_target_weight=1.0 is stable and visually plausible.",
        ],
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    weights = parse_weights()
    results = []
    for weight in weights:
        print(f"[sweep] running weight={weight:.2f}", flush=True)
        result = run_weight(weight)
        print(json.dumps({k: result[k] for k in ["weight", "status", "returncode"]}, ensure_ascii=False), flush=True)
        results.append(result)
    summary = make_summary(results)
    write_json(OUT_ROOT / "clean_walk_model_weight_sweep_summary.json", summary)
    write_tables(results, summary)
    print(json.dumps({"status": summary["status"], "summary_json": summary["outputs"]["summary_json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
