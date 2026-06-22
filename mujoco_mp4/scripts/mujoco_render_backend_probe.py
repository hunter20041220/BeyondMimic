#!/usr/bin/env python3
"""Probe MuJoCo rendering backends in isolated subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, traceback_payload, utc_now, write_json, write_tsv


def run_backend(backend: str) -> dict[str, object]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    env.setdefault("BM_MUJOCO_SMOKE_FRAMES", "60")
    log_dir = PKG / "logs/smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"backend_probe_{backend}.log"
    cmd = [sys.executable, str(SCRIPT_DIR / "mujoco_minimal_video_smoke.py")]
    try:
        proc = subprocess.run(cmd, cwd=str(PKG.parent), env=env, text=True, capture_output=True, timeout=180)
        log_path.write_text(
            "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr,
            encoding="utf-8",
        )
        summary_path = PKG / f"res/smoke/minimal_scene_{backend}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        return {
            "backend": backend,
            "status": "ok" if proc.returncode == 0 and summary.get("status") == "ok" else "failed",
            "returncode": proc.returncode,
            "log_path": str(log_path),
            "summary_path": str(summary_path),
            "mp4_path": summary.get("outputs", {}).get("mp4", ""),
            "keyframe_path": summary.get("outputs", {}).get("keyframe_png", ""),
            "checks": summary.get("checks", {}),
            "error": summary.get("error", {}),
        }
    except Exception as exc:  # noqa: BLE001
        log_path.write_text(json.dumps(traceback_payload(exc), indent=2), encoding="utf-8")
        return {
            "backend": backend,
            "status": "failed",
            "returncode": "",
            "log_path": str(log_path),
            "summary_path": "",
            "mp4_path": "",
            "keyframe_path": "",
            "checks": {},
            "error": traceback_payload(exc),
        }


def main() -> None:
    backends = [b.strip() for b in os.environ.get("BM_MUJOCO_BACKENDS", "egl,osmesa,glfw").split(",") if b.strip()]
    rows = [run_backend(backend) for backend in backends]
    status = "ok" if any(row["status"] == "ok" for row in rows) else "failed"
    payload = {
        "status": status,
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_render_backend_probe",
        "claim_level": "MuJoCo backend availability probe; not robot, not policy, not paper-level result",
        "rows": rows,
        "preferred_backend": next((row["backend"] for row in rows if row["status"] == "ok"), ""),
        "checks": {
            "any_backend_ok": any(row["status"] == "ok" for row in rows),
            "egl_ok": any(row["backend"] == "egl" and row["status"] == "ok" for row in rows),
            "osmesa_ok": any(row["backend"] == "osmesa" and row["status"] == "ok" for row in rows),
            "glfw_ok": any(row["backend"] == "glfw" and row["status"] == "ok" for row in rows),
        },
    }
    out_json = PKG / "res/smoke/mujoco_render_backend_probe.json"
    out_tsv = PKG / "res/smoke/mujoco_render_backend_probe.tsv"
    write_json(out_json, payload)
    write_tsv(
        out_tsv,
        rows,
        ["backend", "status", "returncode", "log_path", "summary_path", "mp4_path", "keyframe_path", "checks", "error"],
    )
    print(json.dumps({"status": status, "preferred_backend": payload["preferred_backend"], "json": str(out_json)}))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
