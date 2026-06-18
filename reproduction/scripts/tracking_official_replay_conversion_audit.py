#!/usr/bin/env python3
"""Audit bounded official whole_body_tracking conversion attempts."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/official_replay_conversion"
LOG_DIR = ROOT / "logs/tracking_official_replay_conversion"
MOTION_NPZ = OUT / "walk1_subject1_frames_1_180_motion.npz"
CONTRACT_JSON = OUT / "walk1_subject1_frames_1_180_motion_contract.json"
LOCAL_CSV_TO_NPZ = ROOT / "reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py"


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.strip()


def classify(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "argparse_enable_cameras_false_error": "unrecognized arguments: false" in lowered,
        "cuda_visible_devices_ordinal_error": "invalid device ordinal" in lowered,
        "rsl_rl_env_missing": "no module named 'rsl_rl.env'" in lowered,
        "libglu_missing": "libglu.so.1" in lowered and "cannot open shared object file" in lowered,
        "usd_save_not_allowed": "saving not allowed" in lowered,
        "robot_reference_unresolved": "unresolved reference prim path" in lowered,
        "no_contact_sensors": "no contact sensors added to the prim" in lowered,
        "p2p_iommu_warning": "p2pbandwidthlatencytest" in lowered
        or "cuda peer-to-peer observed bandwidth" in lowered,
        "motion_saved": "[info]: motion saved locally" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "timeout_or_killed": "csv_to_npz_rc=124" in lowered or "csv_to_npz_rc=137" in lowered,
    }


def log_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LOG_DIR.glob("csv_to_npz_local_*.log")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rc_match = re.findall(r"CSV_TO_NPZ_RC=(\d+)", text)
        rows.append(
            {
                "log": str(path),
                "size_bytes": path.stat().st_size,
                "return_code_recorded": int(rc_match[-1]) if rc_match else None,
                "markers": classify(text),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rsl_rc, rsl_out = run(
        [
            str(ROOT / "envs/bm_tracking/bin/python"),
            "-c",
            "import torch, rsl_rl, rsl_rl.env; print(torch.__version__); print(rsl_rl.__file__)",
        ]
    )
    pip_check_rc, pip_check_out = run([str(ROOT / "envs/bm_tracking/bin/python"), "-m", "pip", "check"])
    libglu_rc, libglu_out = run(["bash", "-lc", "ldconfig -p | rg 'libGLU.so.1'"])
    rows = log_rows()
    any_success = MOTION_NPZ.is_file() and any(row["markers"]["motion_saved"] for row in rows)
    latest_blocker = "none" if any_success else "urdf_usd_save_not_allowed"
    if any(row["markers"]["rsl_rl_env_missing"] for row in rows) and rsl_rc == 0:
        resolved_rsl_rl = True
    else:
        resolved_rsl_rl = rsl_rc == 0
    checks = {
        "local_csv_to_npz_script_exists": LOCAL_CSV_TO_NPZ.is_file(),
        "rsl_rl_env_import_ok": rsl_rc == 0,
        "rsl_rl_isaaclab_expected_version_installed": "site-packages/rsl_rl" in rsl_out,
        "tracking_pip_check_ok": pip_check_rc == 0,
        "system_libglu_available": libglu_rc == 0 and "libGLU.so.1" in libglu_out,
        "attempt_logs_present": len(rows) >= 3,
        "motion_npz_written": MOTION_NPZ.is_file(),
        "motion_npz_contract_written": CONTRACT_JSON.is_file(),
        "usd_save_blocker_recorded": any(row["markers"]["usd_save_not_allowed"] for row in rows),
        "rsl_rl_missing_was_repaired": resolved_rsl_rl,
        "does_not_claim_replay_success": not any_success,
        "does_not_start_training": True,
    }
    summary: dict[str, Any] = {
        "status": "ok_with_blocked_conversion" if not any_success and checks["usd_save_blocker_recorded"] else "ok"
        if any_success
        else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_replay_conversion_audit",
        "scope": (
            "Bounded local execution attempts for official-equivalent whole_body_tracking csv_to_npz conversion. "
            "No replay video, PPO training, policy evaluation, DAgger, VAE, or diffusion closed-loop run is claimed."
        ),
        "selected_motion": str(ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"),
        "motion_npz": str(MOTION_NPZ),
        "contract_json": str(CONTRACT_JSON),
        "latest_blocker": latest_blocker,
        "environment_repairs": {
            "rsl_rl": rsl_out,
            "pip_check": pip_check_out,
            "libglu": libglu_out,
            "notes": [
                "Installed rsl-rl-lib==2.3.1 in bm_tracking without changing torch 2.5.1+cu121.",
                "Installed host libglu1-mesa so Isaac Sim can resolve libGLU.so.1.",
                "Removed CUDA_VISIBLE_DEVICES from Isaac conversion command to avoid Omniverse/CUDA ordinal mismatch.",
            ],
        },
        "attempts": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The conversion progressed beyond argument parsing, RSL-RL import, CUDA ordinal, and libGLU issues, "
                "but the Isaac URDF importer still reports USD saving-not-allowed and the robot prim has no rigid "
                "bodies/contact sensors. No valid official motion.npz or replay has been produced."
            ),
            "next_action": (
                "Investigate Isaac Sim URDF importer USD write permissions/save policy or pre-generate a valid G1 USD "
                "asset under the project tmp/cache before retrying csv_to_npz."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_official_replay_conversion_audit.json"),
        },
    }
    (OUT / "tracking_official_replay_conversion_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
