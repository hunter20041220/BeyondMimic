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
URDF_PROBE = ROOT / "res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json"
URDF_PATH_TINY_PROBE = ROOT / "res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json"
MJCF_STAGE_PROBE = ROOT / "res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json"
USD_SAVE_POLICY_PROBE = ROOT / "res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json"
SIMULATIONAPP_SAVE_POLICY_PROBE = (
    ROOT / "res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json"
)


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    urdf_probe = load_json(URDF_PROBE)
    urdf_path_tiny_probe = load_json(URDF_PATH_TINY_PROBE)
    mjcf_stage_probe = load_json(MJCF_STAGE_PROBE)
    usd_save_policy_probe = load_json(USD_SAVE_POLICY_PROBE)
    simulationapp_save_policy_probe = load_json(SIMULATIONAPP_SAVE_POLICY_PROBE)
    urdf_payload = urdf_probe.get("payload", {})
    any_success = MOTION_NPZ.is_file() and any(row["markers"]["motion_saved"] for row in rows)
    if any_success:
        latest_blocker = "none"
    elif simulationapp_save_policy_probe.get("current_blocker"):
        latest_blocker = simulationapp_save_policy_probe["current_blocker"]
    elif usd_save_policy_probe.get("current_blocker"):
        latest_blocker = usd_save_policy_probe["current_blocker"]
    elif mjcf_stage_probe.get("current_blocker"):
        latest_blocker = mjcf_stage_probe["current_blocker"]
    elif urdf_path_tiny_probe.get("current_blocker"):
        latest_blocker = urdf_path_tiny_probe["current_blocker"]
    elif urdf_payload.get("stage_open_ok") and urdf_payload.get("prim_count") == 0:
        latest_blocker = "urdf_converter_empty_usd"
    else:
        latest_blocker = "urdf_usd_save_not_allowed"
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
        "urdf_conversion_probe_recorded": urdf_probe.get("status") == "ok_with_urdf_usd_blocker",
        "urdf_path_tiny_probe_recorded": urdf_path_tiny_probe.get("status") == "ok_with_blocker_classified",
        "mjcf_stage_probe_recorded": mjcf_stage_probe.get("status") == "ok_with_blocker_classified",
        "usd_save_policy_probe_recorded": usd_save_policy_probe.get("status") == "ok_with_blocker_classified",
        "simulationapp_save_policy_probe_recorded": simulationapp_save_policy_probe.get("status")
        == "ok_with_blocker_classified",
        "urdf_converter_empty_usd_recorded": urdf_payload.get("stage_open_ok") is True
        and urdf_payload.get("prim_count") == 0,
        "urdf_save_forbidden_and_vulkan_device_lost_recorded": urdf_path_tiny_probe.get("markers", {}).get(
            "usd_save_not_allowed"
        )
        is True
        and urdf_path_tiny_probe.get("markers", {}).get("vulkan_device_lost") is True,
        "basic_usd_stage_save_failure_recorded": mjcf_stage_probe.get("checks", {}).get("minimal_stage_save_success")
        is False
        and mjcf_stage_probe.get("markers", {}).get("usd_save_not_allowed") is True,
        "mjcf_bypass_blocked_recorded": mjcf_stage_probe.get("checks", {}).get("g1_mjcf_conversion_success") is False,
        "app_launcher_layers_permission_to_save_false_recorded": usd_save_policy_probe.get("checks", {}).get(
            "permission_false_recorded"
        )
        is True
        and usd_save_policy_probe.get("checks", {}).get("force_permission_attempts_failed") is True,
        "simulationapp_and_applauncher_save_policy_match_recorded": simulationapp_save_policy_probe.get(
            "checks", {}
        ).get("simulationapp_isaaclab_headless_permission_false")
        is True
        and simulationapp_save_policy_probe.get("checks", {}).get("applauncher_permission_false") is True,
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
        "urdf_conversion_probe_json": str(URDF_PROBE),
        "urdf_path_tiny_probe_json": str(URDF_PATH_TINY_PROBE),
        "mjcf_stage_probe_json": str(MJCF_STAGE_PROBE),
        "usd_save_policy_probe_json": str(USD_SAVE_POLICY_PROBE),
        "simulationapp_save_policy_probe_json": str(SIMULATIONAPP_SAVE_POLICY_PROBE),
        "urdf_conversion_probe_summary": {
            "status": urdf_probe.get("status"),
            "returncode": urdf_probe.get("returncode"),
            "converter_usd_mode": urdf_payload.get("converter_usd_mode"),
            "converter_usd_size": urdf_payload.get("converter_usd_size"),
            "stage_open_ok": urdf_payload.get("stage_open_ok"),
            "default_prim_valid": urdf_payload.get("default_prim_valid"),
            "prim_count": urdf_payload.get("prim_count"),
            "rigid_body_like_count": urdf_payload.get("rigid_body_like_count"),
        },
        "urdf_path_tiny_probe_summary": {
            "status": urdf_path_tiny_probe.get("status"),
            "returncode": urdf_path_tiny_probe.get("returncode"),
            "current_blocker": urdf_path_tiny_probe.get("current_blocker"),
            "markers": urdf_path_tiny_probe.get("markers"),
            "checks": urdf_path_tiny_probe.get("checks"),
        },
        "mjcf_stage_probe_summary": {
            "status": mjcf_stage_probe.get("status"),
            "returncode": mjcf_stage_probe.get("returncode"),
            "current_blocker": mjcf_stage_probe.get("current_blocker"),
            "markers": mjcf_stage_probe.get("markers"),
            "checks": mjcf_stage_probe.get("checks"),
        },
        "usd_save_policy_probe_summary": {
            "status": usd_save_policy_probe.get("status"),
            "current_blocker": usd_save_policy_probe.get("current_blocker"),
            "plain_python": usd_save_policy_probe.get("plain_python"),
            "app_launcher_counts": {
                "save_ok_count": usd_save_policy_probe.get("app_launcher", {}).get("save_ok_count"),
                "force_save_ok_count": usd_save_policy_probe.get("app_launcher", {}).get("force_save_ok_count"),
                "export_ok_count": usd_save_policy_probe.get("app_launcher", {}).get("export_ok_count"),
                "permission_false_count": usd_save_policy_probe.get("app_launcher", {}).get(
                    "permission_false_count"
                ),
            },
            "checks": usd_save_policy_probe.get("checks"),
        },
        "simulationapp_save_policy_probe_summary": {
            "status": simulationapp_save_policy_probe.get("status"),
            "current_blocker": simulationapp_save_policy_probe.get("current_blocker"),
            "checks": simulationapp_save_policy_probe.get("checks"),
            "cases": [
                {
                    "name": case.get("name"),
                    "returncode": case.get("returncode"),
                    "save_ok_count": case.get("save_ok_count"),
                    "permission_false_count": case.get("permission_false_count"),
                    "force_after_false_count": case.get("force_after_false_count"),
                    "markers": case.get("markers"),
                    "log": case.get("log"),
                }
                for case in simulationapp_save_policy_probe.get("cases", [])
            ],
        },
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
                "but the isolated G1 URDF conversion probe produces a tiny USD with no default prim, no traversable "
                "robot prims, and no rigid-body-like prims. The path/tiny contrast probe further records project-local "
                "USD layer save-forbidden errors followed by Vulkan device loss before a valid URDF conversion payload. "
                "The MJCF/stage bypass probe shows the same save policy at the basic USD stage-save layer, and the "
                "USD save policy probe shows all AppLauncher-created layers have permissionToSave=False across "
                "tmp/cache/res paths even after SetPermissionToSave(True). The SimulationApp/AppLauncher comparison "
                "probe further shows raw SimulationApp with the IsaacLab headless experience has the same save policy, "
                "while the Isaac Sim base python experience hits a Vulkan device-lost crash before payload. No valid "
                "official motion.npz or replay has been produced."
            ),
            "next_action": (
                "Investigate why AppLauncher/Kit creates local Sdf layers with permissionToSave=False. A valid basic "
                "USD stage save or an external preconverted G1 USD is required before retrying URDF/MJCF conversion "
                "or csv_to_npz."
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
