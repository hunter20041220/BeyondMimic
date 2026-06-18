#!/usr/bin/env python3
"""Audit local preconverted G1 asset candidates for the official replay gate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_preconverted_asset_audit"
LOG_DIR = ROOT / "logs/tracking_g1_preconverted_asset_audit"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ISAACLAB_HEADLESS_KIT = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"
TIMEOUT_SECONDS = 120


SEARCH_ROOTS = [
    ROOT / "download",
    ROOT / "other",
    ROOT / "reproduction",
    ROOT / "res",
]


KIT_STAGE_CODE = r"""
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
ASSET = Path(os.environ["BM_G1_ASSET_PATH"])
EXPERIENCE = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"

print("BM_SENTINEL:before_app", flush=True)
from isaacsim.simulation_app import SimulationApp

config = {
    "headless": True,
    "active_gpu": 6,
    "physics_gpu": 6,
    "multi_gpu": False,
    "fast_shutdown": False,
    "create_new_stage": False,
    "extra_args": [
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        "--/renderer/activeGpu=6",
        "--/physics/cudaDevice=6",
    ],
}
app = SimulationApp(config, experience=str(EXPERIENCE))
print("BM_SENTINEL:after_app", flush=True)

payload = {
    "asset": str(ASSET),
    "experience": str(EXPERIENCE),
    "config": config,
}
try:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(ASSET))
    payload["stage_open_ok"] = stage is not None
    if stage is not None:
        default_prim = stage.GetDefaultPrim()
        payload["default_prim_valid"] = bool(default_prim)
        payload["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
        prim_count = 0
        joint_count = 0
        rigid_body_like_count = 0
        articulation_api_count = 0
        g1_path_like_count = 0
        pelvis_path_present = False
        torso_path_present = False
        sample = []
        for prim in stage.Traverse():
            prim_count += 1
            path_s = str(prim.GetPath())
            type_s = prim.GetTypeName()
            if len(sample) < 100:
                sample.append({"path": path_s, "type": type_s})
            if "Joint" in type_s:
                joint_count += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
                rigid_body_like_count += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_api_count += 1
            if "g1" in path_s.lower():
                g1_path_like_count += 1
            if path_s.endswith("/pelvis") or "pelvis" in path_s.lower():
                pelvis_path_present = True
            if "torso" in path_s.lower():
                torso_path_present = True
        payload.update(
            {
                "prim_count": prim_count,
                "joint_count": joint_count,
                "rigid_body_like_count": rigid_body_like_count,
                "articulation_api_count": articulation_api_count,
                "g1_path_like_count": g1_path_like_count,
                "pelvis_path_present": pelvis_path_present,
                "torso_path_present": torso_path_present,
                "prim_sample": sample,
            }
        )
except Exception as exc:
    payload["exception"] = repr(exc)

print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
app.close()
print("BM_SENTINEL:after_close", flush=True)
"""


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit
        while True:
            if remaining is None:
                chunk = f.read(1024 * 1024)
            elif remaining <= 0:
                break
            else:
                chunk = f.read(min(1024 * 1024, remaining))
                remaining -= len(chunk)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def command_output(args: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def classify_source(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("download/official/whole_body_tracking") or rel.startswith("reproduction/third_party/official/whole_body_tracking"):
        return "official_whole_body_tracking"
    if rel.startswith("download/official/LAFAN1_Retargeting_Dataset") or rel.startswith(
        "reproduction/data/Dataset_beyondmimic"
    ):
        return "official_or_released_dataset_asset"
    if rel.startswith("download/reference_code"):
        return "reference_code_not_beyondmimic_official"
    if rel.startswith("other/"):
        return "old_workspace_snapshot"
    if rel.startswith("res/") or rel.startswith("reproduction/"):
        return "local_reproduction_or_generated"
    return "other"


def is_candidate(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    rel = path.relative_to(ROOT).as_posix().lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return "g1" in rel or "unitree" in rel
    if suffix in {".urdf", ".xml", ".mjcf"}:
        return "g1" in rel or "unitree_g1" in rel
    return False


def collect_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {".git", "__pycache__"}
                and not (Path(dirpath) / d).as_posix().startswith((ROOT / "envs").as_posix())
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if is_candidate(path):
                    candidates.append(path)
    return sorted(set(candidates), key=lambda p: p.relative_to(ROOT).as_posix())


def static_strings(path: Path) -> dict[str, Any]:
    rc, out = command_output(["bash", "-lc", f"strings -n 8 {str(path)!r} | head -400"], timeout=30)
    lower = out.lower()
    markers = {
        "mentions_g1": "g1" in lower,
        "mentions_pelvis": "pelvis" in lower,
        "mentions_torso": "torso" in lower,
        "mentions_physics_scene": "physicsscene" in lower or "physicsScene" in out,
        "mentions_rigid_body_api": "rigidbodyapi" in lower,
        "mentions_articulation": "articulation" in lower,
        "mentions_joint": "joint" in lower,
        "mentions_mesh": "mesh" in lower,
    }
    return {
        "strings_returncode": rc,
        "markers": markers,
        "sample": [line for line in out.splitlines()[:80]],
    }


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def classify_kit_output(text: str, timed_out: bool) -> dict[str, bool]:
    lower = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "vulkan_device_lost": "error_device_lost" in lower or "gpu crash is detected" in lower,
        "traceback": "traceback (most recent call last)" in lower,
    }


def base_env(asset: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BM_G1_ASSET_PATH": str(asset),
            "PYTHONNOUSERSITE": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "ENABLE_CAMERAS": "0",
            "TMPDIR": str(ROOT / "tmp"),
            "HOME": str(ROOT / "cache/home"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "PIP_CACHE_DIR": str(ROOT / "cache/pip"),
            "OMNI_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OMNI_LOGS_DIR": str(ROOT / "logs/omniverse"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "OV_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OV_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def kit_read_usd(asset: Path) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", asset.relative_to(ROOT).as_posix())[-180:]
    log_path = LOG_DIR / f"{safe_name}.log"
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(KIT_STAGE_CODE)],
            cwd=ROOT,
            env=base_env(asset),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
        )
        output = proc.stdout
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        if isinstance(exc.stdout, bytes):
            output = exc.stdout.decode("utf-8", errors="ignore")
        elif isinstance(exc.stdout, str):
            output = exc.stdout
        else:
            output = ""
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    return {
        "returncode": returncode,
        "log": str(log_path),
        "markers": classify_kit_output(output, timed_out),
        "payload": extract_payload(output),
    }


def row_for(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    suffix = path.suffix.lower()
    row: dict[str, Any] = {
        "path": str(path),
        "relative_path": rel,
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_class": classify_source(path),
        "is_usd": suffix in {".usd", ".usda", ".usdc"},
        "is_reference_only": classify_source(path) == "reference_code_not_beyondmimic_official",
        "is_official_beyondmimic_tracking_asset": classify_source(path) == "official_whole_body_tracking",
        "is_mesh_usd": suffix in {".usd", ".usda", ".usdc"} and "/meshes/" in rel and path.name.endswith(".tmp.usd"),
    }
    row["is_full_robot_usd_candidate"] = bool(row["is_usd"] and not row["is_mesh_usd"])
    if row["is_usd"]:
        rc, out = command_output(["file", str(path)], timeout=10)
        row["file_returncode"] = rc
        row["file_type"] = out.strip()
        row["static_strings"] = static_strings(path)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = [row_for(path) for path in collect_candidates()]
    usd_candidates = [row for row in candidates if row["is_usd"]]
    g1_reference_usd = [
        row
        for row in usd_candidates
        if "g1" in row["relative_path"].lower()
        and row["source_class"] == "reference_code_not_beyondmimic_official"
    ]

    kit_validated: list[dict[str, Any]] = []
    for row in sorted(g1_reference_usd, key=lambda r: r["size_bytes"], reverse=True)[:2]:
        probe = kit_read_usd(Path(row["path"]))
        payload = probe.get("payload", {})
        has_robotish_stage = bool(
            payload.get("stage_open_ok")
            and payload.get("prim_count", 0) > 10
            and (
                payload.get("rigid_body_like_count", 0) > 0
                or payload.get("articulation_api_count", 0) > 0
                or payload.get("joint_count", 0) > 0
            )
        )
        kit_validated.append(
            {
                "relative_path": row["relative_path"],
                "source_class": row["source_class"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "probe": probe,
                "has_robotish_stage": has_robotish_stage,
                "usable_as_official_beyondmimic_asset": False,
                "why_not_official": (
                    "This candidate comes from downloaded reference_code, not the official BeyondMimic "
                    "whole_body_tracking repository or released dataset. It may inform a workaround but cannot be "
                    "reported as an official BeyondMimic replay asset."
                ),
            }
        )

    official_mesh_usd = [
        row
        for row in usd_candidates
        if row["source_class"] in {"official_whole_body_tracking", "official_or_released_dataset_asset"}
        and row["is_mesh_usd"]
    ]
    official_full_robot_usd = [
        row
        for row in usd_candidates
        if row["source_class"] in {"official_whole_body_tracking", "official_or_released_dataset_asset"}
        and row["is_full_robot_usd_candidate"]
    ]
    reference_robotish_validated = [row for row in kit_validated if row["has_robotish_stage"]]
    checks = {
        "candidate_count_positive": len(candidates) > 0,
        "usd_candidate_count_positive": len(usd_candidates) > 0,
        "official_mesh_usd_present": len(official_mesh_usd) > 0,
        "official_full_robot_preconverted_g1_usd_absent": len(official_full_robot_usd) == 0,
        "reference_g1_usd_present": len(g1_reference_usd) > 0,
        "kit_validation_attempted_for_reference_usd": len(kit_validated) > 0,
        "reference_robotish_usd_validated_if_available": len(reference_robotish_validated) > 0
        or len(g1_reference_usd) == 0,
        "does_not_modify_download": True,
        "does_not_claim_reference_usd_as_official": all(not row["usable_as_official_beyondmimic_asset"] for row in kit_validated),
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = "ok_with_reference_usd_candidate" if reference_robotish_validated else "ok_with_no_official_preconverted_g1_usd"
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_preconverted_asset_audit",
        "scope": (
            "Local read-only audit for preconverted G1 USD candidates that might help unblock the official replay "
            "conversion gate. Reference-code assets are explicitly not treated as official BeyondMimic results."
        ),
        "candidate_count": len(candidates),
        "usd_candidate_count": len(usd_candidates),
        "official_mesh_usd_count": len(official_mesh_usd),
        "official_full_robot_preconverted_g1_usd_count": len(official_full_robot_usd),
        "reference_g1_usd_count": len(g1_reference_usd),
        "validated_reference_robotish_usd_count": len(reference_robotish_validated),
        "candidates": candidates,
        "official_mesh_usd_candidates": official_mesh_usd,
        "official_full_robot_preconverted_usd_candidates": official_full_robot_usd,
        "kit_validated_reference_usd": kit_validated,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The audit can identify candidate preconverted USD assets, but official replay remains blocked unless "
                "a valid asset is accepted and wired into the official csv_to_npz/replay pipeline. Reference-code USDs "
                "cannot be claimed as official BeyondMimic assets."
            ),
            "next_action": (
                "If a reference USD is structurally valid, evaluate whether a clearly labeled resource-adjusted "
                "conversion path can use it; otherwise build a lower-level offline converter from official URDF/MJCF."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_g1_preconverted_asset_audit.json")},
    }
    (OUT / "tracking_g1_preconverted_asset_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "candidate_count": len(candidates),
                "usd_candidate_count": len(usd_candidates),
                "official_mesh_usd_count": len(official_mesh_usd),
                "official_full_robot_preconverted_g1_usd_count": len(official_full_robot_usd),
                "validated_reference_robotish_usd_count": len(reference_robotish_validated),
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
