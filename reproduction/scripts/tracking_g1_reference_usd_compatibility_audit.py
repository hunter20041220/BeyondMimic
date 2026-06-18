#!/usr/bin/env python3
"""Audit compatibility between the ASAP reference G1 USD and the official BeyondMimic G1 contract."""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_reference_usd_compatibility_audit"
LOG_DIR = ROOT / "logs/tracking_g1_reference_usd_compatibility_audit"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ISAACLAB_HEADLESS_KIT = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"
OFFICIAL_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/"
    "unitree_description/urdf/g1/main.urdf"
)
REFERENCE_USD = ROOT / "download/reference_code/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.usd"
OFFICIAL_SOURCE_AUDIT = ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
TIMEOUT_SECONDS = 120


KIT_USD_CODE = r"""
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
ASSET = Path(os.environ["BM_G1_REFERENCE_USD"])
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

payload = {"asset": str(ASSET), "experience": str(EXPERIENCE), "config": config}

try:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(ASSET))
    payload["stage_open_ok"] = stage is not None
    links = []
    joints = []
    rigid_bodies = []
    articulation_roots = []
    mesh_prims = []
    prims = []
    if stage is not None:
        default_prim = stage.GetDefaultPrim()
        payload["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
        for prim in stage.Traverse():
            path_s = str(prim.GetPath())
            name = prim.GetName()
            type_s = prim.GetTypeName()
            prims.append({"path": path_s, "name": name, "type": type_s})
            if type_s in {"Xform", ""} and path_s.startswith("/g1_29dof/") and path_s.count("/") == 2:
                if name not in {"Looks"}:
                    links.append(name)
            if "Joint" in type_s:
                joints.append({"name": name, "path": path_s, "type": type_s})
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
                rigid_bodies.append({"name": name, "path": path_s, "type": type_s})
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append({"name": name, "path": path_s, "type": type_s})
            if type_s == "Mesh":
                mesh_prims.append({"name": name, "path": path_s})
    payload.update(
        {
            "prim_count": len(prims),
            "links": sorted(set(links)),
            "link_count": len(set(links)),
            "joints": joints,
            "joint_names": sorted({row["name"] for row in joints}),
            "joint_count": len(joints),
            "rigid_bodies": rigid_bodies,
            "rigid_body_count": len(rigid_bodies),
            "articulation_roots": articulation_roots,
            "articulation_api_count": len(articulation_roots),
            "mesh_prim_count": len(mesh_prims),
            "mesh_prims_sample": mesh_prims[:40],
        }
    )
except Exception as exc:
    payload["exception"] = repr(exc)

print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
app.close()
print("BM_SENTINEL:after_close", flush=True)
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_official_urdf() -> dict[str, Any]:
    root = ET.parse(OFFICIAL_URDF).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib.get("link") if joint.find("parent") is not None else None
        child = joint.find("child").attrib.get("link") if joint.find("child") is not None else None
        joints.append({"name": joint.attrib["name"], "type": joint.attrib.get("type"), "parent": parent, "child": child})
    return {
        "links": links,
        "link_count": len(links),
        "joints": joints,
        "joint_names": [row["name"] for row in joints],
        "non_fixed_joint_names": [row["name"] for row in joints if row["type"] != "fixed"],
        "fixed_joint_names": [row["name"] for row in joints if row["type"] == "fixed"],
        "non_fixed_joint_count": sum(1 for row in joints if row["type"] != "fixed"),
    }


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BM_G1_REFERENCE_USD": str(REFERENCE_USD),
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


def classify_output(text: str, timed_out: bool) -> dict[str, bool]:
    lower = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "vulkan_device_lost": "error_device_lost" in lower or "gpu crash is detected" in lower,
        "traceback": "traceback (most recent call last)" in lower,
    }


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def run_usd_probe() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(KIT_USD_CODE)],
            cwd=ROOT,
            env=base_env(),
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
    log_path = LOG_DIR / "tracking_g1_reference_usd_compatibility_audit.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    return {
        "returncode": returncode,
        "log": str(log_path),
        "markers": classify_output(output, timed_out),
        "payload": extract_payload(output),
    }


def sorted_diff(left: list[str], right: list[str]) -> dict[str, list[str]]:
    left_set = set(left)
    right_set = set(right)
    return {
        "missing_from_right": sorted(left_set - right_set),
        "extra_in_right": sorted(right_set - left_set),
        "intersection": sorted(left_set & right_set),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    official = parse_official_urdf()
    official_source = load_json(OFFICIAL_SOURCE_AUDIT)
    action_scale = load_json(ACTION_SCALE_AUDIT)
    probe = run_usd_probe()
    payload = probe.get("payload", {})

    target_bodies = official_source["flat_env"]["body_names"]
    anchor_body = official_source["flat_env"]["anchor_body_name"]
    action_joint_names = [row["joint_name"] for row in action_scale["joint_rows"]]
    reference_links = payload.get("links", [])
    reference_joint_names = payload.get("joint_names", [])
    reference_revolute_joint_names = [
        row["name"] for row in payload.get("joints", []) if row.get("type") == "PhysicsRevoluteJoint"
    ]
    official_link_diff = sorted_diff(official["links"], reference_links)
    nonfixed_joint_diff = sorted_diff(official["non_fixed_joint_names"], reference_revolute_joint_names)
    action_joint_diff = sorted_diff(action_joint_names, reference_revolute_joint_names)
    target_body_diff = sorted_diff(target_bodies, reference_links)

    compatible_for_resource_adjusted_replay = bool(
        payload.get("stage_open_ok")
        and payload.get("articulation_api_count", 0) >= 1
        and len(nonfixed_joint_diff["missing_from_right"]) == 0
        and len(action_joint_diff["missing_from_right"]) == 0
        and len(target_body_diff["missing_from_right"]) == 0
        and anchor_body in reference_links
    )
    checks = {
        "reference_usd_exists": REFERENCE_USD.is_file(),
        "official_urdf_exists": OFFICIAL_URDF.is_file(),
        "kit_probe_reached_payload": probe["markers"]["sentinel_payload"],
        "reference_stage_open_ok": payload.get("stage_open_ok") is True,
        "reference_has_articulation_root": payload.get("articulation_api_count", 0) >= 1,
        "official_nonfixed_joint_count_29": official["non_fixed_joint_count"] == 29,
        "reference_revolute_joint_count_at_least_29": len(reference_revolute_joint_names) >= 29,
        "all_official_nonfixed_joints_in_reference_usd": len(nonfixed_joint_diff["missing_from_right"]) == 0,
        "all_action_joints_in_reference_usd": len(action_joint_diff["missing_from_right"]) == 0,
        "all_target_bodies_in_reference_usd": len(target_body_diff["missing_from_right"]) == 0,
        "anchor_body_in_reference_usd": anchor_body in reference_links,
        "does_not_claim_official_asset": True,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = (
        "ok_with_resource_adjusted_usd_compatible"
        if compatible_for_resource_adjusted_replay
        else "ok_with_reference_usd_incompatible_or_partial"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_reference_usd_compatibility_audit",
        "scope": (
            "Read-only compatibility audit between a non-official ASAP reference G1 USD and the official "
            "BeyondMimic whole_body_tracking G1 URDF/action/target-body contract. No replay, PPO, policy evaluation, "
            "checkpoint, video, or robot execution is performed."
        ),
        "reference_usd": str(REFERENCE_USD),
        "official_urdf": str(OFFICIAL_URDF),
        "compatible_for_resource_adjusted_replay": compatible_for_resource_adjusted_replay,
        "official_contract": {
            "link_count": official["link_count"],
            "non_fixed_joint_count": official["non_fixed_joint_count"],
            "target_bodies": target_bodies,
            "anchor_body": anchor_body,
            "action_joint_count": len(action_joint_names),
        },
        "reference_contract": {
            "stage_open_ok": payload.get("stage_open_ok"),
            "default_prim_path": payload.get("default_prim_path"),
            "link_count": payload.get("link_count"),
            "joint_count": payload.get("joint_count"),
            "revolute_joint_count": len(reference_revolute_joint_names),
            "rigid_body_count": payload.get("rigid_body_count"),
            "articulation_api_count": payload.get("articulation_api_count"),
        },
        "diffs": {
            "official_links_vs_reference_links": official_link_diff,
            "official_nonfixed_joints_vs_reference_revolute_joints": nonfixed_joint_diff,
            "official_action_joints_vs_reference_revolute_joints": action_joint_diff,
            "official_target_bodies_vs_reference_links": target_body_diff,
        },
        "reference_probe": probe,
        "official": official,
        "reference_links": reference_links,
        "reference_joint_names": reference_joint_names,
        "reference_revolute_joint_names": sorted(reference_revolute_joint_names),
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Even if the reference USD is structurally compatible, it is not an official BeyondMimic asset and "
                "does not itself produce a motion.npz, replay video, PPO policy, or paper-level tracking metric."
            ),
            "next_action": (
                "If compatible, attempt a clearly labeled resource-adjusted replay/conversion gate using this USD; "
                "otherwise continue toward an offline converter from the official URDF/MJCF."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_g1_reference_usd_compatibility_audit.json")},
    }
    (OUT / "tracking_g1_reference_usd_compatibility_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "compatible_for_resource_adjusted_replay": compatible_for_resource_adjusted_replay,
                "missing_joints": len(nonfixed_joint_diff["missing_from_right"]),
                "missing_target_bodies": len(target_body_diff["missing_from_right"]),
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
