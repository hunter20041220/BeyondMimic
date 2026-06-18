#!/usr/bin/env python3
"""Build and audit a minimal 29-DoF G1 skeleton USD from the official URDF contract."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_official_urdf_skeleton_usd"
LOG_DIR = ROOT / "logs/tracking_g1_official_urdf_skeleton_usd"
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
OFFICIAL_SOURCE_AUDIT = ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
USD_OUT = OUT / "g1_official_urdf_29dof_skeleton.usda"
TIMEOUT_SECONDS = 120


KIT_BUILD_CODE = r"""
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
URDF_JSON = json.loads(os.environ["BM_G1_URDF_CONTRACT_JSON"])
USD_OUT = Path(os.environ["BM_G1_SKELETON_USD_OUT"])
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

payload = {"usd_out": str(USD_OUT), "experience": str(EXPERIENCE), "config": config}

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory("g1_official_urdf_29dof_skeleton.usda")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/g1_29dof_skeleton")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/g1_29dof_skeleton"))

    links = URDF_JSON["links"]
    joints = URDF_JSON["joints"]
    root_name = "g1_29dof_skeleton"
    for index, link_name in enumerate(links):
        prim_path = f"/{root_name}/{link_name}"
        xform = UsdGeom.Xform.Define(stage, prim_path)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        # Spread placeholder links slightly along x so authored transforms are non-degenerate.
        xform.AddTranslateOp().Set(Gf.Vec3d(index * 0.01, 0.0, 0.0))

    authored_joints = []
    for joint in joints:
        joint_path = f"/{root_name}/{joint['parent']}/{joint['name']}"
        if joint["type"] == "fixed":
            joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)
        else:
            joint_prim = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint_prim.CreateAxisAttr("X")
        joint_prim.CreateBody0Rel().SetTargets([Sdf.Path(f"/{root_name}/{joint['parent']}")])
        joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(f"/{root_name}/{joint['child']}")])
        authored_joints.append({"name": joint["name"], "path": joint_path, "type": joint["type"]})

    export_ok = bool(stage.Export(str(USD_OUT)))
    payload["export_ok"] = export_ok
    payload["authored_joint_count"] = len(authored_joints)
    payload["authored_nonfixed_joint_count"] = sum(1 for row in authored_joints if row["type"] != "fixed")
    payload["authored_link_count"] = len(links)

    opened = Usd.Stage.Open(str(USD_OUT)) if USD_OUT.exists() else None
    payload["reopen_ok"] = opened is not None
    if opened is not None:
        link_names = []
        revolute_joint_names = []
        fixed_joint_names = []
        rigid_body_count = 0
        articulation_api_count = 0
        prim_count = 0
        for prim in opened.Traverse():
            prim_count += 1
            path_s = str(prim.GetPath())
            if path_s.startswith(f"/{root_name}/") and path_s.count("/") == 2:
                link_names.append(prim.GetName())
            if prim.GetTypeName() == "PhysicsRevoluteJoint":
                revolute_joint_names.append(prim.GetName())
            if prim.GetTypeName() == "PhysicsFixedJoint":
                fixed_joint_names.append(prim.GetName())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
                rigid_body_count += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_api_count += 1
        payload.update(
            {
                "prim_count": prim_count,
                "link_names": sorted(set(link_names)),
                "link_count": len(set(link_names)),
                "revolute_joint_names": sorted(set(revolute_joint_names)),
                "revolute_joint_count": len(set(revolute_joint_names)),
                "fixed_joint_names": sorted(set(fixed_joint_names)),
                "fixed_joint_count": len(set(fixed_joint_names)),
                "rigid_body_count": rigid_body_count,
                "articulation_api_count": articulation_api_count,
                "default_prim_path": str(opened.GetDefaultPrim().GetPath()) if opened.GetDefaultPrim() else None,
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
        "joints": joints,
        "non_fixed_joint_names": [row["name"] for row in joints if row["type"] != "fixed"],
        "fixed_joint_names": [row["name"] for row in joints if row["type"] == "fixed"],
    }


def base_env(contract: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BM_G1_URDF_CONTRACT_JSON": json.dumps(contract, sort_keys=True),
            "BM_G1_SKELETON_USD_OUT": str(USD_OUT),
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


def run_builder(contract: dict[str, Any]) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(KIT_BUILD_CODE)],
            cwd=ROOT,
            env=base_env(contract),
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
    log_path = LOG_DIR / "tracking_g1_official_urdf_skeleton_usd_audit.log"
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


def normalize_usda_text(path: Path) -> None:
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    official = parse_official_urdf()
    official_source = load_json(OFFICIAL_SOURCE_AUDIT)
    action_scale = load_json(ACTION_SCALE_AUDIT)
    probe = run_builder(official)
    normalize_usda_text(USD_OUT)
    payload = probe.get("payload", {})
    target_bodies = official_source["flat_env"]["body_names"]
    action_joint_names = [row["joint_name"] for row in action_scale["joint_rows"]]
    link_diff = sorted_diff(official["links"], payload.get("link_names", []))
    action_joint_diff = sorted_diff(action_joint_names, payload.get("revolute_joint_names", []))
    nonfixed_joint_diff = sorted_diff(official["non_fixed_joint_names"], payload.get("revolute_joint_names", []))
    target_body_diff = sorted_diff(target_bodies, payload.get("link_names", []))
    skeleton_contract_ok = bool(
        payload.get("export_ok")
        and payload.get("reopen_ok")
        and len(link_diff["missing_from_right"]) == 0
        and len(nonfixed_joint_diff["missing_from_right"]) == 0
        and len(action_joint_diff["missing_from_right"]) == 0
        and len(target_body_diff["missing_from_right"]) == 0
        and payload.get("articulation_api_count", 0) >= 1
    )
    checks = {
        "official_urdf_exists": OFFICIAL_URDF.is_file(),
        "usd_export_written": USD_OUT.is_file() and USD_OUT.stat().st_size > 0,
        "kit_probe_reached_payload": probe["markers"]["sentinel_payload"],
        "skeleton_export_ok": payload.get("export_ok") is True,
        "skeleton_reopen_ok": payload.get("reopen_ok") is True,
        "official_link_count_40": len(official["links"]) == 40,
        "skeleton_link_count_40": payload.get("link_count") == 40,
        "official_nonfixed_joint_count_29": len(official["non_fixed_joint_names"]) == 29,
        "skeleton_revolute_joint_count_29": payload.get("revolute_joint_count") == 29,
        "all_action_joints_revolute_in_skeleton": len(action_joint_diff["missing_from_right"]) == 0,
        "all_target_bodies_in_skeleton": len(target_body_diff["missing_from_right"]) == 0,
        "has_articulation_root": payload.get("articulation_api_count", 0) >= 1,
        "skeleton_contract_ok": skeleton_contract_ok,
        "does_not_claim_official_converter_success": True,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = "ok_with_minimal_29dof_skeleton_usd" if skeleton_contract_ok else "ok_with_skeleton_usd_partial"
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_urdf_skeleton_usd_audit",
        "scope": (
            "Builds a minimal placeholder skeleton USD from the official G1 URDF link/joint names using "
            "Usd.Stage.Export. It is a structure/contract probe only: no official IsaacLab URDF converter success, "
            "mesh/inertia/collision fidelity, motion.npz, replay, PPO, policy evaluation, video, or robot result is "
            "claimed."
        ),
        "usd_path": str(USD_OUT),
        "official_urdf": str(OFFICIAL_URDF),
        "skeleton_contract_ok": skeleton_contract_ok,
        "official_contract": {
            "link_count": len(official["links"]),
            "non_fixed_joint_count": len(official["non_fixed_joint_names"]),
            "fixed_joint_count": len(official["fixed_joint_names"]),
            "target_body_count": len(target_bodies),
            "action_joint_count": len(action_joint_names),
        },
        "skeleton_contract": {
            "link_count": payload.get("link_count"),
            "revolute_joint_count": payload.get("revolute_joint_count"),
            "fixed_joint_count": payload.get("fixed_joint_count"),
            "rigid_body_count": payload.get("rigid_body_count"),
            "articulation_api_count": payload.get("articulation_api_count"),
            "default_prim_path": payload.get("default_prim_path"),
            "usd_size_bytes": USD_OUT.stat().st_size if USD_OUT.exists() else None,
        },
        "diffs": {
            "official_links_vs_skeleton_links": link_diff,
            "official_nonfixed_joints_vs_skeleton_revolute_joints": nonfixed_joint_diff,
            "official_action_joints_vs_skeleton_revolute_joints": action_joint_diff,
            "official_target_bodies_vs_skeleton_links": target_body_diff,
        },
        "probe": probe,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This skeleton USD only proves that the official 29-DoF naming/action contract can be authored into "
                "a local USD stage without the crashing URDF importer. It is not a physically faithful robot asset and "
                "cannot be used for paper-level replay/training claims without mesh, collision, inertia, drive, and "
                "csv_to_npz/replay validation."
            ),
            "next_action": (
                "Use this skeleton as a scaffolding check for an offline converter that adds official meshes, "
                "collisions, inertias, and drives, then rerun the replay conversion gate."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_urdf_skeleton_usd_audit.json"),
            "usd": str(USD_OUT),
        },
    }
    (OUT / "tracking_g1_official_urdf_skeleton_usd_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "skeleton_contract_ok": skeleton_contract_ok,
                "usd": str(USD_OUT),
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
