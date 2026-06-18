#!/usr/bin/env python3
"""Author and audit a resource-adjusted enriched G1 USD scaffold from public URDF fields."""

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
OUT = ROOT / "res/tracking/g1_resource_adjusted_enriched_usd"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_enriched_usd"
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
UNITREE_DESCRIPTION_ROOT = OFFICIAL_URDF.parents[2]
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
PHYSICAL_CONTRACT_AUDIT = (
    ROOT / "res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json"
)
SKELETON_USD = ROOT / "res/tracking/g1_official_urdf_skeleton_usd/g1_official_urdf_29dof_skeleton.usda"
ENRICHED_USD = OUT / "g1_resource_adjusted_29dof_enriched_scaffold.usda"
CONTRACT_JSON = OUT / "g1_resource_adjusted_enrichment_contract.json"
TIMEOUT_SECONDS = 180


KIT_ENRICH_CODE = r"""
import json
import math
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
CONTRACT = json.loads(Path(os.environ["BM_G1_ENRICH_CONTRACT_JSON"]).read_text(encoding="utf-8"))
SKELETON_USD = Path(os.environ["BM_G1_SKELETON_USD"])
ENRICHED_USD = Path(os.environ["BM_G1_ENRICHED_USD"])
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
    "skeleton_usd": str(SKELETON_USD),
    "enriched_usd": str(ENRICHED_USD),
    "experience": str(EXPERIENCE),
    "config": config,
}

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    root_path = "/g1_29dof_skeleton"
    stage = Usd.Stage.Open(str(SKELETON_USD))
    if stage is None:
        raise RuntimeError(f"failed to open skeleton USD: {SKELETON_USD}")
    stage.SetMetadata("comment", "Resource-adjusted enrichment scaffold; not official converter output or replay validation.")

    def vec3(values, default=(0.0, 0.0, 0.0)):
        vals = list(values or default)
        vals = (vals + list(default))[:3]
        return Gf.Vec3f(float(vals[0]), float(vals[1]), float(vals[2]))

    def create_attr(prim, name, type_name, value):
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, type_name, custom=True)
        attr.Set(value)

    authored = {
        "mass_api": 0,
        "inertia_attrs": 0,
        "visual_proxy": 0,
        "visual_mesh_reference": 0,
        "collision_proxy": 0,
        "collision_api": 0,
        "joint_axis": 0,
        "joint_limits": 0,
        "joint_drive_metadata": 0,
        "joint_origin_metadata": 0,
    }

    for link in CONTRACT["links"]:
        link_name = link["name"]
        prim = stage.GetPrimAtPath(f"{root_path}/{link_name}")
        if not prim:
            continue
        create_attr(prim, "bm:enrichment:source", Sdf.ValueTypeNames.String, "official_g1_urdf")
        inertial = link.get("inertial")
        if inertial:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(float(inertial["mass"]))
            mass_api.CreateCenterOfMassAttr(vec3(inertial.get("origin", {}).get("xyz")))
            inertia = inertial["inertia"]
            mass_api.CreateDiagonalInertiaAttr(
                Gf.Vec3f(float(inertia["ixx"]), float(inertia["iyy"]), float(inertia["izz"]))
            )
            create_attr(prim, "bm:urdf:inertia:ixy", Sdf.ValueTypeNames.Float, float(inertia["ixy"]))
            create_attr(prim, "bm:urdf:inertia:ixz", Sdf.ValueTypeNames.Float, float(inertia["ixz"]))
            create_attr(prim, "bm:urdf:inertia:iyz", Sdf.ValueTypeNames.Float, float(inertia["iyz"]))
            authored["mass_api"] += 1
            authored["inertia_attrs"] += 1
        for visual in link.get("visuals", []):
            visual_path = f"{root_path}/{link_name}/visual_{visual['index']}"
            xform = UsdGeom.Xform.Define(stage, visual_path)
            xform.AddTranslateOp().Set(vec3(visual.get("origin", {}).get("xyz")))
            prim_v = xform.GetPrim()
            create_attr(prim_v, "bm:urdf:mesh:filename", Sdf.ValueTypeNames.String, visual["mesh_filename"])
            create_attr(prim_v, "bm:urdf:mesh:stl_path", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(visual["stl_path"]))
            create_attr(prim_v, "bm:urdf:visual:proxy", Sdf.ValueTypeNames.Bool, True)
            tmp_usd = visual.get("tmp_usd_path")
            if tmp_usd and Path(tmp_usd).is_file():
                prim_v.GetReferences().AddReference(tmp_usd)
                create_attr(prim_v, "bm:urdf:mesh:tmp_usd_path", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(tmp_usd))
                authored["visual_mesh_reference"] += 1
            authored["visual_proxy"] += 1
        for collision in link.get("collisions", []):
            coll_path = f"{root_path}/{link_name}/collision_{collision['index']}"
            geom = collision.get("geometry", {})
            if geom.get("type") == "sphere":
                prim_c = UsdGeom.Sphere.Define(stage, coll_path).GetPrim()
                prim_c.GetAttribute("radius").Set(float(geom.get("radius", 0.0)))
            elif geom.get("type") == "cylinder":
                prim_c = UsdGeom.Cylinder.Define(stage, coll_path).GetPrim()
                prim_c.GetAttribute("radius").Set(float(geom.get("radius", 0.0)))
                prim_c.GetAttribute("height").Set(float(geom.get("length", 0.0)))
            else:
                prim_c = UsdGeom.Xform.Define(stage, coll_path).GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim_c)
            create_attr(prim_c, "bm:urdf:collision:geometry_type", Sdf.ValueTypeNames.String, geom.get("type", "unknown"))
            create_attr(prim_c, "bm:urdf:collision:proxy", Sdf.ValueTypeNames.Bool, True)
            authored["collision_proxy"] += 1
            authored["collision_api"] += 1

    for joint in CONTRACT["joints"]:
        joint_path = f"{root_path}/{joint['parent']}/{joint['name']}"
        prim = stage.GetPrimAtPath(joint_path)
        if not prim:
            continue
        create_attr(prim, "bm:urdf:origin:xyz", Sdf.ValueTypeNames.Float3, vec3(joint.get("origin", {}).get("xyz")))
        create_attr(prim, "bm:urdf:origin:rpy", Sdf.ValueTypeNames.Float3, vec3(joint.get("origin", {}).get("rpy")))
        authored["joint_origin_metadata"] += 1
        if joint["type"] != "fixed":
            axis = joint.get("axis", [])
            axis_token = "X"
            if len(axis) == 3:
                max_i = max(range(3), key=lambda i: abs(axis[i]))
                axis_token = ["X", "Y", "Z"][max_i]
            attr = prim.GetAttribute("physics:axis") or prim.CreateAttribute("physics:axis", Sdf.ValueTypeNames.Token)
            attr.Set(axis_token)
            create_attr(prim, "bm:urdf:axis:xyz", Sdf.ValueTypeNames.Float3, vec3(axis))
            authored["joint_axis"] += 1
            limit = joint.get("limit", {})
            for name, key in [("physics:lowerLimit", "lower"), ("physics:upperLimit", "upper")]:
                attr_l = prim.GetAttribute(name) or prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
                # UsdPhysics revolute joint limits are authored in degrees; URDF stores radians.
                attr_l.Set(math.degrees(float(limit[key])))
            create_attr(prim, "bm:urdf:limit:effort", Sdf.ValueTypeNames.Float, float(limit["effort"]))
            create_attr(prim, "bm:urdf:limit:velocity", Sdf.ValueTypeNames.Float, float(limit["velocity"]))
            create_attr(prim, "bm:urdf:limit:lower_rad", Sdf.ValueTypeNames.Float, float(limit["lower"]))
            create_attr(prim, "bm:urdf:limit:upper_rad", Sdf.ValueTypeNames.Float, float(limit["upper"]))
            authored["joint_limits"] += 1
            drive = joint.get("drive", {})
            if drive:
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive_api.CreateStiffnessAttr(float(drive["stiffness"]))
                drive_api.CreateDampingAttr(float(drive["damping"]))
                drive_api.CreateMaxForceAttr(float(drive["effort_limit_sim"]))
                create_attr(prim, "bm:drive:armature", Sdf.ValueTypeNames.Float, float(drive["armature"]))
                create_attr(prim, "bm:drive:action_scale", Sdf.ValueTypeNames.Float, float(drive["action_scale"]))
                create_attr(prim, "bm:drive:actuator_group", Sdf.ValueTypeNames.String, drive["actuator_group"])
                authored["joint_drive_metadata"] += 1

    export_ok = bool(stage.GetRootLayer().Export(str(ENRICHED_USD)))
    payload["export_ok"] = export_ok
    payload["authored"] = authored

    opened = Usd.Stage.Open(str(ENRICHED_USD)) if ENRICHED_USD.exists() else None
    payload["reopen_ok"] = opened is not None
    if opened is not None:
        readback = {
            "prim_count": 0,
            "link_count": 0,
            "mass_api_count": 0,
            "visual_proxy_count": 0,
            "visual_mesh_reference_count": 0,
            "collision_proxy_count": 0,
            "collision_api_count": 0,
            "revolute_joint_count": 0,
            "joint_limit_count": 0,
            "joint_drive_metadata_count": 0,
            "joint_origin_metadata_count": 0,
            "articulation_api_count": 0,
        }
        for prim in opened.Traverse():
            readback["prim_count"] += 1
            path_s = str(prim.GetPath())
            if path_s.startswith(root_path + "/") and path_s.count("/") == 2:
                readback["link_count"] += 1
            if prim.HasAPI(UsdPhysics.MassAPI):
                readback["mass_api_count"] += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                readback["collision_api_count"] += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                readback["articulation_api_count"] += 1
            if prim.GetAttribute("bm:urdf:visual:proxy"):
                readback["visual_proxy_count"] += 1
            if prim.GetAttribute("bm:urdf:mesh:tmp_usd_path"):
                readback["visual_mesh_reference_count"] += 1
            if prim.GetAttribute("bm:urdf:collision:proxy"):
                readback["collision_proxy_count"] += 1
            if prim.GetTypeName() == "PhysicsRevoluteJoint":
                readback["revolute_joint_count"] += 1
            if prim.GetAttribute("physics:lowerLimit") and prim.GetAttribute("physics:upperLimit"):
                readback["joint_limit_count"] += 1
            if prim.GetAttribute("bm:drive:action_scale"):
                readback["joint_drive_metadata_count"] += 1
            if prim.GetAttribute("bm:urdf:origin:xyz"):
                readback["joint_origin_metadata_count"] += 1
        payload["readback"] = readback
except Exception as exc:
    payload["exception"] = repr(exc)

print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
app.close()
print("BM_SENTINEL:after_close", flush=True)
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(x) for x in text.split()]


def parse_origin(element: ET.Element | None) -> dict[str, list[float]]:
    if element is None:
        return {"xyz": [], "rpy": []}
    return {"xyz": floats(element.attrib.get("xyz")), "rpy": floats(element.attrib.get("rpy"))}


def resolve_mesh(filename: str) -> Path:
    prefix = "package://unitree_description/"
    if filename.startswith(prefix):
        return UNITREE_DESCRIPTION_ROOT / filename[len(prefix) :]
    return (OFFICIAL_URDF.parent / filename).resolve()


def parse_geometry(geometry: ET.Element | None) -> dict[str, Any]:
    if geometry is None:
        return {"type": "missing"}
    sphere = geometry.find("sphere")
    if sphere is not None:
        return {"type": "sphere", "radius": float(sphere.attrib["radius"])}
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return {
            "type": "cylinder",
            "radius": float(cylinder.attrib["radius"]),
            "length": float(cylinder.attrib["length"]),
        }
    mesh = geometry.find("mesh")
    if mesh is not None:
        return {"type": "mesh", "filename": mesh.attrib.get("filename", "")}
    return {"type": "unknown"}


def parse_urdf_contract() -> dict[str, Any]:
    action_scale = load_json(ACTION_SCALE_AUDIT)
    action_by_name = {row["joint_name"]: row for row in action_scale["joint_rows"]}
    root = ET.parse(OFFICIAL_URDF).getroot()
    links: list[dict[str, Any]] = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        inertial_row = None
        if inertial is not None:
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            if mass is not None and inertia is not None:
                inertial_row = {
                    "origin": parse_origin(inertial.find("origin")),
                    "mass": float(mass.attrib["value"]),
                    "inertia": {key: float(inertia.attrib[key]) for key in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]},
                }
        visuals = []
        for idx, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            filename = mesh.attrib.get("filename") if mesh is not None else ""
            stl_path = resolve_mesh(filename) if filename else None
            tmp_usd = stl_path.with_suffix(".tmp.usd") if stl_path is not None else None
            visuals.append(
                {
                    "index": idx,
                    "mesh_filename": filename,
                    "stl_path": str(stl_path) if stl_path else "",
                    "tmp_usd_path": str(tmp_usd) if tmp_usd and tmp_usd.is_file() else "",
                    "origin": parse_origin(visual.find("origin")),
                }
            )
        collisions = []
        for idx, collision in enumerate(link.findall("collision")):
            collisions.append(
                {
                    "index": idx,
                    "origin": parse_origin(collision.find("origin")),
                    "geometry": parse_geometry(collision.find("geometry")),
                }
            )
        links.append({"name": link.attrib["name"], "inertial": inertial_row, "visuals": visuals, "collisions": collisions})

    joints = []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        limit = joint.find("limit")
        action = action_by_name.get(name)
        joints.append(
            {
                "name": name,
                "type": joint.attrib.get("type"),
                "parent": joint.find("parent").attrib.get("link") if joint.find("parent") is not None else "",
                "child": joint.find("child").attrib.get("link") if joint.find("child") is not None else "",
                "origin": parse_origin(joint.find("origin")),
                "axis": floats(joint.find("axis").attrib.get("xyz")) if joint.find("axis") is not None else [],
                "limit": {
                    "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
                    "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
                    "effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
                    "velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
                },
                "drive": {
                    key: action[key]
                    for key in [
                        "actuator_group",
                        "stiffness",
                        "damping",
                        "armature",
                        "action_scale",
                        "effort_limit_sim",
                        "velocity_limit_sim",
                    ]
                }
                if action
                else None,
            }
        )
    return {"links": links, "joints": joints}


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BM_G1_ENRICH_CONTRACT_JSON": str(CONTRACT_JSON),
            "BM_G1_SKELETON_USD": str(SKELETON_USD),
            "BM_G1_ENRICHED_USD": str(ENRICHED_USD),
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


def normalize_usda_text(path: Path) -> None:
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_enrichment() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(KIT_ENRICH_CODE)],
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
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_enriched_usd_probe.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    return {
        "returncode": returncode,
        "log": str(log_path),
        "markers": classify_output(output, timed_out),
        "payload": extract_payload(output),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    physical_contract = load_json(PHYSICAL_CONTRACT_AUDIT)
    contract = parse_urdf_contract()
    CONTRACT_JSON.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    probe = run_enrichment()
    normalize_usda_text(ENRICHED_USD)
    payload = probe.get("payload", {})
    readback = payload.get("readback", {})
    authored = payload.get("authored", {})
    checks = {
        "skeleton_usd_exists": SKELETON_USD.is_file(),
        "physical_contract_ready": physical_contract.get("checks", {}).get("ready_for_offline_converter_scaffold") is True,
        "kit_probe_reached_payload": probe["markers"]["sentinel_payload"],
        "kit_probe_closed": probe["markers"]["sentinel_after_close"],
        "export_ok": payload.get("export_ok") is True,
        "reopen_ok": payload.get("reopen_ok") is True,
        "enriched_usd_written": ENRICHED_USD.is_file() and ENRICHED_USD.stat().st_size > 0,
        "readback_link_count_40": readback.get("link_count") == 40,
        "readback_mass_api_count_37": readback.get("mass_api_count") == 37,
        "readback_visual_proxy_count_35": readback.get("visual_proxy_count") == 35,
        "readback_visual_mesh_reference_count_35": readback.get("visual_mesh_reference_count") == 35,
        "readback_collision_proxy_count_29": readback.get("collision_proxy_count") == 29,
        "readback_collision_api_count_29": readback.get("collision_api_count") == 29,
        "readback_revolute_joint_count_29": readback.get("revolute_joint_count") == 29,
        "readback_joint_limit_count_29": readback.get("joint_limit_count") == 29,
        "readback_joint_drive_metadata_count_29": readback.get("joint_drive_metadata_count") == 29,
        "readback_articulation_root_present": readback.get("articulation_api_count") == 1,
        "does_not_claim_official_converter_success": True,
        "does_not_claim_motion_npz": True,
        "does_not_claim_replay_success": True,
        "does_not_start_training": True,
    }
    expected_ok = all(
        checks[key]
        for key in [
            "physical_contract_ready",
            "kit_probe_reached_payload",
            "kit_probe_closed",
            "export_ok",
            "reopen_ok",
            "enriched_usd_written",
            "readback_link_count_40",
            "readback_mass_api_count_37",
            "readback_visual_proxy_count_35",
            "readback_visual_mesh_reference_count_35",
            "readback_collision_proxy_count_29",
            "readback_collision_api_count_29",
            "readback_revolute_joint_count_29",
            "readback_joint_limit_count_29",
            "readback_joint_drive_metadata_count_29",
            "readback_articulation_root_present",
            "does_not_claim_replay_success",
            "does_not_start_training",
        ]
    )
    status = "ok_with_resource_adjusted_enriched_usd_scaffold" if expected_ok else "ok_with_enriched_usd_partial"
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_resource_adjusted_enriched_usd_probe",
        "scope": (
            "Authors a resource-adjusted enriched USD scaffold by adding public URDF physical metadata/proxies to the "
            "minimal 29-DoF skeleton USD. This is not official IsaacLab URDF converter output and does not claim "
            "motion.npz, replay, PPO, policy evaluation, video, or robot success."
        ),
        "sources": {
            "skeleton_usd": str(SKELETON_USD),
            "official_urdf": str(OFFICIAL_URDF),
            "physical_contract_audit": str(PHYSICAL_CONTRACT_AUDIT),
            "action_scale_audit": str(ACTION_SCALE_AUDIT),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_enriched_usd_probe.json"),
            "contract_json": str(CONTRACT_JSON),
            "enriched_usd": str(ENRICHED_USD),
        },
        "authored": authored,
        "readback": readback,
        "probe": probe,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The enriched USD scaffold records public URDF physical metadata and proxy geometry, but it has not "
                "been accepted by official csv_to_npz.py/replay_npz.py and is not proven to be a physically faithful "
                "IsaacLab articulation asset."
            ),
            "next_action": (
                "Run a bounded official csv_to_npz/replay preflight against the enriched scaffold or refine the USD "
                "authoring until the official preprocessing path accepts it."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_enriched_usd_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "json": summary["outputs"]["json"],
                "enriched_usd": summary["outputs"]["enriched_usd"],
                "readback": readback,
            },
            sort_keys=True,
        )
    )
    if status == "ok_with_enriched_usd_partial":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
