#!/usr/bin/env python3
"""Load G1 MuJoCo assets and render a short offscreen smoke video."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, ROOT, render_frame, traceback_payload, utc_now, write_json, write_tsv


DEFAULT_CANDIDATES = [
    "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml",
    "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof_with_hands.xml",
    "mujoco_mp4/assets/work_g1/pbhc_g1/g1_29dof_rev_1_0.xml",
    "mujoco_mp4/assets/work_g1/unitree_rl_mjlab_unitree_g1/xmls/scene_g1.xml",
    "mujoco_mp4/assets/work_g1/unitree_rl_mjlab_unitree_g1/xmls/g1.xml",
    "download/reference_code/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
    "download/reference_code/PBHC/description/robots/g1/g1_29dof_rev_1_0.xml",
    "download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml",
]


def candidate_paths() -> list[Path]:
    override = os.environ.get("BM_MUJOCO_G1_XML", "").strip()
    if override:
        return [Path(override).expanduser()]
    return [ROOT / rel for rel in DEFAULT_CANDIDATES]


def render_g1(model_path: Path, backend: str) -> dict[str, object]:
    import mujoco

    frames = int(os.environ.get("BM_MUJOCO_G1_FRAMES", "120"))
    width = int(os.environ.get("BM_MUJOCO_WIDTH", "1280"))
    height = int(os.environ.get("BM_MUJOCO_HEIGHT", "720"))
    fps = int(os.environ.get("BM_MUJOCO_FPS", "30"))
    out_dir = PKG / "res/g1_import"
    stem = model_path.stem.replace(" ", "_")
    mp4_path = out_dir / f"{stem}_{backend}_g1_import_smoke.mp4"
    keyframe_path = out_dir / f"{stem}_{backend}_g1_import_keyframe.png"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nq >= 7:
        data.qpos[2] = max(float(data.qpos[2]), 0.85)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = -1
    frame0 = render_frame(model, data, renderer, camera=camera)
    imageio.imwrite(keyframe_path, frame0)
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for _ in range(frames):
            if model.nu:
                data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            writer.append_data(render_frame(model, data, renderer, camera=camera))
    renderer.close()
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}" for i in range(model.njnt)]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)]
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}" for i in range(model.nu)
    ]
    return {
        "status": "ok",
        "source_xml": str(model_path),
        "backend": backend,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "ngeom": model.ngeom,
        "nmesh": model.nmesh,
        "joint_names": joint_names,
        "body_names_sample": body_names[:80],
        "actuator_names": actuator_names,
        "frames_written": frames,
        "mp4_path": str(mp4_path),
        "keyframe_path": str(keyframe_path),
        "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
        "keyframe_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
        "file_sizes": {
            "mp4": mp4_path.stat().st_size if mp4_path.exists() else 0,
            "keyframe_png": keyframe_path.stat().st_size if keyframe_path.exists() else 0,
        },
    }


def main() -> None:
    backend = os.environ.get("MUJOCO_GL", "egl")
    out_dir = PKG / "res/g1_import"
    log_dir = PKG / "logs/g1_import"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for path in candidate_paths():
        row: dict[str, object] = {"source_xml": str(path), "exists": path.is_file(), "backend": backend}
        if not path.is_file():
            row.update({"status": "missing", "error": "file_missing"})
            rows.append(row)
            continue
        try:
            result = render_g1(path, backend)
            row.update(result)
            rows.append(row)
            selected = result
            break
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "failed", "error": traceback_payload(exc)})
            rows.append(row)

    payload = {
        "status": "ok" if selected else "failed",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_g1_import_smoke",
        "claim_level": "MuJoCo G1 asset import/render smoke; not policy, not IsaacLab, not real robot",
        "backend": backend,
        "selected": selected or {},
        "rows": rows,
        "checks": {
            "any_g1_model_loaded": selected is not None,
            "mp4_exists": bool(selected and selected.get("mp4_exists")),
            "keyframe_exists": bool(selected and selected.get("keyframe_exists")),
            "does_not_claim_policy_rollout": True,
            "does_not_claim_isaaclab": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(out_dir / "mujoco_g1_import_smoke.json", payload)
    write_tsv(
        out_dir / "mujoco_g1_import_smoke.tsv",
        rows,
        ["source_xml", "exists", "backend", "status", "nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "mp4_path", "keyframe_path", "error"],
    )
    print(json.dumps({"status": payload["status"], "backend": backend, "selected": selected.get("source_xml") if selected else ""}))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
