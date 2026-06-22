#!/usr/bin/env python3
"""Render a minimal MuJoCo offscreen MP4 smoke test."""

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

from mujoco_common import PKG, minimal_scene_xml, render_frame, traceback_payload, utc_now, write_json


def main() -> None:
    backend = os.environ.get("MUJOCO_GL", "egl")
    frames = int(os.environ.get("BM_MUJOCO_SMOKE_FRAMES", "180"))
    width = int(os.environ.get("BM_MUJOCO_WIDTH", "960"))
    height = int(os.environ.get("BM_MUJOCO_HEIGHT", "540"))
    fps = int(os.environ.get("BM_MUJOCO_FPS", "30"))
    out_dir = PKG / "res/smoke"
    log_dir = PKG / "logs/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"minimal_scene_{backend}.mp4"
    keyframe_path = out_dir / f"minimal_scene_{backend}_keyframe.png"
    summary_path = out_dir / f"minimal_scene_{backend}_summary.json"

    summary: dict[str, object] = {
        "status": "started",
        "timestamp_utc": utc_now(),
        "backend": backend,
        "claim_level": "MuJoCo minimal offscreen rendering smoke; not robot, not policy, not paper-level result",
        "frames_requested": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "outputs": {
            "mp4": str(mp4_path),
            "keyframe_png": str(keyframe_path),
            "summary_json": str(summary_path),
        },
        "checks": {},
    }
    try:
        import mujoco

        summary["mujoco_version"] = mujoco.__version__
        model = mujoco.MjModel.from_xml_string(minimal_scene_xml())
        data = mujoco.MjData(model)
        data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        if model.nq > 7:
            data.qpos[7] = 0.55
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        frame0 = render_frame(model, data, renderer)
        imageio.imwrite(keyframe_path, frame0)
        with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for step in range(frames):
                if model.nu:
                    data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)
                frame = render_frame(model, data, renderer)
                writer.append_data(frame)
        renderer.close()
        summary["status"] = "ok"
        summary["frames_written"] = frames
        summary["checks"] = {
            "import_mujoco": True,
            "model_created": True,
            "data_created": True,
            "renderer_created": True,
            "rgb_frame_rendered": True,
            "keyframe_png_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
        }
        summary["file_sizes"] = {
            "mp4": mp4_path.stat().st_size if mp4_path.exists() else 0,
            "keyframe_png": keyframe_path.stat().st_size if keyframe_path.exists() else 0,
        }
    except Exception as exc:  # noqa: BLE001 - smoke should retain failure detail.
        summary["status"] = "failed"
        summary["error"] = traceback_payload(exc)
    write_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "backend": backend, "summary": str(summary_path), "mp4": str(mp4_path)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
