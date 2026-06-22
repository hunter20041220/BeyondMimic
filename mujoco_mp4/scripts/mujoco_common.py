#!/usr/bin/env python3
"""Shared helpers for the independent MuJoCo MP4 experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).expanduser().resolve()
PKG = ROOT / "mujoco_mp4"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def traceback_payload(exc: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
    }


def minimal_scene_xml() -> str:
    return """
<mujoco model="bm_mujoco_minimal_scene">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="960" offheight="540"/>
    <quality shadowsize="1024"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1="0.22 0.24 0.26" rgb2="0.38 0.40 0.42"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" reflectance="0.15"/>
    <material name="blue" rgba="0.1 0.35 0.9 1"/>
    <material name="red" rgba="0.9 0.2 0.12 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -3 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="track" pos="3 -5 2.2" xyaxes="1 0 0 0 0.36 0.93" fovy="45"/>
    <geom name="floor" type="plane" size="8 8 0.1" material="grid_mat"/>
    <body name="box_body" pos="0 0 1.0">
      <joint name="box_free" type="free"/>
      <geom name="box" type="box" size="0.25 0.25 0.25" material="blue" mass="1.0"/>
    </body>
    <body name="pendulum" pos="1.0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.08"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.8" size="0.04" material="red" mass="0.3"/>
      <geom name="bob" type="sphere" pos="0 0 -0.85" size="0.12" material="red" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def render_frame(model: Any, data: Any, renderer: Any, camera: str | int | None = "track") -> Any:
    renderer.update_scene(data, camera=camera)
    return renderer.render()
