#!/usr/bin/env python3
"""Non-Kit smoke checks for the official whole_body_tracking extension."""

from __future__ import annotations

import importlib.metadata as metadata
import re
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PKG_ROOT = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
)


def require(path: Path) -> None:
    print(f"{path.relative_to(ROOT)}\texists={path.exists()}\tsize={path.stat().st_size if path.exists() else 'NA'}")
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    print("whole_body_tracking version:", metadata.version("whole_body_tracking"))
    urdf = PKG_ROOT / "assets/unitree_description/urdf/g1/main.urdf"
    require(PKG_ROOT / "__init__.py")
    require(PKG_ROOT / "robots/g1.py")
    require(PKG_ROOT / "tasks/tracking/tracking_env_cfg.py")
    require(PKG_ROOT / "tasks/tracking/mdp/commands.py")
    require(PKG_ROOT / "tasks/tracking/mdp/rewards.py")
    require(urdf)
    mesh_refs = re.findall(r'filename="package://unitree_description/([^"]+)"', urdf.read_text())
    if not mesh_refs:
        raise RuntimeError("No package://unitree_description mesh references found in G1 URDF")
    first_mesh = PKG_ROOT / "assets/unitree_description" / mesh_refs[0]
    print("first URDF mesh reference:", mesh_refs[0])
    require(first_mesh)
    print("non-kit package and asset smoke: OK")


if __name__ == "__main__":
    main()
