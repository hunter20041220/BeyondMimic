#!/usr/bin/env python3
"""Probe CUDA peer-access behavior that blocks Isaac Sim Kit close sentinel."""

from __future__ import annotations

import csv
import ctypes
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/cuda_p2p_runtime_probe"


def command_output(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def probe_pair(src: int, dst: int) -> dict[str, Any]:
    cuda = ctypes.CDLL("libcudart.so.12")
    cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
    cuda.cudaGetErrorString.restype = ctypes.c_char_p
    can = ctypes.c_int()
    cuda.cudaDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuda.cudaDeviceCanAccessPeer.restype = ctypes.c_int
    can_rc = int(cuda.cudaDeviceCanAccessPeer(ctypes.byref(can), src, dst))
    cuda.cudaSetDevice.argtypes = [ctypes.c_int]
    cuda.cudaSetDevice.restype = ctypes.c_int
    set_rc = int(cuda.cudaSetDevice(src))
    cuda.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    cuda.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
    enable_rc = int(cuda.cudaDeviceEnablePeerAccess(dst, 0))
    cuda.cudaGetLastError.restype = ctypes.c_int
    last_rc = int(cuda.cudaGetLastError())
    cuda.cudaDeviceDisablePeerAccess.argtypes = [ctypes.c_int]
    cuda.cudaDeviceDisablePeerAccess.restype = ctypes.c_int
    disable_rc = int(cuda.cudaDeviceDisablePeerAccess(dst)) if enable_rc == 0 else None

    def err(rc: int | None) -> str | None:
        if rc is None:
            return None
        raw = cuda.cudaGetErrorString(rc)
        return raw.decode("utf-8", errors="replace") if raw else None

    return {
        "src": src,
        "dst": dst,
        "can_access_rc": can_rc,
        "can_access": int(can.value),
        "set_device_rc": set_rc,
        "enable_peer_access_rc": enable_rc,
        "enable_peer_access_error": err(enable_rc),
        "last_error_after_enable_rc": last_rc,
        "last_error_after_enable": err(last_rc),
        "disable_peer_access_rc": disable_rc,
        "disable_peer_access_error": err(disable_rc),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "src",
        "dst",
        "can_access",
        "can_access_rc",
        "set_device_rc",
        "enable_peer_access_rc",
        "enable_peer_access_error",
        "last_error_after_enable_rc",
        "last_error_after_enable",
        "disable_peer_access_rc",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nvidia_rc, nvidia_out = command_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    pairs = [(6, 6), (6, 7), (7, 6)]
    rows: list[dict[str, Any]] = []
    for src, dst in pairs:
        try:
            rows.append(probe_pair(src, dst))
        except Exception as exc:
            rows.append({"src": src, "dst": dst, "exception": repr(exc)})

    checks = {
        "nvidia_smi_ok": nvidia_rc == 0,
        "records_peer_access_results": bool(rows),
        "has_peer_access_already_enabled_signature": any(
            row.get("enable_peer_access_rc") == 704 or row.get("last_error_after_enable_rc") == 704 for row in rows
        ),
        "does_not_launch_kit_or_training": True,
        "does_not_claim_isaaclab_gate_passed": True,
    }
    summary: dict[str, Any] = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "cuda_p2p_runtime_probe",
        "scope": "ctypes CUDA runtime peer-access probe only; no Isaac Sim Kit launch and no training",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": {"returncode": nvidia_rc, "stdout": nvidia_out},
        "rows": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This only probes CUDA peer-access runtime behavior. IsaacLab AppLauncher, official replay, PPO, "
                "closed-loop diffusion evaluation, and robot deployment remain separate gates."
            ),
        },
        "outputs": {
            "json": str(OUT / "cuda_p2p_runtime_probe.json"),
            "tsv": str(OUT / "cuda_p2p_runtime_probe.tsv"),
        },
    }
    (OUT / "cuda_p2p_runtime_probe.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "cuda_p2p_runtime_probe.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "has_peer_access_already_enabled_signature": checks["has_peer_access_already_enabled_signature"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
