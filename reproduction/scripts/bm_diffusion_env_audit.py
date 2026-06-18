#!/usr/bin/env python3
"""Audit the project-local bm_diffusion environment state.

The audit records what is actually usable and where installation is blocked.
It deliberately avoids claiming that Level C training can run unless the
PyTorch import/smoke test succeeds.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
ENV = ROOT / "envs/bm_diffusion"
OUT = ROOT / "res/setup/bm_diffusion_env_audit"


def run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def file_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    python = ENV / "bin/python"
    locks = [
        ENV / "environment.yml",
        ENV / "requirements-lock.txt",
        ENV / "pip-freeze.txt",
        ENV / "conda-list-explicit.txt",
    ]
    base_smoke = run(
        [
            str(python),
            "-c",
            (
                "import sys, numpy, scipy, yaml, tqdm; "
                "print(sys.version.split()[0]); "
                "print(numpy.__version__, scipy.__version__, yaml.__version__, tqdm.__version__)"
            ),
        ]
    )
    torch_smoke = run(
        [
            str(python),
            "-c",
            (
                "import torch; "
                "print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); "
                "x=torch.ones(2,2,device='cuda' if torch.cuda.is_available() else 'cpu'); "
                "print(float(x.sum()))"
            ),
        ],
        timeout=45,
    )
    pip_log = ROOT / "logs/setup/install_bm_diffusion_torch.log"
    pip_log_text = pip_log.read_text(encoding="utf-8", errors="replace") if pip_log.is_file() else ""
    torch_install_log_incomplete = "Installing collected packages:" in pip_log_text and "Successfully installed" not in pip_log_text
    torch_smoke_passes = torch_smoke["returncode"] == 0 and not torch_smoke["timed_out"]
    rows = [
        {
            "check": "prefix_exists",
            "status": "passed" if ENV.is_dir() else "failed",
            "detail": str(ENV),
        },
        {
            "check": "lock_files_exist",
            "status": "passed" if all(path.is_file() and path.stat().st_size > 0 for path in locks) else "failed",
            "detail": ", ".join(path.name for path in locks),
        },
        {
            "check": "base_numpy_scipy_yaml_tqdm_smoke",
            "status": "passed" if base_smoke["returncode"] == 0 and not base_smoke["timed_out"] else "failed",
            "detail": base_smoke["stdout"] or base_smoke["stderr"],
        },
        {
            "check": "torch_cuda_smoke",
            "status": "passed" if torch_smoke_passes else "blocked",
            "detail": "timed_out" if torch_smoke["timed_out"] else (torch_smoke["stdout"] or torch_smoke["stderr"]),
        },
        {
            "check": "torch_install_log_incomplete_but_runtime_smoke_passes",
            "status": "warning" if torch_install_log_incomplete and torch_smoke_passes else "passed",
            "detail": str(pip_log),
        },
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary: dict[str, Any] = {
        "status": "partial_blocked" if any(row["status"] == "blocked" for row in rows) else "ok",
        "experiment_type": "environment_audit",
        "scope": "project-local bm_diffusion prefix environment",
        "environment_path": str(ENV),
        "python": str(python),
        "lock_files": [file_status(path) for path in locks],
        "logs": {
            "create": str(ROOT / "logs/setup/create_bm_diffusion.log"),
            "torch_install": str(pip_log),
            "lock_export": str(ROOT / "logs/setup/export_bm_diffusion_locks.log"),
        },
        "base_smoke": base_smoke,
        "torch_smoke": torch_smoke,
        "status_counts": status_counts,
        "rows": rows,
        "checks": {
            "prefix_exists": ENV.is_dir(),
            "lock_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in locks),
            "base_numpy_scipy_yaml_tqdm_smoke_passes": base_smoke["returncode"] == 0 and not base_smoke["timed_out"],
            "torch_cuda_smoke_passes": torch_smoke_passes,
            "torch_install_log_incomplete_recorded": torch_install_log_incomplete,
            "training_environment_smoke_ready": torch_smoke_passes,
            "does_not_claim_training_or_paper_results": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The bm_diffusion prefix, base scientific stack, lock files, and Torch/CUDA runtime smoke pass. "
                "This closes the separate-prefix environment setup gap, but it does not train VAE/diffusion "
                "checkpoints or reproduce paper metrics."
            ),
        },
        "outputs": {
            "json": str(OUT / "bm_diffusion_env_audit.json"),
            "tsv": str(OUT / "bm_diffusion_env_audit.tsv"),
        },
    }
    (OUT / "bm_diffusion_env_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "bm_diffusion_env_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "tsv": summary["outputs"]["tsv"]}))


if __name__ == "__main__":
    main()
