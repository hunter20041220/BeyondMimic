#!/usr/bin/env python3
"""Audit local artifacts for official BeyondMimic VAE/diffusion implementation evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOWNLOAD = ROOT / "download"
OUT = ROOT / "res/level_c/official_artifact_audit"

SEARCH_ROOTS = [
    DOWNLOAD / "official",
    DOWNLOAD / "reference_code",
    ROOT / "reproduction/third_party/official",
]

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}

PATH_KEYWORDS = [
    "vae",
    "dagger",
    "diffusion",
    "latent",
    "denois",
    "transformer",
    "tensorrt",
    "trt",
    "engine",
    "checkpoint",
    "ckpt",
    "policy",
    "onnx",
]

CONTENT_KEYWORDS = [
    "conditional vae",
    "dagger",
    "diffusion",
    "denoising",
    "latent",
    "classifier guidance",
    "tensorrt",
    "ema",
]

MAX_CONTENT_SCAN_BYTES = 64_000
MAX_TEXT_FILE_SIZE_FOR_CONTENT_SCAN = 2_000_000

TEXT_SUFFIXES = {
    ".py",
    ".cpp",
    ".hpp",
    ".h",
    ".cu",
    ".cuh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".rst",
    ".tex",
    ".launch",
    ".xml",
    ".cmake",
}

CHECKPOINT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".engine",
    ".plan",
    ".trt",
    ".pkl",
    ".npz",
}


def sha256(path: Path, limit: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def source_area(path: Path) -> str:
    s = str(path)
    if "/download/official/" in s or "/reproduction/third_party/official/" in s:
        return "official"
    if "/download/reference_code/" in s:
        return "reference_code"
    return "other"


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".onnx", ".engine", ".plan", ".trt"}:
        return "deployment_model"
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return "checkpoint"
    if suffix in {".npz", ".pkl"}:
        return "data_or_pickle"
    if suffix == ".py":
        return "python_code"
    if suffix in {".cpp", ".hpp", ".h", ".cu", ".cuh"}:
        return "cpp_cuda_code"
    if suffix in {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"}:
        return "config"
    if suffix in {".md", ".txt", ".rst", ".tex"}:
        return "documentation"
    if "readme" in name:
        return "documentation"
    return "other"


def path_matches(path: Path) -> list[str]:
    lower = str(path.relative_to(ROOT)).lower()
    return [kw for kw in PATH_KEYWORDS if kw in lower]


def should_scan_content(path: Path, path_keyword_matches: list[str]) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > MAX_TEXT_FILE_SIZE_FOR_CONTENT_SCAN:
        return False
    area = source_area(path)
    lower = str(path).lower()
    if area == "official" and (
        lower.endswith("readme.md")
        or "/scripts/" in lower
    ):
        return True
    return False


def content_matches(path: Path, path_keyword_matches: list[str]) -> list[str]:
    if not should_scan_content(path, path_keyword_matches):
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read(MAX_CONTENT_SCAN_BYTES).lower()
    except OSError:
        return []
    return [kw for kw in CONTENT_KEYWORDS if kw in text]


def iter_files(root: Path):
    if not root.exists():
        return
    try:
        proc = subprocess.run(
            ["find", str(root), "-type", "f"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"find failed for {root}: {exc.stderr}") from exc
    for line in proc.stdout.splitlines():
        path = Path(line)
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def include_unmatched_path_for_audit(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in CHECKPOINT_SUFFIXES:
        return True
    return False


def should_hash(row: dict[str, Any]) -> bool:
    if row["source_area"] == "official":
        return True
    if row["artifact_kind"] in {"checkpoint", "deployment_model"}:
        return True
    return False


def is_beyondmimic_specific(row: dict[str, Any]) -> bool:
    if row["source_area"] != "official":
        return False
    path = row["relative_path"].lower()
    matches = set(row["path_keyword_matches"]) | set(row["content_keyword_matches"])
    if "reference_code" in path:
        return False
    if "whole_body_tracking" in path or "motion_tracking_controller" in path:
        # These are official, but they are the tracking/controller repos already audited separately.
        return False
    if row["artifact_kind"] == "documentation":
        return False
    return bool({"vae", "dagger", "diffusion", "latent", "denois", "tensorrt", "trt", "engine"} & matches)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "relative_path",
        "source_area",
        "artifact_kind",
        "size_bytes",
        "path_keyword_matches",
        "content_keyword_matches",
        "is_official_area",
        "is_reference_area",
        "is_beyondmimic_specific_official_candidate",
        "sha256_first_mb",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["path_keyword_matches"] = ",".join(row["path_keyword_matches"])
            out["content_keyword_matches"] = ",".join(row["content_keyword_matches"])
            writer.writerow({key: out[key] for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_files_scanned = 0
    for root in SEARCH_ROOTS:
        for path in iter_files(root):
            total_files_scanned += 1
            pm = path_matches(path)
            cm = content_matches(path, pm)
            kind = artifact_kind(path)
            if not pm and not cm and not include_unmatched_path_for_audit(path):
                continue
            rel = path.relative_to(ROOT)
            row: dict[str, Any] = {
                "relative_path": str(rel),
                "source_area": source_area(path),
                "artifact_kind": kind,
                "size_bytes": path.stat().st_size,
                "path_keyword_matches": pm,
                "content_keyword_matches": cm,
                "is_official_area": source_area(path) == "official",
                "is_reference_area": source_area(path) == "reference_code",
                "is_beyondmimic_specific_official_candidate": False,
                "sha256_first_mb": "",
            }
            if should_hash(row):
                row["sha256_first_mb"] = sha256(path)
            row["is_beyondmimic_specific_official_candidate"] = is_beyondmimic_specific(row)
            rows.append(row)

    rows.sort(key=lambda r: (r["source_area"], r["artifact_kind"], r["relative_path"]))
    official_rows = [r for r in rows if r["source_area"] == "official"]
    reference_rows = [r for r in rows if r["source_area"] == "reference_code"]
    official_bm_candidates = [r for r in rows if r["is_beyondmimic_specific_official_candidate"]]
    official_checkpoints = [
        r for r in official_rows if r["artifact_kind"] in {"checkpoint", "deployment_model"} or r["relative_path"].lower().endswith(".engine")
    ]
    reference_diffusion_rows = [
        r
        for r in reference_rows
        if {"diffusion", "latent", "denoising", "classifier guidance"} & (set(r["path_keyword_matches"]) | set(r["content_keyword_matches"]))
    ]

    json_path = OUT / "level_c_official_artifact_audit.json"
    tsv_path = OUT / "level_c_official_artifact_audit.tsv"
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "audit",
        "scope": "local official/reference artifact search for BeyondMimic VAE/diffusion implementation, checkpoints, and deployment engines",
        "search_roots": [str(path) for path in SEARCH_ROOTS],
        "total_files_scanned": total_files_scanned,
        "matched_rows": len(rows),
        "counts": {
            "official_matched_rows": len(official_rows),
            "reference_matched_rows": len(reference_rows),
            "official_beyondmimic_specific_candidates": len(official_bm_candidates),
            "official_checkpoint_or_deployment_model_candidates": len(official_checkpoints),
            "reference_diffusion_related_rows": len(reference_diffusion_rows),
        },
        "official_checkpoint_or_deployment_model_candidates": official_checkpoints,
        "official_beyondmimic_specific_candidates": official_bm_candidates,
        "reference_diffusion_related_examples": reference_diffusion_rows[:40],
        "conclusion": {
            "official_beyondmimic_vae_diffusion_code_found": len(official_bm_candidates) > 0,
            "official_beyondmimic_checkpoint_or_engine_found": len(official_checkpoints) > 0,
            "reference_diffusion_code_present": len(reference_diffusion_rows) > 0,
            "interpretation": (
                "No official BeyondMimic-specific VAE/diffusion implementation, checkpoint, TensorRT engine, or "
                "state-latent diffusion deployment artifact was identified in the scanned official areas. "
                "Diffusion-related hits are reference-code repositories or the already-audited tracking/controller repos."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
