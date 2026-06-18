#!/usr/bin/env python3
"""Generate a local inventory for downloaded BeyondMimic reproduction assets.

The output is a TSV required by goal.md. It records repositories, archives,
papers, datasets, checkpoints, and other relevant files found under
DOWNLOAD_ROOT without modifying the download directory.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOWNLOAD_ROOT = ROOT / "download"
OUT = ROOT / "reproduction" / "docs" / "local_inventory.tsv"

ARCHIVE_SUFFIXES = {
    ".zip",
    ".rar",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
}
DATA_SUFFIXES = {
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".onnx",
    ".csv",
    ".json",
    ".h5",
    ".hdf5",
    ".bag",
    ".db3",
}


@dataclass(frozen=True)
class Row:
    item_name: str
    detected_path: str
    type: str
    size: str
    git_remote: str
    git_commit: str
    git_branch: str
    archive_hash: str
    usable: str
    duplicate: str
    selected_copy: str
    notes: str


def run_git(repo: Path, args: list[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return out.strip()
    except Exception:
        return ""


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def shallow_size(path: Path) -> int:
    """Return a bounded size estimate without expensive recursive NFS scans."""
    total = 0
    try:
        entries = list(path.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        return f"ERROR:{exc}"
    return digest.hexdigest()


def is_archive(path: Path) -> bool:
    suffixes = path.suffixes
    if path.suffix.lower() in ARCHIVE_SUFFIXES:
        return True
    return len(suffixes) >= 2 and "".join(suffixes[-2:]).lower() in {".tar.gz", ".tar.xz", ".tar.bz2"}


def classify_file(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "paper"
    if is_archive(path):
        return "archive"
    if suffix in DATA_SUFFIXES:
        return "data_or_checkpoint"
    if name.startswith("readme") or suffix in {".md", ".txt"}:
        return "documentation"
    return "file"


def note_for(path: Path) -> str:
    rel = str(path.relative_to(DOWNLOAD_ROOT)).lower()
    tags: list[str] = []
    for key in [
        "beyondmimic",
        "whole_body_tracking",
        "motion_tracking_controller",
        "isaaclab",
        "rsl_rl",
        "unitree",
        "lafan",
        "zenodo",
        "mjlab",
        "pbhc",
        "asap",
        "gmr",
        "diffuser",
        "guided-motion-diffusion",
        "motion-diffusion-model",
        "latent-diffusion",
    ]:
        if key in rel:
            tags.append(key)
    return ",".join(tags)


def discover_git_repos() -> list[Path]:
    repos = []
    for git_dir in DOWNLOAD_ROOT.rglob(".git"):
        if git_dir.is_dir():
            repos.append(git_dir.parent)
    return sorted(repos)


def discover_files(repo_roots: Iterable[Path]) -> list[Path]:
    repo_set = set(repo_roots)
    files: list[Path] = []
    for path in DOWNLOAD_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part == ".git" for part in path.parts):
            continue
        if classify_file(path) in {"file", "documentation"} and note_for(path) == "":
            continue
        # Avoid listing every source file inside repos; repo row covers them.
        if any(repo in path.parents for repo in repo_set) and classify_file(path) == "file":
            continue
        files.append(path)
    return sorted(files)


def mark_duplicates(rows: list[Row]) -> list[Row]:
    by_name: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_name.setdefault(row.item_name.lower(), []).append(idx)

    output = list(rows)
    for indices in by_name.values():
        if len(indices) <= 1:
            continue
        selected = max(indices, key=lambda i: int(output[i].size) if output[i].size.isdigit() else -1)
        for i in indices:
            row = output[i]
            output[i] = Row(
                row.item_name,
                row.detected_path,
                row.type,
                row.size,
                row.git_remote,
                row.git_commit,
                row.git_branch,
                row.archive_hash,
                row.usable,
                "yes",
                "yes" if i == selected else "no",
                (row.notes + "; " if row.notes else "") + "duplicate_name",
            )
    return output


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []

    repos = discover_git_repos()
    for repo in repos:
        remote = run_git(repo, ["remote", "get-url", "origin"])
        commit = run_git(repo, ["rev-parse", "HEAD"])
        branch = run_git(repo, ["branch", "--show-current"])
        rows.append(
            Row(
                item_name=repo.name,
                detected_path=str(repo),
                type="git_repo",
                size=str(shallow_size(repo)),
                git_remote=remote,
                git_commit=commit,
                git_branch=branch,
                archive_hash="",
                usable="unknown",
                duplicate="no",
                selected_copy="yes",
                notes=note_for(repo),
            )
        )

    for path in discover_files(repos):
        kind = classify_file(path)
        digest = sha256(path) if kind == "archive" else ""
        rows.append(
            Row(
                item_name=path.name,
                detected_path=str(path),
                type=kind,
                size=str(file_size(path)),
                git_remote="",
                git_commit="",
                git_branch="",
                archive_hash=digest,
                usable="unknown",
                duplicate="no",
                selected_copy="yes",
                notes=note_for(path),
            )
        )

    rows = mark_duplicates(rows)
    fields = list(Row.__dataclass_fields__.keys())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    print(f"wrote {OUT} with {len(rows)} rows")


if __name__ == "__main__":
    main()
