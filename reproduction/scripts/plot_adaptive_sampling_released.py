#!/usr/bin/env python3
"""Plot converted adaptive-sampling released-data matrices."""

from __future__ import annotations

import csv
import hashlib
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
CONVERTED = ROOT / "res/released_figures/adaptive_sampling_converted"
OUT = ROOT / "res/released_figures/adaptive_sampling"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["pdf", "svg", "png"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=240 if ext == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_condition(condition: str) -> tuple[Path, list[Path]]:
    cdir = CONVERTED / condition
    matrix_path = cdir / "all_motions_iter_number_matrix_normalized.csv"
    labels = (cdir / "motion_labels.txt").read_text(encoding="utf-8").splitlines()
    mat = np.loadtxt(matrix_path, delimiter=",")
    if mat.ndim == 1:
        mat = mat[None, :]
    fig, ax = plt.subplots(1, 1, figsize=(8, 2.4), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", origin="upper", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Adaptive sampling iteration matrix: {condition}")
    ax.set_xlabel("Normalized time bin")
    ax.set_ylabel("Motion")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Iteration / motion max")
    outputs = save(fig, OUT / f"adaptive_sampling_{condition}_iter_matrix")
    return matrix_path, outputs


def plot_probability_evolution() -> tuple[list[Path], list[Path]]:
    source = ROOT / "reproduction/data/Dataset_beyondmimic/adaptive_sample/sampling_prob_over_time.pkl"
    with source.open("rb") as f:
        obj = pickle.load(f)
    times = np.asarray([t for t, _ in obj], dtype=np.float64)
    probs = np.stack([p for _, p in obj], axis=0).astype(np.float64)
    rel_min = (times - times[0]) / 60.0

    out_npz = OUT / "adaptive_sampling_probability_evolution_processed.npz"
    np.savez_compressed(out_npz, rel_min=rel_min, probs=probs)

    # Save a compact CSV with ten evenly spaced snapshots for audit-friendly inspection.
    indices = np.linspace(0, len(rel_min) - 1, num=min(10, len(rel_min)), dtype=int)
    snapshot_csv = OUT / "adaptive_sampling_probability_snapshots.csv"
    with snapshot_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot_index", "source_index", "minutes", *[f"class_{i}" for i in range(probs.shape[1])]])
        for k, idx in enumerate(indices):
            writer.writerow([k, int(idx), float(rel_min[idx]), *[float(x) for x in probs[idx]]])

    outputs: list[Path] = []
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.0), constrained_layout=True)
    im = ax.imshow(
        probs.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[float(rel_min[0]), float(rel_min[-1]), 0, probs.shape[1] - 1],
        cmap="viridis",
    )
    ax.set_title("Adaptive sampling probability evolution")
    ax.set_xlabel("Training time (min)")
    ax.set_ylabel("Class id")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability")
    outputs.extend(save(fig, OUT / "adaptive_sampling_probability_evolution_heatmap"))

    xmin, xmax = 90, min(130, probs.shape[1] - 1)
    fig, axes = plt.subplots(5, 2, figsize=(9.0, 10.0), constrained_layout=True)
    for k, (idx, ax) in enumerate(zip(indices, axes.flat)):
        xs = np.arange(xmin, xmax + 1)
        ax.bar(xs, probs[idx, xmin : xmax + 1], width=0.9, color="tab:blue", alpha=0.75)
        ax.set_title(f"idx={int(idx)}, t={rel_min[idx]:.1f} min", fontsize=9)
        ax.set_xlim(xmin, xmax)
        ax.grid(True, axis="y", alpha=0.25)
        if k >= 8:
            ax.set_xlabel("Class id")
        if k % 2 == 0:
            ax.set_ylabel("Prob.")
    outputs.extend(save(fig, OUT / "adaptive_sampling_probability_snapshots_90_130"))
    return [source, out_npz, snapshot_csv], outputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    source_paths = []
    for condition in ["w", "wo"]:
        matrix_path, outputs = plot_condition(condition)
        summary = CONVERTED / condition / "failure_rate_summary.tsv"
        source_paths.extend([matrix_path, summary])
        rows.append(
            {
                "figure_id": f"adaptive_sampling_{condition}",
                "processed_csv": str(matrix_path),
                "outputs": ",".join(str(p) for p in outputs),
                "log": str(OUT / "run.log"),
                "source_count": 2,
            }
        )
    prob_sources, prob_outputs = plot_probability_evolution()
    source_paths.extend(prob_sources)
    rows.append(
        {
            "figure_id": "adaptive_sampling_probability_evolution",
            "processed_csv": str(OUT / "adaptive_sampling_probability_snapshots.csv"),
            "outputs": ",".join(str(p) for p in prob_outputs),
            "log": str(OUT / "run.log"),
            "source_count": len(prob_sources),
        }
    )
    with (OUT / "adaptive_sampling_summary.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["figure_id", "processed_csv", "outputs", "log", "source_count"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "source_hashes.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"], delimiter="\t")
        writer.writeheader()
        for path in source_paths:
            writer.writerow({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (OUT / "paper_panel_mapping.md").write_text(
        "# Paper Panel Mapping\n\nAdaptive sampling failure/progress maps from released `adaptive_sample/w` and `adaptive_sample/wo` torch-save files, plus probability evolution from `sampling_prob_over_time.pkl`.\n",
        encoding="utf-8",
    )
    (OUT / "run.log").write_text(
        "figure_id=adaptive_sampling\nkind=adaptive_sampling_converted_and_probability_evolution\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'adaptive_sampling_summary.tsv'}")


if __name__ == "__main__":
    main()
