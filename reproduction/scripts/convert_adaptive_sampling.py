#!/usr/bin/env python3
"""Convert adaptive-sampling torch-save files to plain CSV matrices."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch


def parse_iter(name: str) -> int:
    match = re.search(r"(\d+)(?=\.pt$)", name)
    return int(match.group(1)) if match else -1


def parse_motion(path: Path) -> str:
    match = re.search(r"motion_(\d+)", path.name)
    return f"motion_{match.group(1)}" if match else path.stem


def resample_to_bins(vec: np.ndarray, bins: int) -> np.ndarray:
    if len(vec) == bins:
        return vec.astype(np.float32)
    idx = np.rint(np.linspace(0, len(vec) - 1, bins)).astype(int)
    return vec[np.clip(idx, 0, len(vec) - 1)].astype(np.float32)


def convert_dir(in_dir: Path, out_dir: Path, bins: int, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    matrix = []
    labels = []
    for path in sorted(in_dir.glob("*_ckpt_failure_*.npy")):
        motion = parse_motion(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        parsed = []
        for name, fail_t, total_t in data:
            fail = fail_t.to(torch.float32).cpu().numpy()
            total = total_t.to(torch.float32).cpu().numpy()
            valid = total > 0
            rate = np.full_like(fail, np.nan, dtype=np.float32)
            rate[valid] = fail[valid] / total[valid]
            parsed.append((parse_iter(name), name, rate))
        parsed.sort(key=lambda x: x[0] if x[0] >= 0 else 10**12)
        max_t = max(len(rate) for _, _, rate in parsed)
        rates = np.full((len(parsed), max_t), np.nan, dtype=np.float32)
        iter_ids = np.zeros(len(parsed), dtype=np.float32)
        for i, (it, name, rate) in enumerate(parsed):
            rates[i, : len(rate)] = rate
            iter_ids[i] = it if it >= 0 else i
            rows.append(
                {
                    "condition": in_dir.name,
                    "motion": motion,
                    "iteration": iter_ids[i],
                    "checkpoint": name,
                    "time_bins": len(rate),
                    "mean_failure_rate": float(np.nanmean(rate)),
                    "max_failure_rate": float(np.nanmax(rate)),
                }
            )
        cond = rates < threshold
        max_iter = float(np.nanmax(iter_ids))
        iter_vec = np.full(max_t, max_iter, dtype=np.float32)
        for t in range(max_t):
            hits = np.where(cond[:, t])[0]
            if hits.size:
                iter_vec[t] = float(np.min(iter_ids[hits]))
        row_max = max(max_iter, 1.0)
        matrix.append(resample_to_bins(iter_vec / row_max, bins))
        labels.append(motion)
        np.savetxt(out_dir / f"{motion}_failure_rates.csv", rates, delimiter=",", fmt="%.6f")
    if matrix:
        np.savetxt(out_dir / "all_motions_iter_number_matrix_normalized.csv", np.vstack(matrix), delimiter=",", fmt="%.6f")
        (out_dir / "motion_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    with (out_dir / "failure_rate_summary.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "motion",
                "iteration",
                "checkpoint",
                "time_bins",
                "mean_failure_rate",
                "max_failure_rate",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/infini-data/test/BeyondMimic/reproduction/data/Dataset_beyondmimic/adaptive_sample"))
    parser.add_argument("--out", type=Path, default=Path("/mnt/infini-data/test/BeyondMimic/res/released_figures/adaptive_sampling_converted"))
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    for condition in ["w", "wo"]:
        convert_dir(args.root / condition, args.out / condition, args.bins, args.threshold)
    print(f"converted adaptive sampling files to {args.out}")


if __name__ == "__main__":
    main()
