#!/usr/bin/env python3
"""Debug-only optimizer/LR/EMA schedule probe for Level C diffusion training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/training_schedule_probe"


@dataclass(frozen=True)
class DiffusionTrainingScheduleConfig:
    batch_size: int = 512
    epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    scheduler: str = "cosine_with_linear_warmup"
    warmup_gradient_steps: int = 10000
    ema_power: float = 0.75
    ema_max: float = 0.9999
    probe_total_gradient_steps: int = 50000


def learning_rate_at_step(step: int, cfg: DiffusionTrainingScheduleConfig) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if step < cfg.warmup_gradient_steps:
        return cfg.learning_rate * float(step + 1) / float(cfg.warmup_gradient_steps)
    decay_span = max(cfg.probe_total_gradient_steps - cfg.warmup_gradient_steps, 1)
    progress = min(max((step - cfg.warmup_gradient_steps) / decay_span, 0.0), 1.0)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_decay_at_step(step: int, cfg: DiffusionTrainingScheduleConfig) -> float:
    # Candidate power schedule with explicit cap from the paper table. It starts permissive and approaches ema_max.
    # The paper reports power/max but not implementation code, so this is a reproducible local schedule probe.
    value = 1.0 - (1.0 + float(step)) ** (-cfg.ema_power)
    return min(cfg.ema_max, value)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "learning_rate",
        "ema_decay",
        "warmup_phase",
        "cosine_phase",
        "at_ema_cap",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-total-gradient-steps", type=int, default=50000)
    parser.add_argument(
        "--sample-steps",
        type=str,
        default="0,1,10,100,1000,9999,10000,20000,40000,49999",
    )
    args = parser.parse_args()

    cfg = DiffusionTrainingScheduleConfig(probe_total_gradient_steps=args.probe_total_gradient_steps)
    OUT.mkdir(parents=True, exist_ok=True)
    sample_steps = sorted({int(item.strip()) for item in args.sample_steps.split(",") if item.strip()})
    if min(sample_steps) < 0 or max(sample_steps) >= cfg.probe_total_gradient_steps:
        raise ValueError("sample steps must lie inside [0, probe_total_gradient_steps)")

    rows: list[dict[str, Any]] = []
    for step in sample_steps:
        lr = learning_rate_at_step(step, cfg)
        ema = ema_decay_at_step(step, cfg)
        rows.append(
            {
                "step": step,
                "learning_rate": lr,
                "ema_decay": ema,
                "warmup_phase": step < cfg.warmup_gradient_steps,
                "cosine_phase": step >= cfg.warmup_gradient_steps,
                "at_ema_cap": math.isclose(ema, cfg.ema_max, rel_tol=0.0, abs_tol=1e-15),
            }
        )

    dense_steps = np.arange(cfg.probe_total_gradient_steps, dtype=np.int64)
    dense_lr = np.asarray([learning_rate_at_step(int(step), cfg) for step in dense_steps], dtype=np.float64)
    dense_ema = np.asarray([ema_decay_at_step(int(step), cfg) for step in dense_steps], dtype=np.float64)
    json_path = OUT / "level_c_training_schedule_probe.json"
    tsv_path = OUT / "level_c_training_schedule_probe.tsv"
    npz_path = OUT / "level_c_training_schedule_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(npz_path, steps=dense_steps, learning_rate=dense_lr, ema_decay=dense_ema)

    warmup_lr = dense_lr[: cfg.warmup_gradient_steps]
    cosine_lr = dense_lr[cfg.warmup_gradient_steps :]
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "paper-parameter optimizer, cosine warmup LR, and EMA schedule probe",
        "paper_evidence": {
            "diffusion_training_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:845-854"),
        },
        "not_a_replacement_for": [
            "full diffusion training",
            "official optimizer implementation",
            "checkpoint reproduction",
            "validation/test metrics",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": asdict(cfg),
        "sample_rows": rows,
        "metrics": {
            "initial_learning_rate": float(dense_lr[0]),
            "warmup_final_learning_rate": float(dense_lr[cfg.warmup_gradient_steps - 1]),
            "first_cosine_learning_rate": float(dense_lr[cfg.warmup_gradient_steps]),
            "final_learning_rate": float(dense_lr[-1]),
            "max_learning_rate": float(dense_lr.max()),
            "initial_ema_decay": float(dense_ema[0]),
            "final_ema_decay": float(dense_ema[-1]),
            "max_ema_decay": float(dense_ema.max()),
            "ema_cap_first_step": int(dense_steps[np.argmax(dense_ema >= cfg.ema_max)])
            if bool(np.any(dense_ema >= cfg.ema_max))
            else None,
        },
        "checks": {
            "paper_batch_size_recorded": cfg.batch_size == 512,
            "paper_epochs_recorded": cfg.epochs == 1000,
            "paper_learning_rate_recorded": math.isclose(cfg.learning_rate, 1e-4),
            "paper_weight_decay_recorded": math.isclose(cfg.weight_decay, 0.001),
            "paper_warmup_recorded": cfg.warmup_gradient_steps == 10000,
            "paper_ema_power_recorded": math.isclose(cfg.ema_power, 0.75),
            "paper_ema_max_recorded": math.isclose(cfg.ema_max, 0.9999),
            "warmup_lr_monotone_non_decreasing": bool(np.all(np.diff(warmup_lr) >= -1e-18)),
            "cosine_lr_monotone_non_increasing": bool(np.all(np.diff(cosine_lr) <= 1e-18)),
            "lr_nonnegative": bool(np.all(dense_lr >= 0.0)),
            "lr_peak_matches_paper_lr": math.isclose(float(dense_lr.max()), cfg.learning_rate, rel_tol=1e-12, abs_tol=1e-12),
            "ema_decay_monotone_non_decreasing": bool(np.all(np.diff(dense_ema) >= -1e-18)),
            "ema_decay_capped_by_max": bool(np.all(dense_ema <= cfg.ema_max + 1e-15)),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
