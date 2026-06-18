#!/usr/bin/env python3
"""Create debug GIF previews for the resource-adjusted tiny diffusion run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_DIR = ROOT / "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500"
SOURCE_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
    / "level_c_resource_adjusted_tiny_diffusion_training_run.json"
)
SOURCE_NPZ = (
    ROOT
    / "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
    / "level_c_resource_adjusted_tiny_diffusion_training_run.npz"
)
OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_video_preview"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_row(values: np.ndarray, *, width: int, height: int, lo: float, hi: float) -> Image.Image:
    denom = max(hi - lo, 1e-8)
    norm = np.clip((values - lo) / denom, 0.0, 1.0)
    rgb = np.zeros((1, len(values), 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * norm, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255 * (1.0 - np.abs(norm - 0.5) * 2.0), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1.0 - norm), 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((width, height), resample=Image.Resampling.NEAREST)


def make_frame(
    clean: np.ndarray,
    noisy: np.ndarray,
    pred: np.ndarray,
    *,
    frame_index: int,
    split: str,
    sample_index: int,
    width: int = 960,
    height: int = 540,
) -> Image.Image:
    state_dim = 99
    pad = 24
    row_h = 72
    label_w = 150
    bar_w = width - label_w - 2 * pad
    image = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.text((pad, 16), f"Level C tiny diffusion debug preview | {split} sample {sample_index}", fill=(20, 24, 28))
    draw.text((pad, 40), f"token {frame_index:02d}: clean vs noisy vs predicted tau", fill=(70, 72, 76))
    rows = [
        ("clean state", clean[frame_index, :state_dim]),
        ("noisy state", noisy[frame_index, :state_dim]),
        ("pred state", pred[frame_index, :state_dim]),
        ("clean latent", clean[frame_index, state_dim:]),
        ("noisy latent", noisy[frame_index, state_dim:]),
        ("pred latent", pred[frame_index, state_dim:]),
    ]
    all_vals = np.concatenate([row for _, row in rows])
    lo, hi = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))
    y = 86
    for label, values in rows:
        draw.text((pad, y + 22), label, fill=(28, 32, 36))
        row_img = scale_row(values, width=bar_w, height=row_h - 14, lo=lo, hi=hi)
        image.paste(row_img, (pad + label_w, y))
        draw.rectangle((pad + label_w, y, pad + label_w + bar_w, y + row_h - 14), outline=(42, 42, 42))
        y += row_h
    token_mse_noisy = float(np.mean(np.square(noisy[frame_index] - clean[frame_index])))
    token_mse_pred = float(np.mean(np.square(pred[frame_index] - clean[frame_index])))
    draw.text(
        (pad, height - 42),
        f"token MSE noisy={token_mse_noisy:.6f} pred={token_mse_pred:.6f} | debug-only, not closed-loop video",
        fill=(32, 32, 32),
    )
    return image


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "gif_path",
        "poster_path",
        "frame_count",
        "clean_vs_noisy_mse",
        "clean_vs_pred_mse",
        "pred_reduction_vs_noisy",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-ms", type=int, default=140)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "videos").mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    with np.load(SOURCE_NPZ) as data:
        clean = data["clean_tau"].astype(np.float64)
        noisy = data["noisy_tau"].astype(np.float64)
        pred = data["predicted_tau"].astype(np.float64)
        split_labels = data["split_labels"].astype(str)
    rows: list[dict[str, Any]] = []
    for split in ["validation", "test"]:
        sample_index = int(np.nonzero(split_labels == split)[0][0])
        frames = [
            make_frame(
                clean[sample_index],
                noisy[sample_index],
                pred[sample_index],
                frame_index=i,
                split=split,
                sample_index=sample_index,
            )
            for i in range(clean.shape[1])
        ]
        gif_path = RUN_DIR / "videos" / f"tiny_diffusion_{split}_debug_preview.gif"
        poster_path = OUT / f"tiny_diffusion_{split}_debug_preview_poster.png"
        frames[0].save(poster_path)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=args.duration_ms,
            loop=0,
            optimize=False,
        )
        mse_noisy = float(np.mean(np.square(noisy[sample_index] - clean[sample_index])))
        mse_pred = float(np.mean(np.square(pred[sample_index] - clean[sample_index])))
        rows.append(
            {
                "split": split,
                "sample_index": sample_index,
                "gif_path": str(gif_path),
                "poster_path": str(poster_path),
                "frame_count": len(frames),
                "clean_vs_noisy_mse": mse_noisy,
                "clean_vs_pred_mse": mse_pred,
                "pred_reduction_vs_noisy": (mse_noisy - mse_pred) / mse_noisy if mse_noisy > 0 else 0.0,
            }
        )
    tsv_path = OUT / "level_c_resource_adjusted_tiny_diffusion_video_preview.tsv"
    json_path = OUT / "level_c_resource_adjusted_tiny_diffusion_video_preview.json"
    write_tsv(tsv_path, rows)
    checks = {
        "source_training_status_ok": source["status"] == "ok",
        "two_debug_gifs_written": len(rows) == 2 and all(Path(row["gif_path"]).is_file() for row in rows),
        "posters_written": all(Path(row["poster_path"]).is_file() for row in rows),
        "all_gifs_nonempty": all(Path(row["gif_path"]).stat().st_size > 0 for row in rows),
        "all_posters_nonempty": all(Path(row["poster_path"]).stat().st_size > 0 for row in rows),
        "all_previews_have_21_frames": all(row["frame_count"] == 21 for row in rows),
        "all_pred_improves_vs_noisy": all(row["pred_reduction_vs_noisy"] > 0 for row in rows),
        "run_videos_dir_nonempty": any((RUN_DIR / "videos").iterdir()),
        "does_not_claim_closed_loop_video": True,
        "does_not_claim_paper_figures": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "resource_adjusted_tiny_diffusion_video_preview",
        "scope": "debug GIF previews of offline clean/noisy/predicted state-latent tokens from the tiny denoiser run",
        "source_training_json": str(SOURCE_JSON),
        "source_training_npz": str(SOURCE_NPZ),
        "run_dir": str(RUN_DIR),
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "videos_dir": str(RUN_DIR / "videos"),
            "gif_sha256": {row["split"]: sha256(Path(row["gif_path"])) for row in rows},
            "poster_sha256": {row["split"]: sha256(Path(row["poster_path"])) for row in rows},
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "debug_offline_visual_preview_only",
            "why_not_complete": (
                "These GIFs visualize offline token denoising for the small debug model. They are not IsaacLab "
                "closed-loop rollouts, not robot videos, not paper Fig. 5/Fig. 6 evidence, and do not make the run a "
                "valid full training run."
            ),
        },
    }
    write_json_atomic(json_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "videos": len(rows)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
