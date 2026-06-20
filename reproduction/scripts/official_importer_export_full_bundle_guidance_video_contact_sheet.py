#!/usr/bin/env python3
"""Build report-facing video/keyframe index for importer-export guidance rollouts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
    "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet"
ORDERED_TASKS = ["joystick", "waypoint", "obstacle_avoidance", "composed"]
ORDERED_SEEDS = ["seed_group_0_existing", "seed_group_1", "seed_group_2"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def pick_representative_frame(keyframes_png: Path) -> Image.Image:
    image = Image.open(keyframes_png).convert("RGB")
    width, height = image.size
    crop_width = min(height, width)
    left = max(0, (width - crop_width) // 2)
    return image.crop((left, 0, left + crop_width, height))


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary["rows"]:
        mp4 = Path(row["mp4"])
        keyframes_png = Path(row["keyframes_png"])
        metrics_png = Path(row["metrics_png"])
        metrics_csv = Path(row["metrics_csv"])
        source_json = Path(row["summary_json"])
        rows.append(
            {
                "task": row["task"],
                "seed_group": row["seed_group"],
                "seed": "" if row.get("seed") is None else row["seed"],
                "rollout_steps": row["rollout_steps"],
                "guided_reward_mean": row["guided_reward_mean"],
                "guided_target_body_error_mean": row["guided_target_body_error_mean"],
                "guided_done_count_total": row["guided_done_count_total"],
                "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
                "guided_base_action_mse_mean": row["guided_base_action_mse_mean"],
                "mp4": rel(mp4),
                "mp4_size_bytes": mp4.stat().st_size if mp4.exists() else 0,
                "mp4_sha256": sha256_file(mp4) if mp4.exists() else "",
                "keyframes_png": rel(keyframes_png),
                "keyframes_png_size_bytes": keyframes_png.stat().st_size if keyframes_png.exists() else 0,
                "metrics_png": rel(metrics_png),
                "metrics_csv": rel(metrics_csv),
                "source_json": rel(source_json),
                "github_policy": "do_not_commit_mp4_commit_index_and_contact_sheet_only",
                "claim_level": "local_virtual_official_importer_export_guidance_video_evidence_not_paper_fig5_fig6",
                "limitation": (
                    "Local official-importer-export IsaacLab rollout with local PPO/VAE/denoiser checkpoints "
                    "and proxy costs; not official BeyondMimic Fig. 5/Fig. 6, TensorRT deployment, or real-robot "
                    "evidence."
                ),
            }
        )
    return rows


def make_contact_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    by_key = {(row["seed_group"], row["task"]): row for row in rows}
    tile_w, tile_h = 270, 216
    label_h = 50
    header_h = 74
    left_w = 148
    gap = 8
    sheet_w = left_w + len(ORDERED_TASKS) * tile_w + (len(ORDERED_TASKS) + 1) * gap
    sheet_h = header_h + len(ORDERED_SEEDS) * (tile_h + label_h) + (len(ORDERED_SEEDS) + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "#f7f7f4")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    draw.text(
        (18, 18),
        "Official-importer-export guidance: local video/keyframe evidence",
        fill="#111111",
        font=font,
    )
    draw.text(
        (18, 40),
        "Local proxy rollouts only; not official BeyondMimic Fig. 5/Fig. 6.",
        fill="#555555",
        font=font,
    )
    for col, task in enumerate(ORDERED_TASKS):
        x = left_w + gap + col * (tile_w + gap)
        draw.text((x + 8, header_h - 24), task.replace("_", " "), fill="#111111", font=font)

    for row_idx, seed in enumerate(ORDERED_SEEDS):
        y = header_h + gap + row_idx * (tile_h + label_h + gap)
        draw.text((18, y + tile_h // 2 - 8), seed.replace("_", " "), fill="#111111", font=font)
        for col, task in enumerate(ORDERED_TASKS):
            x = left_w + gap + col * (tile_w + gap)
            item = by_key[(seed, task)]
            frame = pick_representative_frame(ROOT / item["keyframes_png"])
            frame.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), "#deded8")
            tile.paste(frame, ((tile_w - frame.width) // 2, (tile_h - frame.height) // 2))
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w, y + tile_h), outline="#333333", width=1)
            label_y = y + tile_h + 6
            draw.text(
                (x + 6, label_y),
                f"R {item['guided_reward_mean']:.4f} | E {item['guided_target_body_error_mean']:.4f}",
                fill="#111111",
                font=font,
            )
            draw.text(
                (x + 6, label_y + 16),
                f"done {item['guided_done_count_total']} | steps {item['rollout_steps']}",
                fill="#555555",
                font=font,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(SOURCE_JSON)
    rows = build_rows(summary)
    contact_png = OUT / "importer_export_guidance_video_contact_sheet.png"
    rows_csv = OUT / "importer_export_guidance_video_index.csv"
    rows_json = OUT / "importer_export_guidance_video_index.json"
    readme = OUT / "README.md"
    fields = [
        "task",
        "seed_group",
        "seed",
        "rollout_steps",
        "guided_reward_mean",
        "guided_target_body_error_mean",
        "guided_done_count_total",
        "guidance_cost_delta_mean",
        "guided_base_action_mse_mean",
        "mp4",
        "mp4_size_bytes",
        "mp4_sha256",
        "keyframes_png",
        "keyframes_png_size_bytes",
        "metrics_png",
        "metrics_csv",
        "source_json",
        "github_policy",
        "claim_level",
        "limitation",
    ]
    write_csv(rows_csv, rows, fields)
    make_contact_sheet(rows, contact_png)
    checks = {
        "source_status_ok": summary.get("status")
        == "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
        "row_count_12": len(rows) == 12,
        "seed_group_count_3": len({row["seed_group"] for row in rows}) == 3,
        "task_count_4": len({row["task"] for row in rows}) == 4,
        "all_mp4_exist": all((ROOT / row["mp4"]).is_file() and (ROOT / row["mp4"]).stat().st_size > 0 for row in rows),
        "all_keyframes_exist": all(
            (ROOT / row["keyframes_png"]).is_file() and (ROOT / row["keyframes_png"]).stat().st_size > 0
            for row in rows
        ),
        "contact_sheet_exists": contact_png.is_file() and contact_png.stat().st_size > 0,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    result = {
        "status": "ok_official_importer_export_full_bundle_guidance_video_contact_sheet"
        if all(checks.values())
        else "failed_official_importer_export_full_bundle_guidance_video_contact_sheet",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": rel(SOURCE_JSON),
        "rows": rows,
        "metrics": {
            "row_count": len(rows),
            "seed_group_count": len({row["seed_group"] for row in rows}),
            "task_count": len({row["task"] for row in rows}),
            "video_count": sum(1 for row in rows if (ROOT / row["mp4"]).is_file()),
            "total_mp4_size_bytes": sum(row["mp4_size_bytes"] for row in rows),
            "contact_sheet_size_bytes": contact_png.stat().st_size if contact_png.exists() else 0,
        },
        "checks": checks,
        "assets": {
            "json": str(rows_json),
            "csv": str(rows_csv),
            "contact_sheet_png": str(contact_png),
            "readme": str(readme),
        },
        "interpretation": {
            "claim_level": "local_virtual_official_importer_export_guidance_video_evidence_not_paper_fig5_fig6",
            "mp4_github_policy": "do_not_commit_large_mp4_files",
            "goal_complete": False,
        },
    }
    write_json(rows_json, result)
    readme.write_text(
        "\n".join(
            [
                "# Official-importer-export guidance video contact sheet",
                "",
                "This directory indexes 12 local closed-loop MP4 rollouts from the three-seed official-importer-export task-conditioned guidance evaluation.",
                "",
                "The contact sheet is generated from existing keyframe PNGs. MP4 files remain local and should not be committed to GitHub.",
                "",
                "Claim boundary: local virtual official-importer-export IsaacLab evidence only. This is not official BeyondMimic Fig. 5/Fig. 6, TensorRT deployment, or real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "json": str(rows_json), "contact_sheet": str(contact_png)}, sort_keys=True))


if __name__ == "__main__":
    main()
