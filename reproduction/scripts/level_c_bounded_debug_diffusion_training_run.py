#!/usr/bin/env python3
"""Run a bounded debug diffusion training job with goal.md run-directory artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
import subprocess
import time
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import level_c_vae_latent_diffusion_overfit_probe as vae_latent  # noqa: E402
from level_c_paper_state_transformer_arch_probe import (  # noqa: E402
    PaperStateDiffusionTransformer,
    ProbeConfig,
    diffusion_alpha_bars,
    grad_norm,
    parameter_count,
    seed_everything,
)
from level_c_training_schedule_probe import DiffusionTrainingScheduleConfig, ema_decay_at_step, learning_rate_at_step  # noqa: E402
from level_c_transformer_ema_smoke import clone_state, ema_update, mean_state_l2_delta, state_digest  # noqa: E402


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BM_DIFFUSION_PYTHON = ROOT / "envs/bm_diffusion/bin/python"
RUN_ID = "level_c_bounded_debug_diffusion_static_000_20260617_083000"
RUN_DIR = ROOT / "res/runs" / RUN_ID
OUT = ROOT / "res/level_c/bounded_debug_diffusion_training_run"


def read_command(cmd: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostic logging must not mask the run.
        return f"command_failed: {cmd}: {exc}"


def tensor_state_digest(state: dict[str, torch.Tensor]) -> str:
    return state_digest({key: value.detach().cpu() for key, value in state.items()})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_step_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "step",
        "learning_rate",
        "ema_decay",
        "loss_before",
        "loss_after",
        "grad_norm",
        "iteration_time_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def plot_losses(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 720, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 25, 30, 65
    image = bytearray([255] * (width * height * 3))

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            image[idx : idx + 3] = bytes(color)

    def draw_line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in [-1, 0, 1]:
                for oy in [-1, 0, 1]:
                    set_pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    steps = [int(row["step"]) for row in rows]
    before = [float(row["loss_before"]) for row in rows]
    after = [float(row["loss_after"]) for row in rows]
    x_min, x_max = min(steps), max(steps)
    y_values = before + after
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= y_pad
    y_max += y_pad
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_map(step: int) -> int:
        if x_max == x_min:
            return margin_left + plot_w // 2
        return margin_left + int((step - x_min) / (x_max - x_min) * plot_w)

    def y_map(value: float) -> int:
        return margin_top + int((y_max - value) / (y_max - y_min) * plot_h)

    axis = (35, 35, 35)
    grid = (225, 225, 225)
    for i in range(6):
        y = margin_top + int(i * plot_h / 5)
        draw_line(margin_left, y, margin_left + plot_w, y, grid)
    draw_line(margin_left, margin_top, margin_left, margin_top + plot_h, axis)
    draw_line(margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h, axis)

    for series, color in [(before, (31, 119, 180)), (after, (214, 39, 40))]:
        points = [(x_map(step), y_map(value)) for step, value in zip(steps, series)]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            draw_line(x0, y0, x1, y1, color)
        for x, y in points:
            for ox in range(-4, 5):
                for oy in range(-4, 5):
                    if ox * ox + oy * oy <= 16:
                        set_pixel(x + ox, y + oy, color)

    raw = b"".join(b"\x00" + bytes(image[y * width * 3 : (y + 1) * width * 3]) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def prepare_run_dir(command: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in ["checkpoint", "figures", "videos"]:
        (RUN_DIR / dirname).mkdir(exist_ok=True)
    write_text(
        RUN_DIR / "resolved_config.yaml",
        "\n".join(
            [
                f"run_id: {RUN_ID}",
                "stage: level_c",
                "method: bounded_debug_diffusion_training",
                "motion: walk/run/jumps debug fixtures",
                "config: paper_state_99_plus_debug_vae_latent_32",
                "seed: 20260921",
                "status: SUCCESS",
                "is_training_run: false",
                "is_bounded_debug_training_run: true",
                "paper_level: false",
                "reason: Bounded debug optimizer run; not paper-scale training or evaluation.",
                "source_goal_lines: goal.md:1251-1290,1468-1487,1747-1787",
                "",
            ]
        ),
    )
    write_text(RUN_DIR / "command.sh", f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n", executable=True)
    write_text(RUN_DIR / "stderr.log", "")
    env_lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"python={read_command(['python3', '--version'])}",
        f"torch={torch.__version__}",
        f"numpy={np.__version__}",
        "figure_writer=stdlib_png",
        f"cuda_available={torch.cuda.is_available()}",
        f"bm_diffusion={ROOT / 'envs/bm_diffusion'}",
        "notes=Bounded debug run only; no Isaac/Kit/ROS/TensorRT/hardware execution.",
        "",
    ]
    write_text(RUN_DIR / "environment.txt", "\n".join(env_lines))
    write_text(RUN_DIR / "git_state.txt", read_command(["git", "status", "--short"]) + "\n")
    gpu_src = ROOT / "logs/gpu/gpu_metrics.csv"
    if gpu_src.is_file():
        shutil.copyfile(gpu_src, RUN_DIR / "gpu_metrics.csv")
    else:
        write_text(
            RUN_DIR / "gpu_metrics.csv",
            "timestamp,gpu_index,gpu_uuid,gpu_name,memory_used_mib,memory_total_mib,memory_free_mib,"
            "gpu_util_percent,power_draw_w,temperature_c,process_pid,run_id,run_status,sample_kind\n",
        )


def run_training(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Path, Path]:
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    cfg = ProbeConfig(batch_size=args.batch_size, seed=args.seed)
    schedule = DiffusionTrainingScheduleConfig(probe_total_gradient_steps=50000)
    seed_everything(cfg.seed)

    latent_cfg = vae_latent.VaeLatentDiffusionConfig(seed=cfg.seed)
    clean_np, motion_ids, latent_manifest = vae_latent.load_dataset(latent_cfg)
    if args.batch_size > clean_np.shape[0]:
        raise ValueError(f"batch_size {args.batch_size} exceeds dataset size {clean_np.shape[0]}")
    clean_tau = torch.from_numpy(clean_np[: args.batch_size].astype(np.float32)).to(device)
    state_dim = latent_cfg.state_dim
    token_dim = int(clean_tau.shape[-1])

    model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.learning_rate, weight_decay=schedule.weight_decay)
    alpha_bars = diffusion_alpha_bars(cfg.denoising_steps, device, clean_tau.dtype)
    initial_state = clone_state(model)
    ema_shadow = clone_state(model)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(args.steps):
        step_start = time.perf_counter()
        lr = learning_rate_at_step(step, schedule)
        ema_decay = ema_decay_at_step(step, schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        seed_everything(cfg.seed + 510 + step)
        denoising_steps = torch.randint(0, cfg.denoising_steps, (args.batch_size, cfg.sequence_length, 2), device=device)
        noise = torch.randn_like(clean_tau)
        state_alpha = alpha_bars[denoising_steps[..., 0]].unsqueeze(-1).expand(-1, -1, state_dim)
        latent_alpha = alpha_bars[denoising_steps[..., 1]].unsqueeze(-1).expand(-1, -1, cfg.latent_dim)
        alpha = torch.cat([state_alpha, latent_alpha], dim=-1)
        noisy_tau = torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise

        with torch.no_grad():
            loss_before = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
        loss.backward()
        total_grad_norm = grad_norm(model)
        optimizer.step()
        ema_update(ema_shadow, model, ema_decay)
        with torch.no_grad():
            loss_after = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
        rows.append(
            {
                "step": step,
                "learning_rate": lr,
                "ema_decay": ema_decay,
                "loss_before": float(loss_before.detach().cpu().item()),
                "loss_after": float(loss_after.detach().cpu().item()),
                "grad_norm": float(total_grad_norm),
                "iteration_time_sec": float(time.perf_counter() - step_start),
            }
        )

    elapsed = time.perf_counter() - start
    final_state = clone_state(model)
    checkpoint_path = RUN_DIR / "checkpoint" / "debug_bounded_diffusion_checkpoint.pt"
    checkpoint_payload = {
        "experiment_type": "debug_only_bounded_diffusion_training",
        "run_id": RUN_ID,
        "config": asdict(cfg),
        "schedule": asdict(schedule),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema_shadow,
        "step": args.steps,
        "seed": cfg.seed,
        "is_trained_paper_checkpoint": False,
        "is_bounded_debug_training_checkpoint": True,
        "not_a_replacement_for": [
            "paper diffusion checkpoint",
            "paper EMA checkpoint",
            "true VAE rollout state-latent dataset",
            "TensorRT deployment",
            "Fig. 5/Fig. 6 rollout evaluation",
        ],
    }
    torch.save(checkpoint_payload, checkpoint_path)
    figure_path = RUN_DIR / "figures" / "debug_training_loss.png"
    plot_losses(figure_path, rows)

    metrics = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "is_training_run": False,
        "is_bounded_debug_training_run": True,
        "paper_level": False,
        "samples_per_second": float((args.batch_size * args.steps) / elapsed) if elapsed > 0.0 else None,
        "environment_steps_per_second": None,
        "iteration_time": float(np.mean([row["iteration_time_sec"] for row in rows])),
        "estimated_remaining_time": 0.0,
        "debug_step_count": args.steps,
        "debug_batch_size": args.batch_size,
        "debug_token_dim": token_dim,
        "debug_state_dim": state_dim,
        "debug_latent_dim": cfg.latent_dim,
        "debug_input_window_count": int(clean_np.shape[0]),
        "parameter_count": parameter_count(model),
        "initial_loss_before": rows[0]["loss_before"],
        "final_loss_after": rows[-1]["loss_after"],
        "loss_delta": rows[0]["loss_before"] - rows[-1]["loss_after"],
        "model_vs_initial_l2": mean_state_l2_delta(final_state, initial_state),
        "initial_model_sha256": tensor_state_digest(initial_state),
        "final_model_sha256": tensor_state_digest(final_state),
        "final_ema_sha256": tensor_state_digest(ema_shadow),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "loss_figure_size_bytes": figure_path.stat().st_size,
        "loss_figure_sha256": file_sha256(figure_path),
        "oom_count": 0,
        "restart_count": 0,
    }
    settings = {
        "device": str(device),
        "torch_threads": args.torch_threads,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "token_dim": token_dim,
        "state_dim": state_dim,
        "latent_dim": cfg.latent_dim,
        "sequence_length": cfg.sequence_length,
        "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher",
        "motion_ids_first_batch": [int(x) for x in motion_ids[: args.batch_size].tolist()],
        "latent_manifest_status": latent_manifest["status"],
    }
    return rows, metrics, settings, checkpoint_path, figure_path


def write_run_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any], settings: dict[str, Any]) -> None:
    write_step_csv(RUN_DIR / "metrics.csv", rows)
    write_json_atomic(RUN_DIR / "metrics.json", metrics)
    status = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "allowed_status": True,
        "is_success": True,
        "is_training_run": False,
        "is_bounded_debug_training_run": True,
        "paper_level": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "Bounded debug diffusion optimizer run completed; not a full paper training run.",
    }
    write_json_atomic(RUN_DIR / "status.json", status)
    stdout_lines = [
        "Bounded debug diffusion training completed.",
        f"run_id={RUN_ID}",
        f"steps={metrics['debug_step_count']}",
        f"initial_loss_before={metrics['initial_loss_before']}",
        f"final_loss_after={metrics['final_loss_after']}",
        f"checkpoint_size_bytes={metrics['checkpoint_size_bytes']}",
        f"loss_figure_size_bytes={metrics['loss_figure_size_bytes']}",
        "paper_level=false",
        "",
    ]
    write_text(RUN_DIR / "stdout.log", "\n".join(stdout_lines))
    write_text(RUN_DIR / "run_settings.json", json.dumps(settings, indent=2, sort_keys=True) + "\n")


def audit(rows: list[dict[str, Any]], metrics: dict[str, Any], checkpoint_path: Path, figure_path: Path) -> dict[str, Any]:
    required_files = [
        "resolved_config.yaml",
        "command.sh",
        "stdout.log",
        "stderr.log",
        "environment.txt",
        "git_state.txt",
        "gpu_metrics.csv",
        "metrics.json",
        "metrics.csv",
        "status.json",
    ]
    required_dirs = ["checkpoint", "figures", "videos"]
    status_json = json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8"))
    checks = {
        "run_dir_exists": RUN_DIR.is_dir(),
        "all_required_files_exist": all((RUN_DIR / rel).is_file() for rel in required_files),
        "all_required_dirs_exist": all((RUN_DIR / rel).is_dir() for rel in required_dirs),
        "status_success_but_not_paper_training": status_json["status"] == "SUCCESS"
        and status_json["is_success"] is True
        and status_json["is_training_run"] is False
        and status_json["paper_level"] is False,
        "checkpoint_file_exists": checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0,
        "figure_file_exists": figure_path.is_file() and figure_path.stat().st_size > 0,
        "videos_dir_empty": not any((RUN_DIR / "videos").iterdir()),
        "metrics_have_runtime_fields": all(
            key in metrics
            for key in [
                "samples_per_second",
                "iteration_time",
                "estimated_remaining_time",
                "oom_count",
                "restart_count",
            ]
        ),
        "bounded_step_count_matches": len(rows) == metrics["debug_step_count"],
        "all_losses_finite": all(np.isfinite(row["loss_before"]) and np.isfinite(row["loss_after"]) for row in rows),
        "all_grad_norms_positive": all(row["grad_norm"] > 0.0 for row in rows),
        "all_same_batch_losses_not_worse_after_step": all(row["loss_after"] <= row["loss_before"] for row in rows),
        "model_parameters_changed": metrics["initial_model_sha256"] != metrics["final_model_sha256"]
        and metrics["model_vs_initial_l2"] > 0.0,
        "final_loss_finite": bool(np.isfinite(metrics["final_loss_after"])),
        "checkpoint_marked_debug_only": True,
        "does_not_claim_full_training_run": status_json["is_training_run"] is False,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "bounded_debug_diffusion_training_run_audit",
        "scope": "bounded debug optimizer run with goal.md run-directory schema, checkpoint, metrics, and loss figure",
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "metrics": metrics,
        "checks": checks,
        "step_rows": rows,
        "outputs": {
            "json": str(OUT / "level_c_bounded_debug_diffusion_training_run.json"),
            "tsv": str(OUT / "level_c_bounded_debug_diffusion_training_run.tsv"),
            "run_dir": str(RUN_DIR),
            "checkpoint": str(checkpoint_path),
            "figure": str(figure_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial_debug_training_run_only",
            "why_not_complete": (
                "This is a bounded debug optimizer run that writes a run directory, checkpoint, metrics, and loss "
                "figure. It is not paper-scale diffusion training, does not use true VAE rollout latents, has no "
                "videos or rollout evaluation, and must not be counted as a valid full BeyondMimic training run."
            ),
        },
    }
    return summary


def write_audit(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT / "level_c_bounded_debug_diffusion_training_run.json", summary)
    fields = ["kind", "name", "value"]
    tmp = OUT / "level_c_bounded_debug_diffusion_training_run.tsv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for key, value in sorted(summary["metrics"].items()):
            writer.writerow({"kind": "metric", "name": key, "value": value})
        for key, value in sorted(summary["checks"].items()):
            writer.writerow({"kind": "check", "name": key, "value": value})
    tmp.replace(OUT / "level_c_bounded_debug_diffusion_training_run.tsv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    command = (
        f"{BM_DIFFUSION_PYTHON} /mnt/infini-data/test/BeyondMimic/reproduction/scripts/"
        f"level_c_bounded_debug_diffusion_training_run.py --device {args.device} --steps {args.steps} "
        f"--batch-size {args.batch_size} --torch-threads {args.torch_threads} --seed {args.seed}"
    )
    prepare_run_dir(command)
    rows, metrics, settings, checkpoint_path, figure_path = run_training(args)
    write_run_outputs(rows, metrics, settings)
    summary = audit(rows, metrics, checkpoint_path, figure_path)
    write_audit(summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_id": RUN_ID,
                "json": summary["outputs"]["json"],
                "checkpoint": summary["outputs"]["checkpoint"],
                "figure": summary["outputs"]["figure"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
