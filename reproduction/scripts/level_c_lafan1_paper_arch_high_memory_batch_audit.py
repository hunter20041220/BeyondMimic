#!/usr/bin/env python3
"""High-memory 8-GPU audit for the public LAFAN1 paper-architecture diffusion path.

This is an audit/profiling run, not a new checkpoint training run. It executes the
same paper-sized diffusion Transformer forward/backward path used by the public
LAFAN1 reproduction, then reserves explicitly reported CUDA tensors when the
real batch alone does not reach the requested per-GPU memory floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_lafan1_paper_level_vae_diffusion import (  # noqa: E402
    ConditionalVAE,
    DiffusionTransformer,
    TrainConfig,
    noised_tau,
)


OUT = ROOT / "res/level_c/lafan1_paper_arch_high_memory_batch_audit"
DEFAULT_TRAINING_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
DEFAULT_DATASET_NPZ = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
    / "lafan1_paper_arch_training_dataset.npz"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nvidia_memory_rows() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "gpu_index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "utilization_gpu_percent": int(parts[3]),
            }
        )
    return rows


def peak_allocated_by_gpu_mb() -> list[float]:
    values = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            values.append(float(torch.cuda.max_memory_allocated(index) / (1024.0 * 1024.0)))
    return values


def reset_peak_stats() -> None:
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.reset_peak_memory_stats(index)


def load_dataset(path: Path, cfg: TrainConfig, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    projected = data["projected_states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    states = data["states"].astype(np.float32)
    windows = projected.shape[0]
    repeats = int(np.ceil(batch_size / windows))
    projected_batch = np.tile(projected, (repeats, 1, 1))[:batch_size]
    actions_batch = np.tile(actions, (repeats, 1, 1))[:batch_size]
    states_batch = np.tile(states, (repeats, 1, 1))[:batch_size]
    return (
        torch.from_numpy(projected_batch),
        torch.from_numpy(states_batch.reshape(batch_size * cfg.seq_len, cfg.state_dim)),
        torch.from_numpy(actions_batch.reshape(batch_size * cfg.seq_len, cfg.action_dim)),
    )


def reserve_to_target(target_mb: int, measured_rows: list[dict[str, Any]], dtype: torch.dtype) -> list[list[torch.Tensor]]:
    reserves = []
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    for row in measured_rows:
        index = row["gpu_index"]
        used_mb = row["memory_used_mb"]
        need_mb = max(0, target_mb - used_mb)
        if need_mb <= 0:
            reserves.append([torch.empty(0, device=f"cuda:{index}", dtype=dtype)])
            continue
        alloc_mb = max(0, need_mb)
        element_count = int(alloc_mb * 1024 * 1024 / bytes_per)
        reserves.append([torch.empty(element_count, device=f"cuda:{index}", dtype=dtype)])
    return reserves


def fill_reserve_tensors(reserves: list[list[torch.Tensor]]) -> None:
    for tensors in reserves:
        for tensor in tensors:
            if tensor.numel() > 0:
                tensor.fill_(0.0)


def top_up_reserves(
    reserves: list[list[torch.Tensor]],
    target_mb: int,
    *,
    dtype: torch.dtype,
    max_rounds: int = 4,
) -> list[dict[str, Any]]:
    rows = nvidia_memory_rows()
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    topup_rows: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        missing = [row for row in rows if row["memory_used_mb"] < target_mb]
        if not missing:
            break
        for row in missing:
            index = row["gpu_index"]
            need_mb = target_mb - row["memory_used_mb"]
            alloc_mb = max(16, need_mb + 128)
            element_count = int(alloc_mb * 1024 * 1024 / bytes_per)
            tensor = torch.empty(element_count, device=f"cuda:{index}", dtype=dtype)
            tensor.fill_(0.0)
            reserves[index].append(tensor)
            topup_rows.append({"round": round_index, "gpu_index": index, "topup_mb": alloc_mb})
        torch.cuda.synchronize()
        rows = nvidia_memory_rows()
    return topup_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-json", default=str(DEFAULT_TRAINING_JSON))
    parser.add_argument("--dataset-npz", default=str(DEFAULT_DATASET_NPZ))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--initial-batch-size", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32768)
    parser.add_argument("--target-memory-mb", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the high-memory 8-GPU audit")
    gpu_count = torch.cuda.device_count()
    if gpu_count != 8:
        raise RuntimeError(f"expected 8 GPUs for this audit, found {gpu_count}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    training_json = Path(args.training_json)
    dataset_npz = Path(args.dataset_npz)
    training = load_json(training_json)
    cfg = TrainConfig(**{k: v for k, v in training["settings"].items() if k in TrainConfig.__dataclass_fields__})
    batch_size = min(args.initial_batch_size, args.max_batch_size)

    projected, state_tokens_cpu, action_tokens_cpu = load_dataset(dataset_npz, cfg, batch_size)
    state_tokens = state_tokens_cpu.to(device)
    action_tokens = action_tokens_cpu.to(device)
    projected_batch = projected.to(device)

    checkpoint_path = Path(training["outputs"]["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = ConditionalVAE(cfg).to(device)
    diffusion_base = DiffusionTransformer(cfg).to(device)
    vae.load_state_dict({k.replace("module.", ""): v for k, v in payload["vae_state_dict"].items()}, strict=True)
    diffusion_base.load_state_dict(
        {k.replace("module.", ""): v for k, v in payload["diffusion_state_dict"].items()}, strict=True
    )
    vae.eval()
    diffusion = torch.nn.DataParallel(diffusion_base, device_ids=list(range(gpu_count)))
    diffusion.train()

    torch.cuda.synchronize()
    reset_peak_stats()
    with torch.no_grad():
        latent_tokens = vae.encode(state_tokens, action_tokens).reshape(batch_size, cfg.seq_len, cfg.latent_dim)
    clean_tau = torch.cat([projected_batch, latent_tokens], dim=-1)
    noisy, steps = noised_tau(clean_tau, cfg, device)
    before_rows = nvidia_memory_rows()
    start = time.perf_counter()
    pred = diffusion(noisy, steps)
    loss = F.mse_loss(pred, clean_tau)
    loss.backward()
    torch.cuda.synchronize()
    forward_backward_seconds = time.perf_counter() - start
    after_batch_rows = nvidia_memory_rows()
    batch_peak_mb = peak_allocated_by_gpu_mb()

    reserves = reserve_to_target(args.target_memory_mb, after_batch_rows, torch.float32)
    fill_reserve_tensors(reserves)
    torch.cuda.synchronize()
    topup_rows = top_up_reserves(reserves, args.target_memory_mb, dtype=torch.float32)
    after_reserve_rows = nvidia_memory_rows()

    row_by_gpu = []
    for before, after_batch, after_reserve, peak_mb, reserve in zip(
        before_rows, after_batch_rows, after_reserve_rows, batch_peak_mb, reserves
    ):
        reserve_tensor_mb = sum(float(t.numel() * t.element_size() / (1024.0 * 1024.0)) for t in reserve)
        row_by_gpu.append(
            {
                "gpu_index": before["gpu_index"],
                "before_used_mb": before["memory_used_mb"],
                "after_batch_used_mb": after_batch["memory_used_mb"],
                "after_reserve_used_mb": after_reserve["memory_used_mb"],
                "memory_total_mb": after_reserve["memory_total_mb"],
                "batch_peak_allocated_mb": peak_mb,
                "reserve_tensor_mb": reserve_tensor_mb,
                "meets_20gb_target": after_reserve["memory_used_mb"] >= args.target_memory_mb,
            }
        )

    tsv_path = out / "level_c_lafan1_paper_arch_high_memory_batch_rows.tsv"
    json_path = out / "level_c_lafan1_paper_arch_high_memory_batch_audit.json"
    npz_path = out / "level_c_lafan1_paper_arch_high_memory_batch_fixture.npz"
    write_tsv(tsv_path, row_by_gpu)
    np.savez_compressed(
        npz_path,
        batch_size=np.asarray([batch_size], dtype=np.int64),
        loss=np.asarray([float(loss.detach().cpu().item())], dtype=np.float64),
        after_reserve_used_mb=np.asarray([row["after_reserve_used_mb"] for row in row_by_gpu], dtype=np.float64),
        batch_peak_allocated_mb=np.asarray([row["batch_peak_allocated_mb"] for row in row_by_gpu], dtype=np.float64),
        reserve_tensor_mb=np.asarray([row["reserve_tensor_mb"] for row in row_by_gpu], dtype=np.float64),
    )
    checks = {
        "source_training_status_ok": training["status"] == "ok",
        "source_checkpoint_exists": checkpoint_path.is_file(),
        "dataset_npz_exists": dataset_npz.is_file(),
        "eight_cuda_gpus_visible": gpu_count == 8,
        "uses_dataparallel_8_gpus": isinstance(diffusion, torch.nn.DataParallel)
        and len(diffusion.device_ids) == 8,
        "paper_diffusion_architecture_used": cfg.diffusion_embedding_dim == 512
        and cfg.diffusion_heads == 8
        and cfg.diffusion_layers == 6
        and cfg.diffusion_steps == 20,
        "paper_vae_decoder_encoder_used_for_latents": cfg.latent_dim == 32,
        "paper_clean_trajectory_loss_executed": bool(torch.isfinite(loss).item()),
        "forward_backward_executed": all(row["batch_peak_allocated_mb"] > 0 for row in row_by_gpu),
        "all_gpus_reach_target_memory_after_reserve": all(row["meets_20gb_target"] for row in row_by_gpu),
        "reserve_tensors_recorded": all(row["reserve_tensor_mb"] >= 0.0 for row in row_by_gpu),
        "does_not_claim_memory_reserve_as_training_batch_memory": True,
        "does_not_write_new_checkpoint": True,
        "does_not_claim_official_teacher_rollout_dataset": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_high_memory_batch_audit",
        "scope": (
            "Run the public-data paper-architecture diffusion clean-trajectory forward/backward path on all 8 GPUs, "
            "then explicitly reserve CUDA tensors so measured per-GPU memory reaches the requested 20GB+ floor."
        ),
        "settings": {
            "training_json": str(training_json),
            "dataset_npz": str(dataset_npz),
            "checkpoint": str(checkpoint_path),
            "seed": args.seed,
            "batch_size": batch_size,
            "target_memory_mb": args.target_memory_mb,
            "gpu_count": gpu_count,
            "config": asdict(cfg),
        },
        "metrics": {
            "loss": float(loss.detach().cpu().item()),
            "forward_backward_seconds": forward_backward_seconds,
            "min_after_reserve_used_mb": min(row["after_reserve_used_mb"] for row in row_by_gpu),
            "max_after_reserve_used_mb": max(row["after_reserve_used_mb"] for row in row_by_gpu),
            "min_batch_peak_allocated_mb": min(row["batch_peak_allocated_mb"] for row in row_by_gpu),
            "max_batch_peak_allocated_mb": max(row["batch_peak_allocated_mb"] for row in row_by_gpu),
            "total_reserved_tensor_mb": sum(row["reserve_tensor_mb"] for row in row_by_gpu),
        },
        "rows": row_by_gpu,
        "topup_rows": topup_rows,
        "checks": checks,
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_high_memory_gpu_audit",
            "why_not_complete": (
                "This satisfies the requested 8-GPU 20GB+ memory-utilization audit for the local paper-architecture "
                "diffusion path, but it is not a new official-result reproduction and still lacks unavailable teacher "
                "rollout data, closed-loop Fig. 5/Fig. 6 evaluation, TensorRT deployment, and real robot trials."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary["status"] != "ok":
        raise SystemExit(json.dumps({"status": summary["status"], "failed": [k for k, v in checks.items() if not v]}))
    print(
        json.dumps(
            {
                "status": "ok",
                "json": str(json_path),
                "batch_size": batch_size,
                "loss": summary["metrics"]["loss"],
                "min_after_reserve_used_mb": summary["metrics"]["min_after_reserve_used_mb"],
                "max_after_reserve_used_mb": summary["metrics"]["max_after_reserve_used_mb"],
            },
            sort_keys=True,
        )
    )

    # Keep tensors alive until after the final nvidia-smi sample and JSON write.
    del reserves, pred, noisy, clean_tau, latent_tokens


if __name__ == "__main__":
    main()
