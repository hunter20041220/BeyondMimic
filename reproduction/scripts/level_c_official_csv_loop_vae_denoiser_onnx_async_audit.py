#!/usr/bin/env python3
"""Audit ONNXRuntime export and async inference for local official-csv-loop VAE/denoiser.

This is a local deployment-path audit for the resource-adjusted official-csv-loop
and official-importer-export conditional action VAE/state-latent denoiser
variants. It intentionally does not claim TensorRT, paper Mini-PC latency,
official BeyondMimic checkpoints, or robot deployment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
VARIANT = os.environ.get("BM_OFFICIAL_CSV_LOOP_ONNX_VARIANT", "standard")
if VARIANT not in {"standard", "full_bundle", "importer_export_full_bundle"}:
    raise ValueError(f"Unsupported BM_OFFICIAL_CSV_LOOP_ONNX_VARIANT={VARIANT!r}")
IS_FULL_BUNDLE = VARIANT == "full_bundle"
IS_IMPORTER_EXPORT_FULL_BUNDLE = VARIANT == "importer_export_full_bundle"
if IS_IMPORTER_EXPORT_FULL_BUNDLE:
    NAME_STEM = "official_importer_export_full_bundle"
elif IS_FULL_BUNDLE:
    NAME_STEM = "official_csv_loop_full_bundle"
else:
    NAME_STEM = "official_csv_loop"
EXPERIMENT_TYPE = f"{NAME_STEM}_vae_denoiser_onnx_async_audit"
OK_STATUS = f"ok_{EXPERIMENT_TYPE}"
OUT = ROOT / f"res/level_c/{NAME_STEM}_vae_denoiser_onnx_async"
if IS_IMPORTER_EXPORT_FULL_BUNDLE:
    VAE_TRAIN_JSON = (
        ROOT
        / "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
    )
    DENOISER_TRAIN_JSON = (
        ROOT
        / "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
        "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
    )
elif IS_FULL_BUNDLE:
    VAE_TRAIN_JSON = (
        ROOT
        / "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
        "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
    )
    DENOISER_TRAIN_JSON = (
        ROOT
        / "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
        "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
    )
else:
    VAE_TRAIN_JSON = (
        ROOT
        / "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
        "level_c_official_csv_loop_teacher_rollout_vae_training.json"
    )
    DENOISER_TRAIN_JSON = (
        ROOT
        / "res/level_c/official_csv_loop_state_latent_diffusion_training/"
        "level_c_official_csv_loop_state_latent_diffusion_training.json"
    )
SEED = 20260619
OPSET = 17


class ConditionalActionVAE(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(torch.cat([obs, action], dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        pred = self.decoder(torch.cat([obs, mu], dim=-1))
        return pred, mu, logvar


class VaeEncoderWrapper(nn.Module):
    def __init__(self, vae: ConditionalActionVAE) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.vae.encoder(torch.cat([obs, action], dim=-1))
        return stats.chunk(2, dim=-1)


class VaeDecoderWrapper(nn.Module):
    def __init__(self, vae: ConditionalActionVAE) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(torch.cat([obs, latent], dim=-1))


class StateLatentDenoiser(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(token_dim + steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, noisy: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(step_idx, num_classes=self.steps).to(noisy.dtype)
        return self.net(torch.cat([noisy, onehot], dim=-1))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def summarize_times_ms(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "stdev_ms": float(statistics.pstdev(times_ms)),
    }


def benchmark(label: str, fn: Callable[[], Any], warmup: int = 20, repeats: int = 120) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return {"component": label, "warmup": warmup, "repeats": repeats, **summarize_times_ms(times_ms)}


def export_and_check(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    metadata: dict[str, str],
) -> dict[str, Any]:
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
    )
    proto = onnx.load(str(onnx_path))
    existing = {item.key: item for item in proto.metadata_props}
    for key, value in metadata.items():
        prop = existing.get(key) or proto.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(proto, str(onnx_path))
    checked = onnx.load(str(onnx_path))
    onnx.checker.check_model(checked)
    return {
        "path": str(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "sha256": sha256(onnx_path),
        "metadata_keys": sorted(item.key for item in checked.metadata_props),
    }


def load_models() -> tuple[ConditionalActionVAE, StateLatentDenoiser, dict[str, Any]]:
    vae_train = load_json(VAE_TRAIN_JSON)
    denoiser_train = load_json(DENOISER_TRAIN_JSON)
    vae_run = Path(vae_train["outputs"]["run_dir"])
    denoiser_run = Path(denoiser_train["outputs"]["run_dir"])
    vae_checkpoint = vae_run / "resource_adjusted_teacher_rollout_action_vae.pt"
    denoiser_checkpoint = denoiser_run / "resource_adjusted_state_latent_denoiser.pt"

    vae_payload = torch.load(vae_checkpoint, map_location="cpu", weights_only=False)
    denoiser_payload = torch.load(denoiser_checkpoint, map_location="cpu", weights_only=False)
    vae_cfg = vae_payload["config"]
    denoiser_cfg = denoiser_payload["config"]

    vae = ConditionalActionVAE(
        int(vae_cfg["obs_dim"]),
        int(vae_cfg["action_dim"]),
        int(vae_cfg["latent_dim"]),
        int(vae_cfg["hidden_dim"]),
    )
    vae.load_state_dict(vae_payload["model_state_dict"])
    vae.eval()

    denoiser = StateLatentDenoiser(
        int(denoiser_cfg["token_dim"]),
        int(denoiser_cfg["hidden_dim"]),
        int(denoiser_cfg["denoising_steps"]),
    )
    denoiser.load_state_dict(denoiser_payload["model_state_dict"])
    denoiser.eval()

    model_info = {
        "vae_training_json": str(VAE_TRAIN_JSON),
        "denoiser_training_json": str(DENOISER_TRAIN_JSON),
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": sha256(vae_checkpoint),
        "denoiser_checkpoint": str(denoiser_checkpoint),
        "denoiser_checkpoint_sha256": sha256(denoiser_checkpoint),
        "vae_config": vae_cfg,
        "denoiser_config": denoiser_cfg,
        "vae_parameter_count": int(sum(p.numel() for p in vae.parameters())),
        "denoiser_parameter_count": int(sum(p.numel() for p in denoiser.parameters())),
    }
    return vae, denoiser, model_info


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t" if path.suffix == ".tsv" else ",",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(SEED)
    vae, denoiser, model_info = load_models()
    vae_cfg = model_info["vae_config"]
    denoiser_cfg = model_info["denoiser_config"]

    obs = torch.from_numpy(rng.normal(0.0, 0.25, (8, int(vae_cfg["obs_dim"]))).astype(np.float32))
    action = torch.from_numpy(rng.normal(0.0, 0.25, (8, int(vae_cfg["action_dim"]))).astype(np.float32))
    latent = torch.from_numpy(rng.normal(0.0, 0.25, (8, int(vae_cfg["latent_dim"]))).astype(np.float32))
    noisy = torch.from_numpy(
        rng.normal(
            0.0,
            0.25,
            (8, int(denoiser_cfg["sequence_length"]), int(denoiser_cfg["token_dim"])),
        ).astype(np.float32)
    )
    step_idx_np = rng.integers(
        0,
        int(denoiser_cfg["denoising_steps"]),
        size=(8, int(denoiser_cfg["sequence_length"])),
        dtype=np.int64,
    )
    step_idx = torch.from_numpy(step_idx_np)

    encoder = VaeEncoderWrapper(vae).eval()
    decoder = VaeDecoderWrapper(vae).eval()
    with torch.inference_mode():
        torch_mu, torch_logvar = encoder(obs, action)
        torch_action = decoder(obs, latent)
        torch_denoised = denoiser(noisy, step_idx)

    metadata = {
        "project": "BeyondMimic local reproduction",
        "experiment_type": EXPERIMENT_TYPE,
        "variant": VARIANT,
        "paper_level": "false",
        "official_checkpoint": "false",
        "tensorrt": "false",
        "real_robot": "false",
    }
    onnx_exports = {
        "vae_encoder": export_and_check(
            encoder,
            (obs, action),
            OUT / f"{NAME_STEM}_vae_encoder_local.onnx",
            ["obs", "action"],
            ["latent_mu", "latent_logvar"],
            {
                "obs": {0: "batch"},
                "action": {0: "batch"},
                "latent_mu": {0: "batch"},
                "latent_logvar": {0: "batch"},
            },
            metadata,
        ),
        "vae_decoder": export_and_check(
            decoder,
            (obs, latent),
            OUT / f"{NAME_STEM}_vae_decoder_local.onnx",
            ["obs", "latent"],
            ["decoded_action"],
            {
                "obs": {0: "batch"},
                "latent": {0: "batch"},
                "decoded_action": {0: "batch"},
            },
            metadata,
        ),
        "state_latent_denoiser": export_and_check(
            denoiser,
            (noisy, step_idx),
            OUT / f"{NAME_STEM}_state_latent_denoiser_local.onnx",
            ["noisy_state_latent", "diffusion_step"],
            ["predicted_clean_state_latent"],
            {
                "noisy_state_latent": {0: "batch", 1: "sequence"},
                "diffusion_step": {0: "batch", 1: "sequence"},
                "predicted_clean_state_latent": {0: "batch", 1: "sequence"},
            },
            metadata,
        ),
    }

    providers = ort.get_available_providers()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    provider_list = ["CPUExecutionProvider"]
    encoder_session = ort.InferenceSession(
        onnx_exports["vae_encoder"]["path"],
        sess_options=session_options,
        providers=provider_list,
    )
    decoder_session = ort.InferenceSession(
        onnx_exports["vae_decoder"]["path"],
        sess_options=session_options,
        providers=provider_list,
    )
    denoiser_session = ort.InferenceSession(
        onnx_exports["state_latent_denoiser"]["path"],
        sess_options=session_options,
        providers=provider_list,
    )

    obs_np = obs.numpy().astype(np.float32)
    action_np = action.numpy().astype(np.float32)
    latent_np = latent.numpy().astype(np.float32)
    noisy_np = noisy.numpy().astype(np.float32)
    step_np = step_idx.numpy().astype(np.int64)
    ort_mu, ort_logvar = encoder_session.run(None, {"obs": obs_np, "action": action_np})
    (ort_action,) = decoder_session.run(None, {"obs": obs_np, "latent": latent_np})
    (ort_denoised,) = denoiser_session.run(None, {"noisy_state_latent": noisy_np, "diffusion_step": step_np})

    consistency = {
        "vae_encoder_mu_max_abs_onnx_vs_torch": float(np.max(np.abs(ort_mu - torch_mu.numpy()))),
        "vae_encoder_logvar_max_abs_onnx_vs_torch": float(np.max(np.abs(ort_logvar - torch_logvar.numpy()))),
        "vae_decoder_action_max_abs_onnx_vs_torch": float(np.max(np.abs(ort_action - torch_action.numpy()))),
        "denoiser_token_max_abs_onnx_vs_torch": float(np.max(np.abs(ort_denoised - torch_denoised.numpy()))),
    }

    latency_rows = [
        {"backend": "pytorch_cpu", **benchmark("vae_encoder", lambda: encoder(obs, action))},
        {"backend": "pytorch_cpu", **benchmark("vae_decoder", lambda: decoder(obs, latent))},
        {"backend": "pytorch_cpu", **benchmark("state_latent_denoiser", lambda: denoiser(noisy, step_idx))},
        {
            "backend": "onnxruntime_cpu",
            **benchmark("vae_encoder", lambda: encoder_session.run(None, {"obs": obs_np, "action": action_np})),
        },
        {
            "backend": "onnxruntime_cpu",
            **benchmark("vae_decoder", lambda: decoder_session.run(None, {"obs": obs_np, "latent": latent_np})),
        },
        {
            "backend": "onnxruntime_cpu",
            **benchmark(
                "state_latent_denoiser",
                lambda: denoiser_session.run(
                    None,
                    {"noisy_state_latent": noisy_np, "diffusion_step": step_np},
                ),
            ),
        },
    ]

    def run_pipeline_once() -> None:
        mu, _ = encoder_session.run(None, {"obs": obs_np, "action": action_np})
        denoiser_session.run(None, {"noisy_state_latent": noisy_np, "diffusion_step": step_np})
        decoder_session.run(None, {"obs": obs_np, "latent": mu.astype(np.float32)})

    sequential_pipeline = benchmark("onnxruntime_pipeline_sequential", run_pipeline_once, warmup=10, repeats=80)
    request_count = 80
    worker_count = 4
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_pipeline_once) for _ in range(request_count)]
        for future in futures:
            future.result()
    async_elapsed = time.perf_counter() - start
    async_summary = {
        "component": "onnxruntime_pipeline_async_thread_pool",
        "requests": request_count,
        "workers": worker_count,
        "elapsed_seconds": async_elapsed,
        "throughput_requests_per_second": request_count / max(async_elapsed, 1e-12),
        "sequential_mean_ms_per_request": sequential_pipeline["mean_ms"],
        "async_mean_ms_per_request": (async_elapsed * 1000.0) / request_count,
        "throughput_speedup_vs_sequential_mean": sequential_pipeline["mean_ms"]
        / max((async_elapsed * 1000.0) / request_count, 1e-12),
    }

    checks = {
        "vae_training_source_ok": load_json(VAE_TRAIN_JSON)["status"]
        == {
            "standard": "ok_official_csv_loop_teacher_rollout_vae_training",
            "full_bundle": "ok_official_csv_loop_full_bundle_teacher_rollout_vae_training",
            "importer_export_full_bundle": (
                "ok_official_importer_export_full_bundle_teacher_rollout_vae_training"
            ),
        }[VARIANT],
        "denoiser_training_source_ok": load_json(DENOISER_TRAIN_JSON)["status"]
        == {
            "standard": "ok_official_csv_loop_state_latent_diffusion_training",
            "full_bundle": "ok_official_csv_loop_full_bundle_state_latent_diffusion_training",
            "importer_export_full_bundle": (
                "ok_official_importer_export_full_bundle_state_latent_diffusion_training"
            ),
        }[VARIANT],
        "onnx_files_written": all(Path(item["path"]).is_file() and item["size_bytes"] > 0 for item in onnx_exports.values()),
        "onnx_checker_passed": True,
        "onnxruntime_cpu_available": "CPUExecutionProvider" in providers,
        "onnxruntime_cuda_unavailable_recorded": "CUDAExecutionProvider" not in providers,
        "onnxruntime_tensorrt_unavailable_recorded": "TensorrtExecutionProvider" not in providers,
        "vae_encoder_matches_torch": consistency["vae_encoder_mu_max_abs_onnx_vs_torch"] <= 1e-5
        and consistency["vae_encoder_logvar_max_abs_onnx_vs_torch"] <= 1e-5,
        "vae_decoder_matches_torch": consistency["vae_decoder_action_max_abs_onnx_vs_torch"] <= 1e-5,
        "denoiser_matches_torch": consistency["denoiser_token_max_abs_onnx_vs_torch"] <= 1e-5,
        "latency_rows_written": True,
        "async_pipeline_completed": async_summary["requests"] == request_count,
        "does_not_claim_tensorrt": True,
        "does_not_claim_paper_latency": True,
        "does_not_claim_official_checkpoint": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }

    rows: list[dict[str, Any]] = []
    for row in latency_rows:
        rows.append({**row, "claim_level": "local_cpu_onnxruntime_or_pytorch_microbenchmark"})
    rows.append(
        {
            "backend": "onnxruntime_cpu",
            **sequential_pipeline,
            "claim_level": "local_cpu_onnxruntime_sequential_pipeline_microbenchmark",
        }
    )
    rows.append(
        {
            "backend": "onnxruntime_cpu_threadpool",
            "component": async_summary["component"],
            "mean_ms": async_summary["async_mean_ms_per_request"],
            "median_ms": "",
            "p95_ms": "",
            "min_ms": "",
            "max_ms": "",
            "stdev_ms": "",
            "warmup": 0,
            "repeats": async_summary["requests"],
            "claim_level": "local_cpu_onnxruntime_async_threadpool_throughput_probe",
        }
    )

    csv_path = OUT / f"level_c_{NAME_STEM}_vae_denoiser_onnx_async_latency.csv"
    tsv_path = OUT / f"level_c_{NAME_STEM}_vae_denoiser_onnx_async_audit.tsv"
    write_rows(csv_path, rows)
    write_rows(tsv_path, rows)

    summary = {
        "status": OK_STATUS if all(checks.values()) else "check_failed",
        "experiment_type": EXPERIMENT_TYPE,
        "scope": (
            f"Exports the local {NAME_STEM.replace('_', '-')} conditional action VAE encoder/decoder and state-latent denoiser "
            "to ONNX, verifies ONNXRuntime CPU outputs against PyTorch, and measures local sequential/async "
            "microbenchmarks. This is not TensorRT or paper Mini-PC deployment."
        ),
        "settings": {
            "python": str(BM_DIFFUSION_PY),
            "variant": VARIANT,
            "seed": SEED,
            "opset": OPSET,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "onnxruntime_available_providers": providers,
            "onnxruntime_execution_providers_used": provider_list,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "formal_gpu_experiment": False,
            "formal_gpu_experiment_reason": (
                "This is an export/runtime microbenchmark using ONNXRuntime CPU providers; the local ORT build has "
                "no CUDAExecutionProvider or TensorRT provider, and the audit is not a training or full rollout run."
            ),
        },
        "models": model_info,
        "onnx_exports": onnx_exports,
        "consistency": consistency,
        "latency_rows": rows,
        "async_summary": async_summary,
        "checks": checks,
        "outputs": {
            "json": str(OUT / f"level_c_{NAME_STEM}_vae_denoiser_onnx_async_audit.json"),
            "tsv": str(tsv_path),
            "csv": str(csv_path),
            "onnx_files": {key: value["path"] for key, value in onnx_exports.items()},
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_cpu_onnxruntime_deployment_path_audit",
            "why_not_complete": (
                f"The audit uses locally trained resource-adjusted {NAME_STEM.replace('_', '-')} VAE and denoiser checkpoints. "
                "It verifies graph export and CPU ONNXRuntime execution, plus a thread-pool async proxy. It does "
                "not use official BeyondMimic checkpoints, TensorRT, CppAD guidance, RTX 4060 Mini-PC hardware, "
                "IsaacLab live control-loop deployment, Fig. 5/Fig. 6 task metrics, or a real robot."
            ),
        },
    }
    write_json_atomic(OUT / f"level_c_{NAME_STEM}_vae_denoiser_onnx_async_audit.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "providers": providers,
                "decoder_max_abs": consistency["vae_decoder_action_max_abs_onnx_vs_torch"],
                "denoiser_max_abs": consistency["denoiser_token_max_abs_onnx_vs_torch"],
                "async_speedup": async_summary["throughput_speedup_vs_sequential_mean"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
