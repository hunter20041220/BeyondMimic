#!/usr/bin/env python3
"""Debug-only receding-horizon VAE decoder probe for Level C inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_REVERSE_NPZ = ROOT / "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.npz"
OUT = ROOT / "res/level_c/receding_horizon_decoder_probe"


@dataclass(frozen=True)
class DecoderProbeConfig:
    seed: int = 20260831
    history: int = 4
    horizon: int = 16
    state_dim: int = 181
    latent_dim: int = 32
    proprioception_dim: int = 96
    action_dim: int = 29
    decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    control_frequency_hz: float = 25.0

    @property
    def sequence_length(self) -> int:
        return self.history + 1 + self.horizon

    @property
    def current_index(self) -> int:
        return self.history

    @property
    def decoder_input_dim(self) -> int:
        return self.latent_dim + self.proprioception_dim

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_frequency_hz


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, hidden), nn.ELU()])
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def make_candidate_proprioception(current_state: np.ndarray, cfg: DecoderProbeConfig) -> np.ndarray:
    # Candidate proprioception vector matching the synthetic VAE smoke dimension:
    # projected gravity + IMU velocities + joint position/velocity + last action/padding.
    # This is a schema probe; the exact paper proprioception layout remains checkpoint/code dependent.
    proprio = np.zeros(cfg.proprioception_dim, dtype=np.float32)
    fill = min(cfg.proprioception_dim, current_state.shape[0])
    proprio[:fill] = current_state[:fill].astype(np.float32)
    return proprio


def parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def write_tsv(path: Path, rows: dict[str, Any]) -> None:
    flat: list[tuple[str, str]] = []

    def rec(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                rec(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, list):
            flat.append((prefix, json.dumps(value, sort_keys=True)))
        else:
            flat.append((prefix, str(value)))

    rec("", rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse-npz", type=Path, default=DEFAULT_REVERSE_NPZ)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    cfg = DecoderProbeConfig(seed=args.seed)
    seed_everything(cfg.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(args.reverse_npz)
    tau_final = data["tau_guided_final"] if "tau_guided_final" in data.files else data["tau_after_full_oracle_reverse"]
    if tau_final.shape != (cfg.sequence_length, cfg.state_dim + cfg.latent_dim):
        raise ValueError(f"unexpected tau shape {tau_final.shape}")

    current_token = tau_final[cfg.current_index]
    future_token = tau_final[min(cfg.current_index + 1, cfg.sequence_length - 1)]
    current_state = current_token[: cfg.state_dim]
    current_latent = current_token[cfg.state_dim :]
    future_latent = future_token[cfg.state_dim :]
    proprioception = make_candidate_proprioception(current_state, cfg)

    decoder = mlp(cfg.decoder_input_dim, cfg.decoder_hidden_dims, cfg.action_dim).to(device)
    decoder.eval()
    with torch.no_grad():
        current_input = torch.from_numpy(np.concatenate([current_latent, proprioception]).astype(np.float32)).to(device)
        future_input = torch.from_numpy(np.concatenate([future_latent, proprioception]).astype(np.float32)).to(device)
        current_action = decoder(current_input.unsqueeze(0)).squeeze(0).cpu().numpy()
        next_latent_action_for_contrast = decoder(future_input.unsqueeze(0)).squeeze(0).cpu().numpy()

    action_delta_if_wrong_future_latent = float(np.linalg.norm(current_action - next_latent_action_for_contrast))
    json_path = OUT / "level_c_receding_horizon_decoder_probe.json"
    tsv_path = OUT / "level_c_receding_horizon_decoder_probe.tsv"
    npz_path = OUT / "level_c_receding_horizon_decoder_probe.npz"
    np.savez_compressed(
        npz_path,
        tau_final=tau_final,
        current_state=current_state,
        current_latent=current_latent,
        candidate_proprioception=proprioception,
        current_action=current_action,
        next_latent_action_for_contrast=next_latent_action_for_contrast,
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "receding-horizon current-latent VAE decoder inference schema probe",
        "paper_evidence": {
            "decoder_equation": str(ROOT / "reproduction/paper/source/tex/method.tex:157-162"),
            "current_latent_action": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "control_frequency_and_cpu_decoder": str(ROOT / "reproduction/paper/source/root.tex:589-593"),
        },
        "not_a_replacement_for": [
            "trained VAE decoder checkpoint",
            "paper-exact proprioception layout",
            "TensorRT/asynchronous deployment",
            "closed-loop policy rollout",
            "real-time latency benchmark",
        ],
        "settings": asdict(cfg) | {"device": str(device), "source_npz": str(args.reverse_npz)},
        "model": {
            "decoder_parameter_count": parameter_count(decoder),
            "decoder_input_dim": cfg.decoder_input_dim,
            "decoder_output_dim": cfg.action_dim,
        },
        "metrics": {
            "tau_shape": list(tau_final.shape),
            "current_index": cfg.current_index,
            "control_dt_seconds": cfg.control_dt,
            "current_latent_norm": float(np.linalg.norm(current_latent)),
            "candidate_proprioception_norm": float(np.linalg.norm(proprioception)),
            "current_action_norm": float(np.linalg.norm(current_action)),
            "action_delta_if_wrong_future_latent": action_delta_if_wrong_future_latent,
        },
        "checks": {
            "tau_shape_matches_history_horizon": list(tau_final.shape) == [cfg.sequence_length, cfg.state_dim + cfg.latent_dim],
            "current_index_is_history": cfg.current_index == cfg.history,
            "latent_dim_matches_paper": current_latent.shape[0] == cfg.latent_dim,
            "proprioception_dim_matches_smoke_contract": proprioception.shape[0] == cfg.proprioception_dim,
            "action_dim_matches_g1_joint_count": current_action.shape[0] == cfg.action_dim,
            "decoder_runs_on_cpu_by_default": str(device) == "cpu",
            "control_frequency_is_25hz": math.isclose(cfg.control_frequency_hz, 25.0),
            "uses_current_latent_only_for_action": bool(action_delta_if_wrong_future_latent > 0.0),
            "action_is_finite": bool(np.all(np.isfinite(current_action))),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
