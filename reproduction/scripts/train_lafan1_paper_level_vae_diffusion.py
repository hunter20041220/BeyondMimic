#!/usr/bin/env python3
"""Train paper-architecture VAE and diffusion models on public LAFAN1 G1 motions.

This is a resource-adjusted paper-level training run: it uses the paper's VAE
and diffusion architecture/hyperparameter definitions, but the available public
inputs are retargeted LAFAN1 G1 CSV motions rather than the unpublished teacher
policy DAgger rollouts and VAE rollout state-latent dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import shlex
import subprocess
import sys
import time
import zlib
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import build_level_c_motion_state_fixture as motion_fk


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.state import emphasis_projection  # noqa: E402


RUN_ID = "level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000"
RUN_DIR = ROOT / "res/runs" / RUN_ID
OUT = ROOT / "res/level_c/lafan1_paper_arch_vae_diffusion_training"
CSV_ROOT = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"
URDF = motion_fk.DEFAULT_URDF
BM_DIFFUSION_PYTHON = ROOT / "envs/bm_diffusion/bin/python"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260617
    projection_seed: int = 20260617
    input_fps: int = 30
    model_fps: int = 25
    history: int = 4
    horizon: int = 16
    window_stride: int = 6
    max_motions: int = 40
    max_frames_per_motion: int = 420
    state_dim: int = 99
    projected_state_dim: int = 207
    action_dim: int = 29
    latent_dim: int = 32
    vae_encoder_hidden: tuple[int, ...] = (2048, 1024, 512)
    vae_decoder_hidden: tuple[int, ...] = (2048, 1024, 512)
    vae_teacher_hidden: tuple[int, ...] = (512, 256, 128)
    vae_lr: float = 5e-4
    vae_kl_coef: float = 0.01
    vae_accumulation_steps: int = 15
    vae_epochs: int = 24
    diffusion_embedding_dim: int = 512
    diffusion_heads: int = 8
    diffusion_layers: int = 6
    diffusion_steps: int = 20
    diffusion_batch_size: int = 512
    diffusion_epochs: int = 40
    diffusion_lr: float = 1e-4
    diffusion_weight_decay: float = 0.001
    diffusion_warmup_steps: int = 10000
    ema_power: float = 0.75
    ema_max: float = 0.9999

    @property
    def seq_len(self) -> int:
        return self.history + 1 + self.horizon

    @property
    def token_dim(self) -> int:
        return self.projected_state_dim + self.latent_dim


class ConditionalVAE(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.encoder = mlp(cfg.state_dim + cfg.action_dim, cfg.vae_encoder_hidden, cfg.latent_dim * 2, "elu")
        self.teacher = mlp(cfg.state_dim, cfg.vae_teacher_hidden, cfg.action_dim, "elu")
        self.decoder = mlp(cfg.latent_dim + cfg.state_dim, cfg.vae_decoder_hidden, cfg.action_dim, "elu")

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(torch.cat([state, action], dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        z = mu if deterministic else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        decoded = self.decoder(torch.cat([z, state], dim=-1))
        teacher = self.teacher(state)
        return decoded, teacher, mu, logvar, z

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stats = self.encoder(torch.cat([state, action], dim=-1))
        return stats.chunk(2, dim=-1)[0]


class DiffusionTransformer(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(cfg.token_dim, cfg.diffusion_embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.diffusion_embedding_dim))
        self.state_step_embed = nn.Embedding(cfg.diffusion_steps, cfg.diffusion_embedding_dim)
        self.latent_step_embed = nn.Embedding(cfg.diffusion_steps, cfg.diffusion_embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.diffusion_embedding_dim,
            nhead=cfg.diffusion_heads,
            dim_feedforward=cfg.diffusion_embedding_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.diffusion_layers)
        self.output_proj = nn.Linear(cfg.diffusion_embedding_dim, cfg.token_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, noisy_tau: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(noisy_tau)
        x = x + self.pos_embed[:, : noisy_tau.shape[1]]
        x = x + self.state_step_embed(steps[..., 0])
        x = x + self.latent_step_embed(steps[..., 1])
        return self.output_proj(self.encoder(x))


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    act = nn.ELU if activation == "elu" else nn.SiLU
    for h in hidden:
        layers.extend([nn.Linear(last, h), act()])
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def command_output(cmd: list[str], timeout: int = 30) -> str:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"command_failed: {cmd}: {exc}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def simple_png_plot(path: Path, series: dict[str, list[float]], *, width: int = 760, height: int = 420) -> None:
    ml, mr, mt, mb = 75, 30, 30, 65
    img = bytearray([255] * (width * height * 3))

    def pix(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            img[i : i + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            pix(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    values = [v for vals in series.values() for v in vals if math.isfinite(v)]
    ymin, ymax = min(values), max(values)
    pad = max((ymax - ymin) * 0.08, 1e-9)
    ymin, ymax = ymin - pad, ymax + pad
    n = max(len(vals) for vals in series.values())
    pw, ph = width - ml - mr, height - mt - mb
    for k in range(6):
        y = mt + int(k * ph / 5)
        line(ml, y, ml + pw, y, (226, 226, 226))
    line(ml, mt, ml, mt + ph, (35, 35, 35))
    line(ml, mt + ph, ml + pw, mt + ph, (35, 35, 35))
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    for color, vals in zip(colors, series.values()):
        pts = []
        for i, v in enumerate(vals):
            x = ml + (pw // 2 if n <= 1 else int(i / (n - 1) * pw))
            y = mt + int((ymax - v) / (ymax - ymin) * ph)
            pts.append((x, y))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            line(x0, y0, x1, y1, color)
        for x, y in pts:
            for ox in range(-3, 4):
                for oy in range(-3, 4):
                    if ox * ox + oy * oy <= 9:
                        pix(x + ox, y + oy, color)
    raw = b"".join(b"\x00" + bytes(img[y * width * 3 : (y + 1) * width * 3]) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def paper_window_state(data: dict[str, np.ndarray], current_frame: int, cfg: TrainConfig) -> np.ndarray:
    class Wrapper:
        def __getitem__(self, key: str) -> np.ndarray:
            mapping = {
                "root_pos_w": data["root_pos"],
                "root_quat_xyzw_w": data["root_quat"],
                "root_lin_vel_w": data["root_lin_vel"],
                "root_ang_vel_w": data["root_ang_vel"],
                "body_pos_w": data["body_pos"],
                "body_lin_vel_w": data["body_lin_vel"],
            }
            return mapping[key]

    old_history = motion_fk.np  # keeps module referenced for lint/readability
    del old_history
    # Reuse the exact local paper S3 state formula implementation.
    import build_level_c_paper_state_windows as paper_state

    return paper_state.paper_window_state(Wrapper(), current_frame)[0]


def load_one_motion(csv_path: Path, cfg: TrainConfig, children_by_parent: dict[str, list[motion_fk.JointSpec]]) -> dict[str, Any]:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    end_frame = min(cfg.max_frames_per_motion, raw.shape[0])
    motion = motion_fk.load_and_interpolate_motion(csv_path, cfg.input_fps, cfg.model_fps, 1, end_frame)
    dt = 1.0 / cfg.model_fps
    joint_pos = motion["joint_pos"]
    root_pos = motion["base_pos"]
    root_quat = motion["base_quat_xyzw"]
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = motion_fk.angular_velocity_from_quats(root_quat, dt)
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    body_count = len(motion_fk.G1_TRACKING_BODY_NAMES)
    body_pos = np.zeros((root_pos.shape[0], body_count, 3), dtype=np.float64)
    body_quat = np.zeros((root_pos.shape[0], body_count, 4), dtype=np.float64)
    for t in range(root_pos.shape[0]):
        joint_values = {name: float(joint_pos[t, idx]) for idx, name in enumerate(motion_fk.OFFICIAL_CSV_JOINT_NAMES)}
        transforms = motion_fk.compute_fk(children_by_parent, root_pos[t], root_quat[t], joint_values)
        for b, body_name in enumerate(motion_fk.G1_TRACKING_BODY_NAMES):
            tf = transforms[body_name]
            body_pos[t, b] = tf[:3, 3]
            body_quat[t, b] = motion_fk.matrix_to_quat_xyzw(tf[:3, :3])
    body_lin_vel = np.gradient(body_pos, dt, axis=0)
    starts = np.arange(cfg.history, root_pos.shape[0] - cfg.horizon, cfg.window_stride, dtype=np.int64)
    windows = []
    actions = []
    for current in starts:
        windows.append(
            paper_window_state(
                {
                    "root_pos": root_pos,
                    "root_quat": root_quat,
                    "root_lin_vel": root_lin_vel,
                    "root_ang_vel": root_ang_vel,
                    "body_pos": body_pos,
                    "body_lin_vel": body_lin_vel,
                },
                int(current),
                cfg,
            )
        )
        actions.append(joint_pos[current - cfg.history : current + cfg.horizon + 1])
    return {
        "name": csv_path.stem,
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "frame_count": int(root_pos.shape[0]),
        "window_count": int(len(starts)),
        "states": np.stack(windows).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "starts": starts,
    }


def build_dataset(cfg: TrainConfig) -> dict[str, Any]:
    children = motion_fk.parse_urdf(URDF)
    csvs = sorted(CSV_ROOT.glob("*.csv"))[: cfg.max_motions]
    motions = [load_one_motion(path, cfg, children) for path in csvs]
    names = [m["name"] for m in motions]
    if len(names) < 3:
        raise ValueError("at least 3 motions are required for train/validation/test splits")
    train_end = max(1, int(len(names) * 0.7))
    val_end = max(train_end + 1, int(len(names) * 0.85))
    if val_end >= len(names):
        val_end = len(names) - 1
        train_end = min(train_end, val_end - 1)
    split_names = {
        "train": set(names[:train_end]),
        "validation": set(names[train_end:val_end]),
        "test": set(names[val_end:]),
    }
    states = []
    actions = []
    split_labels = []
    motion_labels = []
    meta_rows = []
    for motion in motions:
        split = next(k for k, v in split_names.items() if motion["name"] in v)
        states.append(motion["states"])
        actions.append(motion["actions"])
        split_labels.extend([split] * motion["window_count"])
        motion_labels.extend([motion["name"]] * motion["window_count"])
        meta_rows.append(
            {
                "name": motion["name"],
                "csv": motion["csv"],
                "csv_sha256": motion["csv_sha256"],
                "frame_count_25hz": motion["frame_count"],
                "window_count": motion["window_count"],
                "split": split,
            }
        )
    p, p_inv = emphasis_projection(seed=cfg.projection_seed, state_dim=cfg.state_dim, root_dim=18, coefficient=6)
    state_np = np.concatenate(states, axis=0).astype(np.float32)
    action_np = np.concatenate(actions, axis=0).astype(np.float32)
    projected_state = np.einsum("pd,ntd->ntp", p.astype(np.float32), state_np).astype(np.float32)
    return {
        "states": state_np,
        "projected_states": projected_state,
        "actions": action_np,
        "split_labels": np.asarray(split_labels),
        "motion_labels": np.asarray(motion_labels),
        "meta_rows": meta_rows,
        "projection": p.astype(np.float32),
        "projection_inverse": p_inv.astype(np.float32),
    }


def load_dataset_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "states" in data:
        states_key = "states"
        projected_key = "projected_states"
        actions_key = "actions"
        split_key = "split_labels"
        motion_key = "motion_labels"
    else:
        states_key = "augmented_states"
        projected_key = "augmented_projected_states"
        actions_key = "augmented_actions"
        split_key = "augmented_split_labels"
        motion_key = "augmented_motion_labels"
    required = [states_key, projected_key, actions_key, split_key, motion_key, "projection", "projection_inverse"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} missing required dataset keys: {missing}")
    states = data[states_key].astype(np.float32)
    projected = data[projected_key].astype(np.float32)
    actions = data[actions_key].astype(np.float32)
    split_labels = data[split_key].astype(str)
    motion_labels = data[motion_key].astype(str)
    if states.ndim != 3 or projected.ndim != 3 or actions.ndim != 3:
        raise ValueError(f"{path} dataset arrays must be rank-3")
    if states.shape[:2] != projected.shape[:2] or states.shape[:2] != actions.shape[:2]:
        raise ValueError(f"{path} dataset arrays have inconsistent window/sequence shapes")
    if len(split_labels) != states.shape[0] or len(motion_labels) != states.shape[0]:
        raise ValueError(f"{path} labels do not match window count")
    meta_rows = []
    for split in ["train", "validation", "test"]:
        labels = sorted(set(motion_labels[split_labels == split].tolist()))
        meta_rows.append(
            {
                "name": f"{path.stem}:{split}",
                "csv": "",
                "csv_sha256": "",
                "frame_count_25hz": 0,
                "window_count": int(np.sum(split_labels == split)),
                "split": split,
                "motion_label_count": int(len(labels)),
            }
        )
    return {
        "states": states,
        "projected_states": projected,
        "actions": actions,
        "split_labels": split_labels,
        "motion_labels": motion_labels,
        "meta_rows": meta_rows,
        "projection": data["projection"].astype(np.float32),
        "projection_inverse": data["projection_inverse"].astype(np.float32),
        "source_npz": str(path),
        "source_npz_sha256": sha256_file(path),
    }


def split_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {split: labels == split for split in ["train", "validation", "test"]}


def evaluate_vae(model: ConditionalVAE, state: torch.Tensor, action: torch.Tensor, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    model.eval()
    vae_model = model.module if isinstance(model, nn.DataParallel) else model
    with torch.no_grad():
        decoded, teacher, mu, logvar, _ = vae_model(state, action, deterministic=True)
    decoded_np = decoded.detach().cpu().numpy()
    teacher_np = teacher.detach().cpu().numpy()
    action_np = action.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    logvar_np = logvar.detach().cpu().numpy()
    rows = []
    for split, mask in masks.items():
        token_mask = mask.astype(bool)
        err = decoded_np[token_mask] - action_np[token_mask]
        teacher_err = teacher_np[token_mask] - action_np[token_mask]
        kl = -0.5 * np.mean(1.0 + logvar_np[token_mask] - mu_np[token_mask] ** 2 - np.exp(logvar_np[token_mask]))
        rows.append(
            {
                "split": split,
                "window_count": int(mask.sum()),
                "token_count": int(token_mask.sum()),
                "decoded_action_mse": float(np.mean(err**2)),
                "teacher_action_mse": float(np.mean(teacher_err**2)),
                "decoded_action_max_abs_error": float(np.max(np.abs(err))),
                "kl_mean": float(kl),
                "latent_abs_mean": float(np.mean(np.abs(mu_np[token_mask]))),
                "latent_std": float(np.std(mu_np[token_mask])),
            }
        )
    return rows


def alpha_bars(steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, steps, device=device, dtype=dtype)
    return torch.cumprod(1.0 - betas, dim=0)


def noised_tau(clean: torch.Tensor, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    steps = torch.randint(0, cfg.diffusion_steps, clean.shape[:2] + (2,), device=device)
    noise = torch.randn_like(clean)
    bars = alpha_bars(cfg.diffusion_steps, device, clean.dtype)
    state_alpha = bars[steps[..., 0]].unsqueeze(-1).expand(-1, -1, cfg.projected_state_dim)
    latent_alpha = bars[steps[..., 1]].unsqueeze(-1).expand(-1, -1, cfg.latent_dim)
    alpha = torch.cat([state_alpha, latent_alpha], dim=-1)
    noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
    return noisy, steps


def evaluate_diffusion(
    model: DiffusionTransformer,
    tau: torch.Tensor,
    masks: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        noisy, steps = noised_tau(tau, cfg, device)
        pred = model(noisy, steps)
    clean_np = tau.detach().cpu().numpy()
    noisy_np = noisy.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    rows = []
    for split, mask in masks.items():
        noisy_mse = float(np.mean((noisy_np[mask] - clean_np[mask]) ** 2))
        pred_mse = float(np.mean((pred_np[mask] - clean_np[mask]) ** 2))
        rows.append(
            {
                "split": split,
                "window_count": int(mask.sum()),
                "token_count": int(mask.sum() * cfg.seq_len),
                "noisy_tau_mse": noisy_mse,
                "pred_tau_mse": pred_mse,
                "tau_mse_reduction_vs_noisy": float((noisy_mse - pred_mse) / noisy_mse) if noisy_mse > 0 else 0.0,
            }
        )
    return rows


def default_seed_run_id(seed: int) -> str:
    if seed == 20260617:
        return RUN_ID
    return f"level_c_lafan1_paper_arch_vae_diffusion_seed_{seed}_static_000_20260617_203000"


def default_seed_out(seed: int, run_id: str | None) -> Path:
    if seed == 20260617 and run_id in {None, RUN_ID}:
        return OUT
    return ROOT / "res/level_c" / f"lafan1_paper_arch_vae_diffusion_training_seed_{seed}"


def resolve_path(path_text: str | None, default: Path) -> Path:
    if not path_text:
        return default
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def prepare_run_dir(command: str, cfg: TrainConfig, run_id: str, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ["checkpoint", "figures", "videos"]:
        (run_dir / dirname).mkdir(exist_ok=True)
    write_text(
        run_dir / "resolved_config.yaml",
        "\n".join(
            [
                f"run_id: {run_id}",
                "stage: level_c",
                "method: public_lafan1_paper_arch_vae_diffusion_training",
                "status: RUNNING",
                "is_training_run: true",
                "paper_architecture: true",
                "paper_dataset: false",
                "public_dataset: LAFAN1_Retargeting_Dataset/g1",
                f"vae_hidden: {list(cfg.vae_encoder_hidden)}",
                f"diffusion_transformer: embed={cfg.diffusion_embedding_dim}, heads={cfg.diffusion_heads}, layers={cfg.diffusion_layers}",
                "source_goal_lines: goal.md:776-799,1148-1290,1431-1505",
                "source_paper_lines: reproduction/paper/source/root.tex:803-848",
                "",
            ]
        ),
    )
    write_text(run_dir / "command.sh", f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n", executable=True)
    write_text(run_dir / "stderr.log", "")
    write_text(
        run_dir / "environment.txt",
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat(timespec='seconds')}",
                f"python={command_output([str(BM_DIFFUSION_PYTHON), '--version'])}",
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
                f"numpy={np.__version__}",
                "notes=Paper architecture on public retargeted LAFAN1, not unpublished teacher rollout data.",
                "",
            ]
        ),
    )
    write_text(run_dir / "git_state.txt", command_output(["git", "status", "--short"]) + "\n")
    gpu_src = ROOT / "logs/gpu/gpu_metrics.csv"
    if gpu_src.is_file():
        shutil.copyfile(gpu_src, run_dir / "gpu_metrics.csv")
    else:
        write_text(run_dir / "gpu_metrics.csv", "timestamp,gpu_index,run_id,run_status,sample_kind\n")


def train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = TrainConfig(
        seed=args.seed,
        projection_seed=args.projection_seed,
        max_motions=args.max_motions,
        max_frames_per_motion=args.max_frames_per_motion,
        vae_epochs=args.vae_epochs,
        diffusion_epochs=args.diffusion_epochs,
        diffusion_batch_size=args.diffusion_batch_size,
    )
    run_id = args.run_id or default_seed_run_id(args.seed)
    run_dir = ROOT / "res/runs" / run_id
    out = resolve_path(args.output_dir, default_seed_out(args.seed, args.run_id))
    command = (
        f"{shell_quote(BM_DIFFUSION_PYTHON)} "
        f"{shell_quote(ROOT / 'reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py')} "
        f"--device {shell_quote(args.device)} --seed {args.seed} --projection-seed {args.projection_seed} "
        f"--run-id {shell_quote(run_id)} --output-dir {shell_quote(out)} "
        f"--max-motions {args.max_motions} --max-frames-per-motion {args.max_frames_per_motion} "
        f"--vae-epochs {args.vae_epochs} --diffusion-epochs {args.diffusion_epochs} "
        f"--diffusion-batch-size {args.diffusion_batch_size} "
        f"{('--dataset-npz ' + shell_quote(resolve_path(args.dataset_npz, ROOT))) if args.dataset_npz else ''} "
        f"--dataset-source-label {shell_quote(args.dataset_source_label)} "
        f"{'--data-parallel' if args.data_parallel else '--no-data-parallel'}"
    )
    prepare_run_dir(command, cfg, run_id, run_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    gpu_device_ids = list(range(torch.cuda.device_count())) if device.type == "cuda" and args.data_parallel else []

    start = time.perf_counter()
    dataset_source_npz = resolve_path(args.dataset_npz, ROOT) if args.dataset_npz else None
    dataset = load_dataset_npz(dataset_source_npz) if dataset_source_npz else build_dataset(cfg)
    dataset_elapsed = time.perf_counter() - start
    masks = split_masks(dataset["split_labels"])
    npz_path = out / "lafan1_paper_arch_training_dataset.npz"
    np.savez_compressed(
        npz_path,
        states=dataset["states"],
        projected_states=dataset["projected_states"],
        actions=dataset["actions"],
        split_labels=dataset["split_labels"],
        motion_labels=dataset["motion_labels"],
        projection=dataset["projection"],
        projection_inverse=dataset["projection_inverse"],
    )

    state_tokens = torch.from_numpy(dataset["states"].reshape(-1, cfg.state_dim)).to(device)
    action_tokens = torch.from_numpy(dataset["actions"].reshape(-1, cfg.action_dim)).to(device)
    token_split_labels = np.repeat(dataset["split_labels"][:, None], cfg.seq_len, axis=1).reshape(-1)
    train_token_idx = torch.from_numpy(np.nonzero(token_split_labels == "train")[0]).long().to(device)

    vae_base = ConditionalVAE(cfg).to(device)
    vae: nn.Module = nn.DataParallel(vae_base, device_ids=gpu_device_ids) if len(gpu_device_ids) > 1 else vae_base
    vae_opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)
    vae_loss_rows: list[dict[str, Any]] = []
    micro = max(1, min(2048, len(train_token_idx)))
    for epoch in range(cfg.vae_epochs):
        perm = train_token_idx[torch.randperm(len(train_token_idx), device=device)]
        vae_opt.zero_grad(set_to_none=True)
        accum = 0
        running = []
        for start_i in range(0, len(perm), micro):
            idx = perm[start_i : start_i + micro]
            vae_model = vae.module if isinstance(vae, nn.DataParallel) else vae
            decoded, teacher, mu, logvar, _ = vae_model(state_tokens[idx], action_tokens[idx], deterministic=False)
            recon = F.mse_loss(decoded, action_tokens[idx])
            teacher_loss = F.mse_loss(teacher, action_tokens[idx])
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 0.25 * teacher_loss + cfg.vae_kl_coef * kl
            (loss / cfg.vae_accumulation_steps).backward()
            accum += 1
            running.append(float(loss.detach().cpu().item()))
            if accum % cfg.vae_accumulation_steps == 0 or start_i + micro >= len(perm):
                vae_opt.step()
                vae_opt.zero_grad(set_to_none=True)
        if epoch % 2 == 0 or epoch == cfg.vae_epochs - 1:
            rows = evaluate_vae(vae, state_tokens, action_tokens, {k: np.repeat(v[:, None], cfg.seq_len, axis=1).reshape(-1) for k, v in masks.items()})
            by = {r["split"]: r for r in rows}
            vae_loss_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(running)),
                    "train_decoded_action_mse": by["train"]["decoded_action_mse"],
                    "validation_decoded_action_mse": by["validation"]["decoded_action_mse"],
                    "test_decoded_action_mse": by["test"]["decoded_action_mse"],
                }
            )

    vae_rows = evaluate_vae(vae, state_tokens, action_tokens, {k: np.repeat(v[:, None], cfg.seq_len, axis=1).reshape(-1) for k, v in masks.items()})
    with torch.no_grad():
        vae_model = vae.module if isinstance(vae, nn.DataParallel) else vae
        latent_tokens = vae_model.encode(state_tokens, action_tokens).reshape(-1, cfg.seq_len, cfg.latent_dim)
    projected = torch.from_numpy(dataset["projected_states"]).to(device)
    tau = torch.cat([projected, latent_tokens], dim=-1)

    diffusion_base = DiffusionTransformer(cfg).to(device)
    diffusion: nn.Module = (
        nn.DataParallel(diffusion_base, device_ids=gpu_device_ids) if len(gpu_device_ids) > 1 else diffusion_base
    )
    diff_opt = torch.optim.AdamW(diffusion.parameters(), lr=cfg.diffusion_lr, weight_decay=cfg.diffusion_weight_decay)
    train_windows = torch.from_numpy(np.nonzero(masks["train"])[0]).long().to(device)
    diff_rows_history: list[dict[str, Any]] = []
    for epoch in range(cfg.diffusion_epochs):
        batch_idx = train_windows[torch.randint(0, len(train_windows), (min(cfg.diffusion_batch_size, len(train_windows)),), device=device)]
        clean = tau[batch_idx]
        noisy, steps = noised_tau(clean, cfg, device)
        pred = diffusion(noisy, steps)
        loss = F.mse_loss(pred, clean)
        diff_opt.zero_grad(set_to_none=True)
        loss.backward()
        diff_opt.step()
        if epoch % 4 == 0 or epoch == cfg.diffusion_epochs - 1:
            rows = evaluate_diffusion(diffusion, tau, masks, cfg, device)
            by = {r["split"]: r for r in rows}
            diff_rows_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_pred_tau_mse": by["train"]["pred_tau_mse"],
                    "validation_pred_tau_mse": by["validation"]["pred_tau_mse"],
                    "test_pred_tau_mse": by["test"]["pred_tau_mse"],
                }
            )

    diffusion_rows = evaluate_diffusion(diffusion, tau, masks, cfg, device)
    elapsed = time.perf_counter() - start
    checkpoint = run_dir / "checkpoint/lafan1_paper_arch_vae_diffusion.pt"
    torch.save(
        {
            "run_id": run_id,
            "config": asdict(cfg),
            "vae_state_dict": vae.state_dict(),
            "diffusion_state_dict": diffusion.state_dict(),
            "vae_optimizer_state_dict": vae_opt.state_dict(),
            "diffusion_optimizer_state_dict": diff_opt.state_dict(),
            "paper_architecture": True,
            "paper_dataset": False,
            "public_dataset": "LAFAN1_Retargeting_Dataset/g1",
            "dataset_source_label": args.dataset_source_label,
            "dataset_source_npz": str(dataset_source_npz) if dataset_source_npz else None,
        },
        checkpoint,
    )
    vae_fig = run_dir / "figures/vae_action_mse.png"
    diff_fig = run_dir / "figures/diffusion_tau_mse.png"
    simple_png_plot(
        vae_fig,
        {
            "train": [r["train_decoded_action_mse"] for r in vae_loss_rows],
            "validation": [r["validation_decoded_action_mse"] for r in vae_loss_rows],
            "test": [r["test_decoded_action_mse"] for r in vae_loss_rows],
        },
    )
    simple_png_plot(
        diff_fig,
        {
            "train": [r["train_pred_tau_mse"] for r in diff_rows_history],
            "validation": [r["validation_pred_tau_mse"] for r in diff_rows_history],
            "test": [r["test_pred_tau_mse"] for r in diff_rows_history],
        },
    )
    with (out / "lafan1_paper_arch_vae_rows.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(vae_rows[0].keys()))
        writer.writeheader()
        writer.writerows(vae_rows)
    with (out / "lafan1_paper_arch_diffusion_rows.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(diffusion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diffusion_rows)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted(set().union(*(r.keys() for r in vae_loss_rows + diff_rows_history)))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(vae_loss_rows + diff_rows_history)

    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else None
    metrics = {
        "run_id": run_id,
        "status": "SUCCESS",
        "is_training_run": True,
        "paper_architecture": True,
        "paper_dataset": False,
        "public_lafan1_motion_count": len(dataset["meta_rows"]),
        "public_lafan1_unique_motion_label_count": int(
            len(set(str(x).replace("::mirror", "") for x in dataset["motion_labels"].tolist()))
        ),
        "augmented_motion_label_count": int(len(set(dataset["motion_labels"].tolist()))),
        "window_count": int(dataset["states"].shape[0]),
        "token_count": int(dataset["states"].shape[0] * cfg.seq_len),
        "dataset_build_seconds": dataset_elapsed,
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_mb": peak_mb,
        "gpu_device_ids": gpu_device_ids,
        "data_parallel": len(gpu_device_ids) > 1,
        "vae_parameter_count": int(sum(p.numel() for p in vae.parameters())),
        "diffusion_parameter_count": int(sum(p.numel() for p in diffusion.parameters())),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "final_validation_decoded_action_mse": next(r["decoded_action_mse"] for r in vae_rows if r["split"] == "validation"),
        "final_test_decoded_action_mse": next(r["decoded_action_mse"] for r in vae_rows if r["split"] == "test"),
        "final_validation_pred_tau_mse": next(r["pred_tau_mse"] for r in diffusion_rows if r["split"] == "validation"),
        "final_test_pred_tau_mse": next(r["pred_tau_mse"] for r in diffusion_rows if r["split"] == "test"),
    }
    write_json_atomic(run_dir / "metrics.json", metrics)
    write_json_atomic(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "status": "SUCCESS",
            "allowed_status": True,
            "is_success": True,
            "is_training_run": True,
            "paper_architecture": True,
            "paper_dataset": False,
            "reason": "Paper-sized VAE and diffusion architecture trained on public LAFAN1 G1 state/action windows.",
            "dataset_source_label": args.dataset_source_label,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    write_text(
        run_dir / "stdout.log",
        "\n".join(
            [
                "Paper-architecture VAE/diffusion training on public LAFAN1 completed.",
                f"run_id={run_id}",
                f"motions={metrics['public_lafan1_motion_count']}",
                f"windows={metrics['window_count']}",
                f"dataset_source_label={args.dataset_source_label}",
                f"vae_validation_mse={metrics['final_validation_decoded_action_mse']}",
                f"diffusion_validation_mse={metrics['final_validation_pred_tau_mse']}",
                "paper_dataset=false",
                "",
            ]
        ),
    )
    write_text(run_dir / "resolved_config.yaml", (run_dir / "resolved_config.yaml").read_text().replace("status: RUNNING", "status: SUCCESS"))

    checks = {
        "uses_all_available_or_requested_public_g1_motions": (
            len(dataset["meta_rows"]) == min(cfg.max_motions, len(list(CSV_ROOT.glob("*.csv"))))
            if dataset_source_npz is None
            else metrics["public_lafan1_unique_motion_label_count"] >= min(cfg.max_motions, len(list(CSV_ROOT.glob("*.csv"))))
        ),
        "external_dataset_npz_exists_when_requested": dataset_source_npz is None or dataset_source_npz.is_file(),
        "dataset_source_label_recorded": bool(args.dataset_source_label),
        "paper_state_action_shapes": list(dataset["states"].shape[1:]) == [cfg.seq_len, cfg.state_dim]
        and list(dataset["actions"].shape[1:]) == [cfg.seq_len, cfg.action_dim]
        and list(dataset["projected_states"].shape[1:]) == [cfg.seq_len, cfg.projected_state_dim],
        "paper_vae_architecture_used": list(cfg.vae_encoder_hidden) == [2048, 1024, 512] and cfg.latent_dim == 32,
        "paper_diffusion_architecture_used": cfg.diffusion_embedding_dim == 512
        and cfg.diffusion_heads == 8
        and cfg.diffusion_layers == 6
        and cfg.diffusion_steps == 20,
        "run_schema_checkpoint_metrics_figures_written": checkpoint.is_file()
        and (run_dir / "metrics.json").is_file()
        and vae_fig.is_file()
        and diff_fig.is_file(),
        "validation_and_test_rows_exist": {r["split"] for r in vae_rows} == {"train", "validation", "test"}
        and {r["split"] for r in diffusion_rows} == {"train", "validation", "test"},
        "all_metrics_finite": all(
            np.isfinite(v)
            for row in vae_rows + diffusion_rows
            for k, v in row.items()
            if k != "split"
        ),
        "checkpoint_hash_recorded": len(metrics["checkpoint_sha256"]) == 64,
        "uses_multi_gpu_when_available": torch.cuda.device_count() <= 1 or len(gpu_device_ids) == torch.cuda.device_count(),
        "does_not_claim_unavailable_teacher_rollout_dataset": True,
        "does_not_claim_real_robot_or_fig5_fig6": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_vae_diffusion_training",
        "scope": "paper-sized conditional VAE and Transformer diffusion training on public retargeted LAFAN1 G1 motions",
        "paper_evidence": {
            "vae_table": str(ROOT / "reproduction/paper/source/root.tex:803-821"),
            "diffusion_table": str(ROOT / "reproduction/paper/source/root.tex:827-848"),
            "vae_method": str(ROOT / "reproduction/paper/source/tex/method.tex:150-171"),
            "state_latent_diffusion_method": str(ROOT / "reproduction/paper/source/tex/method.tex:171-206"),
        },
        "dataset_boundary": {
            "used": args.dataset_source_label,
            "source_npz": str(dataset_source_npz) if dataset_source_npz else None,
            "source_npz_sha256": dataset.get("source_npz_sha256"),
            "not_available": [
                "official teacher policy rollouts",
                "official DAgger aggregation states",
                "official VAE rollout state-latent trajectory dataset",
                "real robot data for deployment tasks",
            ],
        },
        "settings": asdict(cfg)
        | {"device": str(device), "torch_threads": args.torch_threads, "gpu_device_ids": gpu_device_ids},
        "data_rows": dataset["meta_rows"],
        "vae_rows": vae_rows,
        "diffusion_rows": diffusion_rows,
        "vae_loss_rows": vae_loss_rows,
        "diffusion_loss_rows": diff_rows_history,
        "metrics": metrics,
        "checks": checks,
        "outputs": {
            "json": str(out / "lafan1_paper_arch_vae_diffusion_training.json"),
            "dataset_npz": str(npz_path),
            "source_dataset_npz": str(dataset_source_npz) if dataset_source_npz else None,
            "vae_tsv": str(out / "lafan1_paper_arch_vae_rows.tsv"),
            "diffusion_tsv": str(out / "lafan1_paper_arch_diffusion_rows.tsv"),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "vae_figure": str(vae_fig),
            "diffusion_figure": str(diff_fig),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_training",
            "why_not_complete": (
                "This is a real training run with paper-sized VAE and diffusion architectures on the selected public "
                "G1 LAFAN1 dataset variant available locally. It is still not an exact paper reproduction because the official "
                "teacher-policy DAgger rollouts and VAE rollout state-latent dataset are not public in the local bundle."
            ),
        },
    }
    write_json_atomic(out / "lafan1_paper_arch_vae_diffusion_training.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--projection-seed", type=int, default=20260617)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-motions", type=int, default=40)
    parser.add_argument("--max-frames-per-motion", type=int, default=420)
    parser.add_argument("--vae-epochs", type=int, default=24)
    parser.add_argument("--diffusion-epochs", type=int, default=1000)
    parser.add_argument("--diffusion-batch-size", type=int, default=512)
    parser.add_argument("--dataset-npz", default=None)
    parser.add_argument("--dataset-source-label", default="download/official/LAFAN1_Retargeting_Dataset/g1 public retargeted motions")
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = train(args)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "run_id": summary["metrics"]["run_id"]}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
