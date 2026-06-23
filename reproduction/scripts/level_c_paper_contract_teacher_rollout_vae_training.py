#!/usr/bin/env python3
"""Train a paper-contract conditional VAE on local teacher rollout shards.

The important correction versus the older resource-adjusted VAE is the input
contract:

* encoder: reference intent only, E(psi, e_anchor)
* decoder: current proprioception plus latent, D(z, proprioception)

The data source is still local teacher rollout data, not official BeyondMimic
DAgger logs.  This script therefore repairs the formula-level VAE interface,
but it does not by itself unblock paper-level videos when the teacher data is
weak.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/paper_contract_teacher_rollout_vae_training"
RUN_ROOT = ROOT / "res/runs/level_c_paper_contract_teacher_rollout_vae_training"
LOG_DIR = ROOT / "logs/level_c_paper_contract_teacher_rollout_vae_training"
FAILED_DIR = ROOT / "res/failed_runs/level_c_paper_contract_teacher_rollout_vae_training"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"
DEFAULT_TEACHER_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
    "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
)
SCHEMA_JSON = ROOT / "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json"

SEED = int(os.environ.get("BM_PAPER_CONTRACT_VAE_SEED", "20260921"))
EPOCHS = int(os.environ.get("BM_PAPER_CONTRACT_VAE_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BM_PAPER_CONTRACT_VAE_BATCH_SIZE", "65536"))
LATENT_DIM = int(os.environ.get("BM_PAPER_CONTRACT_VAE_LATENT_DIM", "32"))
HIDDEN_DIMS = tuple(
    int(item.strip())
    for item in os.environ.get("BM_PAPER_CONTRACT_VAE_HIDDEN_DIMS", "2048,1024,512").split(",")
    if item.strip()
)
GRAD_ACCUM_STEPS = int(os.environ.get("BM_PAPER_CONTRACT_VAE_GRAD_ACCUM_STEPS", "15"))
KL_COEF = float(os.environ.get("BM_PAPER_CONTRACT_VAE_KL_COEF", "0.01"))
LR = float(os.environ.get("BM_PAPER_CONTRACT_VAE_LR", "5e-4"))
CANDIDATE_GPUS = [
    int(item.strip())
    for item in os.environ.get("BM_PAPER_CONTRACT_VAE_GPUS", "5,6").split(",")
    if item.strip()
]


WORKER_CODE = r"""
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
TEACHER_JSON = Path(os.environ["BM_TEACHER_JSON"])
SCHEMA_JSON = Path(os.environ["BM_SCHEMA_JSON"])
SEED = int(os.environ["BM_PAPER_CONTRACT_VAE_SEED"])
EPOCHS = int(os.environ["BM_PAPER_CONTRACT_VAE_EPOCHS"])
BATCH_SIZE = int(os.environ["BM_PAPER_CONTRACT_VAE_BATCH_SIZE"])
LATENT_DIM = int(os.environ["BM_PAPER_CONTRACT_VAE_LATENT_DIM"])
HIDDEN_DIMS = [
    int(item.strip())
    for item in os.environ["BM_PAPER_CONTRACT_VAE_HIDDEN_DIMS"].split(",")
    if item.strip()
]
GRAD_ACCUM_STEPS = int(os.environ["BM_PAPER_CONTRACT_VAE_GRAD_ACCUM_STEPS"])
KL_COEF = float(os.environ["BM_PAPER_CONTRACT_VAE_KL_COEF"])
LR = float(os.environ["BM_PAPER_CONTRACT_VAE_LR"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def obs_slices_from_schema():
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    rows = [row for row in schema["observation_rows"] if row["group"] == "policy"]
    rows = sorted(rows, key=lambda row: row["order"])
    offset = 0
    slices = {}
    for row in rows:
        dim = int(row["dimension"])
        slices[row["term"]] = (offset, offset + dim)
        offset += dim
    if offset != 160:
        raise RuntimeError(f"Expected policy obs dim 160, got {offset}")
    return slices


class PaperContractVAE(nn.Module):
    def __init__(self, encoder_dim, proprio_dim, action_dim, latent_dim, hidden_dims):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        encoder_layers = []
        in_dim = encoder_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ELU()])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim * 2))
        decoder_layers = []
        in_dim = latent_dim + proprio_dim
        for hidden_dim in self.hidden_dims:
            decoder_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ELU()])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, action_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, encoder_x, proprio_x, deterministic=False):
        stats = self.encoder(encoder_x)
        mu, logvar = stats.chunk(2, dim=-1)
        z = mu if deterministic else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        pred = self.decoder(torch.cat([z, proprio_x], dim=-1))
        return pred, mu, logvar


def teacher_shard_paths(summary):
    paths = []
    if "run" in summary and "shard_metrics" in summary["run"]:
        for row in summary["run"]["shard_metrics"]:
            path = row.get("dataset_npz")
            if path:
                paths.append(path)
    if not paths and "run" in summary and "shard_npz_paths" in summary["run"]:
        paths = list(summary["run"]["shard_npz_paths"])
    if not paths:
        raise RuntimeError("No teacher rollout shard paths found")
    return [Path(path) for path in paths]


def split_obs(obs, slices):
    command = obs[:, slices["command"][0] : slices["command"][1]]
    anchor_pos = obs[:, slices["motion_anchor_pos_b"][0] : slices["motion_anchor_pos_b"][1]]
    anchor_ori = obs[:, slices["motion_anchor_ori_b"][0] : slices["motion_anchor_ori_b"][1]]
    encoder_x = np.concatenate([command, anchor_pos, anchor_ori], axis=-1).astype(np.float32)
    proprio_terms = []
    for term in ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions"]:
        start, end = slices[term]
        proprio_terms.append(obs[:, start:end])
    proprio_x = np.concatenate(proprio_terms, axis=-1).astype(np.float32)
    return encoder_x, proprio_x


def load_dataset():
    summary = json.loads(TEACHER_JSON.read_text(encoding="utf-8"))
    slices = obs_slices_from_schema()
    enc_chunks = []
    prop_chunks = []
    action_chunks = []
    reward_chunks = []
    done_chunks = []
    time_chunks = []
    shard_rows = []
    for path in teacher_shard_paths(summary):
        with np.load(path) as data:
            obs = data["policy_obs"].reshape(-1, data["policy_obs"].shape[-1]).astype(np.float32)
            actions = data["actions"].reshape(-1, data["actions"].shape[-1]).astype(np.float32)
            rewards = data["rewards"].reshape(-1).astype(np.float32)
            dones = data["dones"].reshape(-1).astype(np.bool_)
            motion_time_steps = data["motion_time_steps"].reshape(-1).astype(np.int64)
        enc, prop = split_obs(obs, slices)
        enc_chunks.append(enc)
        prop_chunks.append(prop)
        action_chunks.append(actions)
        reward_chunks.append(rewards)
        done_chunks.append(dones)
        time_chunks.append(motion_time_steps)
        shard_rows.append(
            {
                "path": str(path),
                "sample_count": int(obs.shape[0]),
                "obs_dim": int(obs.shape[1]),
                "encoder_dim": int(enc.shape[1]),
                "proprio_dim": int(prop.shape[1]),
                "action_dim": int(actions.shape[1]),
            }
        )
    return (
        summary,
        slices,
        shard_rows,
        np.concatenate(enc_chunks, axis=0),
        np.concatenate(prop_chunks, axis=0),
        np.concatenate(action_chunks, axis=0),
        np.concatenate(reward_chunks, axis=0),
        np.concatenate(done_chunks, axis=0),
        np.concatenate(time_chunks, axis=0),
    )


def split_indices(count):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(count)
    train_end = int(count * 0.8)
    val_end = int(count * 0.9)
    return {
        "train": perm[:train_end],
        "validation": perm[train_end:val_end],
        "test": perm[val_end:],
    }


def evaluate(model, encoder_x, proprio_x, actions, indices, device):
    model.eval()
    losses, abs_errs, kl_vals, mu_abs = [], [], [], []
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(encoder_x[indices]),
            torch.from_numpy(proprio_x[indices]),
            torch.from_numpy(actions[indices]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.inference_mode():
        for enc_b, prop_b, act_b in loader:
            enc_b = enc_b.to(device, non_blocking=True)
            prop_b = prop_b.to(device, non_blocking=True)
            act_b = act_b.to(device, non_blocking=True)
            pred, mu, logvar = model(enc_b, prop_b, deterministic=True)
            err = pred - act_b
            losses.append(float(F.mse_loss(pred, act_b).detach().cpu()))
            abs_errs.append(float(err.abs().mean().detach().cpu()))
            kl_vals.append(float((-0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())).detach().cpu()))
            mu_abs.append(float(mu.abs().mean().detach().cpu()))
    return {
        "sample_count": int(len(indices)),
        "action_mse": float(np.mean(losses)),
        "action_abs_error_mean": float(np.mean(abs_errs)),
        "kl_mean": float(np.mean(kl_vals)),
        "latent_mu_abs_mean": float(np.mean(mu_abs)),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    start = time.time()
    teacher_summary, slices, shard_rows, encoder_x, proprio_x, actions, rewards, dones, motion_steps = load_dataset()
    splits = split_indices(encoder_x.shape[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PaperContractVAE(encoder_x.shape[1], proprio_x.shape[1], actions.shape[1], LATENT_DIM, HIDDEN_DIMS).to(device)
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(encoder_x[splits["train"]]),
            torch.from_numpy(proprio_x[splits["train"]]),
            torch.from_numpy(actions[splits["train"]]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    epoch_rows = []
    for epoch in range(EPOCHS):
        model.train()
        recon_values, kl_values, total_values = [], [], []
        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        step_counter = 0
        for batch_index, (enc_b, prop_b, act_b) in enumerate(train_loader):
            enc_b = enc_b.to(device, non_blocking=True)
            prop_b = prop_b.to(device, non_blocking=True)
            act_b = act_b.to(device, non_blocking=True)
            pred, mu, logvar = model(enc_b, prop_b, deterministic=False)
            recon = F.mse_loss(pred, act_b)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + KL_COEF * kl
            (loss / GRAD_ACCUM_STEPS).backward()
            accum_counter += 1
            is_last_batch = batch_index == len(train_loader) - 1
            if accum_counter == GRAD_ACCUM_STEPS or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_counter += 1
                accum_counter = 0
            recon_values.append(float(recon.detach().cpu()))
            kl_values.append(float(kl.detach().cpu()))
            total_values.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            row = {
                "epoch": epoch + 1,
                "train_reconstruction_mse": float(np.mean(recon_values)),
                "train_kl_mean": float(np.mean(kl_values)),
                "train_total_loss": float(np.mean(total_values)),
            }
            epoch_rows.append(row)
            print("BM_SENTINEL:epoch:" + json.dumps(row, sort_keys=True), flush=True)
    base = model.module if isinstance(model, nn.DataParallel) else model
    evaluations = {split: evaluate(base, encoder_x, proprio_x, actions, idx, device) for split, idx in splits.items()}
    checkpoint = RUN_DIR / "paper_contract_teacher_rollout_vae.pt"
    torch.save(
        {
            "model_state_dict": base.state_dict(),
            "config": {
                "encoder_dim": int(encoder_x.shape[1]),
                "proprio_dim": int(proprio_x.shape[1]),
                "action_dim": int(actions.shape[1]),
                "latent_dim": LATENT_DIM,
                "hidden_dims": list(HIDDEN_DIMS),
                "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "kl_coef": KL_COEF,
                "learning_rate": LR,
                "obs_slices": {k: list(v) for k, v in slices.items()},
                "encoder_input_terms": ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"],
                "decoder_proprioception_terms": ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions"],
            },
        },
        checkpoint,
    )
    tsv = RUN_DIR / "paper_contract_teacher_rollout_vae_epochs.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_reconstruction_mse", "train_kl_mean", "train_total_loss"], delimiter="\t")
        writer.writeheader()
        writer.writerows(epoch_rows)
    done_rate = float(np.mean(dones))
    timeout_free_quality_proxy = {
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "done_rate": done_rate,
        "motion_time_step_jump_fraction": float(np.mean(np.abs(np.diff(motion_steps[: min(len(motion_steps), 200000)])) > 1))
        if len(motion_steps) > 1
        else 0.0,
    }
    summary = {
        "status": "ok",
        "duration_seconds": round(time.time() - start, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "data_parallel_used": torch.cuda.device_count() >= 2,
        "source_teacher_rollout": {
            "json": str(TEACHER_JSON),
            "status": teacher_summary.get("status"),
            "claim_level": teacher_summary.get("interpretation", {}).get("claim_level"),
            "official_dagger_dataset": False,
            "paper_level_teacher_rollout": False,
        },
        "dataset": {
            "sample_count": int(encoder_x.shape[0]),
            "encoder_dim": int(encoder_x.shape[1]),
            "proprio_dim": int(proprio_x.shape[1]),
            "action_dim": int(actions.shape[1]),
            "split_counts": {k: int(len(v)) for k, v in splits.items()},
            "shards": shard_rows,
            "teacher_quality_proxy": timeout_free_quality_proxy,
        },
        "paper_contract": {
            "encoder_equation": "z = E(psi, e_anchor)",
            "decoder_equation": "a_hat = D(z, [g/V_imu/proprioception, theta, theta_dot, a_last])",
            "encoder_terms": ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"],
            "decoder_proprioception_terms": ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions"],
            "paper_source": str(ROOT / "reproduction/paper/source/tex/method.tex:150-170"),
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "hidden_dims": list(HIDDEN_DIMS),
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "kl_coef": KL_COEF,
            "learning_rate": LR,
            "epoch_rows": epoch_rows,
        },
        "evaluation": evaluations,
        "outputs": {
            "checkpoint": str(checkpoint),
            "epoch_tsv": str(tsv),
            "worker_summary_json": str(RUN_DIR / "paper_contract_teacher_rollout_vae_worker_summary.json"),
        },
        "checks": {
            "encoder_uses_reference_intent_only": encoder_x.shape[1] == 67,
            "decoder_uses_proprioception_plus_latent": proprio_x.shape[1] == 93,
            "action_dim_29": actions.shape[1] == 29,
            "hidden_dims_match_appendix": list(HIDDEN_DIMS) == [2048, 1024, 512],
            "gradient_accumulation_steps_15": GRAD_ACCUM_STEPS == 15,
            "data_parallel_used": torch.cuda.device_count() >= 2,
            "test_action_mse_below_action_variance": evaluations["test"]["action_mse"] < float(np.var(actions)),
            "source_teacher_done_rate_low_enough_for_downstream": done_rate < 0.05,
            "does_not_claim_official_dagger": True,
            "does_not_claim_paper_level_closed_loop": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "formula_level_vae_contract_repaired": True,
            "appendix_architecture_contract_repaired": list(HIDDEN_DIMS) == [2048, 1024, 512],
            "appendix_gradient_accumulation_repaired": GRAD_ACCUM_STEPS == 15,
            "downstream_video_allowed": bool(done_rate < 0.05),
            "why_downstream_may_remain_blocked": (
                "The VAE input contract is now paper-aligned, but the source teacher rollout may still contain frequent "
                "resets/failures. If the teacher done rate is high, do not use this checkpoint to claim successful "
                "single-leg or walking videos."
            ),
        },
    }
    (RUN_DIR / "paper_contract_teacher_rollout_vae_worker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("BM_SENTINEL:summary:" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_dir() -> Path:
    tag = os.environ.get("BM_PAPER_CONTRACT_VAE_RUN_TAG", "").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return RUN_ROOT / f"paper_contract_teacher_rollout_vae_{stamp}_seed{SEED}{suffix}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    teacher_json = Path(os.environ.get("BM_PAPER_CONTRACT_VAE_TEACHER_JSON", str(DEFAULT_TEACHER_JSON))).resolve()
    rd = run_dir()
    worker = OUT / "paper_contract_teacher_rollout_vae_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    log = LOG_DIR / "level_c_paper_contract_teacher_rollout_vae_training.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "BM_RUN_DIR": str(rd),
            "BM_TEACHER_JSON": str(teacher_json),
            "BM_SCHEMA_JSON": str(SCHEMA_JSON),
            "BM_PAPER_CONTRACT_VAE_SEED": str(SEED),
            "BM_PAPER_CONTRACT_VAE_EPOCHS": str(EPOCHS),
            "BM_PAPER_CONTRACT_VAE_BATCH_SIZE": str(BATCH_SIZE),
            "BM_PAPER_CONTRACT_VAE_LATENT_DIM": str(LATENT_DIM),
            "BM_PAPER_CONTRACT_VAE_HIDDEN_DIMS": ",".join(str(dim) for dim in HIDDEN_DIMS),
            "BM_PAPER_CONTRACT_VAE_GRAD_ACCUM_STEPS": str(GRAD_ACCUM_STEPS),
            "BM_PAPER_CONTRACT_VAE_KL_COEF": str(KL_COEF),
            "BM_PAPER_CONTRACT_VAE_LR": str(LR),
        }
    )
    start = time.time()
    proc = subprocess.Popen(
        [str(BM_DIFFUSION_PY), str(worker)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    captured: list[str] = []
    try:
        assert proc.stdout is not None
        with log.open("w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                captured.append(line.rstrip())
                print(line, end="")
        rc = proc.wait()
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGINT)
        rc = proc.wait()
    worker_summary = rd / "paper_contract_teacher_rollout_vae_worker_summary.json"
    summary: dict[str, Any]
    if rc == 0 and worker_summary.is_file():
        summary = json.loads(worker_summary.read_text(encoding="utf-8"))
        summary["status"] = "ok_paper_contract_teacher_rollout_vae_training_completed"
    else:
        fail_path = FAILED_DIR / f"paper_contract_teacher_rollout_vae_failed_{int(time.time())}.log"
        fail_path.write_text("\n".join(captured[-500:]), encoding="utf-8")
        summary = {
            "status": "failed_paper_contract_teacher_rollout_vae_training",
            "returncode": rc,
            "failed_log_tail": str(fail_path),
            "goal_complete": False,
        }
    summary.setdefault("outputs", {})
    summary["outputs"].update(
        {
            "json": str(OUT / "level_c_paper_contract_teacher_rollout_vae_training.json"),
            "run_dir": str(rd),
            "log": str(log),
            "worker_script": str(worker),
        }
    )
    summary["duration_wall_seconds"] = round(time.time() - start, 3)
    summary["generated_at"] = utc_now()
    write_json(OUT / "level_c_paper_contract_teacher_rollout_vae_training.json", summary)
    print(json.dumps({"status": summary["status"], "json": str(OUT / "level_c_paper_contract_teacher_rollout_vae_training.json")}, sort_keys=True))
    raise SystemExit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
