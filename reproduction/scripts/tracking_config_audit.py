#!/usr/bin/env python3
"""Non-Kit audit for official whole_body_tracking configuration and assets."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
OUT = ROOT / "res/tracking/smoke_config_audit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_snippets(path: Path, patterns: dict[str, str]) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.S)
        out[key] = match.group(1).strip() if match else "NOT_FOUND"
    return out


def parse_ppo(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    values = {}
    for key in ["num_steps_per_env", "max_iterations", "save_interval", "experiment_name", "empirical_normalization"]:
        m = re.search(rf"{key}\s*=\s*([^\n]+)", text)
        if m:
            raw = m.group(1).strip()
            try:
                values[key] = ast.literal_eval(raw)
            except Exception:
                values[key] = raw
    for key in ["actor_hidden_dims", "critic_hidden_dims"]:
        m = re.search(rf"{key}\s*=\s*(\[[^\]]+\])", text)
        if m:
            values[key] = ast.literal_eval(m.group(1))
    m = re.search(r'activation\s*=\s*["\']([^"\']+)["\']', text)
    if m:
        values["activation"] = m.group(1)
    for key in [
        "value_loss_coef",
        "use_clipped_value_loss",
        "clip_param",
        "entropy_coef",
        "num_learning_epochs",
        "num_mini_batches",
        "learning_rate",
        "schedule",
        "gamma",
        "lam",
        "desired_kl",
        "max_grad_norm",
    ]:
        m = re.search(rf"{key}\s*=\s*([^,\n\)]+)", text)
        if m:
            raw = m.group(1).strip()
            try:
                values[key] = ast.literal_eval(raw)
            except Exception:
                values[key] = raw.strip('"')
    return values


def parse_tracking_env(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    values = {}
    for key in ["num_envs", "env_spacing", "episode_length_s", "decimation"]:
        m = re.search(rf"{key}\s*=\s*([^,\n\)]+)", text)
        if m:
            try:
                values[key] = ast.literal_eval(m.group(1).strip())
            except Exception:
                values[key] = m.group(1).strip()
    m = re.search(r"self\.sim\.dt\s*=\s*([-+0-9.eE]+)", text)
    if m:
        values["sim_dt"] = float(m.group(1))
        if "decimation" in values:
            values["control_frequency_hz"] = 1.0 / (values["sim_dt"] * values["decimation"])
    values["velocity_range"] = source_snippets(path, {"VELOCITY_RANGE": r"VELOCITY_RANGE\s*=\s*(\{.*?\})\n\n"})[
        "VELOCITY_RANGE"
    ]
    values["reward_weights_present"] = sorted(set(re.findall(r"weight\s*=\s*([-+0-9.eE]+)", text)))
    values["termination_thresholds_present"] = sorted(set(re.findall(r"threshold[\"']?\s*[:=]\s*([-+0-9.eE]+)", text)))
    return values


def parse_g1_flat_cfg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    anchor = re.search(r"anchor_body_name\s*=\s*['\"]([^'\"]+)['\"]", text)
    body_names_match = re.search(r"body_names\s*=\s*(\[[^\]]+\])", text, flags=re.S)
    body_names = ast.literal_eval(body_names_match.group(1)) if body_names_match else []
    return {
        "anchor_body_name": anchor.group(1) if anchor else "NOT_FOUND",
        "body_names": body_names,
        "body_count": len(body_names),
        "anchor_in_body_names": (anchor.group(1) in body_names) if anchor else False,
        "body_names_source": str(path),
    }


def parse_termination_bodies(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'"body_names":\s*(\[[^\]]+\])', text, flags=re.S)
    body_names = ast.literal_eval(match.group(1)) if match else []
    return {
        "termination_body_names": body_names,
        "termination_body_count": len(body_names),
    }


def parse_adaptive_cfg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    values = {}
    for key in ["adaptive_kernel_size", "adaptive_lambda", "adaptive_uniform_ratio", "adaptive_alpha"]:
        match = re.search(rf"{key}:\s*[^=]+=\s*([-+0-9.eE]+)", text)
        if match:
            raw = match.group(1)
            values[key] = int(raw) if raw.isdigit() else float(raw)
    return values


def adaptive_sampling_probe() -> dict:
    """Compare official-code default kernel with paper-source three-bin look-ahead."""
    failure = np.zeros(8, dtype=float)
    failure[3] = 1.0
    uniform_ratio = 0.1
    rho = 0.8

    def probs(kernel_size: int) -> list[float]:
        kernel = np.array([rho**i for i in range(kernel_size)], dtype=float)
        kernel = kernel / kernel.sum()
        padded = np.pad(failure + uniform_ratio / len(failure), (0, kernel_size - 1), mode="edge")
        conv = np.convolve(padded, kernel, mode="valid")
        conv = conv / conv.sum()
        return conv.tolist()

    code_probs = probs(1)
    paper_probs = probs(3)
    return {
        "failure_vector": failure.tolist(),
        "code_default_kernel_size": 1,
        "paper_kernel_size_from_u_0_1_2": 3,
        "rho": rho,
        "uniform_ratio": uniform_ratio,
        "code_default_probabilities": code_probs,
        "paper_three_bin_probabilities": paper_probs,
        "l1_difference": float(np.abs(np.array(code_probs) - np.array(paper_probs)).sum()),
        "code_argmax": int(np.argmax(code_probs)),
        "paper_argmax": int(np.argmax(paper_probs)),
        "paper_spreads_mass_to_preceding_bins": bool(paper_probs[1] > code_probs[1] and paper_probs[2] > code_probs[2]),
        "prob_sums": {
            "code": float(math.fsum(code_probs)),
            "paper": float(math.fsum(paper_probs)),
        },
    }


def asset_audit() -> dict:
    urdf = WBT / "assets/unitree_description/urdf/g1/main.urdf"
    mesh_refs = re.findall(r'filename="package://unitree_description/([^"]+)"', urdf.read_text(encoding="utf-8"))
    missing = []
    for rel in mesh_refs:
        if not (WBT / "assets/unitree_description" / rel).exists():
            missing.append(rel)
    return {
        "g1_urdf": str(urdf),
        "g1_urdf_exists": urdf.exists(),
        "g1_urdf_bytes": urdf.stat().st_size,
        "mesh_references": len(mesh_refs),
        "missing_mesh_references": missing,
        "urdf_sha256": sha256_file(urdf),
    }


def lafan_audit() -> dict:
    csv_path = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"
    df = pd.read_csv(csv_path, header=None)
    return {
        "sample_csv": str(csv_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256": sha256_file(csv_path),
        "first_root_xyz": [float(x) for x in df.iloc[0, :3]],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "tracking_env_cfg": WBT / "tasks/tracking/tracking_env_cfg.py",
        "g1_flat_env_cfg": WBT / "tasks/tracking/config/g1/flat_env_cfg.py",
        "g1_ppo_cfg": WBT / "tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
        "commands": WBT / "tasks/tracking/mdp/commands.py",
        "rewards": WBT / "tasks/tracking/mdp/rewards.py",
        "g1_robot": WBT / "robots/g1.py",
    }
    audit = {
        "ppo": parse_ppo(sources["g1_ppo_cfg"]),
        "tracking_env": parse_tracking_env(sources["tracking_env_cfg"]),
        "g1_tracking_target_bodies": parse_g1_flat_cfg(sources["g1_flat_env_cfg"]),
        "termination_bodies": parse_termination_bodies(sources["tracking_env_cfg"]),
        "adaptive_sampling_code": parse_adaptive_cfg(sources["commands"]),
        "adaptive_sampling_probe": adaptive_sampling_probe(),
        "assets": asset_audit(),
        "lafan_sample": lafan_audit(),
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
    }
    json_path = OUT / "tracking_config_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "tracking_config_audit.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["section", "key", "value"])
        for section, vals in audit.items():
            if isinstance(vals, dict):
                for key, value in vals.items():
                    writer.writerow([section, key, json.dumps(value, sort_keys=True)])
    (OUT / "run.log").write_text("kind=whole_body_tracking_nonkit_config_audit\nstatus=ok\n", encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
