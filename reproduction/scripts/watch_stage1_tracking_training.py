#!/usr/bin/env python3
"""Live read-only monitor for the two current Stage-1 PPO teacher runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class RunSpec:
    label: str
    short_name: str
    gpus: tuple[int, ...]
    log_path: Path
    run_dir: Path
    wrapper_script: str
    worker_path_fragment: str
    contract_json: Path | None = None
    bundle_json: Path | None = None
    notes: str = ""


RUNS = (
    RunSpec(
        label="GPU 4+7 LAFAN1 paper-contract PPO teacher",
        short_name="lafan1_40_motion",
        gpus=(4, 7),
        log_path=ROOT
        / "logs/tracking_g1_official_importer_export_paper_contract_ppo_training_run"
        / "tracking_g1_resource_adjusted_ppo_training_run.log",
        run_dir=ROOT
        / "res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training"
        / "resource_adjusted_ppo_20260622_084243_seed20260801",
        wrapper_script="tracking_g1_official_importer_export_paper_contract_ppo_training_run.py",
        worker_path_fragment=(
            "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
            "tracking_g1_resource_adjusted_ppo_worker.py"
        ),
        contract_json=ROOT
        / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run"
        / "paper_contract_tracking_parameters.json",
        notes="40 Unitree-retargeted LAFAN1 motions; local teacher retraining, not official checkpoint.",
    ),
    RunSpec(
        label="GPU 5+6 multi-source public-plus-available PPO teacher",
        short_name="multisource_49_motion",
        gpus=(5, 6),
        log_path=ROOT
        / "logs/tracking_stage1_multisource_paper_contract_ppo_training_run"
        / "tracking_g1_resource_adjusted_ppo_training_run.log",
        run_dir=ROOT
        / "res/runs/stage1_multisource_paper_contract_ppo_training"
        / "resource_adjusted_ppo_20260622_114146_seed20260851",
        wrapper_script="tracking_stage1_multisource_paper_contract_ppo_training_run.py",
        worker_path_fragment=(
            "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
            "tracking_g1_resource_adjusted_ppo_worker.py"
        ),
        contract_json=ROOT
        / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run"
        / "paper_contract_tracking_parameters.json",
        bundle_json=ROOT
        / "res/tracking/stage1_multisource_motion_bundle"
        / "tracking_stage1_multisource_motion_bundle.json",
        notes=(
            "49 motions, 2.49 h public-plus-available bundle; local reconstruction candidate, "
            "not the complete official paper motion set."
        ),
    ),
)


def read_tail(path: Path, max_bytes: int = 4_000_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    return ANSI_RE.sub("", data.decode("utf-8", errors="replace"))


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_latest_block(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seeds = [int(match.group(1)) for match in re.finditer(r"Setting seed:\s+(\d+)", text)]
    if seeds:
        out["seeds"] = sorted(set(seeds))
    starts = list(re.finditer(r"Learning iteration\s+(\d+)/(\d+)", text))
    if not starts:
        return out
    last = starts[-1]
    out["iteration"] = int(last.group(1))
    out["max_iterations"] = int(last.group(2))
    block = text[last.start() :]

    patterns = {
        "computation_steps_per_s": rf"Computation:\s+({FLOAT_RE})\s+steps/s",
        "collection_s": rf"collection:\s+({FLOAT_RE})s",
        "learning_s": rf"learning\s+({FLOAT_RE})s",
        "mean_action_noise_std": rf"Mean action noise std:\s+({FLOAT_RE})",
        "mean_value_function_loss": rf"Mean value_function loss:\s+({FLOAT_RE})",
        "mean_surrogate_loss": rf"Mean surrogate loss:\s+({FLOAT_RE})",
        "mean_entropy_loss": rf"Mean entropy loss:\s+({FLOAT_RE})",
        "mean_reward": rf"Mean reward:\s+({FLOAT_RE})",
        "mean_episode_length": rf"Mean episode length:\s+({FLOAT_RE})",
        "reward_anchor_pos": rf"Episode_Reward/motion_global_anchor_pos:\s+({FLOAT_RE})",
        "reward_anchor_ori": rf"Episode_Reward/motion_global_anchor_ori:\s+({FLOAT_RE})",
        "reward_body_pos": rf"Episode_Reward/motion_body_pos:\s+({FLOAT_RE})",
        "reward_body_ori": rf"Episode_Reward/motion_body_ori:\s+({FLOAT_RE})",
        "reward_body_lin_vel": rf"Episode_Reward/motion_body_lin_vel:\s+({FLOAT_RE})",
        "reward_body_ang_vel": rf"Episode_Reward/motion_body_ang_vel:\s+({FLOAT_RE})",
        "reward_action_rate_l2": rf"Episode_Reward/action_rate_l2:\s+({FLOAT_RE})",
        "reward_joint_limit": rf"Episode_Reward/joint_limit:\s+({FLOAT_RE})",
        "reward_undesired_contacts": rf"Episode_Reward/undesired_contacts:\s+({FLOAT_RE})",
        "error_anchor_pos": rf"Metrics/motion/error_anchor_pos:\s+({FLOAT_RE})",
        "error_anchor_rot": rf"Metrics/motion/error_anchor_rot:\s+({FLOAT_RE})",
        "error_anchor_lin_vel": rf"Metrics/motion/error_anchor_lin_vel:\s+({FLOAT_RE})",
        "error_anchor_ang_vel": rf"Metrics/motion/error_anchor_ang_vel:\s+({FLOAT_RE})",
        "error_body_pos": rf"Metrics/motion/error_body_pos:\s+({FLOAT_RE})",
        "error_body_rot": rf"Metrics/motion/error_body_rot:\s+({FLOAT_RE})",
        "error_joint_pos": rf"Metrics/motion/error_joint_pos:\s+({FLOAT_RE})",
        "error_joint_vel": rf"Metrics/motion/error_joint_vel:\s+({FLOAT_RE})",
        "sampling_entropy": rf"Metrics/motion/sampling_entropy:\s+({FLOAT_RE})",
        "sampling_top1_prob": rf"Metrics/motion/sampling_top1_prob:\s+({FLOAT_RE})",
        "sampling_top1_bin": rf"Metrics/motion/sampling_top1_bin:\s+({FLOAT_RE})",
        "error_body_lin_vel": rf"Metrics/motion/error_body_lin_vel:\s+({FLOAT_RE})",
        "error_body_ang_vel": rf"Metrics/motion/error_body_ang_vel:\s+({FLOAT_RE})",
        "termination_timeout": rf"Episode_Termination/time_out:\s+({FLOAT_RE})",
        "termination_anchor_pos": rf"Episode_Termination/anchor_pos:\s+({FLOAT_RE})",
        "termination_anchor_ori": rf"Episode_Termination/anchor_ori:\s+({FLOAT_RE})",
        "termination_ee_body_pos": rf"Episode_Termination/ee_body_pos:\s+({FLOAT_RE})",
        "total_timesteps": r"Total timesteps:\s+(\d+)",
        "iteration_time_s": rf"Iteration time:\s+({FLOAT_RE})s",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, block)
        if not match:
            continue
        raw = match.group(1)
        out[key] = int(raw) if key == "total_timesteps" else float(raw)

    for key, pattern in {
        "time_elapsed": r"Time elapsed:\s+([0-9:]+)",
        "eta": r"ETA:\s+([0-9:]+)",
    }.items():
        match = re.search(pattern, block)
        if match:
            out[key] = match.group(1)

    out["progress_percent"] = (
        100.0 * out["iteration"] / out["max_iterations"]
        if out.get("max_iterations")
        else None
    )
    return out


def latest_checkpoint(run_dir: Path) -> dict[str, Any]:
    rank0 = run_dir / "rank_0"
    checkpoints = []
    for path in rank0.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt$", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return {"count": 0}
    iteration, path = max(checkpoints, key=lambda item: item[0])
    return {
        "count": len(checkpoints),
        "iteration": iteration,
        "path": str(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_mb": path.stat().st_size / 1024 / 1024,
    }


def gpu_status(gpus: tuple[int, ...]) -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(map(str, gpus)),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return [{"error": str(exc)}]
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or proc.stdout.strip()}]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "mem_used_mib": parts[1],
                    "mem_total_mib": parts[2],
                    "util_percent": parts[3],
                    "power_w": parts[4],
                }
            )
    return rows


def process_status(spec: RunSpec) -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in proc.stdout.splitlines()[1:]:
        if spec.wrapper_script not in line and spec.worker_path_fragment not in line:
            continue
        parts = line.split(None, 4)
        if len(parts) == 5:
            rows.append(
                {
                    "pid": parts[0],
                    "ppid": parts[1],
                    "stat": parts[2],
                    "etime": parts[3],
                    "cmd": parts[4],
                }
            )
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def short_path(path: str | Path | None) -> str:
    if not path:
        return "n/a"
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def contract_summary(spec: RunSpec) -> list[str]:
    contract = read_json(spec.contract_json)
    bundle = read_json(spec.bundle_json)
    sim = contract.get("simulation", {})
    ppo = contract.get("ppo_contract", {})
    task = contract.get("tracking_task", {})
    lines = []
    lines.append(
        "params: "
        f"seed={ppo.get('seed', 'n/a')}, "
        f"max_iter={ppo.get('max_iterations', 'n/a')}, "
        f"steps/env={ppo.get('num_steps_per_env', 'n/a')}, "
        f"envs/rank={sim.get('num_envs_per_rank', 'n/a')}, "
        f"total_envs={sim.get('target_total_envs', 'n/a')}, "
        f"obs={task.get('policy_obs_dim', 'n/a')}, critic_obs={task.get('critic_obs_dim', 'n/a')}, "
        f"action={task.get('action_dim', 'n/a')}"
    )
    lines.append(
        "ppo: "
        f"lr={ppo.get('learning_rate', 'n/a')}, gamma={ppo.get('gamma', 'n/a')}, "
        f"lambda={ppo.get('lam', 'n/a')}, clip={ppo.get('clip_param', 'n/a')}, "
        f"entropy={ppo.get('entropy_coef', 'n/a')}, desired_kl={ppo.get('desired_kl', 'n/a')}"
    )
    if bundle:
        metrics = bundle.get("metrics", {})
        counts = metrics.get("source_counts", {})
        lines.append(
            "bundle: "
            f"motions={metrics.get('motion_count', 'n/a')}, "
            f"frames={metrics.get('total_frames', 'n/a')}, "
            f"duration_h={format_float(metrics.get('total_duration_hours'), 3)}, "
            f"sources={counts}"
        )
    else:
        lines.append("bundle: 40 Unitree-retargeted LAFAN1 motions from the current paper-contract run input.")
    return lines


def params_line(spec: RunSpec, latest: dict[str, Any]) -> str:
    contract = read_json(spec.contract_json)
    sim = contract.get("simulation", {})
    ppo = contract.get("ppo_contract", {})
    task = contract.get("tracking_task", {})
    seed = latest.get("seeds") or ppo.get("seed", "n/a")
    max_iter = latest.get("max_iterations") or ppo.get("max_iterations", "n/a")
    return (
        "params: "
        f"seeds={seed}, "
        f"max_iter={max_iter}, "
        f"steps/env={ppo.get('num_steps_per_env', 'n/a')}, "
        f"envs/rank={sim.get('num_envs_per_rank', 'n/a')}, "
        f"total_envs={sim.get('target_total_envs', 'n/a')}, "
        f"obs={task.get('policy_obs_dim', 'n/a')}, critic_obs={task.get('critic_obs_dim', 'n/a')}, "
        f"action={task.get('action_dim', 'n/a')}"
    )


def render_run(spec: RunSpec) -> str:
    text = read_tail(spec.log_path)
    latest = parse_latest_block(text)
    ckpt = latest_checkpoint(spec.run_dir)
    gpus = gpu_status(spec.gpus)
    procs = process_status(spec)
    lines = []
    lines.append("=" * 100)
    lines.append(f"{spec.label}")
    lines.append(f"claim: local Stage-1 teacher training monitor; no official checkpoint or paper-level claim")
    lines.append(f"note: {spec.notes}")
    lines.append(f"log: {short_path(spec.log_path)}")
    lines.append(f"run: {short_path(spec.run_dir)}")
    contract_lines = contract_summary(spec)
    if contract_lines:
        contract_lines[0] = params_line(spec, latest)
    lines.extend(contract_lines)
    lines.append(
        "process: "
        + (
            ", ".join(f"pid={p['pid']} etime={p['etime']} stat={p['stat']}" for p in procs[:6])
            if procs
            else "not found by monitor filter"
        )
    )
    gpu_parts = []
    for row in gpus:
        if "error" in row:
            gpu_parts.append(f"error={row['error']}")
        else:
            gpu_parts.append(
                f"GPU{row['index']} mem={row['mem_used_mib']}/{row['mem_total_mib']}MiB "
                f"util={row['util_percent']}% power={row['power_w']}W"
            )
    lines.append("gpu: " + "; ".join(gpu_parts))
    if latest:
        progress = latest.get("progress_percent")
        progress_s = f"{progress:.2f}%" if isinstance(progress, float) else "n/a"
        lines.append(
            "train: "
            f"iter={latest.get('iteration', 'n/a')}/{latest.get('max_iterations', 'n/a')} "
            f"({progress_s}), timesteps={latest.get('total_timesteps', 'n/a')}, "
            f"speed={format_float(latest.get('computation_steps_per_s'), 0)} steps/s, "
            f"elapsed={latest.get('time_elapsed', 'n/a')}, ETA={latest.get('eta', 'n/a')}"
        )
        lines.append(
            "loss/reward: "
            f"reward={format_float(latest.get('mean_reward'))}, "
            f"ep_len={format_float(latest.get('mean_episode_length'), 2)}, "
            f"value_loss={format_float(latest.get('mean_value_function_loss'))}, "
            f"surrogate={format_float(latest.get('mean_surrogate_loss'))}, "
            f"entropy={format_float(latest.get('mean_entropy_loss'))}, "
            f"noise_std={format_float(latest.get('mean_action_noise_std'))}"
        )
        lines.append(
            "motion error: "
            f"anchor_pos={format_float(latest.get('error_anchor_pos'))}, "
            f"anchor_rot={format_float(latest.get('error_anchor_rot'))}, "
            f"body_pos={format_float(latest.get('error_body_pos'))}, "
            f"body_rot={format_float(latest.get('error_body_rot'))}, "
            f"joint_pos={format_float(latest.get('error_joint_pos'))}, "
            f"joint_vel={format_float(latest.get('error_joint_vel'))}"
        )
        lines.append(
            "reward terms: "
            f"anchor_pos={format_float(latest.get('reward_anchor_pos'))}, "
            f"anchor_ori={format_float(latest.get('reward_anchor_ori'))}, "
            f"body_pos={format_float(latest.get('reward_body_pos'))}, "
            f"body_vel={format_float(latest.get('reward_body_lin_vel'))}, "
            f"action_rate={format_float(latest.get('reward_action_rate_l2'))}, "
            f"joint_limit={format_float(latest.get('reward_joint_limit'))}"
        )
        lines.append(
            "termination/sample: "
            f"timeout={format_float(latest.get('termination_timeout'))}, "
            f"anchor_pos={format_float(latest.get('termination_anchor_pos'))}, "
            f"anchor_ori={format_float(latest.get('termination_anchor_ori'))}, "
            f"ee_body_pos={format_float(latest.get('termination_ee_body_pos'))}, "
            f"sample_entropy={format_float(latest.get('sampling_entropy'))}, "
            f"top1_prob={format_float(latest.get('sampling_top1_prob'))}"
        )
    else:
        lines.append("train: no learning-iteration block parsed yet")
    if ckpt.get("count"):
        lines.append(
            "checkpoint: "
            f"latest=model_{ckpt['iteration']}.pt, count={ckpt['count']}, "
            f"mtime={ckpt['mtime']}, size={ckpt['size_mb']:.1f}MiB, "
            f"path={short_path(ckpt['path'])}"
        )
    else:
        lines.append("checkpoint: none found")
    return "\n".join(lines)


def render_screen(args: argparse.Namespace) -> str:
    header = [
        "BeyondMimic Stage-1 PPO teacher live monitor",
        f"root: {ROOT}",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  interval={args.interval}s",
        "read-only: parses logs/checkpoints/nvidia-smi only; it does not touch running jobs",
    ]
    body = [render_run(spec) for spec in RUNS]
    footer = [
        "=" * 100,
        "Boundary: these are local teacher retraining jobs. They are not official BeyondMimic weights yet.",
        "Exit: Ctrl-C",
    ]
    return "\n".join(header + body + footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal before refresh.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        while True:
            screen = render_screen(args)
            if not args.no_clear and not args.once:
                print("\033[2J\033[H", end="")
            print(screen, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
