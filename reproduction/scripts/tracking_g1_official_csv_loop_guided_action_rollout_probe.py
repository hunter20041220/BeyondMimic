#!/usr/bin/env python3
"""Run decoded VAE guidance action samples inside the local IsaacLab tracking task.

This is a bridge probe only: it executes short decoded action windows in the
resource-adjusted official-csv-loop tracking environment. It is not a paper-level
guided diffusion rollout, not Fig. 5/Fig. 6 evidence, and not real-robot evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/official_csv_loop_guided_action_rollout_probe"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_guided_action_rollout_probe"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_guided_action_rollout_probe"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_guided_action_rollout_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
ANALYSIS_PY = ROOT / "envs/bm_analysis/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
DECODE_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
    "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
)
DECODE_SAMPLES_NPZ = (
    ROOT
    / "res/runs/level_c_official_csv_loop_guidance_vae_action_decode_eval/"
    "official_csv_loop_guided_decode_20260619_122046_seed20260636/"
    "official_csv_loop_guidance_vae_action_decode_samples.npz"
)
SOURCE_CONTRACT = ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
CANDIDATE_GPUS = [4, 7]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
SEED = 20260638
TASK = "velocity_command"
SAMPLE_INDEX = 0


WORKER_CODE = r"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = "cuda:0"

print(f"BM_SENTINEL:guided_action_probe:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:guided_action_probe:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    samples_npz = Path(os.environ["BM_DECODE_SAMPLES_NPZ"])
    out_npz = Path(os.environ["BM_OUT_NPZ"])
    metrics_path = Path(os.environ["BM_METRICS_JSON"])
    sample_index = int(os.environ["BM_SAMPLE_INDEX"])
    task = os.environ["BM_TASK"]
    seed = int(os.environ["BM_SEED"])

    data = np.load(samples_npz)
    variants = {
        "base": data[f"base_action_{task}"][sample_index].astype(np.float32),
        "guided": data[f"guided_action_{task}"][sample_index].astype(np.float32),
        "teacher": data[f"teacher_action_{task}"][sample_index].astype(np.float32),
    }
    rollout_steps = int(variants["base"].shape[0])

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(enriched_usd),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.seed = seed

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:guided_action_probe:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term("motion")
    device = vec_env.unwrapped.device

    traces = {}
    rollout_metrics = {}
    metric_names = ["error_anchor_pos", "error_body_pos", "error_joint_pos", "sampling_top1_prob"]
    action_delta = {
        "base_guided_max_abs": float(np.max(np.abs(variants["base"] - variants["guided"]))),
        "base_guided_l2_mean": float(np.linalg.norm((variants["base"] - variants["guided"]).reshape(-1, 29), axis=-1).mean()),
        "base_teacher_mse": float(np.mean((variants["base"] - variants["teacher"]) ** 2)),
        "guided_teacher_mse": float(np.mean((variants["guided"] - variants["teacher"]) ** 2)),
    }

    for variant_name, seq_np in variants.items():
        obs, _ = vec_env.reset()
        robot_body_pos = []
        reference_body_pos = []
        rewards = []
        dones = []
        action_abs_mean = []
        action_abs_max = []
        metric_series = {name: [] for name in metric_names}
        with torch.no_grad():
            for step in range(rollout_steps):
                action = torch.as_tensor(seq_np[step], dtype=torch.float32, device=device).reshape(1, -1)
                obs, rew, done, extras = vec_env.step(action)
                robot_body_pos.append(command.robot_body_pos_w[0].detach().cpu().numpy().astype(np.float32))
                reference_body_pos.append(command.body_pos_relative_w[0].detach().cpu().numpy().astype(np.float32))
                rewards.append(float(rew.detach().cpu().mean()))
                dones.append(int(done.detach().cpu().sum()))
                action_abs_mean.append(float(action.abs().mean().detach().cpu()))
                action_abs_max.append(float(action.abs().max().detach().cpu()))
                for name in metric_names:
                    value = command.metrics.get(name)
                    metric_series[name].append(
                        float(value[0].detach().cpu()) if value is not None and value.numel() > 0 else float("nan")
                    )
        robot_arr = np.stack(robot_body_pos, axis=0)
        ref_arr = np.stack(reference_body_pos, axis=0)
        body_error = np.linalg.norm(robot_arr - ref_arr, axis=-1).mean(axis=1)
        traces[f"{variant_name}_robot_body_pos_w"] = robot_arr
        traces[f"{variant_name}_reference_body_pos_w"] = ref_arr
        traces[f"{variant_name}_rewards"] = np.asarray(rewards, dtype=np.float32)
        traces[f"{variant_name}_dones"] = np.asarray(dones, dtype=np.int32)
        traces[f"{variant_name}_action_abs_mean"] = np.asarray(action_abs_mean, dtype=np.float32)
        traces[f"{variant_name}_target_body_error_mean"] = body_error.astype(np.float32)
        rollout_metrics[variant_name] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "done_count_total": int(np.sum(dones)),
            "action_abs_mean": float(np.mean(action_abs_mean)),
            "action_abs_max": float(np.max(action_abs_max)),
            "target_body_error_mean": float(np.mean(body_error)),
            "target_body_error_max": float(np.max(body_error)),
            "motion_metrics": {
                name: {
                    "mean": float(np.nanmean(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
                for name, values in metric_series.items()
            },
        }
        print(f"BM_SENTINEL:guided_action_probe:variant={variant_name}:steps={rollout_steps}", flush=True)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        **traces,
        base_actions=variants["base"],
        guided_actions=variants["guided"],
        teacher_actions=variants["teacher"],
    )

    metrics = {
        "status": "ok",
        "task": task,
        "sample_index": sample_index,
        "device": args.device,
        "num_envs": 1,
        "rollout_steps": rollout_steps,
        "variant_count": len(variants),
        "action_delta": action_delta,
        "variant_metrics": rollout_metrics,
        "uses_resource_adjusted_usd": True,
        "official_csv_loop_motion": True,
        "decoded_action_source": str(samples_npz),
        "paper_level_guidance_rollout": False,
        "fig5_fig6_reproduction": False,
        "real_robot": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:guided_action_probe:npz={out_npz}", flush=True)
    print(f"BM_SENTINEL:guided_action_probe:metrics={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:guided_action_probe:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
"""


PLOT_CODE = r"""
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

npz_path = Path(os.environ["BM_CAPTURE_NPZ"])
metrics_path = Path(os.environ["BM_METRICS_JSON"])
summary_path = Path(os.environ["BM_PLOT_SUMMARY_JSON"])
out_dir = summary_path.parent
metrics = json.loads(metrics_path.read_text())
data = np.load(npz_path)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

out_dir.mkdir(parents=True, exist_ok=True)
timeseries_csv = out_dir / "official_csv_loop_guided_action_rollout_probe_timeseries.csv"
plot_png = out_dir / "official_csv_loop_guided_action_rollout_probe_metrics.png"
readme = out_dir / "README.md"

variants = ["base", "guided", "teacher"]
with timeseries_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["variant", "step", "reward", "done", "action_abs_mean", "target_body_error_mean"],
    )
    writer.writeheader()
    for variant in variants:
        rewards = data[f"{variant}_rewards"]
        dones = data[f"{variant}_dones"]
        actions = data[f"{variant}_action_abs_mean"]
        errors = data[f"{variant}_target_body_error_mean"]
        for step in range(rewards.shape[0]):
            writer.writerow(
                {
                    "variant": variant,
                    "step": step,
                    "reward": float(rewards[step]),
                    "done": int(dones[step]),
                    "action_abs_mean": float(actions[step]),
                    "target_body_error_mean": float(errors[step]),
                }
            )

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
colors = {"base": "#2563eb", "guided": "#dc2626", "teacher": "#059669"}
for variant in variants:
    x = np.arange(data[f"{variant}_rewards"].shape[0])
    axes[0].plot(x, data[f"{variant}_rewards"], label=variant, color=colors[variant])
    axes[1].plot(x, data[f"{variant}_target_body_error_mean"], label=variant, color=colors[variant])
    axes[2].plot(x, data[f"{variant}_action_abs_mean"], label=variant, color=colors[variant])
axes[0].set_ylabel("reward")
axes[1].set_ylabel("target-body error")
axes[2].set_ylabel("|action| mean")
axes[2].set_xlabel("step")
axes[0].legend(loc="best")
fig.suptitle("Decoded VAE Action Rollout Probe in IsaacLab")
fig.tight_layout()
fig.savefig(plot_png, dpi=180)
plt.close(fig)

readme.write_text("\n".join([
    "# Official-CSV-Loop Guided Action Rollout Probe",
    "",
    "This directory contains a short IsaacLab rollout probe for decoded local VAE actions from the offline guidance bridge.",
    "",
    "## Claim Level",
    "",
    "local_virtual_decoded_action_rollout_probe. The probe executes one 21-step decoded-action sample for base, guided, and teacher variants in the resource-adjusted official-csv-loop tracking task.",
    "",
    "It is not a paper-level closed-loop guided diffusion rollout, not Fig. 5/Fig. 6 evidence, and not real-robot evidence.",
    "",
]), encoding="utf-8")

assets = {
    "timeseries_csv": str(timeseries_csv),
    "metrics_png": str(plot_png),
    "readme": str(readme),
}
summary = {
    "status": "ok",
    "experiment_type": "tracking_official_csv_loop_guided_action_rollout_probe_assets",
    "claim_level": "local_virtual_decoded_action_rollout_probe",
    "source_npz": str(npz_path),
    "source_metrics": str(metrics_path),
    "task": metrics["task"],
    "sample_index": metrics["sample_index"],
    "rollout_steps": metrics["rollout_steps"],
    "variant_metrics": metrics["variant_metrics"],
    "action_delta": metrics["action_delta"],
    "assets": assets,
    "asset_sizes": {key: Path(value).stat().st_size for key, value in assets.items()},
    "asset_sha256": {key: sha256_file(Path(value)) for key, value in assets.items()},
    "checks": {
        "base_guided_actions_almost_identical": metrics["action_delta"]["base_guided_max_abs"] < 1e-4,
        "timeseries_csv_exists": timeseries_csv.is_file() and timeseries_csv.stat().st_size > 0,
        "plot_png_exists": plot_png.is_file() and plot_png.stat().st_size > 0,
        "does_not_claim_paper_level_guidance": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
    },
    "interpretation": {
        "goal_complete": False,
        "why_not_complete": "The probe only executes one short decoded-action sample from the local offline guidance bridge. It does not implement receding-horizon diffusion guidance or paper-level task success evaluation.",
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": "ok", "summary": str(summary_path), "plot": str(plot_png)}, sort_keys=True))
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def query_gpus() -> list[dict[str, Any]]:
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 6:
            continue
        index, name, mem_used, mem_total, util, power = [item.strip() for item in raw[:6]]
        mem_used_i = int(float(mem_used))
        mem_total_i = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": mem_used_i,
                "memory_total_mb": mem_total_i,
                "memory_free_mb": mem_total_i - mem_used_i,
                "utilization_gpu_percent": int(float(util)),
                "power_draw_w": float(power),
            }
        )
    return rows


def start_gpu_monitor(path: Path, selected_gpu: int) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv",
            "-i",
            str(selected_gpu),
            "-l",
            "2",
            "-f",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def base_env(selected_gpu: int, capture_npz: Path, metrics_json: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_gpu),
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "WANDB_MODE": "offline",
            "BM_ENRICHED_USD": str(ENRICHED_USD),
            "BM_MOTION_FILE": str(OFFICIAL_LOOP_MOTION_NPZ),
            "BM_DECODE_SAMPLES_NPZ": str(DECODE_SAMPLES_NPZ),
            "BM_OUT_NPZ": str(capture_npz),
            "BM_METRICS_JSON": str(metrics_json),
            "BM_TASK": TASK,
            "BM_SAMPLE_INDEX": str(SAMPLE_INDEX),
            "BM_SEED": str(SEED),
        }
    )
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_official_csv_loop_guided_action_rollout_probe_worker.py"
    plot_path = OUT / "tracking_g1_official_csv_loop_guided_action_rollout_probe_plot.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    plot_path.write_text(textwrap.dedent(PLOT_CODE), encoding="utf-8")

    gpu_snapshot = query_gpus()
    available = [
        row["index"]
        for row in gpu_snapshot
        if "index" in row
        and row.get("memory_free_mb", 0) >= MIN_FREE_MB
        and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
    ]
    selected_gpu = available[0] if available else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"guided_action_rollout_probe_{timestamp}_seed{SEED}"
    capture_npz = run_dir / "official_csv_loop_guided_action_rollout_probe_trace.npz"
    metrics_json = run_dir / "official_csv_loop_guided_action_rollout_probe_metrics.json"
    gpu_metrics = run_dir / "gpu_metrics.csv"
    log_path = LOG_DIR / "tracking_g1_official_csv_loop_guided_action_rollout_probe.log"
    asset_json = OUT / "official_csv_loop_guided_action_rollout_probe_assets.json"

    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "analysis_python_exists": ANALYSIS_PY.is_file(),
        "decode_summary_exists": DECODE_JSON.is_file(),
        "decode_samples_npz_exists": DECODE_SAMPLES_NPZ.is_file(),
        "motion_npz_exists": OFFICIAL_LOOP_MOTION_NPZ.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "source_contract_exists": SOURCE_CONTRACT.is_file(),
        "selected_gpu_available": selected_gpu is not None,
    }
    run_info: dict[str, Any] = {
        "attempted_capture": False,
        "selected_gpu": selected_gpu,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "capture_npz": str(capture_npz),
        "metrics_json": str(metrics_json),
        "gpu_metrics_csv": str(gpu_metrics),
    }
    capture_ok = False
    plot_ok = False

    if all(input_checks.values()) and selected_gpu is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor = start_gpu_monitor(gpu_metrics, selected_gpu)
        start = time.time()
        env = base_env(selected_gpu, capture_npz, metrics_json)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [str(TRACKING_PY), str(worker_path)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            returncode = proc.wait()
        monitor.terminate()
        try:
            monitor.wait(timeout=20)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=20)
        capture_metrics = load_json(metrics_json)
        capture_ok = returncode == 0 and capture_npz.is_file() and capture_metrics.get("status") == "ok"
        run_info.update(
            {
                "attempted_capture": True,
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
                "capture_metrics": capture_metrics,
                "capture_npz_exists": capture_npz.is_file(),
                "metrics_exists": metrics_json.is_file(),
            }
        )
        if capture_ok:
            plot_env = os.environ.copy()
            plot_env.update(
                {
                    "BM_CAPTURE_NPZ": str(capture_npz),
                    "BM_METRICS_JSON": str(metrics_json),
                    "BM_PLOT_SUMMARY_JSON": str(asset_json),
                }
            )
            rc, plot_out = run([str(ANALYSIS_PY), str(plot_path)], env=plot_env, timeout=300)
            run_info["plot_returncode"] = rc
            run_info["plot_output"] = plot_out[-4000:]
            plot_ok = rc == 0 and asset_json.is_file()
    else:
        run_info["reason_not_started"] = "Required inputs missing or no GPU 4/7 satisfied free-memory/utilization preflight."

    if capture_ok and plot_ok:
        status = "ok_official_csv_loop_guided_action_rollout_probe"
    elif capture_ok:
        status = "failed_guided_action_rollout_probe_plot_after_capture"
    elif run_info.get("attempted_capture"):
        status = "failed_guided_action_rollout_probe_capture"
        failed_copy = FAILED_DIR / f"guided_action_rollout_probe_{timestamp}.log"
        failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        run_info["failed_log_copy"] = str(failed_copy)
    else:
        status = "ok_with_resource_unavailable_before_guided_action_rollout_probe"

    capture_metrics = run_info.get("capture_metrics", {})
    action_delta = capture_metrics.get("action_delta", {})
    summary = {
        "status": status,
        "experiment_type": "tracking_official_csv_loop_guided_action_rollout_probe",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Executes one 21-step decoded local VAE action sample for base, guided, and teacher variants inside the "
            "resource-adjusted official-csv-loop IsaacLab tracking task. This is a bridge probe only, not paper-level "
            "closed-loop diffusion guidance."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpu": selected_gpu,
            "cuda_visible_devices": str(selected_gpu) if selected_gpu is not None else "",
            "num_envs": 1,
            "task": TASK,
            "sample_index": SAMPLE_INDEX,
            "seed": SEED,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
            "formal_gpu_experiment": False,
            "why_not_formal_gpu_experiment": (
                "This run is a short IsaacLab bridge/gate probe over one 21-step decoded-action sample, not training "
                "or paper-scale evaluation; therefore the >=10GB per GPU formal-experiment rule is not applicable."
            ),
        },
        "gpu_preflight": {"snapshot": gpu_snapshot, "available_gpus": available},
        "inputs": {
            "decode_summary_json": str(DECODE_JSON),
            "decode_samples_npz": str(DECODE_SAMPLES_NPZ),
            "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
        },
        "input_checks": input_checks,
        "run": run_info,
        "outputs": {
            "json": str(OUT / "tracking_g1_official_csv_loop_guided_action_rollout_probe.json"),
            "asset_json": str(asset_json),
            "worker_script": str(worker_path),
            "plot_script": str(plot_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "capture_npz": str(capture_npz),
            "gpu_metrics_csv": str(gpu_metrics),
        },
        "metrics": {
            "rollout_steps": capture_metrics.get("rollout_steps"),
            "variant_metrics": capture_metrics.get("variant_metrics"),
            "base_guided_max_abs_action_delta": action_delta.get("base_guided_max_abs"),
            "base_guided_l2_mean": action_delta.get("base_guided_l2_mean"),
            "base_teacher_mse": action_delta.get("base_teacher_mse"),
            "guided_teacher_mse": action_delta.get("guided_teacher_mse"),
        },
        "checks": {
            "capture_ok": capture_ok,
            "plot_ok": plot_ok,
            "asset_json_exists": asset_json.is_file(),
            "base_guided_actions_almost_identical": bool(action_delta.get("base_guided_max_abs", 1.0) < 1e-4),
            "does_not_claim_paper_level_guidance": True,
            "does_not_claim_closed_loop_receding_horizon_guidance": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_decoded_action_rollout_probe" if capture_ok and plot_ok else "not_completed",
            "negative_result": (
                "The sampled guided and base decoded actions are numerically almost identical, so this probe mainly "
                "validates the action-to-IsaacLab bridge and does not demonstrate a meaningful guided behavior change."
            ),
            "why_not_complete": (
                "The probe executes only a short local decoded-action sample under a resource-adjusted USD/motion path. "
                "It does not run receding-horizon diffusion guidance in closed loop and does not evaluate paper task "
                "success metrics."
            ),
        },
    }
    (OUT / "tracking_g1_official_csv_loop_guided_action_rollout_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"], "asset_json": str(asset_json)}, sort_keys=True))
    if status.startswith("failed_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
