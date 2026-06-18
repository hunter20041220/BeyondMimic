#!/usr/bin/env python3
"""Reproduce released-data figures from the BeyondMimic Zenodo bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DATASET = ROOT / "reproduction/data/Dataset_beyondmimic"
OUT_ROOT = ROOT / "res/released_figures"

TIME_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})")

PALETTE = {
    "blue": (0.0000, 0.4470, 0.7410),
    "yellow": (0.9290, 0.6940, 0.1250),
    "purple": (0.4940, 0.1840, 0.5560),
    "green": (0.4660, 0.6740, 0.1880),
    "red": (0.6350, 0.0780, 0.1840),
    "black": (0.2500, 0.2500, 0.2500),
}

COLOR_MAP = {
    "pos_err": PALETTE["blue"],
    "ori_err": PALETTE["red"],
    "lin_vel_err": PALETTE["purple"],
    "ang_vel_err": PALETTE["yellow"],
}


@dataclass(frozen=True)
class FigureRecord:
    figure_id: str
    panel_mapping: str
    source_files: list[Path]
    processed_csv: Path
    outputs: list[Path]
    log: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_hash_manifest(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted({x.resolve() for x in paths}):
        rows.append(
            {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, out_stem: Path) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ["pdf", "svg", "png"]:
        out = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 240
        fig.savefig(out, **kwargs)
        outputs.append(out)
    plt.close(fig)
    return outputs


def parse_dir_ts_key(name: str) -> tuple[int, int, int, int, int, int]:
    m = TIME_PATTERN.search(name)
    if not m:
        return (0, 0, 0, 0, 0, 0)
    return tuple(map(int, m.groups()))


def pick_latest_dir(base: Path, prefix: str) -> Path:
    pre = f"{prefix}_rosbag2_"
    cands = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(pre)]
    if not cands:
        raise FileNotFoundError(f"No experiment directory for prefix {prefix!r} under {base}")
    return sorted(cands, key=lambda d: parse_dir_ts_key(d.name))[-1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collapse_segments(segments: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for key in keys:
        vals = [float(seg["errors"][key]) for seg in segments if key in seg.get("errors", {})]
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def ablation_rows(base: Path, prefixes: list[str]) -> tuple[list[dict], list[Path]]:
    rows = []
    sources = []
    for prefix in prefixes:
        exp_dir = pick_latest_dir(base, prefix)
        global_json = exp_dir / "global_pose_overview_errors.json"
        local_json = exp_dir / "res.json"
        sources.extend([global_json, local_json])
        g = collapse_segments(load_json(global_json)["segments"], ["pos_err", "ori_err"])
        l = collapse_segments(load_json(local_json)["segments"], ["pos_err", "ori_err", "lin_vel_err", "ang_vel_err"])
        for scope, stats in [("global", g), ("local", l)]:
            for metric, values in stats.items():
                rows.append(
                    {
                        "experiment": prefix,
                        "directory": exp_dir.name,
                        "scope": scope,
                        "metric": metric,
                        "mean": values["mean"],
                        "min": values["min"],
                        "max": values["max"],
                    }
                )
    return rows, sources


def plot_ablation(csv_path: Path, figure_id: str, title: str, prefixes: list[str], out_dir: Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    baseline = df[df["experiment"] == "origin"]
    if baseline.empty:
        baseline_exp = prefixes[0]
        baseline = df[df["experiment"] == baseline_exp]
    else:
        baseline_exp = "origin"

    outputs = []
    for scope, metrics, figsize in [
        ("global", ["pos_err", "ori_err"], (7.0, 3.2)),
        ("local", ["pos_err", "ori_err", "lin_vel_err", "ang_vel_err"], (11.0, 3.2)),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        x = np.arange(len(prefixes), dtype=float)
        n = len(metrics)
        width = min(0.18, 0.8 / max(n, 1))
        offsets = (np.arange(n) - (n - 1) / 2.0) * width * 1.15
        for i, metric in enumerate(metrics):
            sub = df[(df["scope"] == scope) & (df["metric"] == metric)].set_index("experiment")
            base = float(baseline[(baseline["scope"] == scope) & (baseline["metric"] == metric)]["mean"].iloc[0])
            mean = np.asarray([float(sub.loc[p, "mean"]) / base for p in prefixes])
            lo = np.asarray([float(sub.loc[p, "min"]) / base for p in prefixes])
            hi = np.asarray([float(sub.loc[p, "max"]) / base for p in prefixes])
            xx = x + offsets[i]
            color = COLOR_MAP[metric]
            ax.bar(xx, mean, width=width, color=color, alpha=0.58, edgecolor="none", label=metric)
            ax.errorbar(xx, mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="none", color=color, capsize=3)
            ax.plot(xx, mean, linestyle="--", linewidth=1.7, color=color, alpha=0.85)
        ax.axhline(1.0, color=PALETTE["black"], linewidth=1.0, alpha=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(prefixes, rotation=18, ha="right")
        ax.set_ylabel(f"x {baseline_exp} mean")
        ax.set_title(f"{title}: {scope} tracking error")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(ncol=min(4, n), frameon=False, fontsize=8)
        outputs.extend(save_figure(fig, out_dir / f"{figure_id}_{scope}"))
    return outputs


def reproduce_ablation_group(figure_id: str, title: str, prefixes: list[str], mapping: str) -> FigureRecord:
    out_dir = OUT_ROOT / figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sources = ablation_rows(DATASET / "rosbag_ablation", prefixes)
    csv_path = out_dir / f"{figure_id}_processed.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    outputs = plot_ablation(csv_path, figure_id, title, prefixes, out_dir)
    log = out_dir / "run.log"
    log.write_text(
        f"figure_id={figure_id}\nkind=ablation_error\nprefixes={','.join(prefixes)}\n"
        f"processed_csv={csv_path}\noutputs={','.join(str(p) for p in outputs)}\n",
        encoding="utf-8",
    )
    write_hash_manifest(sources + [csv_path], out_dir / "source_hashes.tsv")
    (out_dir / "paper_panel_mapping.md").write_text(mapping + "\n", encoding="utf-8")
    return FigureRecord(figure_id, mapping, sources, csv_path, outputs, log)


def lowpass_filter(data: np.ndarray, cutoff_freq: float, fs: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = min(0.99, cutoff_freq / nyquist)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def reproduce_imu() -> FigureRecord:
    figure_id = "imu_orientation_accel_angular_velocity"
    out_dir = OUT_ROOT / figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    source = DATASET / "base_imu/rosbag_data_raw_base_imu.csv"
    df = pd.read_csv(source)
    df = df.iloc[7000:-8000].reset_index(drop=True)
    df["time"] = df["__time"] - df["__time"].iloc[0]
    dt = df["time"].diff().dropna().median()
    fs = float(1.0 / dt) if dt > 0 else 100.0

    col = {
        "roll": "/controller_manager/introspection_data/values/state_interface.base_imu/orientation.roll",
        "pitch": "/controller_manager/introspection_data/values/state_interface.base_imu/orientation.pitch",
        "yaw": "/controller_manager/introspection_data/values/state_interface.base_imu/orientation.yaw",
        "acc_x": "/controller_manager/introspection_data/values/state_interface.base_imu/linear_acceleration.x",
        "acc_y": "/controller_manager/introspection_data/values/state_interface.base_imu/linear_acceleration.y",
        "acc_z": "/controller_manager/introspection_data/values/state_interface.base_imu/linear_acceleration.z",
        "ang_x": "/controller_manager/introspection_data/values/state_interface.base_imu/angular_velocity.x",
        "ang_y": "/controller_manager/introspection_data/values/state_interface.base_imu/angular_velocity.y",
        "ang_z": "/controller_manager/introspection_data/values/state_interface.base_imu/angular_velocity.z",
    }
    processed = pd.DataFrame({"time": df["time"]})
    for name, column in col.items():
        values = df[column].astype(float)
        if name.startswith(("acc_", "ang_")) and values.notna().sum() > 10:
            valid = values.notna()
            filtered = np.full(len(values), np.nan)
            filtered[valid.to_numpy()] = lowpass_filter(values[valid].to_numpy(), cutoff_freq=500.0, fs=fs)
            processed[name] = filtered
        else:
            processed[name] = values
    csv_path = out_dir / f"{figure_id}_processed.csv"
    processed.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.0), sharex=True, constrained_layout=True)
    colors = [PALETTE["red"], PALETTE["blue"], PALETTE["purple"]]
    for ax, names, ylabel, title in [
        (axes[0], ["roll", "pitch", "yaw"], "rad", "IMU orientation"),
        (axes[1], ["acc_x", "acc_y", "acc_z"], "m/s^2", "Linear acceleration"),
        (axes[2], ["ang_x", "ang_y", "ang_z"], "rad/s", "Angular velocity"),
    ]:
        for name, color in zip(names, colors):
            ax.plot(processed["time"], processed[name], label=name, linewidth=1.4, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, frameon=False, fontsize=8)
    axes[2].set_xlabel("Time (s)")
    outputs = save_figure(fig, out_dir / figure_id)
    log = out_dir / "run.log"
    log.write_text(
        f"figure_id={figure_id}\nkind=imu\nsource={source}\nsampling_freq={fs:.6f}\n"
        f"samples={len(processed)}\nprocessed_csv={csv_path}\n",
        encoding="utf-8",
    )
    write_hash_manifest([source, csv_path], out_dir / "source_hashes.tsv")
    mapping = (
        "# Paper Panel Mapping\n\n"
        "Level A released-data reproduction for IMU orientation, acceleration, and angular velocity. "
        "Source corresponds to `base_imu/plot_imu_csv.py` in the Zenodo bundle."
    )
    (out_dir / "paper_panel_mapping.md").write_text(mapping + "\n", encoding="utf-8")
    return FigureRecord(figure_id, mapping, [source], csv_path, outputs, log)


def normalize_colname(name: str) -> str:
    value = str(name).strip()
    value = re.sub(r"\s+", "", value).replace(".", "/")
    return re.sub(r"/+", "/", value).lstrip("/").lower()


def first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm:
            return norm[key]
    return None


def lowpass_np(y: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0 or not np.isfinite(cutoff_hz) or fs <= 0 or not np.isfinite(fs):
        return y
    nyquist = 0.5 * fs
    wc = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, wc, btype="low")
    return filtfilt(b, a, y, method="pad")


def plot_grf_processed(csv_path: Path, out_stem: Path, title: str) -> list[Path]:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.0), constrained_layout=True)
    x = df["phase_or_time"]
    for name, color in [("Fx", PALETTE["blue"]), ("Fy", PALETTE["red"]), ("Fz", PALETTE["purple"])]:
        mean_col = f"{name}_mean"
        if mean_col in df:
            ax.plot(x, df[mean_col], color=color, linewidth=2.0, label=name)
            lo_col = f"{name}_min"
            hi_col = f"{name}_max"
            if lo_col in df and hi_col in df:
                ax.fill_between(x, df[lo_col], df[hi_col], color=color, alpha=0.15)
        elif name in df:
            ax.plot(x, df[name], color=color, linewidth=2.0, label=name)
    ax.set_title(title)
    ax.set_xlabel("Stance (%)" if x.max() > 10 else "Time (s)")
    ax.set_ylabel("Force / BW")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    ax.margins(x=0)
    return save_figure(fig, out_stem)


def reproduce_grf_walk_ref() -> FigureRecord:
    figure_id = "grf_walk_human_reference"
    out_dir = OUT_ROOT / figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        DATASET / "GRF/walk_ref/GRF_F_ML_PRO_left.csv",
        DATASET / "GRF/walk_ref/GRF_F_AP_PRO_left.csv",
        DATASET / "GRF/walk_ref/GRF_F_V_PRO_left.csv",
    ]
    prefixes = ["F_ML_PRO_", "F_AP_PRO_", "F_V_PRO_"]
    names = ["Fx", "Fy", "Fz"]
    processed = {"phase_or_time": np.linspace(0, 100, 101)}
    for source, prefix, name in zip(sources, prefixes, names):
        df = pd.read_csv(source, low_memory=False)
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix) and re.search(r"\d+$", c)]
        cols = sorted(cols, key=lambda c: int(re.search(r"(\d+)$", c).group(1)))
        waves = df[cols].to_numpy(dtype=float)
        processed[f"{name}_mean"] = waves.mean(axis=0)
        processed[f"{name}_min"] = waves.min(axis=0)
        processed[f"{name}_max"] = waves.max(axis=0)
    csv_path = out_dir / f"{figure_id}_processed.csv"
    pd.DataFrame(processed).to_csv(csv_path, index=False)
    outputs = plot_grf_processed(csv_path, out_dir / figure_id, "Walking GRF: human reference")
    log = out_dir / "run.log"
    log.write_text(f"figure_id={figure_id}\nkind=grf_walk_ref\nprocessed_csv={csv_path}\n", encoding="utf-8")
    write_hash_manifest(sources + [csv_path], out_dir / "source_hashes.tsv")
    mapping = "# Paper Panel Mapping\n\nWalking GRF human reference from `GRF/walk_ref` released GaitRec PRO CSV files."
    (out_dir / "paper_panel_mapping.md").write_text(mapping + "\n", encoding="utf-8")
    return FigureRecord(figure_id, mapping, sources, csv_path, outputs, log)


def load_force_plate(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=None, engine="python")
    lower = {str(c).lower().strip(): c for c in df.columns}
    tcol = lower.get("time", df.columns[0])
    fxcol = lower.get("fx")
    fycol = lower.get("fy")
    fzcol = lower.get("fz")
    if fxcol is None or fycol is None or fzcol is None:
        raise KeyError(f"Missing Fx/Fy/Fz in {path}")
    return (
        pd.to_numeric(df[tcol], errors="coerce").to_numpy(float),
        pd.to_numeric(df[fxcol], errors="coerce").to_numpy(float),
        pd.to_numeric(df[fycol], errors="coerce").to_numpy(float),
        pd.to_numeric(df[fzcol], errors="coerce").to_numpy(float),
    )


def reproduce_grf_run_ref() -> FigureRecord:
    figure_id = "grf_run_human_reference"
    out_dir = OUT_ROOT / figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    source = DATASET / "GRF/run_ref/RBDS028runT25forces.txt"
    t, fx, fy, fz = load_force_plate(source)
    lb, ub = 2072, 2173
    bw = 58.55 * 9.8
    t_new = t[lb:ub] - t[lb]
    fx_new = -fx[lb:ub] / bw
    fz_new = fy[lb:ub] / bw
    fy_new = -fz[lb:ub] / bw
    fs = 1.0 / max(float(np.median(np.diff(t_new))), 1e-6)
    processed = pd.DataFrame(
        {
            "phase_or_time": t_new,
            "Fx": lowpass_np(fx_new, fs, 0.1),
            "Fy": lowpass_np(fy_new, fs, 0.1),
            "Fz": lowpass_np(fz_new, fs, 0.1),
        }
    )
    csv_path = out_dir / f"{figure_id}_processed.csv"
    processed.to_csv(csv_path, index=False)
    outputs = plot_grf_processed(csv_path, out_dir / figure_id, "Running GRF: human reference")
    log = out_dir / "run.log"
    log.write_text(f"figure_id={figure_id}\nkind=grf_run_ref\nsource={source}\nprocessed_csv={csv_path}\n", encoding="utf-8")
    write_hash_manifest([source, csv_path], out_dir / "source_hashes.tsv")
    mapping = "# Paper Panel Mapping\n\nRunning GRF human reference from `GRF/run_ref/RBDS028runT25forces.txt`."
    (out_dir / "paper_panel_mapping.md").write_text(mapping + "\n", encoding="utf-8")
    return FigureRecord(figure_id, mapping, [source], csv_path, outputs, log)


def read_robot_wrench(path: Path, foot_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=None, engine="python")
    sec = first_column(df, [f"{foot_tag}/header/stamp/sec", f"/{foot_tag}/header/stamp/sec"])
    nsec = first_column(df, [f"{foot_tag}/header/stamp/nanosec", f"/{foot_tag}/header/stamp/nanosec"])
    time_col = first_column(df, ["__time", "/__time"])
    if sec and nsec:
        t = pd.to_numeric(df[sec], errors="coerce") + pd.to_numeric(df[nsec], errors="coerce") * 1e-9
        t = t - t.iloc[0]
    elif time_col:
        t = pd.to_numeric(df[time_col], errors="coerce")
        t = t - t.iloc[0]
    else:
        raise KeyError(f"No time columns found in {path}")
    cols = [
        first_column(df, [f"{foot_tag}/wrench/force/x", f"/{foot_tag}/wrench/force/x"]),
        first_column(df, [f"{foot_tag}/wrench/force/y", f"/{foot_tag}/wrench/force/y"]),
        first_column(df, [f"{foot_tag}/wrench/force/z", f"/{foot_tag}/wrench/force/z"]),
    ]
    if any(c is None for c in cols):
        raise KeyError(f"Missing force columns for {foot_tag} in {path}")
    arrays = [pd.to_numeric(df[c], errors="coerce").interpolate("linear", limit_direction="both").to_numpy(float) for c in cols]
    t_arr = pd.to_numeric(t, errors="coerce").interpolate("linear", limit_direction="both").to_numpy(float)
    mask = np.isfinite(t_arr) & np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])
    return t_arr[mask], arrays[0][mask], arrays[1][mask], arrays[2][mask]


def segment_grf(t: np.ndarray, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> pd.DataFrame:
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / max(dt, 1e-6)
    fx_f, fy_f, fz_f = (lowpass_np(v, fs, 12.0) for v in [fx, fy, fz])
    above = fz_f > 40.0
    edges = np.diff(above.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above) - 1)
    waves = {"Fx": [], "Fy": [], "Fz": []}
    for s, e in zip(starts, ends):
        if e <= s or t[e] - t[s] < 0.10:
            continue
        lo = max(0, s - 80)
        hi = min(len(t) - 1, e + 50)
        if hi - lo < 5:
            continue
        phase_old = (t[lo : hi + 1] - t[lo]) / max(t[hi] - t[lo], 1e-9)
        phase = np.linspace(0, 1, 101)
        waves["Fx"].append(np.interp(phase, phase_old, -fx_f[lo : hi + 1]) / (35 * 9.8))
        waves["Fy"].append(np.interp(phase, phase_old, -fy_f[lo : hi + 1]) / (35 * 9.8))
        waves["Fz"].append(np.interp(phase, phase_old, fz_f[lo : hi + 1]) / (35 * 9.8))
    processed = {"phase_or_time": np.linspace(0, 100, 101)}
    for name in ["Fx", "Fy", "Fz"]:
        arr = np.vstack(waves[name])
        processed[f"{name}_mean"] = arr.mean(axis=0)
        processed[f"{name}_min"] = arr.min(axis=0)
        processed[f"{name}_max"] = arr.max(axis=0)
    return pd.DataFrame(processed)


def reproduce_grf_robot(figure_id: str, source_rel: str, title: str) -> FigureRecord:
    out_dir = OUT_ROOT / figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    source = DATASET / source_rel
    t, fx, fy, fz = read_robot_wrench(source, "wrench_LL_FOOT")
    processed = segment_grf(t, fx, fy, fz)
    csv_path = out_dir / f"{figure_id}_processed.csv"
    processed.to_csv(csv_path, index=False)
    outputs = plot_grf_processed(csv_path, out_dir / figure_id, title)
    log = out_dir / "run.log"
    log.write_text(f"figure_id={figure_id}\nkind=grf_robot\nsource={source}\nprocessed_csv={csv_path}\n", encoding="utf-8")
    write_hash_manifest([source, csv_path], out_dir / "source_hashes.tsv")
    mapping = f"# Paper Panel Mapping\n\n{title} from released robot foot-wrench CSV `{source_rel}`."
    (out_dir / "paper_panel_mapping.md").write_text(mapping + "\n", encoding="utf-8")
    return FigureRecord(figure_id, mapping, [source], csv_path, outputs, log)


def main() -> None:
    global DATASET, OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "reproduction/data/Dataset_beyondmimic")
    parser.add_argument("--out", type=Path, default=ROOT / "res/released_figures")
    args = parser.parse_args()
    DATASET = args.dataset
    OUT_ROOT = args.out
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    records: list[FigureRecord] = []
    records.append(reproduce_imu())
    records.append(
        reproduce_ablation_group(
            "ablation_observation_history",
            "Observation history ablation",
            ["origin", "hist4", "hist8", "hist25"],
            "# Paper Panel Mapping\n\nObservation history ablation using `origin`, `hist4`, `hist8`, and `hist25` released JSON metrics.",
        )
    )
    records.append(
        reproduce_ablation_group(
            "ablation_armature",
            "Armature ablation",
            ["arma0", "arma0.1", "origin", "arma10"],
            "# Paper Panel Mapping\n\nArmature ablation using `arma0`, `arma0.1`, `origin`, and `arma10` released JSON metrics.",
        )
    )
    records.append(
        reproduce_ablation_group(
            "ablation_pd_gain",
            "PD gain sensitivity",
            ["wn5", "origin", "asapgain", "wn25"],
            "# Paper Panel Mapping\n\nPD gain sensitivity using `wn5`, `origin`, `asapgain`, and `wn25` released JSON metrics.",
        )
    )
    records.append(
        reproduce_ablation_group(
            "ablation_orientation_representation",
            "Orientation representation ablation",
            ["origin", "quat", "axisangle"],
            "# Paper Panel Mapping\n\nOrientation representation ablation using `origin`, `quat`, and `axisangle` released JSON metrics.",
        )
    )
    records.append(
        reproduce_ablation_group(
            "ablation_latency",
            "Latency ablation",
            ["origin", "2ms", "5ms", "10ms"],
            "# Paper Panel Mapping\n\nLatency ablation using `origin`, `2ms`, `5ms`, and `10ms` released JSON metrics.",
        )
    )
    records.append(reproduce_grf_walk_ref())
    records.append(reproduce_grf_run_ref())
    records.append(reproduce_grf_robot("grf_walk_robot_real", "GRF/real_walk/walk_bag2.csv", "Walking GRF: robot real"))
    records.append(reproduce_grf_robot("grf_run_robot_real", "GRF/real_run/run_bag.csv", "Running GRF: robot real"))

    summary_path = OUT_ROOT / "released_figure_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["figure_id", "processed_csv", "outputs", "log", "source_count"],
            delimiter="\t",
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "figure_id": r.figure_id,
                    "processed_csv": str(r.processed_csv),
                    "outputs": ",".join(str(p) for p in r.outputs),
                    "log": str(r.log),
                    "source_count": len(r.source_files),
                }
            )
    print(f"wrote {summary_path}")
    print(f"figures reproduced: {len(records)}")


if __name__ == "__main__":
    main()
