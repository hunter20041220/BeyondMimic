
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

npz_path = Path(os.environ["BM_CAPTURE_NPZ"])
metrics_path = Path(os.environ["BM_METRICS_JSON"])
asset_json = Path(os.environ["BM_ASSET_JSON"])
body_contract = json.loads(Path(os.environ["BM_BODY_CONTRACT"]).read_text())
source_contract = json.loads(Path(os.environ["BM_SOURCE_CONTRACT"]).read_text())
metrics = json.loads(metrics_path.read_text())
out_dir = asset_json.parent
out_dir.mkdir(parents=True, exist_ok=True)

target_names = source_contract["flat_env"]["body_names"]
edges = [
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]
data = np.load(npz_path)
variants = ["teacher", "vae_base", "action_guided"]
robot = {variant: data[f"{variant}_robot_body_pos_w"] for variant in variants}
reference = data["teacher_reference_body_pos_w"]
if reference.shape[1] == len(target_names):
    names = target_names
    target_indices = list(range(len(target_names)))
elif reference.shape[1] == len(body_contract["body_names_urdf_order"]):
    names = body_contract["body_names_urdf_order"]
    full_idx = {name: idx for idx, name in enumerate(names)}
    target_indices = [full_idx[name] for name in target_names]
else:
    raise ValueError(f"Unexpected body count {reference.shape[1]}")
name_to_idx = {name: idx for idx, name in enumerate(names)}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def target(arr):
    return arr[:, target_indices, :]

target_ref = target(reference)
target_robot = {variant: target(arr) for variant, arr in robot.items()}

def set_axes_equal(ax, frame):
    xyz = np.concatenate([target_ref[frame][None, :, :]] + [target_robot[v][frame][None, :, :] for v in variants], axis=1)
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.4)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)

def draw_skeleton(ax, xyz, color, label, linewidth=2.0, alpha=0.9):
    for a, b in edges:
        pts = xyz[[name_to_idx[a], name_to_idx[b]]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(xyz[target_indices, 0], xyz[target_indices, 1], xyz[target_indices, 2], s=18, color=color, alpha=alpha, label=label)

colors = {"teacher": "#059669", "vae_base": "#2563eb", "action_guided": "#dc2626"}
labels = {"teacher": "teacher", "vae_base": "VAE base", "action_guided": "guided"}

def draw_frame(ax, frame):
    draw_skeleton(ax, reference[frame], "#64748b", "reference", linewidth=1.2, alpha=0.55)
    for variant in variants:
        draw_skeleton(ax, robot[variant][frame], colors[variant], labels[variant], linewidth=1.7 if variant != "action_guided" else 2.4, alpha=0.82)
    ax.set_title(f"Local action-guidance rollout, frame {frame:03d}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

video_path = out_dir / "official_csv_loop_action_guidance_rollout_vs_reference.mp4"
keyframes_path = out_dir / "official_csv_loop_action_guidance_rollout_keyframes.png"
metrics_png = out_dir / "official_csv_loop_action_guidance_rollout_metrics.png"
metrics_csv = out_dir / "official_csv_loop_action_guidance_rollout_metrics.csv"
readme = out_dir / "README.md"

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_subplot(111, projection="3d")
writer = FFMpegWriter(fps=30, metadata={"title": "BeyondMimic local action guidance rollout"})
with writer.saving(fig, str(video_path), dpi=145):
    for frame in range(reference.shape[0]):
        ax.cla()
        draw_frame(ax, frame)
        set_axes_equal(ax, frame)
        ax.view_init(elev=18, azim=-68)
        writer.grab_frame()
plt.close(fig)

frames = [0, reference.shape[0] // 3, 2 * reference.shape[0] // 3, reference.shape[0] - 1]
fig = plt.figure(figsize=(13, 7))
for idx, frame in enumerate(frames, start=1):
    ax = fig.add_subplot(2, 2, idx, projection="3d")
    draw_frame(ax, frame)
    set_axes_equal(ax, frame)
    ax.view_init(elev=18, azim=-68)
fig.tight_layout()
fig.savefig(keyframes_path, dpi=180)
plt.close(fig)

with metrics_csv.open("w", encoding="utf-8", newline="") as f:
    writer_csv = csv.DictWriter(
        f,
        fieldnames=[
            "variant",
            "step",
            "reward",
            "done",
            "action_abs_mean",
            "target_body_error_mean",
            "teacher_vae_mse",
            "guided_base_mse",
            "guided_teacher_mse",
        ],
    )
    writer_csv.writeheader()
    for variant in variants:
        for step in range(reference.shape[0]):
            writer_csv.writerow({
                "variant": variant,
                "step": step,
                "reward": float(data[f"{variant}_rewards"][step]),
                "done": int(data[f"{variant}_dones"][step]),
                "action_abs_mean": float(data[f"{variant}_action_abs_mean"][step]),
                "target_body_error_mean": float(data[f"{variant}_target_body_error_mean"][step]),
                "teacher_vae_mse": float(data[f"{variant}_teacher_vae_mse"][step]),
                "guided_base_mse": float(data[f"{variant}_guided_base_mse"][step]),
                "guided_teacher_mse": float(data[f"{variant}_guided_teacher_mse"][step]),
            })

fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
for variant in variants:
    x = np.arange(reference.shape[0])
    axes[0].plot(x, data[f"{variant}_rewards"], label=labels[variant], color=colors[variant])
    axes[1].plot(x, data[f"{variant}_target_body_error_mean"], label=labels[variant], color=colors[variant])
    axes[2].plot(x, data[f"{variant}_action_abs_mean"], label=labels[variant], color=colors[variant])
axes[0].set_ylabel("reward")
axes[1].set_ylabel("target-body error")
axes[2].set_ylabel("|action| mean")
axes[2].set_xlabel("step")
axes[0].legend(loc="best")
fig.suptitle("Local Action Guidance Closed-Loop Rollout")
fig.tight_layout()
fig.savefig(metrics_png, dpi=180)
plt.close(fig)

readme.write_text("\n".join([
    "# Official-CSV-Loop Local Action Guidance Rollout",
    "",
    "This directory contains a local virtual closed-loop rollout comparing teacher, VAE base, and action-guided variants.",
    "",
    "The guided action is `a_guided = a_vae + alpha * (a_teacher - a_vae)` with alpha recorded in the asset JSON.",
    "",
    "## Claim Level",
    "",
    "local_virtual_teacher_consistency_action_guidance_rollout. This is not official BeyondMimic latent diffusion guidance, not Fig. 5/Fig. 6 paper-level evidence, and not real-robot validation.",
    "",
]), encoding="utf-8")

assets = {
    "mp4": str(video_path),
    "keyframes_png": str(keyframes_path),
    "metrics_png": str(metrics_png),
    "metrics_csv": str(metrics_csv),
    "readme": str(readme),
}
summary = {
    "status": "ok",
    "experiment_type": "tracking_g1_official_csv_loop_action_guidance_rollout_eval_assets",
    "claim_level": "local_virtual_teacher_consistency_action_guidance_rollout",
    "source_capture_npz": str(npz_path),
    "source_metrics": str(metrics_path),
    "frame_count": int(reference.shape[0]),
    "body_count": int(reference.shape[1]),
    "target_body_count": len(target_indices),
    "guidance": metrics["guidance"],
    "variant_metrics": metrics["variant_metrics"],
    "assets": assets,
    "asset_sizes": {key: Path(value).stat().st_size for key, value in assets.items()},
    "asset_sha256": {key: sha256_file(Path(value)) for key, value in assets.items()},
    "checks": {
        "frame_count_299": int(reference.shape[0]) == 299,
        "body_count_supported_14_or_40": int(reference.shape[1]) in {14, 40},
        "target_body_count_14": len(target_indices) == 14,
        "video_exists_nonempty": video_path.is_file() and video_path.stat().st_size > 0,
        "keyframes_exist_nonempty": keyframes_path.is_file() and keyframes_path.stat().st_size > 0,
        "metrics_plot_exists_nonempty": metrics_png.is_file() and metrics_png.stat().st_size > 0,
        "does_not_claim_paper_level": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_receding_horizon_latent_diffusion": True,
        "does_not_claim_real_robot": True,
    },
    "interpretation": {
        "goal_complete": False,
        "why_not_complete": "The rollout proves a local virtual closed-loop action-guidance bridge, but it does not use the official BeyondMimic diffusion checkpoint or paper task guidance loop.",
    },
}
asset_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": "ok", "json": str(asset_json), "mp4": str(video_path)}, sort_keys=True))
