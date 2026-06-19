
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

root = Path(os.environ["BM_ROOT"])
npz_path = Path(os.environ["BM_CAPTURE_NPZ"])
asset_json = Path(os.environ["BM_ASSET_JSON"])
out_dir = asset_json.parent
body_contract = json.loads(Path(os.environ["BM_BODY_CONTRACT"]).read_text())
source_contract = json.loads(Path(os.environ["BM_SOURCE_CONTRACT"]).read_text())
capture_summary = json.loads(Path(os.environ["BM_CAPTURE_SUMMARY"]).read_text())
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
robot = data["robot_body_pos_w"]
reference = data["reference_body_pos_w"]
rewards = data["rewards"]
actions = data["action_abs_mean"]
if robot.shape[1] == len(target_names):
    names = target_names
    target_indices = list(range(len(target_names)))
elif robot.shape[1] == len(body_contract["body_names_urdf_order"]):
    names = body_contract["body_names_urdf_order"]
    name_to_idx_full = {name: idx for idx, name in enumerate(names)}
    target_indices = [name_to_idx_full[name] for name in target_names]
else:
    raise ValueError(f"Unexpected body count {robot.shape[1]} for policy rollout trace")
name_to_idx = {name: idx for idx, name in enumerate(names)}
target_robot = robot[:, target_indices, :]
target_reference = reference[:, target_indices, :]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def set_axes_equal(ax):
    xyz = np.concatenate([target_robot, target_reference], axis=1)
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.4)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)

def draw_skeleton(ax, xyz, color, label, linewidth=2.0):
    for a, b in edges:
        pts = xyz[[name_to_idx[a], name_to_idx[b]]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth, alpha=0.9)
    ax.scatter(xyz[target_indices, 0], xyz[target_indices, 1], xyz[target_indices, 2], s=22, color=color, alpha=0.9, label=label)

def draw_frame(ax, frame):
    draw_skeleton(ax, reference[frame], "#64748b", "reference", linewidth=1.5)
    draw_skeleton(ax, robot[frame], "#2563eb", "policy robot", linewidth=2.2)
    ax.set_title(f"Policy rollout vs reference, frame {frame:03d}", fontsize=11)
    ax.legend(loc="upper right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / "official_csv_loop_policy_rollout_vs_reference.mp4"
keyframes_path = out_dir / "official_csv_loop_policy_rollout_keyframes.png"
metrics_csv = out_dir / "official_csv_loop_policy_rollout_metrics.csv"
readme = out_dir / "README.md"

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_subplot(111, projection="3d")
writer = FFMpegWriter(fps=30, metadata={"title": "BeyondMimic policy rollout vs reference"})
with writer.saving(fig, str(video_path), dpi=150):
    for frame in range(robot.shape[0]):
        ax.cla()
        draw_frame(ax, frame)
        set_axes_equal(ax)
        ax.view_init(elev=18, azim=-68)
        writer.grab_frame()
plt.close(fig)

frames = [0, robot.shape[0] // 3, 2 * robot.shape[0] // 3, robot.shape[0] - 1]
fig = plt.figure(figsize=(13, 7))
for idx, frame in enumerate(frames, start=1):
    ax = fig.add_subplot(2, 2, idx, projection="3d")
    draw_frame(ax, frame)
    set_axes_equal(ax)
    ax.view_init(elev=18, azim=-68)
fig.tight_layout()
fig.savefig(keyframes_path, dpi=180)
plt.close(fig)

tracking_error = np.linalg.norm(target_robot - target_reference, axis=-1).mean(axis=1)
with metrics_csv.open("w", encoding="utf-8", newline="") as f:
    writer_csv = csv.DictWriter(f, fieldnames=["step", "reward", "action_abs_mean", "target_body_error_mean"])
    writer_csv.writeheader()
    for step in range(robot.shape[0]):
        writer_csv.writerow({
            "step": step,
            "reward": float(rewards[step]),
            "action_abs_mean": float(actions[step]),
            "target_body_error_mean": float(tracking_error[step]),
        })

readme.write_text("\n".join([
    "# Official-Loop Policy Rollout Visualization",
    "",
    "This directory contains a local virtual policy rollout visualization captured from the official csv-loop PPO checkpoint.",
    "",
    "## Claim Level",
    "",
    "local_virtual_resource_adjusted_policy_rollout_video. This is not unpatched official BeyondMimic replay, not paper-level Fig. 5/Fig. 6 guided diffusion, and not real-robot evidence.",
    "",
]), encoding="utf-8")

assets = {
    "mp4": str(video_path),
    "keyframes_png": str(keyframes_path),
    "metrics_csv": str(metrics_csv),
    "readme": str(readme),
}
summary = {
    "status": "ok",
    "experiment_type": "tracking_official_csv_loop_policy_rollout_video_capture",
    "claim_level": "local_virtual_resource_adjusted_policy_rollout_video",
    "source_capture_npz": str(npz_path),
    "source_capture_summary": os.environ["BM_CAPTURE_SUMMARY"],
    "source_capture_status": capture_summary["status"],
    "frame_count": int(robot.shape[0]),
    "body_count": int(robot.shape[1]),
    "target_body_count": len(target_indices),
    "metrics": {
        "reward_mean": float(np.mean(rewards)),
        "done_count_total": int(np.sum(data["dones"])),
        "action_abs_mean": float(np.mean(actions)),
        "target_body_error_mean": float(np.mean(tracking_error)),
        "target_body_error_max": float(np.max(tracking_error)),
    },
    "assets": assets,
    "asset_sizes": {key: Path(value).stat().st_size for key, value in assets.items()},
    "asset_sha256": {key: sha256_file(Path(value)) for key, value in assets.items()},
    "checks": {
        "capture_status_ok": capture_summary["status"] == "ok_official_csv_loop_policy_rollout_video_capture",
        "frame_count_299": int(robot.shape[0]) == 299,
        "body_count_supported_14_or_40": int(robot.shape[1]) in {14, 40},
        "target_body_count_14": len(target_indices) == 14,
        "video_exists_nonempty": video_path.is_file() and video_path.stat().st_size > 0,
        "keyframes_exist_nonempty": keyframes_path.is_file() and keyframes_path.stat().st_size > 0,
        "does_not_claim_paper_level": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
    },
    "interpretation": {
        "goal_complete": False,
        "why_not_complete": "This is a local virtual policy rollout video from the resource-adjusted official csv-loop PPO checkpoint, not paper-level guided diffusion or real-robot validation.",
    },
}
asset_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": "ok", "json": str(asset_json), "mp4": str(video_path)}, sort_keys=True))
