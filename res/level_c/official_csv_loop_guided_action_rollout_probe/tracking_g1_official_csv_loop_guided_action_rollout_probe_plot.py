
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
