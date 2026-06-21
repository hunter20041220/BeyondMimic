
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.loads(Path('/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_fk_repaired_robot_order_split_task_eval/fk_repaired_robot_order_split_task_eval_plot_rows.json').read_text(encoding="utf-8"))
motions = [row["motion"] for row in rows]
x = list(range(len(motions)))

def f(row, key):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(x, [f(row, "reward_mean") for row in rows], marker="o", linewidth=1.4, label="Reward mean", color="#1f77b4")
ax1.set_ylabel("Reward mean")
ax1.set_xticks(x)
ax1.set_xticklabels(motions, rotation=75, ha="right", fontsize=7)
ax2 = ax1.twinx()
ax2.bar(x, [f(row, "done_total") for row in rows], alpha=0.25, label="Done count", color="#d62728")
ax2.set_ylabel("Terminated + truncated count")
ax1.set_title("FK-Repaired Robot-Order Split Task Eval: Reward and Done Counts")
fig.tight_layout()
fig.savefig('/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_fk_repaired_robot_order_split_task_eval/fk_repaired_robot_order_split_task_eval_reward_done.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 5))
for key, label, color in [
    ("error_anchor_pos", "Anchor position", "#1f77b4"),
    ("error_body_pos", "Body position", "#2ca02c"),
    ("error_joint_pos", "Joint position", "#ff7f0e"),
]:
    ax.plot(x, [f(row, key) for row in rows], marker="o", linewidth=1.2, label=label, color=color)
ax.set_xticks(x)
ax.set_xticklabels(motions, rotation=75, ha="right", fontsize=7)
ax.set_ylabel("Final command metric")
ax.set_title("FK-Repaired Robot-Order Split Task Eval: Tracking Error Metrics")
ax.legend()
fig.tight_layout()
fig.savefig('/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_fk_repaired_robot_order_split_task_eval/fk_repaired_robot_order_split_task_eval_tracking_errors.png', dpi=180)
plt.close(fig)
