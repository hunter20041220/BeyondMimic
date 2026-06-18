#!/usr/bin/env python3
"""Plot paper-vs-code adaptive sampling probe from tracking_config_audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
AUDIT = ROOT / "res/tracking/smoke_config_audit/tracking_config_audit.json"
OUT = ROOT / "res/tracking/adaptive_sampling_probe"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    probe = audit["adaptive_sampling_probe"]
    code = probe["code_default_probabilities"]
    paper = probe["paper_three_bin_probabilities"]
    with (OUT / "adaptive_sampling_probe.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["bin", "failure", "code_kernel_size_1", "paper_kernel_size_3"])
        for i, (failure, c, p) in enumerate(zip(probe["failure_vector"], code, paper)):
            writer.writerow([i, failure, c, p])

    xs = list(range(len(code)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar([x - width / 2 for x in xs], code, width=width, label="code default K=1", color="#3b6fb6")
    ax.bar([x + width / 2 for x in xs], paper, width=width, label="paper u={0,1,2}", color="#d0673f")
    ax.plot(xs, probe["failure_vector"], color="#222222", marker="o", linewidth=1.2, label="failure bin")
    ax.set_xlabel("one-second bin")
    ax.set_ylabel("sampling probability")
    ax.set_title("Adaptive sampling kernel probe")
    ax.set_ylim(0, max(max(code), max(paper), 1.0) * 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"adaptive_sampling_probe.{ext}", dpi=180)
    plt.close(fig)
    (OUT / "run.log").write_text(
        "kind=adaptive_sampling_probe_plot\n"
        f"l1_difference={probe['l1_difference']}\n"
        f"code_argmax={probe['code_argmax']}\n"
        f"paper_argmax={probe['paper_argmax']}\n",
        encoding="utf-8",
    )
    print(OUT / "adaptive_sampling_probe.tsv")


if __name__ == "__main__":
    main()
