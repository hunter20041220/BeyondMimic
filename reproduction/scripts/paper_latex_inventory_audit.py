#!/usr/bin/env python3
"""Build an automatic LaTeX inventory for formulas, tables, figures, and experiment settings."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PAPER = ROOT / "reproduction/paper/source"
OUT = ROOT / "res/paper_latex_inventory"
SOURCE_COVERAGE_JSON = ROOT / "res/paper_source_coverage/paper_source_coverage_audit.json"
TABLE_VALUES_JSON = ROOT / "res/paper_table_values/paper_table_value_audit.json"

SETTING_PATTERNS = [
    ("motion_tracking_control_frequency", r"Policies run at 50Hz", ["50Hz"]),
    ("state_estimator_frequency", r"Full-state estimation is provided at \$500\$~Hz", ["500 Hz"]),
    ("onnx_cpu_latency", r"Each inference step takes under \$1\.0\$~ms", ["under 1.0 ms"]),
    ("diffusion_control_frequency", r"train and deploy tracking policies at 25 Hz", ["25 Hz"]),
    ("diffusion_parameter_count", r"parameter count of \\~19\.8M", ["~19.8M"]),
    ("diffusion_tensorrt_hardware", r"RTX 4060 Mobile GPU", ["RTX 4060 Mobile GPU", "TensorRT"]),
    ("diffusion_latency_budget", r"approximately \$20\\,\\mathrm\{ms\}\$ using 20 denoising steps", ["20 ms", "20 denoising steps"]),
    ("ou_noise_parameters", r"theta = 0\.8.*?Delta t = 1\.0.*?sigma = 0\.1", ["theta=0.8", "Delta t=1.0", "sigma=0.1"]),
    ("rollout_collection_windows", r"policy for 2\.5 seconds.*?continue for 5 seconds", ["2.5 seconds", "5 seconds"]),
    ("sample_coverage", r"appears approximately 100 times", ["approximately 100 times"]),
    ("pd_gain_frequency", r"omega = 10 ?\\text\{Hz\}", ["omega=10 Hz"]),
    ("inpainting_keyframe_interval", r"keyframes at 0\.2 s intervals", ["0.2 s"]),
    ("velocity_error_claims", r"average velocity tracking error of 12\.14\\% and 13\.65\\%", ["12.14%", "13.65%"]),
    ("long_horizon_track_claim", r"run continuously for over 50 m", ["over 50 m"]),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def line_number(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


def tex_files() -> list[Path]:
    return sorted(PAPER.glob("*.tex")) + sorted((PAPER / "tex").glob("*.tex"))


def extract_envs(path: Path, env: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", re.S)
    rows = []
    for match in pattern.finditer(text):
        block = match.group(0)
        label_match = re.search(r"\\label\{([^}]+)\}", block)
        caption_match = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|$)", block, re.S)
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", block)
        rows.append(
            {
                "kind": env,
                "source": f"{path.relative_to(ROOT)}:{line_number(text, match.start())}",
                "label": label_match.group(1) if label_match else "",
                "caption_excerpt": clean_text(caption_match.group(1))[:700] if caption_match else "",
                "graphics": graphics,
                "content_excerpt": clean_text(block)[:1200],
            }
        )
    return rows


def extract_sections(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    pattern = re.compile(r"\\(?P<kind>section|subsection|subsubsection)\*?\{(?P<title>[^}]+)\}")
    for match in pattern.finditer(text):
        rows.append(
            {
                "kind": match.group("kind"),
                "source": f"{path.relative_to(ROOT)}:{line_number(text, match.start())}",
                "title": clean_text(match.group("title")),
            }
        )
    return rows


def equation_topic(equation: str, nearby: str) -> str:
    eq = equation.lower()
    if "g_{\\mathrm{js}" in eq:
        return "joystick_cost"
    if "g_{\\mathrm{wp}" in eq:
        return "waypoint_cost"
    if "g_{\\mathrm{sdf" in eq:
        return "sdf_obstacle_cost"
    if "b(x" in eq or "-\\ln" in eq:
        return "sdf_barrier_function"
    if "\\eta_" in eq:
        return "ou_perturbation"
    combined = f"{nearby} {equation}".lower()
    topics = [
        ("ou_perturbation", ["eta_", "ou", "ornstein", "theta", "sigma"]),
        ("joystick_cost", ["g_{\\mathrm{js}", "joystick", "velocity"]),
        ("waypoint_cost", ["g_{\\mathrm{wp}", "waypoint"]),
        ("sdf_barrier_cost", ["g_{\\mathrm{sdf", "sdf", "barrier"]),
        ("state_representation", ["root", "yaw", "local", "relative"]),
        ("reward_or_regularization", ["r_{", "reward", "lambda"]),
        ("diffusion_forward_reverse", ["alpha", "gamma", "sigma", "denoising", "diffusion"]),
        ("vae_objective", ["vae", "kl", "decoder", "encoder", "dagger"]),
    ]
    for topic, needles in topics:
        if any(needle in combined for needle in needles):
            return topic
    return "uncategorized_formula"


def extract_equations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\\begin\{equation\}(.*?)\\end\{equation\}", re.S)
    rows = []
    for idx, match in enumerate(pattern.finditer(text), start=1):
        start_line = line_number(text, match.start())
        before_start = max(0, match.start() - 700)
        nearby = clean_text(text[before_start : match.start()])
        equation = clean_text(match.group(1))
        rows.append(
            {
                "kind": "equation",
                "equation_index": idx,
                "source": f"{path.relative_to(ROOT)}:{start_line}",
                "topic": equation_topic(equation, nearby),
                "equation_excerpt": equation,
                "context_excerpt": nearby[-700:],
            }
        )
    return rows


def extract_setting_rows() -> list[dict[str, Any]]:
    rows = []
    for path in tex_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        plain = clean_text(text)
        for setting_id, pattern, expected_values in SETTING_PATTERNS:
            match = re.search(pattern, plain, flags=re.I | re.S)
            if not match:
                continue
            raw_start = text.lower().find(match.group(0).split()[0].lower())
            if raw_start < 0:
                raw_start = 0
            rows.append(
                {
                    "kind": "experiment_setting",
                    "setting_id": setting_id,
                    "source": f"{path.relative_to(ROOT)}:{line_number(text, raw_start)}",
                    "expected_values": expected_values,
                    "matched_excerpt": match.group(0),
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                out[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
            writer.writerow(out)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Paper LaTeX Inventory Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- TeX files: `{summary['counts']['tex_file_count']}`",
        f"- Equations: `{summary['counts']['equation_count']}`",
        f"- Figures: `{summary['counts']['figure_count']}`",
        f"- Tables: `{summary['counts']['table_count']}`",
        f"- Experiment settings: `{summary['counts']['experiment_setting_count']}`",
        f"- Missing expected settings: `{summary['counts']['missing_expected_setting_count']}`",
        "",
        "## Equation Topics",
    ]
    for topic, count in sorted(summary["equation_topic_counts"].items()):
        lines.append(f"- `{topic}`: `{count}`")
    lines.extend(["", "## Outputs"])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = tex_files()
    file_rows = [
        {
            "kind": "tex_file",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "line_count": path.read_text(encoding="utf-8", errors="ignore").count("\n") + 1,
        }
        for path in files
    ]
    figure_rows = []
    table_rows = []
    section_rows = []
    equation_rows = []
    for path in files:
        figure_rows.extend(extract_envs(path, "figure"))
        table_rows.extend(extract_envs(path, "table"))
        section_rows.extend(extract_sections(path))
        equation_rows.extend(extract_equations(path))
    setting_rows = extract_setting_rows()
    found_settings = {row["setting_id"] for row in setting_rows}
    missing_settings = [setting_id for setting_id, _, _ in SETTING_PATTERNS if setting_id not in found_settings]
    equation_topic_counts: dict[str, int] = {}
    for row in equation_rows:
        equation_topic_counts[row["topic"]] = equation_topic_counts.get(row["topic"], 0) + 1

    source_coverage = load_json(SOURCE_COVERAGE_JSON)
    table_values = load_json(TABLE_VALUES_JSON)
    checks = {
        "source_coverage_audit_ok": source_coverage["status"] == "ok",
        "table_value_audit_ok": table_values["status"] == "ok",
        "all_tex_files_hashed": all(bool(row["sha256"]) for row in file_rows),
        "has_equations": len(equation_rows) >= 8,
        "has_all_expected_figures": len(figure_rows) >= 10,
        "has_all_expected_tables": len(table_rows) >= 8,
        "has_expected_experiment_settings": not missing_settings,
        "table_value_mismatch_zero": table_values["counts"]["mismatch_rows"] == 0,
        "paper_source_unmapped_zero": source_coverage["counts"]["unmapped_rows"] == 0,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "paper_latex_inventory_audit",
        "scope": "automatic inventory of BeyondMimic LaTeX source structure, formulas, tables, figures, and experiment-setting statements",
        "counts": {
            "tex_file_count": len(file_rows),
            "section_count": len(section_rows),
            "equation_count": len(equation_rows),
            "figure_count": len(figure_rows),
            "table_count": len(table_rows),
            "experiment_setting_count": len(setting_rows),
            "expected_setting_count": len(SETTING_PATTERNS),
            "missing_expected_setting_count": len(missing_settings),
        },
        "equation_topic_counts": dict(sorted(equation_topic_counts.items())),
        "missing_expected_settings": missing_settings,
        "file_rows": file_rows,
        "section_rows": section_rows,
        "equation_rows": equation_rows,
        "figure_rows": figure_rows,
        "table_rows": table_rows,
        "experiment_setting_rows": setting_rows,
        "cross_checks": {
            "paper_source_coverage_counts": source_coverage["counts"],
            "paper_table_value_counts": table_values["counts"],
        },
        "checks": checks,
        "interpretation": {
            "latex_inventory_complete_for_current_source": all(
                checks[key]
                for key in [
                    "all_tex_files_hashed",
                    "has_equations",
                    "has_all_expected_figures",
                    "has_all_expected_tables",
                    "has_expected_experiment_settings",
                ]
            ),
            "goal_complete": False,
            "why_not_complete": (
                "This inventory verifies that the local paper LaTeX formulas, tables, figures, and key experiment "
                "settings are indexed and cross-checked against existing audits. It does not create trained checkpoints, "
                "closed-loop rollout logs, TensorRT engines, videos, or real-robot evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "paper_latex_inventory_audit.json"),
            "equations_tsv": str(OUT / "paper_latex_equations.tsv"),
            "tables_tsv": str(OUT / "paper_latex_tables.tsv"),
            "figures_tsv": str(OUT / "paper_latex_figures.tsv"),
            "settings_tsv": str(OUT / "paper_latex_experiment_settings.tsv"),
            "markdown": str(OUT / "paper_latex_inventory_audit.md"),
        },
    }
    (OUT / "paper_latex_inventory_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(
        OUT / "paper_latex_equations.tsv",
        equation_rows,
        ["kind", "equation_index", "source", "topic", "equation_excerpt", "context_excerpt"],
    )
    write_tsv(
        OUT / "paper_latex_tables.tsv",
        table_rows,
        ["kind", "source", "label", "caption_excerpt", "graphics", "content_excerpt"],
    )
    write_tsv(
        OUT / "paper_latex_figures.tsv",
        figure_rows,
        ["kind", "source", "label", "caption_excerpt", "graphics", "content_excerpt"],
    )
    write_tsv(
        OUT / "paper_latex_experiment_settings.tsv",
        setting_rows,
        ["kind", "setting_id", "source", "expected_values", "matched_excerpt"],
    )
    write_markdown(OUT / "paper_latex_inventory_audit.md", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "equations": len(equation_rows),
                "settings": len(setting_rows),
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
