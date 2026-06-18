#!/usr/bin/env python3
"""Audit paper/source vs official-code adaptive-sampling look-back discrepancy."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/adaptive_sampling_discrepancy_audit"
COMMANDS = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    / "whole_body_tracking/tasks/tracking/mdp/commands.py"
)
RAW_COMMANDS = (
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/"
    / "whole_body_tracking/tasks/tracking/mdp/commands.py"
)
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
GOAL = ROOT / "goal.md"
SCAN_ROOTS = [
    ROOT / "reproduction/third_party/official/whole_body_tracking",
    ROOT / "download/official/whole_body_tracking",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_adaptive_values(path: Path) -> dict[str, float | int]:
    text = read_text(path)
    values: dict[str, float | int] = {}
    for key in ["adaptive_kernel_size", "adaptive_lambda", "adaptive_uniform_ratio", "adaptive_alpha"]:
        match = re.search(rf"{key}:\s*[^=]+=\s*([-+0-9.eE]+)", text)
        if not match:
            continue
        raw = match.group(1)
        values[key] = int(raw) if re.fullmatch(r"[-+]?\d+", raw) else float(raw)
    return values


def line_for(path: Path, pattern: str) -> int | None:
    regex = re.compile(pattern)
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if regex.search(line):
            return idx
    return None


def source_checks() -> dict[str, bool]:
    paper = read_text(ROOT_TEX)
    goal = read_text(GOAL)
    commands = read_text(COMMANDS)
    return {
        "paper_source_has_ema_0_999_0_001": "0.999" in paper and "0.001" in paper,
        "paper_source_has_uniform_floor_0_1": "0.1/S" in paper,
        "paper_source_has_rho_0_8": "rho=0.8" in paper or "rho = 0.8" in paper,
        "paper_source_has_u_0_1_2": r"u \in \{0,1,2\}" in paper,
        "goal_records_u_0_1_2": "{0,1,2}" in goal and "rho = 0.8" in goal,
        "official_code_noncausal_kernel": "Non-causal kernel" in commands,
        "official_code_uses_exponential_kernel": "adaptive_lambda**i" in commands,
        "official_code_uses_right_padding_for_lookahead": "(0, self.cfg.adaptive_kernel_size - 1)" in commands,
        "official_code_updates_ema": "adaptive_alpha * self._current_bin_failed" in commands,
    }


def iter_text_files(root: Path) -> list[Path]:
    suffixes = {".py", ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".sh"}
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)


def scan_kernel_size_occurrences() -> dict[str, Any]:
    occurrences: list[dict[str, Any]] = []
    numeric_assignments: list[dict[str, Any]] = []
    override_candidates: list[dict[str, Any]] = []
    assignment_re = re.compile(
        r"(?<![\w.])adaptive_kernel_size(?:\s*:\s*[A-Za-z_][\w\[\], ]*)?\s*[:=]\s*([-+]?\d+)"
    )

    for root in SCAN_ROOTS:
        for path in iter_text_files(root):
            try:
                lines = read_text(path).splitlines()
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(lines, start=1):
                if "adaptive_kernel_size" not in line:
                    continue
                rel = str(path.relative_to(ROOT))
                occurrence = {"path": str(path), "relative_path": rel, "line": lineno, "text": line.strip()}
                occurrences.append(occurrence)
                match = assignment_re.search(line)
                if not match:
                    continue
                assignment = dict(occurrence)
                assignment["value"] = int(match.group(1))
                numeric_assignments.append(assignment)
                is_default_decl = path.name == "commands.py" and "adaptive_kernel_size: int = 1" in line
                if not is_default_decl:
                    override_candidates.append(assignment)

    return {
        "occurrences": occurrences,
        "numeric_assignments": numeric_assignments,
        "override_candidates": override_candidates,
        "occurrence_count": len(occurrences),
        "numeric_assignment_count": len(numeric_assignments),
        "override_candidate_count": len(override_candidates),
    }


def convolved_probabilities(kernel_size: int, failure: list[float], uniform_ratio: float, rho: float) -> list[float]:
    kernel = [rho**i for i in range(kernel_size)]
    kernel_sum = math.fsum(kernel)
    kernel = [value / kernel_sum for value in kernel]
    floored = [value + uniform_ratio / len(failure) for value in failure]
    padded = floored + [floored[-1]] * (kernel_size - 1)
    conv: list[float] = []
    for start in range(len(failure)):
        conv.append(math.fsum(padded[start + offset] * kernel[offset] for offset in range(kernel_size)))
    total = math.fsum(conv)
    return [value / total for value in conv]


def distribution_probe() -> dict[str, Any]:
    failure = [0.0] * 8
    failure[3] = 1.0
    uniform_ratio = 0.1
    rho = 0.8
    code_probs = convolved_probabilities(1, failure, uniform_ratio, rho)
    paper_probs = convolved_probabilities(3, failure, uniform_ratio, rho)
    diffs = [paper - code for paper, code in zip(paper_probs, code_probs)]
    return {
        "failure_vector": failure,
        "code_default_kernel_size": 1,
        "paper_kernel_size_from_u_0_1_2": 3,
        "rho": rho,
        "uniform_ratio": uniform_ratio,
        "code_default_probabilities": code_probs,
        "paper_three_bin_probabilities": paper_probs,
        "probability_differences_paper_minus_code": diffs,
        "l1_difference": math.fsum(abs(value) for value in diffs),
        "code_argmax": max(range(len(code_probs)), key=lambda idx: code_probs[idx]),
        "paper_argmax": max(range(len(paper_probs)), key=lambda idx: paper_probs[idx]),
        "code_pre_failure_mass_bins_1_2": math.fsum(code_probs[1:3]),
        "paper_pre_failure_mass_bins_1_2": math.fsum(paper_probs[1:3]),
        "prob_sums": {"code": math.fsum(code_probs), "paper": math.fsum(paper_probs)},
        "paper_spreads_mass_to_preceding_bins": paper_probs[1] > code_probs[1] and paper_probs[2] > code_probs[2],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    code_values = parse_adaptive_values(COMMANDS)
    raw_values = parse_adaptive_values(RAW_COMMANDS)
    scans = scan_kernel_size_occurrences()
    probe = distribution_probe()
    checks = source_checks()
    checks.update(
        {
            "official_code_default_kernel_size_1": code_values.get("adaptive_kernel_size") == 1,
            "official_code_alpha_matches_paper": code_values.get("adaptive_alpha") == 0.001,
            "official_code_lambda_matches_paper": code_values.get("adaptive_lambda") == 0.8,
            "official_code_uniform_ratio_matches_paper": code_values.get("adaptive_uniform_ratio") == 0.1,
            "raw_download_matches_reproduction_default": raw_values.get("adaptive_kernel_size")
            == code_values.get("adaptive_kernel_size")
            == 1,
            "no_runtime_override_to_kernel_size_3_found": scans["override_candidate_count"] == 0,
            "distribution_difference_detected": probe["l1_difference"] > 1.0,
            "paper_pre_failure_mass_exceeds_code": probe["paper_pre_failure_mass_bins_1_2"]
            > probe["code_pre_failure_mass_bins_1_2"],
            "paper_three_bin_kernel_preserves_single_failure_argmax": probe["paper_argmax"] == probe["code_argmax"],
            "discrepancy_boundary_recorded": True,
        }
    )

    audit = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "static_and_formula_audit",
        "scope": "adaptive sampling paper/source look-back versus official-code default",
        "sources": {
            "paper_source": str(ROOT_TEX),
            "goal": str(GOAL),
            "official_commands": str(COMMANDS),
            "raw_download_commands": str(RAW_COMMANDS),
            "official_commands_sha256": sha256_file(COMMANDS),
            "raw_download_commands_sha256": sha256_file(RAW_COMMANDS),
            "line_refs": {
                "paper_u_0_1_2": line_for(ROOT_TEX, r"u \\in \\{0,1,2\\}"),
                "official_default_kernel_size": line_for(COMMANDS, r"adaptive_kernel_size:\s*int\s*=\s*1"),
                "official_noncausal_padding": line_for(COMMANDS, r"Non-causal kernel"),
                "official_ema_update": line_for(COMMANDS, r"adaptive_alpha \* self\._current_bin_failed"),
            },
        },
        "official_code_values": code_values,
        "raw_download_values": raw_values,
        "kernel_size_scan": scans,
        "probe": probe,
        "metrics": {
            "code_kernel_size": code_values.get("adaptive_kernel_size"),
            "paper_kernel_size": probe["paper_kernel_size_from_u_0_1_2"],
            "l1_difference": probe["l1_difference"],
            "code_argmax": probe["code_argmax"],
            "paper_argmax": probe["paper_argmax"],
            "code_pre_failure_mass_bins_1_2": probe["code_pre_failure_mass_bins_1_2"],
            "paper_pre_failure_mass_bins_1_2": probe["paper_pre_failure_mass_bins_1_2"],
            "override_candidate_count": scans["override_candidate_count"],
        },
        "checks": checks,
        "interpretation": {
            "status": "partial_unresolved_discrepancy",
            "summary": (
                "Paper/source and goal describe a three-bin non-causal look-back u in {0,1,2}; "
                "the inspected official code and raw download default to adaptive_kernel_size=1, "
                "and no numeric runtime override was found in the local official tree."
            ),
            "not_a_replacement_for": [
                "live adaptive-sampling training",
                "paper authors' intended runtime config",
                "IsaacLab rollout evidence",
                "resolving the discrepancy",
            ],
        },
    }

    json_path = OUT / "adaptive_sampling_discrepancy_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    with (OUT / "adaptive_sampling_discrepancy_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["bin", "failure", "code_kernel_size_1", "paper_kernel_size_3", "paper_minus_code"])
        for idx, failure_value in enumerate(probe["failure_vector"]):
            writer.writerow(
                [
                    idx,
                    failure_value,
                    probe["code_default_probabilities"][idx],
                    probe["paper_three_bin_probabilities"][idx],
                    probe["probability_differences_paper_minus_code"][idx],
                ]
            )
    (OUT / "run.log").write_text(
        "kind=adaptive_sampling_discrepancy_audit\n"
        f"status={audit['status']}\n"
        f"l1_difference={probe['l1_difference']}\n",
        encoding="utf-8",
    )
    print(json_path)


if __name__ == "__main__":
    main()
