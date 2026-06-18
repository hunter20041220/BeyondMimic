#!/usr/bin/env python3
"""Audit consistency between the downloaded paper PDF, source tar, and extracted LaTeX audits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PDF = ROOT / "download/papers/BeyondMimic_2508.08241.pdf"
SOURCE_TAR = ROOT / "download/papers/BeyondMimic_2508.08241_source.tar"
EXTRACTED_SOURCE = ROOT / "reproduction/paper/source"
LATEX_INVENTORY_JSON = ROOT / "res/paper_latex_inventory/paper_latex_inventory_audit.json"
SOURCE_COVERAGE_JSON = ROOT / "res/paper_source_coverage/paper_source_coverage_audit.json"
TABLE_VALUES_JSON = ROOT / "res/paper_table_values/paper_table_value_audit.json"
OUT = ROOT / "res/paper_pdf_source_consistency"

PDF_ANCHORS = [
    ("title", r"BeyondMimic"),
    ("motion_tracking_frequency", r"Policies run at\s+50\s*Hz"),
    ("state_estimator_frequency", r"500\s*Hz"),
    ("onnx_latency", r"under\s+1\.0\s*ms"),
    ("diffusion_frequency", r"25\s*Hz"),
    ("diffusion_parameter_count", r"19\.8M"),
    ("diffusion_latency", r"20\s*ms"),
    ("denoising_steps", r"20\s+denoising steps"),
    ("rtx_gpu", r"RTX\s+4060"),
    ("tensorrt", r"TensorRT"),
    ("cppad", r"CppAD"),
    ("ou_theta", r"(?:theta|𝜃)\s*=\s*0\.8"),
    ("ou_sigma", r"(?:sigma|𝜎)\s*=\s*0\.1"),
    ("rollout_2_5_seconds", r"2\.5\s+seconds"),
    ("stability_5_seconds", r"5\s+seconds"),
    ("sample_coverage_100", r"100\s+times"),
    ("keyframe_interval", r"0\.2\s*s"),
    ("velocity_error_walk", r"12\.14%"),
    ("velocity_error_run", r"13\.65%"),
    ("long_track_distance", r"over\s+50\s*m"),
]

EXPECTED_TAR_MEMBERS = [
    "00README.json",
    "bibliography.bib",
    "figures/Fig1.pdf",
    "figures/Fig2.pdf",
    "figures/Fig3.pdf",
    "figures/Fig4.pdf",
    "figures/Fig5.pdf",
    "figures/Fig6.pdf",
    "figures/Fig7.pdf",
    "figures/Fig8.pdf",
    "figures/sup_fig1_tracking_pipeline.pdf",
    "figures/sup_fig2_w.png",
    "root.tex",
    "scicite.sty",
    "sciencemag.bst",
    "tex/discussion.tex",
    "tex/intro.tex",
    "tex/method.tex",
    "tex/results.tex",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pdf_pages() -> tuple[list[str], list[str]]:
    reader = PdfReader(str(PDF))
    pages = []
    extraction_errors = []
    for idx, page in enumerate(reader.pages):
        try:
            pages.append(normalize_text(page.extract_text() or ""))
        except Exception as exc:  # pragma: no cover - defensive against PDF extraction edge cases
            pages.append("")
            extraction_errors.append(f"page_{idx + 1}:{type(exc).__name__}:{exc}")
    return pages, extraction_errors


def first_match_excerpt(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.I)
    if not match:
        return ""
    start = max(match.start() - 60, 0)
    end = min(match.end() + 60, len(text))
    return text[start:end]


def anchor_rows(page_text: list[str]) -> list[dict[str, Any]]:
    rows = []
    for anchor_id, pattern in PDF_ANCHORS:
        matched_pages = []
        excerpt = ""
        for idx, text in enumerate(page_text, start=1):
            if re.search(pattern, text, flags=re.I):
                matched_pages.append(idx)
                if not excerpt:
                    excerpt = first_match_excerpt(text, pattern)
        rows.append(
            {
                "anchor_id": anchor_id,
                "pattern": pattern,
                "present": bool(matched_pages),
                "page_numbers": matched_pages,
                "first_excerpt": excerpt,
            }
        )
    return rows


def tar_rows() -> tuple[list[dict[str, Any]], list[str]]:
    with tarfile.open(SOURCE_TAR, "r") as tar:
        members = sorted(m.name.rstrip("/") for m in tar.getmembers() if m.isfile())
    rows = []
    for member in EXPECTED_TAR_MEMBERS:
        extracted = EXTRACTED_SOURCE / member
        rows.append(
            {
                "member": member,
                "in_tar": member in members,
                "extracted_exists": extracted.exists(),
                "extracted_sha256": sha256_file(extracted) if extracted.is_file() else "",
                "extracted_size": extracted.stat().st_size if extracted.exists() else 0,
            }
        )
    extra_members = [member for member in members if member not in EXPECTED_TAR_MEMBERS]
    return rows, extra_members


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                out[key] = json.dumps(value, sort_keys=True) if isinstance(value, list) else value
            writer.writerow(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages, extraction_errors = pdf_pages()
    anchors = anchor_rows(pages)
    tar_member_rows, extra_tar_members = tar_rows()
    latex_inventory = load_json(LATEX_INVENTORY_JSON)
    source_coverage = load_json(SOURCE_COVERAGE_JSON)
    table_values = load_json(TABLE_VALUES_JSON)
    missing_anchors = [row for row in anchors if not row["present"]]
    missing_tar_rows = [row for row in tar_member_rows if not row["in_tar"] or not row["extracted_exists"]]
    checks = {
        "pdf_exists": PDF.is_file(),
        "source_tar_exists": SOURCE_TAR.is_file(),
        "pdf_sha256_recorded": bool(sha256_file(PDF)),
        "source_tar_sha256_recorded": bool(sha256_file(SOURCE_TAR)),
        "pdf_has_pages": len(pages) >= 1,
        "all_pdf_pages_text_extracted": not extraction_errors and all(bool(text) for text in pages),
        "all_expected_pdf_anchors_present": not missing_anchors,
        "all_expected_source_tar_members_present": not missing_tar_rows,
        "no_unexpected_tar_file_members": not extra_tar_members,
        "latex_inventory_ok": latex_inventory["status"] == "ok",
        "source_coverage_ok": source_coverage["status"] == "ok",
        "table_values_ok": table_values["status"] == "ok",
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "paper_pdf_source_consistency_audit",
        "scope": "downloaded PDF/source-tar integrity and PDF text anchors cross-checked with LaTeX inventory/table audits",
        "metrics": {
            "pdf_page_count": len(pages),
            "pdf_text_page_count": sum(1 for text in pages if text),
            "pdf_anchor_count": len(anchors),
            "pdf_anchor_present_count": sum(1 for row in anchors if row["present"]),
            "source_tar_expected_member_count": len(EXPECTED_TAR_MEMBERS),
            "source_tar_present_member_count": sum(1 for row in tar_member_rows if row["in_tar"]),
            "source_tar_extracted_member_count": sum(1 for row in tar_member_rows if row["extracted_exists"]),
            "unexpected_tar_file_member_count": len(extra_tar_members),
        },
        "source_hashes": {
            "pdf_sha256": sha256_file(PDF),
            "source_tar_sha256": sha256_file(SOURCE_TAR),
        },
        "missing_pdf_anchors": missing_anchors,
        "pdf_anchor_rows": anchors,
        "source_tar_rows": tar_member_rows,
        "extra_tar_file_members": extra_tar_members,
        "extraction_errors": extraction_errors,
        "cross_checks": {
            "latex_inventory_counts": latex_inventory["counts"],
            "paper_source_coverage_counts": source_coverage["counts"],
            "paper_table_value_counts": table_values["counts"],
        },
        "checks": checks,
        "interpretation": {
            "pdf_and_source_consistent_with_current_audits": all(
                checks[key]
                for key in [
                    "all_expected_pdf_anchors_present",
                    "all_expected_source_tar_members_present",
                    "latex_inventory_ok",
                    "source_coverage_ok",
                    "table_values_ok",
                ]
            ),
            "goal_complete": False,
            "why_not_complete": (
                "This verifies that the downloaded paper PDF and source tar contain the paper anchors and source files "
                "used by the current audits. It does not produce trained checkpoints, rollout logs, TensorRT engines, "
                "videos, Fig.5/Fig.6 local results, or real-robot evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "paper_pdf_source_consistency_audit.json"),
            "anchors_tsv": str(OUT / "paper_pdf_anchor_audit.tsv"),
            "source_tar_tsv": str(OUT / "paper_source_tar_audit.tsv"),
        },
    }
    (OUT / "paper_pdf_source_consistency_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(
        OUT / "paper_pdf_anchor_audit.tsv",
        anchors,
        ["anchor_id", "pattern", "present", "page_numbers", "first_excerpt"],
    )
    write_tsv(
        OUT / "paper_source_tar_audit.tsv",
        tar_member_rows,
        ["member", "in_tar", "extracted_exists", "extracted_sha256", "extracted_size"],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "pdf_pages": len(pages),
                "anchors": summary["metrics"]["pdf_anchor_present_count"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
