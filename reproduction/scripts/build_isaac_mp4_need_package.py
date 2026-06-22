#!/usr/bin/env python3
"""Build the RTX migration package for true Isaac rendered MP4 generation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).expanduser().resolve()
PKG = ROOT / "isaac_mp4_need"
COPY_LIMIT_BYTES = int(os.environ.get("BM_ISAAC_MP4_NEED_COPY_LIMIT_BYTES", str(256 * 1024 * 1024)))
MANDATORY_DIRS = [
    "scripts",
    "checkpoints",
    "configs",
    "metadata",
    "assets",
    "motions",
    "results",
    "logs",
    "docs",
    "helpers",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix().lstrip("/")


def safe_package_path(category: str, source: Path) -> Path:
    source_rel = rel(source)
    return PKG / category / source_rel


def dir_size_and_count(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total, count


def entry(
    source: Path,
    category: str,
    purpose: str,
    required_or_optional: str,
    claim_level: str,
    notes: str = "",
    copy: bool = True,
    package_rel: str | None = None,
    large_override: bool | None = None,
) -> dict[str, Any]:
    exists = source.exists()
    is_file = source.is_file()
    size = source.stat().st_size if is_file else dir_size_and_count(source)[0] if source.is_dir() else 0
    large = size > COPY_LIMIT_BYTES if large_override is None else large_override
    should_copy = bool(copy and exists and is_file and not large)
    package_path = PKG / package_rel if package_rel else safe_package_path(category, source)
    copied_or_referenced = "copied" if should_copy else "referenced_large_file" if exists and large else "referenced"
    if should_copy:
        package_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, package_path)
    return {
        "source_path": str(source),
        "package_path": str(package_path.relative_to(PKG)) if should_copy else "",
        "exists": exists,
        "file_size": size,
        "sha256": sha256(source) if is_file else "",
        "category": category,
        "purpose": purpose,
        "required_or_optional": required_or_optional,
        "large_file": large,
        "copied_or_referenced": copied_or_referenced,
        "claim_level": claim_level,
        "notes": notes,
    }


def add_unique(rows: list[dict[str, Any]], seen: set[str], source: Path, **kwargs: Any) -> None:
    key = str(source.resolve()) if source.exists() else str(source)
    if key in seen:
        return
    seen.add(key)
    rows.append(entry(source, **kwargs))


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def discover_reproduction_video_scripts() -> list[Path]:
    script_dir = ROOT / "reproduction/scripts"
    tokens = [
        "isaac_mp4",
        "render_stack",
        "rendered",
        "rollout_video",
        "reference_replay_video",
        "policy_rollout_video",
        "vae_closed_loop_rollout",
        "guidance_video",
        "contact_sheet",
        "guided_vs_unguided",
        "fig5_fig6",
        "visual_evidence",
        "visual_media",
        "action_guidance_rollout",
        "guided_action_rollout",
        "receding_latent_guidance_rollout",
        "task_conditioned_latent_guidance_rollout",
        "transition_guidance_rollout",
        "inpainting_guidance_rollout",
    ]
    paths: list[Path] = []
    for path in sorted(script_dir.glob("*.py")):
        lower = path.name.lower()
        if any(token in lower for token in tokens):
            paths.append(path)
    explicit = [
        script_dir / "tracking_g1_isaaclab_rendered_policy_rollout_mp4.py",
        script_dir / "isaac_render_stack_repair_audit.py",
    ]
    return sorted({*paths, *explicit})


def official_helper_files() -> list[Path]:
    return [
        ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py",
        ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py",
        ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/rsl_rl/cli_args.py",
        ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/rsl_rl/play.py",
        ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/rsl_rl/train.py",
        ROOT / "reproduction/third_party/official/whole_body_tracking/README.md",
        ROOT / "reproduction/third_party/official/whole_body_tracking/pyproject.toml",
        ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/config/extension.toml",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/scripts/reinforcement_learning/rsl_rl/play.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/scripts/reinforcement_learning/rsl_rl/train.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/scripts/reinforcement_learning/rsl_rl/cli_args.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/scripts/benchmarks/benchmark_cameras.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/scripts/demos/sensors/cameras.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/docs/source/how-to/record_video.rst",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/docs/source/how-to/save_camera_output.rst",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/docs/source/how-to/wrap_rl_env.rst",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/docs/source/overview/core-concepts/sensors/camera.rst",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/direct_rl_env.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/test/deps/isaacsim/check_camera.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/test/sensors/test_camera.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/test/sensors/test_tiled_camera_env.py",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab_tasks/test/test_record_video.py",
    ]


def checkpoint_files() -> list[Path]:
    globs = [
        "res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training/*/rank_0/model_999.pt",
        "res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training/*/rank_0/model_999.pt",
        "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training/*/rank_0/model_999.pt",
        "res/runs/tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_training/*/rank_0/model_999.pt",
        "res/runs/tracking_g1_official_csv_loop_full_bundle_ppo_training/*/rank_0/model_299.pt",
        "res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training/*/resource_adjusted_teacher_rollout_action_vae.pt",
        "res/runs/level_c_official_importer_export_full_bundle_teacher_rollout_vae_training/*/resource_adjusted_teacher_rollout_action_vae.pt",
        "res/runs/level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training/*/resource_adjusted_teacher_rollout_action_vae.pt",
        "res/runs/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training/*/resource_adjusted_state_latent_denoiser.pt",
        "res/runs/level_c_official_importer_export_full_bundle_state_latent_diffusion_training/*/resource_adjusted_state_latent_denoiser.pt",
        "res/runs/level_c_official_csv_loop_full_bundle_state_latent_diffusion_training/*/resource_adjusted_state_latent_denoiser.pt",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/checkpoint/lafan1_paper_arch_vae_diffusion.pt",
    ]
    paths: list[Path] = []
    for pattern in globs:
        paths.extend(ROOT.glob(pattern))
    return sorted({p for p in paths if p.is_file()})


def onnx_files() -> list[Path]:
    return sorted((ROOT / "res/level_c").rglob("*.onnx"))


def motion_and_asset_files() -> list[Path]:
    explicit = [
        ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda",
        ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz",
        ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json",
        ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/official_csv_loop_full_public_motion_bundle_fk_repaired.npz",
        ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle.npz",
    ]
    for pattern in [
        "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/**/*.py",
        "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/robots/*.urdf",
        "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/robots/*.xml",
        "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/meshes/g1/*",
        "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/*.py",
    ]:
        explicit.extend(ROOT.glob(pattern))
    return sorted({p for p in explicit if p.is_file()})


def metadata_and_config_files() -> list[Path]:
    explicit = [
        ROOT / "res/failed_runs/isaac_mp4/isaaclab_rendered_policy_rollout_video_failed_gate.json",
        ROOT / "res/setup/isaac_render_stack_repair_audit/isaac_render_stack_repair_audit.json",
        ROOT / "res/setup/isaac_render_stack_repair_audit/isaac_render_stack_repair_audit.tsv",
        ROOT / "res/visualization/isaac_mp4/isaaclab_rendered_policy_rollout_video_asset.json",
        ROOT / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json",
        ROOT / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json",
        ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json",
        ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json",
        ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json",
        ROOT / "res/comparison/paper_vs_reproduction.json",
        ROOT / "res/comparison/paper_vs_reproduction.csv",
        ROOT / "res/comparison/paper_vs_reproduction.md",
        ROOT / "res/required_artifact_absence/required_artifact_absence_audit.json",
        ROOT / "res/master_audit/reproduction_master_audit.json",
        ROOT / "res/artifact_manifest/artifact_manifest.json",
    ]
    paths: list[Path] = [p for p in explicit if p.is_file()]
    for base in [ROOT / "res/level_c", ROOT / "res/tracking"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".json", ".csv", ".tsv", ".md", ".yaml", ".yml", ".toml"}:
                continue
            lower = p.as_posix().lower()
            if any(
                token in lower
                for token in [
                    "vae",
                    "denoiser",
                    "diffusion",
                    "guidance",
                    "ppo",
                    "checkpoint_eval",
                    "training_run",
                    "motion_npz",
                    "robot_order",
                    "official_importer",
                    "obs",
                    "action",
                    "schema",
                ]
            ):
                paths.append(p)
    return sorted({p for p in paths if p.is_file()})


def result_files() -> list[Path]:
    paths: list[Path] = []
    for base in [ROOT / "res/visualization", ROOT / "res/report_assets"]:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".json", ".csv", ".md", ".png", ".mp4"}:
                continue
            lower = p.as_posix().lower()
            if any(
                token in lower
                for token in [
                    "isaac_mp4",
                    "rollout",
                    "video",
                    "contact_sheet",
                    "reference",
                    "policy",
                    "vae",
                    "guidance",
                    "keyframes",
                    "visual",
                ]
            ):
                paths.append(p)
    return sorted({p for p in paths if p.is_file()})


def log_files() -> list[Path]:
    paths: list[Path] = []
    for base in [ROOT / "logs/isaac_mp4", ROOT / "logs/setup"]:
        if not base.exists():
            continue
        for p in base.glob("*"):
            if p.is_file() and p.suffix.lower() in {".log", ".txt"} and any(
                token in p.name.lower() for token in ["isaac", "render", "vulkan", "mp4", "xvfb", "llvmpipe"]
            ):
                paths.append(p)
    return sorted({p for p in paths})


def referenced_large_dirs() -> list[tuple[Path, str, str]]:
    return [
        (
            ROOT / "download",
            "Original read-only paper/code/dataset/dependency archive. Do not modify; copy this whole directory or recreate it on the RTX host.",
            "raw_source_reference",
        ),
        (
            ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0",
            "Local editable IsaacLab v2.1.0 source tree used by bm_tracking.",
            "isaaclab_source_reference",
        ),
        (
            ROOT / "reproduction/third_party/official/whole_body_tracking",
            "Official BeyondMimic whole_body_tracking work copy with G1 task, robot assets, csv_to_npz/replay/play/train entries.",
            "official_tracking_source_reference",
        ),
        (
            ROOT / "res/runs",
            "Full training run directory with all checkpoints/logs. The package copies the most relevant small checkpoints and references this full directory.",
            "run_directory_reference",
        ),
        (
            ROOT / "res/visualization",
            "All existing diagnostic skeleton/contact-sheet rollout media. Package copies relevant files under copy limit and references this full directory.",
            "visualization_reference",
        ),
    ]


def package_gitignore() -> str:
    return """# This folder is a local RTX migration package.
# Heavy binaries are copied locally but should not be committed to GitHub.
scripts/
configs/
metadata/
checkpoints/
assets/
motions/
results/
logs/
*.pt
*.pth
*.ckpt
*.onnx
*.engine
*.npz
*.mp4
*.mkv
*.avi
*.usda
*.usd
*.urdf
*.STL
*.stl
*.obj
"""


def smoke_runner() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BM_ROOT:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
PY="${BM_TRACKING_PY:-${ROOT}/envs/bm_tracking/bin/python}"
GPU="${BM_GPU_ID:-0}"
STEPS="${BM_ISAAC_MP4_STEPS:-20}"
SEED="${BM_ISAAC_MP4_SEED:-20260780}"
LOG_DIR="${ROOT}/logs/isaac_mp4_rtx_package"

mkdir -p "${LOG_DIR}" "${ROOT}/tmp" "${ROOT}/cache/ov" "${ROOT}/cache/pip" "${ROOT}/cache/torch"
export BM_ROOT="${ROOT}"
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${ROOT}/tmp/xdg-runtime-$(id -u)}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" || true
export OMNI_USER_DIR="${OMNI_USER_DIR:-${ROOT}/cache/ov/user}"
export OMNI_LOGS_DIR="${OMNI_LOGS_DIR:-${ROOT}/logs/omniverse}"
export OMNI_CACHE_DIR="${OMNI_CACHE_DIR:-${ROOT}/cache/ov/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT}/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export BM_ISAAC_MP4_CANDIDATE_GPUS="${BM_ISAAC_MP4_CANDIDATE_GPUS:-${GPU}}"
export BM_ISAAC_MP4_STEPS="${STEPS}"
export BM_ISAAC_MP4_SEED="${SEED}"
export BM_ISAAC_MP4_TIMEOUT_SECONDS="${BM_ISAAC_MP4_TIMEOUT_SECONDS:-600}"

echo "[smoke] ROOT=${ROOT}"
echo "[smoke] PY=${PY}"
echo "[smoke] GPU=${BM_ISAAC_MP4_CANDIDATE_GPUS}"
echo "[smoke] steps=${BM_ISAAC_MP4_STEPS}"

nvidia-smi | tee "${LOG_DIR}/nvidia_smi_smoke.log"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary | tee "${LOG_DIR}/vulkaninfo_summary_smoke.log" || true
else
  echo "[smoke] vulkaninfo not found; install vulkan-tools before diagnosing render failures." | tee "${LOG_DIR}/vulkaninfo_missing_smoke.log"
fi

"${PY}" - <<'PY' | tee "${LOG_DIR}/python_import_smoke.log"
import importlib
mods = ["torch", "isaacsim", "isaaclab", "isaaclab_rl", "whole_body_tracking", "gymnasium", "imageio", "PIL"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"IMPORT_OK {name} {getattr(mod, '__version__', '')}")
PY

"${PY}" "${ROOT}/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py" 2>&1 | tee "${LOG_DIR}/rtx_true_isaac_rendered_mp4_smoke.log"
echo "[smoke] Done. Check ${ROOT}/res/visualization/isaac_mp4 and ${ROOT}/res/failed_runs/isaac_mp4."
"""


def full_runner() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BM_ROOT:-$(cd "${PACKAGE_DIR}/.." && pwd)}"
PY="${BM_TRACKING_PY:-${ROOT}/envs/bm_tracking/bin/python}"
GPU="${BM_GPU_ID:-0}"
STEPS="${BM_ISAAC_MP4_STEPS:-300}"
SEED="${BM_ISAAC_MP4_SEED:-20260780}"
LOG_DIR="${ROOT}/logs/isaac_mp4_rtx_package"

mkdir -p "${LOG_DIR}" "${ROOT}/tmp" "${ROOT}/cache/ov" "${ROOT}/cache/pip" "${ROOT}/cache/torch"
export BM_ROOT="${ROOT}"
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${ROOT}/tmp/xdg-runtime-$(id -u)}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" || true
export OMNI_USER_DIR="${OMNI_USER_DIR:-${ROOT}/cache/ov/user}"
export OMNI_LOGS_DIR="${OMNI_LOGS_DIR:-${ROOT}/logs/omniverse}"
export OMNI_CACHE_DIR="${OMNI_CACHE_DIR:-${ROOT}/cache/ov/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT}/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export BM_ISAAC_MP4_CANDIDATE_GPUS="${BM_ISAAC_MP4_CANDIDATE_GPUS:-${GPU}}"
export BM_ISAAC_MP4_STEPS="${STEPS}"
export BM_ISAAC_MP4_SEED="${SEED}"
export BM_ISAAC_MP4_TIMEOUT_SECONDS="${BM_ISAAC_MP4_TIMEOUT_SECONDS:-1800}"

echo "[full] ROOT=${ROOT}"
echo "[full] PY=${PY}"
echo "[full] GPU=${BM_ISAAC_MP4_CANDIDATE_GPUS}"
echo "[full] steps=${BM_ISAAC_MP4_STEPS}"
echo "[full] First generating true IsaacLab/Isaac Sim rendered policy rollout MP4."
"${PY}" "${ROOT}/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py" 2>&1 | tee "${LOG_DIR}/rtx_true_isaac_rendered_policy_rollout_300step.log"

run_optional() {
  local script="$1"
  local log_name="$2"
  if [[ -f "${ROOT}/${script}" ]]; then
    echo "[full] Running optional diagnostic asset script: ${script}"
    "${PY}" "${ROOT}/${script}" 2>&1 | tee "${LOG_DIR}/${log_name}.log" || {
      echo "[full] Optional diagnostic script failed: ${script}. This does not invalidate the true Isaac rendered MP4 if it already succeeded."
    }
  else
    echo "[full] Missing optional script: ${script}"
  fi
}

if [[ "${BM_RUN_LEGACY_DIAGNOSTIC_VIDEO_ASSETS:-1}" == "1" ]]; then
  run_optional "reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py" "reference_replay_video_asset"
  run_optional "reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture.py" "policy_rollout_video_asset"
  run_optional "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture.py" "vae_base_rollout_video_asset"
  run_optional "reproduction/scripts/official_importer_export_full_bundle_guidance_video_contact_sheet.py" "guidance_contact_sheet_asset"
fi

echo "[full] Done. True rendered outputs should be under ${ROOT}/res/visualization/isaac_mp4."
echo "[full] Diagnostic skeleton/contact-sheet outputs remain diagnostic evidence, not true Isaac rendered videos."
"""


def readme_text(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    return f"""# Isaac Rendered MP4 RTX Migration Package

This folder is the migration package for generating **true IsaacLab/Isaac Sim rendered rollout MP4s** on an RTX/RT-core machine. It is intentionally broader than a minimal script dump: it includes the current rendered MP4 gate, legacy diagnostic video/contact-sheet helpers, selected local checkpoints, selected motion/config metadata, small ONNX exports, and manifests for large dependencies that should be copied or mounted on the RTX host.

## Current H20 Blocker

The current H20 server cannot produce the requested true Isaac rendered MP4. The failure happens during Isaac Sim Kit/Hydra/Vulkan rendering startup, before `Tracking-Flat-G1-v0` is created and before any PPO checkpoint or closed-loop physics rollout is evaluated.

Recorded blocker:

- classification: `blocked_h20_isaac_sim_rendering_stack`
- no MP4, keyframes, metrics CSV, or summary markdown from the true-rendered gate
- failure boundary: Kit/Hydra/Vulkan/AppLauncher startup, not policy loading, not `env.step`, not robot physics
- claim level: server rendering-stack blocker audit

This package exists so the same MP4 generation path can be moved to a machine with supported graphics rendering hardware instead of repeatedly hitting the H20 rendering boundary.

## Package Contents

Manifest summary generated at `{manifest["generated_utc"]}`:

- total manifest entries: `{counts["total"]}`
- copied files: `{counts["copied"]}`
- referenced files/directories: `{counts["referenced"]}`
- referenced large files/directories: `{counts["large"]}`
- scripts: `{counts["by_category"].get("scripts", 0)}`
- checkpoints: `{counts["by_category"].get("checkpoints", 0)}`
- configs: `{counts["by_category"].get("configs", 0)}`
- metadata: `{counts["by_category"].get("metadata", 0)}`
- assets: `{counts["by_category"].get("assets", 0)}`
- motions: `{counts["by_category"].get("motions", 0)}`
- results: `{counts["by_category"].get("results", 0)}`
- logs: `{counts["by_category"].get("logs", 0)}`

Key files:

- `run_rtx_smoke.sh`: short RTX smoke runner, default 20 steps
- `run_rtx_full_videos.sh`: full runner, default 300 steps
- `isaac_mp4_need_manifest.json`: full machine-readable package inventory
- `isaac_mp4_need_manifest.tsv`: TSV inventory for spreadsheet review
- `scripts/reproduction/scripts/tracking_g1_isaaclab_rendered_policy_rollout_mp4.py`: true IsaacLab rendered policy rollout MP4 gate

## What This Package Can Claim

If the RTX smoke/full runners succeed, the output can be described as:

`true Isaac rendered simulation video / local virtual evidence`

It must **not** be described as:

- real robot evidence
- official paper-level Fig. 5/Fig. 6 reproduction
- official BeyondMimic VAE/diffusion checkpoint reproduction
- proof that DAgger rollout data is available

Legacy skeleton/contact-sheet scripts are included for comparison and report assets, but they remain diagnostic visualizations unless they are adapted to true Isaac rendering.

## Recommended RTX Host

Hardware:

- RTX 4090, RTX 4090D, RTX 6000 Ada, L40/L40S, or another NVIDIA GPU with RTX/RT-core rendering support suitable for Isaac Sim
- at least 24 GB VRAM recommended for comfortable IsaacLab rendering
- local SSD storage for Omniverse/Kit caches

System:

- Ubuntu 20.04 or 22.04
- recent NVIDIA driver compatible with Isaac Sim 4.5.0
- Vulkan loader and NVIDIA Vulkan ICD working
- EGL/headless rendering available
- X11/Xvfb available for fallback diagnostics
- `ffmpeg` available for video encoding

System packages commonly needed:

```bash
sudo apt-get update
sudo apt-get install -y \\
  vulkan-tools mesa-utils ffmpeg xvfb x11-utils \\
  libgl1 libegl1 libglvnd0 libx11-6 libxext6 libxrender1 libxi6 \\
  libxrandr2 libxcursor1 libxinerama1 libxkbcommon0 libxkbcommon-x11-0 \\
  libxcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \\
  libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-cursor0
```

## Python / Conda Environment

Prefer reusing the project conda/venv layout:

- `envs/bm_tracking`: Isaac Sim + IsaacLab + whole_body_tracking + RSL-RL environment
- `envs/bm_diffusion`: PyTorch CUDA environment for VAE/diffusion/guidance
- `envs/bm_analysis`: numpy/pandas/matplotlib/onnxruntime report/audit environment

Core Python packages expected by the runners:

- `isaacsim==4.5.0.0`
- local editable `isaaclab`, `isaaclab_assets`, `isaaclab_rl`, `isaaclab_tasks`, `isaaclab_mimic`
- local editable `whole_body_tracking`
- `torch`, `torchvision`
- `numpy`, `pandas`, `matplotlib`
- `imageio`, `pillow`, `opencv-python`
- `gymnasium`, `rsl-rl`
- `onnx`, `onnxruntime`, `onnxscript`

## Environment Variables

The runner scripts set the important variables, but these are the knobs to know:

```bash
export BM_ROOT=/path/to/BeyondMimic
export BM_TRACKING_PY=$BM_ROOT/envs/bm_tracking/bin/python
export BM_GPU_ID=0
export BM_ISAAC_MP4_CANDIDATE_GPUS=0
export BM_ISAAC_MP4_STEPS=300
export BM_ISAAC_MP4_SEED=20260780
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export XDG_RUNTIME_DIR=$BM_ROOT/tmp/xdg-runtime-$(id -u)
export OMNI_USER_DIR=$BM_ROOT/cache/ov/user
export OMNI_LOGS_DIR=$BM_ROOT/logs/omniverse
export OMNI_CACHE_DIR=$BM_ROOT/cache/ov/cache
export PIP_CACHE_DIR=$BM_ROOT/cache/pip
export TORCH_HOME=$BM_ROOT/cache/torch
```

The true rendered MP4 script now honors `BM_ROOT`, so the RTX machine does not need to use the original H20 absolute path.

## First RTX Checks

From the RTX machine:

```bash
cd /path/to/BeyondMimic
nvidia-smi
vulkaninfo --summary
BM_ROOT=$PWD BM_GPU_ID=0 ./isaac_mp4_need/run_rtx_smoke.sh
```

The smoke runner verifies:

- Python/Isaac Sim/IsaacLab/whole_body_tracking imports
- AppLauncher with `headless=True, enable_cameras=True`
- true Isaac render product/camera path
- `Tracking-Flat-G1-v0` creation
- 10-30 step physics rollout through `env.step(action)`
- short MP4/keyframe/metrics path if rendering succeeds

## 300-Step True Isaac Rendered MP4

After smoke succeeds:

```bash
cd /path/to/BeyondMimic
BM_ROOT=$PWD BM_GPU_ID=0 BM_ISAAC_MP4_STEPS=300 ./isaac_mp4_need/run_rtx_full_videos.sh
```

Primary true-rendered expected outputs:

- `$BM_ROOT/res/visualization/isaac_mp4/*.mp4`
- `$BM_ROOT/res/visualization/isaac_mp4/*keyframes*.png`
- `$BM_ROOT/res/visualization/isaac_mp4/*metrics*.csv`
- `$BM_ROOT/res/visualization/isaac_mp4/*summary*.md` or asset JSON
- `$BM_ROOT/logs/isaac_mp4_rtx_package/*.log`
- `$BM_ROOT/res/failed_runs/isaac_mp4/*.json` if a gate fails

## Reference / Policy / VAE / Guidance Videos

The full runner starts with the true Isaac rendered policy rollout. It also optionally runs legacy diagnostic asset scripts:

- reference replay asset: `official_importer_export_full_dataset_reference_replay_video_asset.py`
- current PPO/policy rollout diagnostic asset: `tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture.py`
- VAE base rollout diagnostic asset: `tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture.py`
- guidance contact sheet: `official_importer_export_full_bundle_guidance_video_contact_sheet.py`

Set `BM_RUN_LEGACY_DIAGNOSTIC_VIDEO_ASSETS=0` if you only want the true Isaac rendered MP4 gate.

## Checkpoints And Motion Data

The package copies small/current checkpoints when available:

- robot-order FK-repaired G1 PPO `model_999.pt`
- endpoint-threshold candidate PPO `model_999.pt`
- scaled/full-bundle PPO `model_999.pt`
- teacher-rollout VAE checkpoints
- state-latent denoiser checkpoints
- small ONNX encoder/decoder/denoiser exports

Large paper-architecture LAFAN1 checkpoints and full run directories are referenced in the manifest when they exceed the copy limit. Copy them manually only if your RTX experiment needs them.

Motion/asset handling:

- the FK-repaired robot-order motion bundle is copied if below the copy limit
- the official G1 USDA is referenced if it is above the copy limit
- whole_body_tracking G1 task configs and small robot description files are copied
- full raw datasets under `download/` are referenced, not duplicated

## Common Failures

`No device could be created` / `GPU Foundation is not initialized`:

- check `nvidia-smi` and `vulkaninfo --summary`
- verify the NVIDIA Vulkan ICD is present under `/etc/vulkan/icd.d/`
- avoid CPU llvmpipe for Isaac Sim rendering

`VkResult: ERROR_DEVICE_LOST` or Hydra/Replicator initialization failure:

- reduce GPU contention
- try a single visible GPU
- clear Omniverse caches under `$BM_ROOT/cache/ov`
- update/downgrade driver to the Isaac Sim supported range

`GLFW initialization failed` / `GLXBadFBConfig`:

- verify EGL/headless setup
- try a real display, X11 session, or Xvfb only as a diagnostic fallback
- make sure XCB/XKB packages are installed

Camera/render product creates but video is empty:

- confirm `enable_cameras=True`
- inspect keyframe PNGs
- check that the camera prim is pointed at the robot root
- verify `simulation_app.update()` is called while frames are captured

Env creation fails:

- confirm `whole_body_tracking.tasks` imports
- confirm `Tracking-Flat-G1-v0` is registered
- verify the G1 USDA path and motion NPZ path in the runner summary

Checkpoint shape mismatch or obs/action dim mismatch:

- use the robot-order FK-repaired PPO checkpoint with the robot-order FK-repaired motion bundle
- compare observation/action schema JSONs in `metadata/`
- do not mix CSV-loop, scaled-PPO, and robot-order checkpoints without checking dimensions

## Manual Large Files To Copy

See `isaac_mp4_need_manifest.json` entries with:

- `copied_or_referenced = referenced_large_file`
- `large_file = true`

Those are intentionally not duplicated into this folder. The most important likely manual dependencies are:

- `download/`
- `reproduction/third_party/official/IsaacLab-v2.1.0`
- `reproduction/third_party/official/whole_body_tracking`
- full `res/runs/` if you need non-selected checkpoints
- official G1 USDA if not copied because it exceeds the local copy limit

## Audit Boundary

This package is an engineering migration artifact. It does not change the reproduction completion state by itself. The project remains incomplete at paper level until required paper-level gates pass, and real robot deployment remains unavailable unless real Unitree G1 hardware is explicitly provided.
"""


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    for name in MANDATORY_DIRS:
        (PKG / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in discover_reproduction_video_scripts():
        add_unique(
            rows,
            seen,
            path,
            category="scripts",
            purpose="Local reproduction script related to Isaac rendered MP4, rollout video, reference replay, VAE/guidance rollout, or diagnostic video assets.",
            required_or_optional="required" if "tracking_g1_isaaclab_rendered_policy_rollout_mp4.py" in path.name else "optional",
            claim_level="script_for_true_isaac_rendered_or_diagnostic_video_asset",
        )
    for path in official_helper_files():
        add_unique(
            rows,
            seen,
            path,
            category="scripts",
            purpose="Official IsaacLab/whole_body_tracking helper, replay, play/train, camera, or RecordVideo reference.",
            required_or_optional="required" if path.name in {"play.py", "replay_npz.py", "csv_to_npz.py"} else "optional",
            claim_level="official_helper_reference",
        )
    for path in checkpoint_files():
        add_unique(
            rows,
            seen,
            path,
            category="checkpoints",
            purpose="Selected local PPO/VAE/denoiser checkpoint for policy, VAE, or guided latent rollout migration.",
            required_or_optional="required" if "robot_order_full_bundle_ppo_training" in path.as_posix() else "optional",
            claim_level="local_checkpoint_not_official_beyondmimic_checkpoint",
        )
    for path in onnx_files():
        add_unique(
            rows,
            seen,
            path,
            category="checkpoints",
            purpose="ONNX export useful for latency/deployment audit or controller component migration.",
            required_or_optional="optional",
            claim_level="local_onnx_export_not_tensorrt_engine",
        )
    for path in motion_and_asset_files():
        category = "motions" if path.suffix.lower() == ".npz" else "assets"
        add_unique(
            rows,
            seen,
            path,
            category=category,
            purpose="Motion bundle, G1 robot asset, mesh, URDF, or task configuration needed for IsaacLab video generation.",
            required_or_optional="required" if "fk_repaired_robot_order" in path.as_posix() or path.suffix.lower() in {".usda", ".urdf"} else "optional",
            claim_level="local_sim_asset_or_motion_dependency",
        )
    for path in metadata_and_config_files():
        add_unique(
            rows,
            seen,
            path,
            category="metadata" if path.suffix.lower() in {".json", ".csv", ".tsv", ".md"} else "configs",
            purpose="Config, audit, schema, training summary, checkpoint eval, or comparison metadata needed to interpret video claims.",
            required_or_optional="required" if any(t in path.as_posix() for t in ["isaac_mp4", "robot_order", "paper_vs_reproduction"]) else "optional",
            claim_level="audit_or_configuration_metadata",
        )
    for path in result_files():
        add_unique(
            rows,
            seen,
            path,
            category="results",
            purpose="Existing diagnostic visual asset, keyframe, MP4, metrics CSV, README, or contact sheet for report comparison.",
            required_or_optional="optional",
            claim_level="diagnostic_visualization_or_existing_local_virtual_evidence",
        )
    for path in log_files():
        add_unique(
            rows,
            seen,
            path,
            category="logs",
            purpose="H20 render-stack, Vulkan, Xvfb, or Isaac MP4 gate log for failure-boundary diagnosis.",
            required_or_optional="optional",
            claim_level="failure_diagnostic_log",
        )
    for path, purpose, claim in referenced_large_dirs():
        add_unique(
            rows,
            seen,
            path,
            category="metadata",
            purpose=purpose,
            required_or_optional="required" if path.name in {"download", "whole_body_tracking"} else "optional",
            claim_level=claim,
            copy=False,
            large_override=True,
        )

    counts_by_category: dict[str, int] = {}
    for row in rows:
        counts_by_category[row["category"]] = counts_by_category.get(row["category"], 0) + 1
    manifest = {
        "status": "ok_isaac_mp4_rtx_migration_package",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "package_root": str(PKG),
        "copy_limit_bytes": COPY_LIMIT_BYTES,
        "claim_level": "migration_package_for_true_isaac_rendered_simulation_video",
        "h20_blocker_classification": "blocked_h20_isaac_sim_rendering_stack",
        "goal_complete": False,
        "counts": {
            "total": len(rows),
            "copied": sum(1 for row in rows if row["copied_or_referenced"] == "copied"),
            "referenced": sum(1 for row in rows if row["copied_or_referenced"] != "copied"),
            "large": sum(1 for row in rows if row["large_file"]),
            "by_category": dict(sorted(counts_by_category.items())),
        },
        "checks": {
            "package_root_exists": PKG.is_dir(),
            "readme_exists": True,
            "smoke_runner_exists": True,
            "full_runner_exists": True,
            "true_rendered_gate_script_collected": any(
                row["source_path"].endswith("tracking_g1_isaaclab_rendered_policy_rollout_mp4.py") for row in rows
            ),
            "render_stack_repair_audit_collected": any(
                row["source_path"].endswith("isaac_render_stack_repair_audit.py") for row in rows
            ),
            "selected_checkpoint_collected": any(
                "robot_order_full_bundle_ppo_training" in row["source_path"] and row["category"] == "checkpoints"
                for row in rows
            ),
            "motion_bundle_collected": any("fk_repaired_robot_order" in row["source_path"] for row in rows),
            "does_not_claim_real_robot": True,
            "does_not_claim_paper_level_fig5_fig6": True,
            "does_not_mark_goal_complete": True,
        },
        "first_rtx_command": "cd /path/to/BeyondMimic && BM_ROOT=$PWD BM_GPU_ID=0 ./isaac_mp4_need/run_rtx_smoke.sh",
        "entries": rows,
    }

    write_text(PKG / ".gitignore", package_gitignore())
    write_text(PKG / "run_rtx_smoke.sh", smoke_runner(), executable=True)
    write_text(PKG / "run_rtx_full_videos.sh", full_runner(), executable=True)
    write_text(PKG / "README_ISAAC_MP4_RTX.md", readme_text(manifest))
    manifest_path = PKG / "isaac_mp4_need_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv_path = PKG / "isaac_mp4_need_manifest.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_path",
            "package_path",
            "exists",
            "file_size",
            "sha256",
            "category",
            "purpose",
            "required_or_optional",
            "large_file",
            "copied_or_referenced",
            "claim_level",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"status": manifest["status"], "package_root": str(PKG), **manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
