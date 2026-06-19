#!/usr/bin/env python3
"""Audit required trained/deployment artifacts that are still absent."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/required_artifact_absence"
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".plan", ".trt", ".safetensors"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".gif"}
ROLLOUT_SUFFIXES = {".npz", ".pkl", ".json", ".csv", ".tsv", ".bag", ".db3"}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def exists(rel_path: str) -> bool:
    path = ROOT / rel_path
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def scan_tree(base: str, suffixes: set[str]) -> dict[str, list[Path]]:
    root = ROOT / base
    buckets = {suffix: [] for suffix in suffixes}
    if not root.exists():
        return buckets
    ignored_dirs = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in ignored_dirs]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            suffix = path.suffix.lower()
            if suffix in buckets:
                buckets[suffix].append(path)
    return {suffix: sorted(paths) for suffix, paths in buckets.items()}


def from_buckets(buckets: dict[str, list[Path]], suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for suffix in suffixes:
        paths.extend(buckets.get(suffix, []))
    return sorted(paths)


def sample(paths: list[Path], limit: int = 8) -> list[str]:
    return [rel(p) for p in paths[:limit]]


def contains_any(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def classify_reference(paths: list[Path]) -> dict[str, int]:
    counts = {
        "reference_asap": 0,
        "reference_pbhc": 0,
        "reference_unitree": 0,
        "reference_motion_diffusion": 0,
        "isaaclab_docs_or_demo": 0,
        "other_download_reference": 0,
    }
    for path in paths:
        text = rel(path)
        if "/ASAP/" in text:
            counts["reference_asap"] += 1
        elif "/PBHC/" in text:
            counts["reference_pbhc"] += 1
        elif "unitree" in text.lower():
            counts["reference_unitree"] += 1
        elif contains_any(text, ["motion-diffusion", "guided-motion-diffusion", "diffusion-motion-inbetweening", "latent-diffusion"]):
            counts["reference_motion_diffusion"] += 1
        elif "IsaacLab" in text or "isaaclab" in text:
            counts["isaaclab_docs_or_demo"] += 1
        else:
            counts["other_download_reference"] += 1
    return counts


def row(
    artifact_id: str,
    goal_refs: str,
    paper_refs: str,
    required_artifact: str,
    local_patterns_checked: list[str],
    found_local: list[str],
    download_reference_count: int,
    download_reference_samples: list[str],
    status: str,
    evidence: list[str],
    why_missing_or_not_accepted: str,
) -> dict[str, Any]:
    evidence_exists = [exists(path) for path in evidence]
    return {
        "artifact_id": artifact_id,
        "goal_refs": goal_refs,
        "paper_refs": paper_refs,
        "required_artifact": required_artifact,
        "local_patterns_checked": local_patterns_checked,
        "found_local": found_local,
        "local_count": len(found_local),
        "download_reference_count": download_reference_count,
        "download_reference_samples": download_reference_samples,
        "status": status,
        "evidence": evidence,
        "evidence_exists": evidence_exists,
        "all_evidence_exists": all(evidence_exists),
        "why_missing_or_not_accepted": why_missing_or_not_accepted,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    all_suffixes = MODEL_SUFFIXES | VIDEO_SUFFIXES | ROLLOUT_SUFFIXES
    res_buckets = scan_tree("res", all_suffixes)
    reproduction_buckets = scan_tree("reproduction", all_suffixes)
    logs_buckets = scan_tree("logs", all_suffixes)
    download_buckets = scan_tree("download", all_suffixes)

    local_models = (
        from_buckets(res_buckets, MODEL_SUFFIXES)
        + from_buckets(reproduction_buckets, MODEL_SUFFIXES)
        + from_buckets(logs_buckets, MODEL_SUFFIXES)
    )
    official_reference_videos = [
        p
        for p in from_buckets(reproduction_buckets, VIDEO_SUFFIXES)
        if "reproduction/third_party/official" in rel(p)
    ]
    all_local_videos = from_buckets(res_buckets, VIDEO_SUFFIXES) + from_buckets(logs_buckets, VIDEO_SUFFIXES)
    debug_preview_videos = [
        p
        for p in all_local_videos
        if "guidance_debug_visualization" in rel(p)
        or "guidance_checkpoint_visualization" in rel(p)
        or "level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos" in rel(p)
    ]
    local_reference_videos = [
        p for p in all_local_videos if "res/visualization/official_csv_loop_reference_replay" in rel(p)
    ]
    local_videos = [p for p in all_local_videos if p not in debug_preview_videos and p not in local_reference_videos]
    local_rollout_files = (
        [p for p in from_buckets(res_buckets, ROLLOUT_SUFFIXES) if "/res/runs/" in str(p) or "/res/level_c/" in str(p)]
        + [p for p in from_buckets(reproduction_buckets, ROLLOUT_SUFFIXES) if "/reproduction/data/" in str(p)]
    )
    download_models = from_buckets(download_buckets, MODEL_SUFFIXES)
    download_videos = from_buckets(download_buckets, VIDEO_SUFFIXES)
    download_rollout_like = from_buckets(download_buckets, {".npz", ".pkl", ".bag", ".db3"})

    diagnostic_checkpoints = [
        p
        for p in from_buckets(res_buckets, {".npz"})
        if "setup_checkpoint_resume_smoke" in rel(p) and "/checkpoint/" in rel(p)
    ]
    debug_vae_checkpoints = [
        p
        for p in local_models
        if "vae_checkpoint_smoke" in rel(p) and p.name == "debug_conditional_vae_checkpoint_smoke.pt"
    ]
    debug_diffusion_checkpoints = [
        p
        for p in local_models
        if "diffusion_checkpoint_smoke" in rel(p) and p.name == "debug_diffusion_transformer_checkpoint_smoke.pt"
    ]
    bounded_debug_diffusion_checkpoints = [
        p
        for p in local_models
        if "level_c_bounded_debug_diffusion_static_000_20260617_083000" in rel(p)
        and p.name == "debug_bounded_diffusion_checkpoint.pt"
    ]
    resource_adjusted_tiny_checkpoints = [
        p
        for p in local_models
        if "level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500" in rel(p)
        and p.name == "tiny_resource_adjusted_denoiser.pt"
    ]
    debug_motion_policy_onnx_files = [
        p
        for p in local_models
        if "debug_motion_policy_onnx_export" in rel(p) and p.name == "debug_motion_policy_contract.onnx"
    ]
    resource_adjusted_tiny_onnx_files = [
        p
        for p in local_models
        if "resource_adjusted_tiny_diffusion_onnx_export_inference" in rel(p)
        and p.name == "resource_adjusted_tiny_denoiser_debug.onnx"
    ]
    public_lafan1_paper_arch_checkpoints = [
        p
        for p in local_models
        if (
            "level_c_lafan1_paper_arch_vae_diffusion" in rel(p)
            or "level_c_lafan1_paper_arch_symmetry_augmented" in rel(p)
        )
        and p.name == "lafan1_paper_arch_vae_diffusion.pt"
    ]
    public_lafan1_paper_arch_onnx_files = [
        p
        for p in local_models
        if (
            "lafan1_paper_arch_onnx_latency" in rel(p)
            or "lafan1_paper_arch_symmetry_augmented_onnx_latency" in rel(p)
        )
        and p.name in {"lafan1_paper_arch_vae_decoder.onnx", "lafan1_paper_arch_diffusion_denoiser.onnx"}
    ]
    resource_adjusted_tracking_checkpoints = [
        p
        for p in local_models
        if "tracking_g1_resource_adjusted_ppo_training" in rel(p) and p.name.startswith("model_")
    ]
    official_csv_loop_tracking_checkpoints = [
        p
        for p in local_models
        if "tracking_g1_official_csv_loop_ppo_training" in rel(p) and p.name.startswith("model_")
    ]
    resource_adjusted_teacher_rollout_vae_checkpoints = [
        p
        for p in local_models
        if "level_c_resource_adjusted_teacher_rollout_vae_training" in rel(p)
        and p.name == "resource_adjusted_teacher_rollout_action_vae.pt"
    ]
    official_csv_loop_teacher_rollout_vae_checkpoints = [
        p
        for p in local_models
        if "level_c_official_csv_loop_teacher_rollout_vae_training" in rel(p)
        and p.name == "resource_adjusted_teacher_rollout_action_vae.pt"
    ]
    resource_adjusted_state_latent_diffusion_checkpoints = [
        p
        for p in local_models
        if "level_c_resource_adjusted_state_latent_diffusion_training" in rel(p)
        and p.name == "resource_adjusted_state_latent_denoiser.pt"
    ]
    official_csv_loop_state_latent_diffusion_checkpoints = [
        p
        for p in local_models
        if "level_c_official_csv_loop_state_latent_diffusion_training" in rel(p)
        and p.name == "resource_adjusted_state_latent_denoiser.pt"
    ]
    official_csv_loop_teacher_rollout_files = [
        p
        for p in local_rollout_files
        if "tracking_g1_official_csv_loop_teacher_rollout_dataset" in rel(p)
        and p.name in {"teacher_rollout_shard.npz", "teacher_rollout_shard_metrics.json", "gpu_metrics.csv"}
    ]
    reproduction_model_files = [
        p
        for p in local_models
        if "checkpoint_resume_smoke" not in rel(p)
        and "vae_checkpoint_smoke" not in rel(p)
        and "diffusion_checkpoint_smoke" not in rel(p)
        and "level_c_bounded_debug_diffusion_static_000_20260617_083000" not in rel(p)
        and "level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500" not in rel(p)
        and "debug_motion_policy_onnx_export" not in rel(p)
        and "resource_adjusted_tiny_diffusion_onnx_export_inference" not in rel(p)
        and "lafan1_paper_arch_onnx_latency" not in rel(p)
        and "lafan1_paper_arch_symmetry_augmented_onnx_latency" not in rel(p)
        and "tracking_g1_resource_adjusted_ppo_training" not in rel(p)
        and "tracking_g1_official_csv_loop_ppo_training" not in rel(p)
        and "level_c_resource_adjusted_teacher_rollout_vae_training" not in rel(p)
        and "level_c_official_csv_loop_teacher_rollout_vae_training" not in rel(p)
        and "level_c_resource_adjusted_state_latent_diffusion_training" not in rel(p)
        and "level_c_official_csv_loop_state_latent_diffusion_training" not in rel(p)
    ]
    unclassified_reproduction_model_files = [
        p
        for p in reproduction_model_files
        if p not in public_lafan1_paper_arch_checkpoints
        and p not in public_lafan1_paper_arch_onnx_files
    ]
    beyondmimic_named_download_models = [p for p in download_models if "beyondmimic" in rel(p).lower()]
    beyondmimic_named_download_videos = [p for p in download_videos if "beyondmimic" in rel(p).lower()]
    fig56_like_videos = [p for p in download_videos if contains_any(rel(p), ["fig5", "fig6", "beyondmimic"])]
    reference_onnx = [p for p in download_models if p.suffix.lower() == ".onnx"]
    reference_pt = [p for p in download_models if p.suffix.lower() in {".pt", ".pth"}]

    rows = [
        row(
            "tracking_policy_checkpoint",
            "goal.md:747-775,1382-1429,1397-1399",
            "root.tex:585",
            "Trained motion-tracking policy checkpoint/export for BeyondMimic G1 tracking.",
            ["res/**/*.pt", "res/**/*.pth", "res/**/*.ckpt", "res/**/*.onnx"],
            [rel(p) for p in reproduction_model_files if contains_any(rel(p), ["tracking", "ppo", "policy", "model"])],
            len(reference_onnx) + len(reference_pt),
            sample(reference_onnx + reference_pt),
            "missing_required_artifact",
            [
                "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json",
                "res/blocked_gates/blocked_gate_audit.json",
            ],
            "Downloaded ONNX/PT files are reference Unitree/ASAP/PBHC assets, not a newly trained BeyondMimic tracking export from this reproduction.",
        ),
        row(
            "tracking_onnx_export",
            "goal.md:767,1399",
            "root.tex:585",
            "BeyondMimic motion policy ONNX satisfying the official motion_tracking_controller contract.",
            ["res/**/*.onnx", "reproduction/**/*.onnx"],
            [rel(p) for p in reproduction_model_files if p.suffix.lower() == ".onnx"],
            len(reference_onnx),
            sample(reference_onnx),
            "missing_required_artifact",
            ["res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json"],
            "The contract audit and debug ONNX export prove exporter/consumer compatibility, but no live IsaacLab export from a trained BeyondMimic policy exists.",
        ),
        row(
            "vae_checkpoint",
            "goal.md:1148-1190,1431-1447,1825",
            "root.tex:253",
            "Trained conditional VAE checkpoint from true DAgger teacher/student rollouts.",
            ["res/**/*.pt", "res/**/*.pth", "res/**/*.ckpt", "res/**/*.safetensors"],
            [rel(p) for p in reproduction_model_files if contains_any(rel(p), ["vae", "dagger"])],
            len(reference_pt),
            sample(reference_pt),
            "missing_required_artifact",
            [
                "res/level_c/vae_contract_audit/level_c_vae_contract_audit.json",
                "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
                "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json",
            ],
            "Local VAE evidence is architecture/loss/debug accumulation plus a save/load smoke checkpoint only; no trained VAE model checkpoint is present.",
        ),
        row(
            "true_dagger_rollout_logs",
            "goal.md:1181-1185,1437-1444",
            "root.tex:253",
            "True DAgger rollout logs with teacher query, aggregation, closed-loop rollout, stability, and student/teacher comparison.",
            ["res/runs/**/*", "res/level_c/**/*", "reproduction/data/**/*"],
            [rel(p) for p in local_rollout_files if contains_any(rel(p), ["dagger", "teacher"])],
            len(download_rollout_like),
            sample(download_rollout_like),
            "missing_required_artifact",
            [
                "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json",
                "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
            ],
            "Existing DAgger manifest is explicitly synthetic/debug; no live teacher rollout or teacher-query aggregation log exists.",
        ),
        row(
            "state_latent_dataset",
            "goal.md:1191-1231,1448-1467",
            "root.tex:253",
            "State-latent trajectory dataset produced by trained VAE rollouts, with accept/reject provenance and splits.",
            ["reproduction/data/**/*", "res/level_c/**/*"],
            [rel(p) for p in local_rollout_files if contains_any(rel(p), ["state", "latent", "rollout"])],
            len(download_rollout_like),
            sample(download_rollout_like),
            "debug_only_not_required_artifact",
            [
                "res/level_c/paper_state_windows/level_c_paper_state_windows.json",
                "res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json",
            ],
            "Paper-state windows and rejection manifests are debug fixtures with synthetic latents, not VAE rollout latents from trained policies.",
        ),
        row(
            "diffusion_checkpoint",
            "goal.md:1251-1290,1468-1487,1825",
            "root.tex:253,593",
            "Trained state-latent diffusion Transformer checkpoint.",
            ["res/**/*.pt", "res/**/*.pth", "res/**/*.ckpt", "res/**/*.safetensors"],
            [rel(p) for p in reproduction_model_files if contains_any(rel(p), ["diffusion", "transformer"])],
            len(reference_pt),
            sample(reference_pt),
            "missing_required_artifact",
            [
                "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json",
                "res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json",
                "res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json",
                "res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json",
            ],
            "Only architecture/forward-backward/debug overfit probes, a state-dict hash manifest, and a two-step EMA smoke exist; no trained diffusion checkpoint is present.",
        ),
        row(
            "diffusion_tensorrt_engine",
            "goal.md:1538-1631,1825",
            "root.tex:593",
            "TensorRT engine or plan for the trained diffusion policy.",
            ["res/**/*.engine", "res/**/*.plan", "res/**/*.trt"],
            [rel(p) for p in reproduction_model_files if p.suffix.lower() in {".engine", ".plan", ".trt"}],
            len([p for p in download_models if p.suffix.lower() in {".engine", ".plan", ".trt"}]),
            sample([p for p in download_models if p.suffix.lower() in {".engine", ".plan", ".trt"}]),
            "missing_required_artifact",
            ["res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json"],
            "Paper TensorRT deployment is indexed, but no TensorRT engine/plan exists locally.",
        ),
        row(
            "closed_loop_tracking_eval_logs",
            "goal.md:747-775,1382-1429,1550-1631",
            "root.tex:585",
            "Closed-loop tracking evaluation logs/metrics from live IsaacLab/Kit rollout.",
            ["res/runs/**/*", "logs/**/*"],
            [rel(p) for p in local_rollout_files if contains_any(rel(p), ["tracking", "rollout", "eval"])],
            len(download_rollout_like),
            sample(download_rollout_like),
            "missing_required_artifact",
            ["res/blocked_gates/blocked_gate_audit.json", "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json"],
            "Static/source contracts and released-data plots exist, but live Kit rollout evaluation remains blocked by the inotify gate.",
        ),
        row(
            "fig5_rollout_artifacts",
            "goal.md:1488-1507,1538-1631",
            "root.tex:223-234",
            "Figure 5 joystick/waypoint/latent visualization rollout data, metrics, and visuals.",
            ["res/**/*fig5*", "res/**/*joystick*", "res/**/*waypoint*"],
            [rel(p) for p in local_rollout_files if contains_any(rel(p), ["fig5", "joystick", "waypoint"])],
            len(fig56_like_videos),
            sample(fig56_like_videos),
            "missing_required_artifact",
            ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"],
            "Debug guidance probes exist, but no paper Fig. 5 rollout logs, trained checkpoints, or released Fig. 5 data are present.",
        ),
        row(
            "fig6_rollout_artifacts",
            "goal.md:1488-1507,1538-1631",
            "root.tex:237-243",
            "Figure 6 inpainting/obstacle rollout data, metrics, and visuals.",
            ["res/**/*fig6*", "res/**/*inpaint*", "res/**/*obstacle*", "res/**/*sdf*"],
            [rel(p) for p in local_rollout_files if contains_any(rel(p), ["fig6", "inpaint", "obstacle", "sdf"])],
            len(fig56_like_videos),
            sample(fig56_like_videos),
            "missing_required_artifact",
            ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"],
            "Debug mask/SDF probes exist, but no paper Fig. 6 closed-loop rollout logs or released Fig. 6 data are present.",
        ),
        row(
            "guidance_success_failure_videos",
            "goal.md:1505,1783,1827",
            "root.tex:223-243",
            "Success and failure videos for all Phase 8 guidance tasks.",
            ["res/videos/**/*", "res/runs/**/videos/**/*"],
            [rel(p) for p in local_videos if "/res/" in "/" + rel(p)],
            len(download_videos),
            sample(download_videos),
            "missing_required_artifact",
            ["res/guidance_task_coverage/guidance_task_coverage_audit.json"],
            "Downloaded GIF/MP4 files are reference-project docs/demos, not reproduced BeyondMimic task videos.",
        ),
        row(
            "sim_to_sim_ros_bag_or_log",
            "goal.md:767-768,808,1382-1429",
            "root.tex:585",
            "MuJoCo/ROS sim-to-sim execution logs, bags, or deployment evaluation from motion_tracking_controller.",
            ["res/runs/**/*", "logs/**/*"],
            [rel(p) for p in from_buckets(res_buckets, {".bag", ".db3"}) + from_buckets(logs_buckets, {".bag", ".db3"})],
            0,
            [],
            "missing_required_artifact",
            ["res/blocked_gates/blocked_gate_audit.json", "res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.json"],
            "ROS 2 Jazzy/Noble deployment stack is unavailable on this host; only static controller audits exist.",
        ),
        row(
            "paper_level_multiseed_training_metrics",
            "goal.md:1550-1631",
            "root.tex:223-243,585-593",
            "Paper-level multi-seed tracking/VAE/diffusion metrics from completed training/evaluation runs.",
            ["res/runs/**/metrics.*", "res/**/*.json", "res/**/*.csv", "res/**/*.tsv"],
            [rel(p) for p in local_rollout_files if contains_any(rel(p), ["multiseed", "heldout", "metrics"])],
            0,
            [],
            "debug_only_not_required_artifact",
            ["res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json"],
            "Current multi-seed evidence is smoke/debug statistics; paper-level tracking/VAE/diffusion training metrics remain missing.",
        ),
        row(
            "completed_training_run_directory",
            "goal.md:1747-1787",
            "",
            "At least one completed full training run directory with config, logs, metrics, checkpoint, figures, videos, and SUCCESS status.",
            ["res/runs/*"],
            [
                rel(p)
                for p in sorted((ROOT / "res/runs").glob("*"))
                if p.is_dir()
                and (p / "status.json").exists()
                and json.loads((p / "status.json").read_text(encoding="utf-8")).get("is_training_run") is True
                and json.loads((p / "status.json").read_text(encoding="utf-8")).get("status") == "SUCCESS"
            ],
            0,
            [],
            "missing_required_artifact",
            ["res/run_management_audit/run_management_audit.json", "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json"],
            "Existing run directories are diagnostic or checkpoint-resume plumbing, not completed PPO/VAE/diffusion training runs.",
        ),
        row(
            "resource_adjusted_tracking_checkpoint_excluded",
            "goal.md:747-775,1382-1429,1397-1399",
            "root.tex:585",
            "Resource-adjusted G1 PPO checkpoints are present but must not be counted as official BeyondMimic tracking checkpoints.",
            [
                "res/runs/tracking_g1_resource_adjusted_ppo_training/*/rank_0/model_*.pt",
                "res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.json",
            ],
            [rel(p) for p in resource_adjusted_tracking_checkpoints],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/tracking/g1_resource_adjusted_ppo_training_run/tracking_g1_resource_adjusted_ppo_training_run.json"
            ],
            "The checkpoints come from the generated resource-adjusted USD and official-CSV-derived motion path. They prove local virtual PPO execution but are not official replay outputs, not a paper-scale teacher, and not paper-level BeyondMimic tracking results.",
        ),
        row(
            "official_csv_loop_tracking_checkpoint_excluded",
            "goal.md:747-775,1382-1429,1397-1399",
            "root.tex:585",
            "Official-csv-loop-motion G1 PPO checkpoints are present but must not be counted as official BeyondMimic tracking checkpoints.",
            [
                "res/runs/tracking_g1_official_csv_loop_ppo_training/*/rank_0/model_*.pt",
                "res/tracking/g1_official_csv_loop_ppo_training_run/tracking_g1_official_csv_loop_ppo_training_run.json",
            ],
            [rel(p) for p in official_csv_loop_tracking_checkpoints],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/tracking/g1_official_csv_loop_ppo_training_run/tracking_g1_official_csv_loop_ppo_training_run.json",
                "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/tracking_g1_official_csv_loop_ppo_checkpoint_eval.json",
            ],
            "The checkpoints come from a 300-iteration local PPO run using official csv-loop motion under the enriched-USD runtime patch. They prove a stronger local virtual tracking chain but are not the paper-scale official BeyondMimic teacher checkpoint.",
        ),
        row(
            "diagnostic_checkpoint_excluded",
            "goal.md:1712",
            "",
            "Diagnostic checkpoint/resume, VAE smoke checkpoint, diffusion smoke checkpoint, bounded debug diffusion checkpoint, and resource-adjusted tiny debug checkpoint artifacts are present but must not be counted as trained model checkpoints.",
            [
                "res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500/checkpoint/*.npz",
                "res/level_c/vae_checkpoint_smoke/debug_conditional_vae_checkpoint_smoke.pt",
                "res/level_c/diffusion_checkpoint_smoke/debug_diffusion_transformer_checkpoint_smoke.pt",
                "res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/checkpoint/debug_bounded_diffusion_checkpoint.pt",
                "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/checkpoint/tiny_resource_adjusted_denoiser.pt",
            ],
            [
                rel(p)
                for p in diagnostic_checkpoints
                + debug_vae_checkpoints
                + debug_diffusion_checkpoints
                + bounded_debug_diffusion_checkpoints
                + resource_adjusted_tiny_checkpoints
            ],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json",
                "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json",
                "res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json",
                "res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json",
                "res/level_c/resource_adjusted_tiny_diffusion_training_run/level_c_resource_adjusted_tiny_diffusion_training_run.json",
            ],
            "The diagnostic NPZ proves resume plumbing, the debug VAE PT proves save/load consistency, the debug diffusion PT proves save/load/resume consistency, the bounded debug diffusion PT proves a 3-step optimizer-run checkpoint only, and the resource-adjusted tiny PT proves a small debug denoiser only; all are explicitly excluded from trained-model counts.",
        ),
        row(
            "resource_adjusted_teacher_rollout_vae_checkpoint_excluded",
            "goal.md:1148-1190,1431-1447,1825",
            "root.tex:253",
            "Resource-adjusted teacher-rollout conditional action VAE checkpoint is present but must not be counted as the official BeyondMimic DAgger/VAE checkpoint.",
            [
                "res/runs/level_c_resource_adjusted_teacher_rollout_vae_training/*/resource_adjusted_teacher_rollout_action_vae.pt",
                "res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json",
            ],
            [rel(p) for p in resource_adjusted_teacher_rollout_vae_checkpoints],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json",
                "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_dataset.json",
            ],
            "The checkpoint is trained on the local generated-asset/resource-adjusted teacher rollout shards. It advances local downstream experimentation but is not trained from the paper's official DAgger rollout dataset and is not an official BeyondMimic VAE checkpoint.",
        ),
        row(
            "official_csv_loop_teacher_rollout_vae_checkpoint_excluded",
            "goal.md:1148-1190,1431-1447,1825",
            "root.tex:253",
            "Official-csv-loop teacher-rollout conditional action VAE checkpoint is present but must not be counted as the official BeyondMimic DAgger/VAE checkpoint.",
            [
                "res/runs/level_c_official_csv_loop_teacher_rollout_vae_training/*/resource_adjusted_teacher_rollout_action_vae.pt",
                "res/level_c/official_csv_loop_teacher_rollout_vae_training/level_c_official_csv_loop_teacher_rollout_vae_training.json",
            ],
            [rel(p) for p in official_csv_loop_teacher_rollout_vae_checkpoints],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/level_c/official_csv_loop_teacher_rollout_vae_training/level_c_official_csv_loop_teacher_rollout_vae_training.json",
                "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/tracking_g1_official_csv_loop_teacher_rollout_dataset.json",
            ],
            "The checkpoint is trained on local official-loop-motion teacher rollout shards from the enriched-USD runtime patch. It is useful downstream evidence but is not trained from official DAgger logs and is not an official BeyondMimic VAE checkpoint.",
        ),
        row(
            "official_csv_loop_teacher_rollout_dataset_excluded",
            "goal.md:1181-1185,1437-1444",
            "root.tex:253",
            "Official-csv-loop-motion teacher rollout shards are present but must not be counted as the official BeyondMimic DAgger rollout dataset.",
            [
                "res/runs/tracking_g1_official_csv_loop_teacher_rollout_dataset/**/*",
                "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/tracking_g1_official_csv_loop_teacher_rollout_dataset.json",
            ],
            [rel(p) for p in official_csv_loop_teacher_rollout_files],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/tracking_g1_official_csv_loop_teacher_rollout_dataset.json",
                "res/tracking/g1_official_csv_loop_ppo_training_run/tracking_g1_official_csv_loop_ppo_training_run.json",
            ],
            "The shards come from a local iteration-299 PPO checkpoint trained on official-loop motion under the enriched-USD runtime patch. They are useful local teacher-rollout data but are not the paper's official DAgger logs.",
        ),
        row(
            "resource_adjusted_state_latent_diffusion_checkpoint_excluded",
            "goal.md:1251-1290,1468-1487,1825",
            "root.tex:253,593",
            "Resource-adjusted state-latent denoiser checkpoint is present but must not be counted as the official BeyondMimic diffusion checkpoint.",
            [
                "res/runs/level_c_resource_adjusted_state_latent_diffusion_training/*/resource_adjusted_state_latent_denoiser.pt",
                "res/level_c/resource_adjusted_state_latent_diffusion_training/level_c_resource_adjusted_state_latent_diffusion_training.json",
            ],
            [rel(p) for p in resource_adjusted_state_latent_diffusion_checkpoints],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/level_c/resource_adjusted_state_latent_diffusion_training/level_c_resource_adjusted_state_latent_diffusion_training.json",
                "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json",
            ],
            "The checkpoint is trained on local resource-adjusted state-latent windows. It proves a downstream virtual denoising run, but it is not the official paper diffusion checkpoint, not TensorRT deployment, and not Fig.5/Fig.6 closed-loop guidance.",
        ),
        row(
            "official_csv_loop_state_latent_diffusion_checkpoint_excluded",
            "goal.md:1251-1290,1468-1487,1825",
            "root.tex:253,593",
            "Official-csv-loop state-latent denoiser checkpoint is present but must not be counted as the official BeyondMimic diffusion checkpoint.",
            [
                "res/runs/level_c_official_csv_loop_state_latent_diffusion_training/*/resource_adjusted_state_latent_denoiser.pt",
                "res/level_c/official_csv_loop_state_latent_diffusion_training/level_c_official_csv_loop_state_latent_diffusion_training.json",
            ],
            [rel(p) for p in official_csv_loop_state_latent_diffusion_checkpoints],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/level_c/official_csv_loop_state_latent_diffusion_training/level_c_official_csv_loop_state_latent_diffusion_training.json",
                "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json",
            ],
            "The checkpoint is trained on local official-loop state-latent windows derived from the enriched-USD virtual tracking chain. It proves a downstream virtual denoising run, but it is not the official paper diffusion checkpoint, not TensorRT deployment, and not Fig.5/Fig.6 closed-loop guidance.",
        ),
        row(
            "debug_guidance_visualization_excluded",
            "goal.md:1505,1783,1827",
            "root.tex:223-243",
            "Debug-only guidance and tiny diffusion preview GIF/figures are present but must not be counted as paper success/failure videos.",
            [
                "res/level_c/guidance_debug_visualization/*",
                "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/*",
            ],
            [rel(p) for p in debug_preview_videos],
            0,
            [],
            "present_but_not_required_artifact",
            ["res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json"],
            "These GIFs are generated from formula-level guidance fixtures or offline tiny-diffusion token previews and are explicitly marked not trained rollouts or paper Fig. 5/Fig. 6 videos.",
        ),
        row(
            "official_reference_doc_videos_excluded",
            "goal.md:1505,1783,1827",
            "root.tex:223-243",
            "Bundled official/dependency documentation GIFs are present but must not be counted as reproduced BeyondMimic videos.",
            ["reproduction/third_party/official/**/*"],
            [rel(p) for p in official_reference_videos],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/source_integrity/download_source_integrity/download_source_integrity_audit.json",
                "res/required_artifact_absence/required_artifact_absence_audit.json",
            ],
            "These GIFs come from bundled IsaacLab/reference documentation under reproduction/third_party/official, not from a local BeyondMimic rollout or paper Fig. 5/Fig. 6 reproduction.",
        ),
        row(
            "local_reference_video_excluded",
            "goal.md:1505,1783,1827",
            "root.tex:223-243",
            "Local kinematic reference replay video is present but must not be counted as a paper-level closed-loop or success/failure video.",
            ["res/visualization/official_csv_loop_reference_replay/*"],
            [rel(p) for p in local_reference_videos],
            0,
            [],
            "present_but_not_required_artifact",
            [
                "res/visualization/official_csv_loop_reference_replay/official_csv_loop_reference_replay_video_asset.json",
                "res/visual_media_inventory/visual_media_inventory_audit.json",
            ],
            "This MP4 visualizes saved reference body positions from the official-loop motion NPZ. It is explicitly labeled as a kinematic report asset, not an IsaacLab closed-loop rollout, not Fig. 5/Fig. 6 evidence, and not real-robot validation.",
        ),
    ]

    missing = [r for r in rows if not r["all_evidence_exists"]]
    status_counts: dict[str, int] = {}
    for item in rows:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    summary = {
        "status": "ok" if not missing else "failed",
        "experiment_type": "required_artifact_absence_audit",
        "scope": "goal.md/paper-required trained, rollout, deployment, and video artifacts with reference-artifact separation",
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "local_scan_counts": {
            "local_model_files": len(local_models),
            "local_reproduction_model_files_excluding_diagnostic": len(reproduction_model_files),
            "local_video_files": len(local_videos),
            "debug_preview_video_files_excluded": len(debug_preview_videos),
            "local_reference_video_files_excluded": len(local_reference_videos),
            "official_reference_video_files_excluded": len(official_reference_videos),
            "local_rollout_like_files": len(local_rollout_files),
            "diagnostic_checkpoint_files": len(diagnostic_checkpoints),
            "debug_vae_checkpoint_files": len(debug_vae_checkpoints),
            "debug_diffusion_checkpoint_files": len(debug_diffusion_checkpoints),
            "bounded_debug_diffusion_checkpoint_files": len(bounded_debug_diffusion_checkpoints),
            "resource_adjusted_tiny_checkpoint_files": len(resource_adjusted_tiny_checkpoints),
            "resource_adjusted_tracking_checkpoint_files": len(resource_adjusted_tracking_checkpoints),
            "official_csv_loop_tracking_checkpoint_files": len(official_csv_loop_tracking_checkpoints),
            "resource_adjusted_teacher_rollout_vae_checkpoint_files": len(
                resource_adjusted_teacher_rollout_vae_checkpoints
            ),
            "official_csv_loop_teacher_rollout_vae_checkpoint_files": len(
                official_csv_loop_teacher_rollout_vae_checkpoints
            ),
            "resource_adjusted_state_latent_diffusion_checkpoint_files": len(
                resource_adjusted_state_latent_diffusion_checkpoints
            ),
            "official_csv_loop_state_latent_diffusion_checkpoint_files": len(
                official_csv_loop_state_latent_diffusion_checkpoints
            ),
            "official_csv_loop_teacher_rollout_files": len(official_csv_loop_teacher_rollout_files),
            "debug_motion_policy_onnx_files": len(debug_motion_policy_onnx_files),
            "resource_adjusted_tiny_onnx_files": len(resource_adjusted_tiny_onnx_files),
            "public_lafan1_paper_arch_checkpoint_files": len(public_lafan1_paper_arch_checkpoints),
            "public_lafan1_paper_arch_onnx_files": len(public_lafan1_paper_arch_onnx_files),
            "unclassified_reproduction_model_files": len(unclassified_reproduction_model_files),
        },
        "download_reference_counts": {
            "download_model_files": len(download_models),
            "download_video_files": len(download_videos),
            "download_rollout_like_files": len(download_rollout_like),
            "download_model_categories": classify_reference(download_models),
            "download_video_categories": classify_reference(download_videos),
            "beyondmimic_named_download_models": len(beyondmimic_named_download_models),
            "beyondmimic_named_download_videos": len(beyondmimic_named_download_videos),
        },
        "missing_evidence_rows": missing,
        "rows": rows,
        "checks": {
            "all_evidence_paths_exist": not missing,
            "required_artifact_rows_with_debug_and_reference_exclusion": len(rows) == 25,
            "reference_download_models_separated": len(download_models) > 0
            and all(r["download_reference_count"] >= 0 for r in rows),
            "no_beyondmimic_named_model_in_download": len(beyondmimic_named_download_models) == 0,
            "no_beyondmimic_named_video_in_download": len(beyondmimic_named_download_videos) == 0,
            "no_local_reproduction_model_checkpoint": len(unclassified_reproduction_model_files) == 0,
            "public_lafan1_paper_arch_checkpoint_present": len(public_lafan1_paper_arch_checkpoints) >= 3,
            "public_lafan1_paper_arch_onnx_exports_present": len(public_lafan1_paper_arch_onnx_files) == 4,
            "no_unclassified_local_reproduction_model_checkpoint": len(unclassified_reproduction_model_files) == 0,
            "no_local_paper_level_reproduction_video": len(local_videos) == 0,
            "diagnostic_checkpoint_excluded": len(diagnostic_checkpoints) >= 1
            and len(debug_vae_checkpoints) == 1
            and len(debug_diffusion_checkpoints) == 1
            and len(bounded_debug_diffusion_checkpoints) == 1
            and len(resource_adjusted_tiny_checkpoints) == 1
            and len(debug_motion_policy_onnx_files) == 1
            and len(resource_adjusted_tiny_onnx_files) == 1
            and any(r["artifact_id"] == "diagnostic_checkpoint_excluded" for r in rows),
            "resource_adjusted_tracking_checkpoint_excluded": len(resource_adjusted_tracking_checkpoints) >= 1
            and any(r["artifact_id"] == "resource_adjusted_tracking_checkpoint_excluded" for r in rows),
            "official_csv_loop_tracking_checkpoint_excluded": len(official_csv_loop_tracking_checkpoints) >= 1
            and any(r["artifact_id"] == "official_csv_loop_tracking_checkpoint_excluded" for r in rows),
            "resource_adjusted_teacher_rollout_vae_checkpoint_excluded": (
                len(resource_adjusted_teacher_rollout_vae_checkpoints) == 1
                and any(
                    r["artifact_id"] == "resource_adjusted_teacher_rollout_vae_checkpoint_excluded"
                    for r in rows
                )
            ),
            "official_csv_loop_teacher_rollout_vae_checkpoint_excluded": (
                len(official_csv_loop_teacher_rollout_vae_checkpoints) == 1
                and any(
                    r["artifact_id"] == "official_csv_loop_teacher_rollout_vae_checkpoint_excluded"
                    for r in rows
                )
            ),
            "resource_adjusted_state_latent_diffusion_checkpoint_excluded": (
                len(resource_adjusted_state_latent_diffusion_checkpoints) == 1
                and any(
                    r["artifact_id"] == "resource_adjusted_state_latent_diffusion_checkpoint_excluded"
                    for r in rows
                )
            ),
            "official_csv_loop_state_latent_diffusion_checkpoint_excluded": (
                len(official_csv_loop_state_latent_diffusion_checkpoints) == 1
                and any(
                    r["artifact_id"] == "official_csv_loop_state_latent_diffusion_checkpoint_excluded"
                    for r in rows
                )
            ),
            "official_csv_loop_teacher_rollout_dataset_excluded": (
                len(official_csv_loop_teacher_rollout_files) >= 3
                and any(r["artifact_id"] == "official_csv_loop_teacher_rollout_dataset_excluded" for r in rows)
            ),
            "debug_preview_videos_excluded": len(debug_preview_videos) >= 3
            and any(r["artifact_id"] == "debug_guidance_visualization_excluded" for r in rows),
            "official_reference_doc_videos_excluded": len(official_reference_videos) >= 1
            and any(r["artifact_id"] == "official_reference_doc_videos_excluded" for r in rows),
            "local_reference_video_excluded": len(local_reference_videos) >= 1
            and any(r["artifact_id"] == "local_reference_video_excluded" for r in rows),
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The workspace contains reference-project ONNX/PT/GIF assets and many debug NPZ artifacts, but it does "
                "not contain the required official/teacher-rollout BeyondMimic tracking/VAE/diffusion checkpoints, "
                "TensorRT engine, closed-loop rollout logs, Fig.5/Fig.6 artifacts, or reproduced success/failure videos. "
                "It does contain one public-LAFAN1 paper-architecture VAE/diffusion checkpoint, which is recorded "
                "separately and must not be counted as an official DAgger/closed-loop paper checkpoint. It also "
                "contains resource-adjusted G1 PPO checkpoints, which prove local virtual training execution but are "
                "separately excluded from official paper-level tracking artifacts. The resource-adjusted teacher-rollout "
                "VAE checkpoint and state-latent denoiser checkpoint are also separately classified as local evidence "
                "rather than official DAgger/VAE/diffusion artifacts. A local kinematic reference MP4 is present for "
                "reporting, but it is excluded from paper-level closed-loop/video evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "required_artifact_absence_audit.json"),
            "tsv": str(OUT / "required_artifact_absence_audit.tsv"),
        },
    }

    (OUT / "required_artifact_absence_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "required_artifact_absence_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "artifact_id",
            "status",
            "local_count",
            "download_reference_count",
            "goal_refs",
            "paper_refs",
            "required_artifact",
            "found_local",
            "download_reference_samples",
            "why_missing_or_not_accepted",
        ]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "artifact_id": item["artifact_id"],
                    "status": item["status"],
                    "local_count": item["local_count"],
                    "download_reference_count": item["download_reference_count"],
                    "goal_refs": item["goal_refs"],
                    "paper_refs": item["paper_refs"],
                    "required_artifact": item["required_artifact"],
                    "found_local": ";".join(item["found_local"]),
                    "download_reference_samples": ";".join(item["download_reference_samples"]),
                    "why_missing_or_not_accepted": item["why_missing_or_not_accepted"],
                }
            )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
