#!/usr/bin/env python3
"""Machine-readable audit of external gates blocking full BeyondMimic reproduction."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/blocked_gates"


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if limit is None:
        return text
    return text[:limit]


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def command_output(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def os_release() -> dict[str, str]:
    data: dict[str, str] = {}
    for line in read_text(Path("/etc/os-release")).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


def file_exists(path: str) -> bool:
    return (ROOT / path).exists()


def load_json(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def make_gate(
    gate_id: str,
    status: str,
    blocks: list[str],
    evidence: dict[str, Any],
    next_action: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": status,
        "blocks": blocks,
        "evidence": evidence,
        "next_action": next_action,
    }


def audit_inotify_gate() -> dict[str, Any]:
    max_watches = read_int("/proc/sys/fs/inotify/max_user_watches")
    max_instances = read_int("/proc/sys/fs/inotify/max_user_instances")
    retry_log = ROOT / "logs/setup/isaaclab_headless_smoke_retry.log"
    log_text = read_text(retry_log)
    errno28_count = log_text.count("errno=28")
    change_watch_fail_count = log_text.count("Failed to create change watch")
    proc = command_output(["sysctl", "fs.inotify.max_user_watches", "fs.inotify.max_user_instances"])
    live = load_json("res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json")
    live_checks = live.get("checks", {})
    limits_meet_targets = (max_watches or 0) >= 524288 and (max_instances or 0) >= 1024
    live_headless_passed = bool(live_checks.get("app_launcher_reached_success_sentinel"))
    historical_failure_recorded = errno28_count > 0 or change_watch_fail_count > 0
    if limits_meet_targets and live_headless_passed:
        status = "clear_with_historical_failure" if historical_failure_recorded else "clear"
    elif errno28_count > 0 and not limits_meet_targets:
        status = "blocked"
    else:
        status = "needs_review"
    return make_gate(
        "isaaclab_kit_inotify",
        status,
        []
        if status.startswith("clear")
        else [
            "IsaacLab/Kit headless smoke",
            "official csv_to_npz.py motion preprocessing",
            "official replay_npz.py reference replay",
            "PPO motion-tracking smoke",
            "live simulation rollout/evaluation",
        ],
        {
            "max_user_watches": max_watches,
            "max_user_instances": max_instances,
            "sysctl_return_code": proc[0],
            "sysctl_output": proc[1],
            "retry_log": str(retry_log),
            "errno28_count_in_retry_log": errno28_count,
            "change_watch_fail_count_in_retry_log": change_watch_fail_count,
            "retry_log_has_change_watch_failure": "Failed to create change watch" in log_text,
            "retry_log_has_no_space_errno28": "errno=28" in log_text,
            "limits_meet_targets": limits_meet_targets,
            "live_headless_passed": live_headless_passed,
            "live_gate_probe_json": str(ROOT / "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json"),
            "live_gate_status": live.get("status"),
            "historical_failure_retained": historical_failure_recorded,
        },
        (
            "Inotify is no longer the active headless-gate blocker when status is clear_with_historical_failure; keep "
            "the retained failed-run evidence, monitor watch headroom, and focus official replay recovery on the USD "
            "conversion path."
            if status.startswith("clear")
            else "Ask an administrator to raise fs.inotify.max_user_watches to at least 524288 and "
            "fs.inotify.max_user_instances to at least 1024, then rerun the IsaacLab/Kit and tracking smoke commands."
        ),
    )


def audit_official_g1_usd_conversion_replay_gate() -> dict[str, Any]:
    conversion = load_json("res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json")
    official_entry = load_json(
        "res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json"
    )
    official_csv_loop = load_json(
        "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
    )
    official_replay_loop = load_json(
        "res/tracking/official_replay_npz_loop_with_enriched_usd/"
        "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
    )
    import_config_probe = load_json(
        "res/tracking/g1_urdf_import_config_variant_probe/"
        "tracking_g1_urdf_import_config_variant_probe.json"
    )
    source_equivalence = load_json(
        "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json"
    )
    csv_task = load_json(
        "res/tracking/g1_resource_adjusted_csv_task_eval/tracking_g1_resource_adjusted_csv_task_eval_audit.json"
    )
    train_entry = load_json(
        "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
        "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
    )
    ppo_harness = load_json(
        "res/tracking/g1_resource_adjusted_ppo_training_run/"
        "tracking_g1_resource_adjusted_ppo_training_run.json"
    )
    blocked = conversion.get("status") == "ok_with_blocked_conversion"
    return make_gate(
        "official_g1_usd_conversion_replay",
        "blocked" if blocked else "needs_review",
        [
            "official csv_to_npz.py motion preprocessing success",
            "official replay_npz.py reference replay",
            "official G1 USD/URDF converter output",
            "paper-level tracking replay/evaluation",
            "formal PPO tracking training on official assets",
        ],
        {
            "official_replay_conversion_json": str(
                ROOT / "res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json"
            ),
            "official_replay_conversion_status": conversion.get("status"),
            "latest_blocker": conversion.get("latest_blocker"),
            "next_action": conversion.get("interpretation", {}).get("next_action"),
            "why_not_complete": conversion.get("interpretation", {}).get("why_not_complete"),
            "official_replay_npz_entry_diagnostic_json": str(
                ROOT
                / "res/tracking/official_replay_npz_entry_diagnostic/"
                "tracking_official_replay_npz_entry_diagnostic_audit.json"
            ),
            "official_replay_npz_entry_diagnostic_status": official_entry.get("status"),
            "official_replay_npz_entry_latest_blocker": official_entry.get("latest_blocker"),
            "official_replay_npz_entry_app_launcher_constructed": official_entry.get("checks", {}).get(
                "app_launcher_constructed"
            ),
            "official_replay_npz_entry_blocked_before_artifact_download": official_entry.get("checks", {}).get(
                "fake_wandb_download_seen"
            )
            is False,
            "official_replay_npz_entry_layer_save_blocker": official_entry.get("run", {})
            .get("markers", {})
            .get("failed_to_save_layer"),
            "official_replay_npz_entry_empty_robot_after_converter": official_entry.get("run", {})
            .get("markers", {})
            .get("empty_robot_after_converter"),
            "official_csv_to_npz_loop_patch_json": str(
                ROOT
                / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
            ),
            "official_csv_to_npz_loop_patch_status": official_csv_loop.get("status"),
            "official_csv_to_npz_loop_patch_latest_blocker": official_csv_loop.get("latest_blocker"),
            "official_csv_to_npz_loop_patch_joint_shape": official_csv_loop.get("metrics", {}).get("joint_pos_shape"),
            "official_csv_to_npz_loop_patch_is_resource_adjusted": official_csv_loop.get("metrics", {}).get(
                "uses_resource_adjusted_usd"
            ),
            "official_replay_loop_patch_json": str(
                ROOT
                / "res/tracking/official_replay_npz_loop_with_enriched_usd/"
                "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
            ),
            "official_replay_loop_patch_status": official_replay_loop.get("status"),
            "official_replay_loop_patch_latest_blocker": official_replay_loop.get("latest_blocker"),
            "g1_urdf_import_config_variant_probe_json": str(
                ROOT
                / "res/tracking/g1_urdf_import_config_variant_probe/"
                "tracking_g1_urdf_import_config_variant_probe.json"
            ),
            "g1_urdf_import_config_variant_probe_status": import_config_probe.get("status"),
            "g1_urdf_import_config_has_make_instanceable_setter": import_config_probe.get("method_probe", {})
            .get("payload", {})
            .get("has_set_make_instanceable"),
            "g1_urdf_import_config_has_instanceable_usd_path_setter": import_config_probe.get("method_probe", {})
            .get("payload", {})
            .get("has_set_instanceable_usd_path"),
            "g1_urdf_import_config_baseline_empty_usd": import_config_probe.get("variant_summary", {})
            .get("variant_baseline_make_instanceable_false", {})
            .get("usd", {})
            .get("prim_count")
            == 0,
            "g1_urdf_source_equivalence_json": str(
                ROOT
                / "res/tracking/g1_urdf_source_equivalence_audit/"
                "tracking_g1_urdf_source_equivalence_audit.json"
            ),
            "g1_urdf_source_equivalence_status": source_equivalence.get("status"),
            "download_reprodata_urdf_identical": source_equivalence.get("checks", {}).get(
                "download_and_reproduction_data_structurally_identical"
            ),
            "wbt_same_29_nonfixed_action_joints": source_equivalence.get("checks", {}).get(
                "whole_body_tracking_has_same_29_nonfixed_action_joints"
            ),
            "wbt_support_link_diff": source_equivalence.get("comparisons", {})
            .get("download_vs_whole_body_tracking", {})
            .get("link_set_diff"),
            "wbt_support_joint_diff": source_equivalence.get("comparisons", {})
            .get("download_vs_whole_body_tracking", {})
            .get("joint_set_diff"),
            "resource_adjusted_csv_task_eval_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_csv_task_eval/"
                "tracking_g1_resource_adjusted_csv_task_eval_audit.json"
            ),
            "resource_adjusted_csv_task_eval_status": csv_task.get("status"),
            "resource_adjusted_train_entry_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
                "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
            ),
            "resource_adjusted_train_entry_status": train_entry.get("status"),
            "resource_adjusted_ppo_training_run_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_ppo_training_run/"
                "tracking_g1_resource_adjusted_ppo_training_run.json"
            ),
            "resource_adjusted_ppo_training_run_status": ppo_harness.get("status"),
            "resource_adjusted_ppo_training_run_attempted": ppo_harness.get("run", {}).get("attempted_training"),
            "resource_adjusted_ppo_gpu_resource_ready": ppo_harness.get("gpu_preflight", {}).get("resource_ready"),
            "resource_adjusted_ppo_checkpoint_eval_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
                "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
            ),
            "resource_adjusted_ppo_checkpoint_eval_status": load_json(
                "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
                "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
            ).get("status"),
            "resource_adjusted_evidence_does_not_clear_official_gate": True,
        },
        (
            "Continue official G1 USD conversion recovery or produce a separately audited offline physical USD "
            "converter scaffold. The official replay_npz entry diagnostic shows the entry reaches AppLauncher but "
            "blocks in the official URDF converter layer-save path before artifact download. Runtime-patched "
            "official csv_to_npz.py and replay_npz.py loops now complete with the enriched USD, which narrows the "
            "blocker to the unpatched official converter/output path rather than the loop bodies themselves. The URDF "
            "source-equivalence audit can justify action-joint alignment only; do not report resource-adjusted CSV "
            "loop/replay/task/train-entry gates as unpatched official converter output or paper-level PPO/tracking "
            "performance."
        ),
    )


def audit_isaaclab_vulkan_gate() -> dict[str, Any]:
    live = load_json("res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json")
    vulkan = load_json("res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json")
    cuda_p2p = load_json("res/setup/cuda_p2p_runtime_probe/cuda_p2p_runtime_probe.json")
    blocker = live.get("current_blocker")
    app_ok = bool(live.get("checks", {}).get("app_launcher_reached_success_sentinel"))
    p2p_warning = bool(live.get("checks", {}).get("cuda_p2p_iommu_runtime_warning_retained"))
    status = (
        "clear_with_runtime_warning"
        if app_ok and p2p_warning
        else "clear"
        if app_ok
        else ("blocked" if blocker in {"vulkan_incompatible_driver", "cuda_p2p_iommu_validation"} else "needs_review")
    )
    return make_gate(
        "isaaclab_kit_vulkan_cuda_runtime",
        status,
        [
            "IsaacLab AppLauncher success sentinel",
            "official whole_body_tracking replay_npz.py live replay",
            "tracking task smoke/evaluation inside Kit",
            "PPO motion-tracking training/evaluation",
            "closed-loop VAE/diffusion rollout evaluation",
        ],
        {
            "live_gate_probe_json": str(ROOT / "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json"),
            "live_gate_status": live.get("status"),
            "current_blocker": blocker,
            "app_launcher_reached_success_sentinel": app_ok,
            "cuda_p2p_iommu_runtime_warning_retained": p2p_warning,
            "vulkaninfo_path": live.get("host", {}).get("vulkaninfo_path"),
            "max_user_watches": live.get("host", {}).get("max_user_watches"),
            "max_user_instances": live.get("host", {}).get("max_user_instances"),
            "probe_logs": live.get("outputs", {}).get("log_dir"),
            "vulkan_runtime_probe_json": str(ROOT / "res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json"),
            "vulkan_runtime_status": vulkan.get("status"),
            "system_loader_create_instance_ok": vulkan.get("checks", {}).get("system_loader_create_instance_ok"),
            "isaac_bundled_loader_create_instance_ok": vulkan.get("checks", {}).get(
                "isaac_bundled_loader_create_instance_ok"
            ),
            "project_egl_icd_removes_vulkan_error": live.get("checks", {}).get("project_egl_icd_removes_vulkan_error"),
            "single_gpu_renderer_limits_active_gpu": live.get("checks", {}).get("single_gpu_renderer_limits_active_gpu"),
            "cuda_visible_devices_single_gpu_not_viable": live.get("checks", {}).get(
                "cuda_visible_devices_single_gpu_not_viable"
            ),
            "cuda_p2p_runtime_probe_json": str(ROOT / "res/setup/cuda_p2p_runtime_probe/cuda_p2p_runtime_probe.json"),
            "cuda_p2p_runtime_has_already_enabled_signature": cuda_p2p.get("checks", {}).get(
                "has_peer_access_already_enabled_signature"
            ),
        },
        (
            "If clear_with_runtime_warning, proceed only to official replay/smoke while monitoring retained CUDA "
            "P2P/IOMMU warnings; do not start PPO, DAgger, or closed-loop paper experiments until replay/task smoke "
            "artifacts pass."
        ),
    )


def audit_ros_gate() -> dict[str, Any]:
    release = os_release()
    ros2_path = shutil.which("ros2")
    colcon_path = shutil.which("colcon")
    readme = ROOT / "reproduction/third_party/official/motion_tracking_controller/README.md"
    readme_text = read_text(readme)
    mentions_jazzy = "ROS 2 Jazzy" in readme_text
    mentions_colcon = "colcon build" in readme_text
    host_ubuntu = release.get("VERSION_ID", "")
    status = "blocked" if not ros2_path or host_ubuntu != "24.04" else "needs_review"
    return make_gate(
        "ros2_jazzy_noble_controller",
        status,
        [
            "MuJoCo sim-to-sim launch from motion_tracking_controller",
            "real.launch.py deployment path",
            "ROS bag recording/evaluation through official deployment package",
        ],
        {
            "host_pretty_name": release.get("PRETTY_NAME", ""),
            "host_version_id": host_ubuntu,
            "ros2_path": ros2_path,
            "colcon_path": colcon_path,
            "official_readme": str(readme),
            "official_readme_mentions_ros2_jazzy": mentions_jazzy,
            "official_readme_mentions_colcon": mentions_colcon,
            "mujoco_launch_exists": file_exists("reproduction/third_party/official/motion_tracking_controller/launch/mujoco.launch.py"),
            "real_launch_exists": file_exists("reproduction/third_party/official/motion_tracking_controller/launch/real.launch.py"),
        },
        (
            "Use an Ubuntu Noble / ROS 2 Jazzy workspace with legged_control2, unitree_bringup, "
            "unitree_description, and mujoco_sim_ros2 before executing official MuJoCo or real launch files."
        ),
    )


def audit_hardware_gate() -> dict[str, Any]:
    return make_gate(
        "unitree_g1_hardware",
        "out_of_scope",
        ["real robot deployment and hardware robustness claims"],
        {
            "real_robot_declared_out_of_scope_doc": str(ROOT / "reproduction/docs/known_limitations.md"),
            "known_limitations_mentions_no_unitree_g1": "no Unitree G1 hardware" in read_text(ROOT / "reproduction/docs/known_limitations.md"),
            "real_launch_exists": file_exists("reproduction/third_party/official/motion_tracking_controller/launch/real.launch.py"),
        },
        "Do not run real-robot launch files on this machine; obtain explicit hardware access and safety procedure first.",
    )


def audit_level_c_artifact_gate() -> dict[str, Any]:
    artifact = load_json("res/level_c/official_artifact_audit/level_c_official_artifact_audit.json")
    conclusion = artifact.get("conclusion", {})
    code_found = bool(conclusion.get("official_beyondmimic_vae_diffusion_code_found"))
    checkpoint_found = bool(conclusion.get("official_beyondmimic_checkpoint_or_engine_found"))
    status = "blocked" if not code_found and not checkpoint_found else "needs_review"
    return make_gate(
        "official_level_c_artifacts",
        status,
        [
            "paper-level VAE training reproduction",
            "paper-level diffusion training reproduction",
            "trained checkpoint evaluation",
            "TensorRT/deployment reproduction for Level C",
        ],
        {
            "artifact_audit_json": str(ROOT / "res/level_c/official_artifact_audit/level_c_official_artifact_audit.json"),
            "audit_status": artifact.get("status"),
            "total_files_scanned": artifact.get("total_files_scanned"),
            "official_beyondmimic_specific_candidates": artifact.get("counts", {}).get("official_beyondmimic_specific_candidates"),
            "official_checkpoint_or_deployment_model_candidates": artifact.get("counts", {}).get("official_checkpoint_or_deployment_model_candidates"),
            "official_beyondmimic_vae_diffusion_code_found": code_found,
            "official_beyondmimic_checkpoint_or_engine_found": checkpoint_found,
            "reference_diffusion_code_present": bool(conclusion.get("reference_diffusion_code_present")),
        },
        (
            "Use released official Level C code/checkpoints if they become available, or continue only as a clearly "
            "marked clean-room/debug reimplementation until real rollout data and checkpoints exist."
        ),
    )


def audit_fig56_gate() -> dict[str, Any]:
    fig = load_json("res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json")
    conclusion = fig.get("conclusion", {})
    possible = bool(conclusion.get("fig5_fig6_paper_reproduction_possible_from_current_local_artifacts"))
    status = "blocked" if not possible else "needs_review"
    return make_gate(
        "fig5_fig6_paper_results",
        status,
        [
            "Figure 5 paper reproduction",
            "Figure 6 paper reproduction",
            "joystick/inpainting/SDF/latent result claims beyond debug mechanics",
        ],
        {
            "fig5_fig6_audit_json": str(ROOT / "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"),
            "audit_status": fig.get("status"),
            "panels_audited": fig.get("counts", {}).get("panels_audited"),
            "paper_pdf_figures_present": fig.get("counts", {}).get("paper_pdf_figures_present"),
            "released_data_match_count": fig.get("counts", {}).get("released_data_match_count"),
            "panels_blocked_for_paper_reproduction": fig.get("counts", {}).get("panels_blocked_for_paper_reproduction"),
            "debug_mechanism_evidence_available": bool(conclusion.get("debug_mechanism_evidence_available")),
            "paper_reproduction_possible": possible,
        },
        (
            "Generate or obtain trained VAE/diffusion checkpoints, state-latent rollout logs, closed-loop task logs, "
            "and figure-specific data before claiming Fig. 5/Fig. 6 reproduction."
        ),
    )


def audit_long_training_gate() -> dict[str, Any]:
    proc = command_output(["pgrep", "-af", "BeyondMimic|whole_body_tracking|rsl_rl|level_c|train.py|csv_to_npz.py|replay_npz.py"])
    current_pid = str(os.getpid())
    lines = [
        line
        for line in proc[1].splitlines()
        if line.strip()
        and current_pid not in line.split()
        and str(ROOT) in line
        and "blocked_gate_audit.py" not in line
        and "pgrep -af" not in line
    ]
    status = "clear" if not lines else "needs_review"
    return make_gate(
        "long_training_safety_gate",
        status,
        ["unexpected long PPO/VAE/diffusion training before smoke gates"],
        {
            "pgrep_return_code": proc[0],
            "matching_processes": lines,
            "no_matching_training_processes": not lines,
        },
        "Keep long training disabled until smoke gates pass; investigate any listed process before starting new training.",
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["gate_id", "status", "blocks", "next_action", "evidence_json"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "gate_id": row["gate_id"],
                    "status": row["status"],
                    "blocks": "; ".join(row["blocks"]),
                    "next_action": row["next_action"],
                    "evidence_json": json.dumps(row["evidence"], sort_keys=True),
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gates = [
        audit_inotify_gate(),
        audit_isaaclab_vulkan_gate(),
        audit_official_g1_usd_conversion_replay_gate(),
        audit_ros_gate(),
        audit_hardware_gate(),
        audit_level_c_artifact_gate(),
        audit_fig56_gate(),
        audit_long_training_gate(),
    ]
    counts: dict[str, int] = {}
    for gate in gates:
        counts[gate["status"]] = counts.get(gate["status"], 0) + 1

    json_path = OUT / "blocked_gate_audit.json"
    tsv_path = OUT / "blocked_gate_audit.tsv"
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "audit",
        "scope": "external/system gates that prevent complete BeyondMimic reproduction on the current host",
        "gate_count": len(gates),
        "gate_status_counts": dict(sorted(counts.items())),
        "gates": gates,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Current evidence still has blocked/out-of-scope gates for official G1 USD conversion/replay, official "
                "deployment runtime, real hardware, official Level C artifacts, and paper-level Fig. 5/Fig. 6 results. "
                "The older inotify failure is retained as historical evidence, but the current live headless gate has "
                "passed with runtime warnings."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, gates)
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
