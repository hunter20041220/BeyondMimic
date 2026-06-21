#!/usr/bin/env python3
"""Master evidence audit for the current BeyondMimic reproduction state."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Tuple


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/master_audit"


JsonCheck = Callable[[Dict[str, Any]], Tuple[bool, str]]


def status_ok(data: dict[str, Any]) -> tuple[bool, str]:
    return data.get("status") == "ok", f"status={data.get('status')!r}"


def json_path(path: str) -> Path:
    return ROOT / path


def has_file(path: Path) -> tuple[bool, str]:
    if path.exists() and path.is_file() and path.stat().st_size > 0:
        return True, f"size={path.stat().st_size}"
    return False, "missing_or_empty"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_counts() -> tuple[Counter[str], list[dict[str, str]]]:
    path = ROOT / "reproduction/docs/completion_matrix.md"
    counts: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 3 or cols[0] == "Requirement":
            continue
        status = cols[1]
        rows.append({"requirement": cols[0], "status": status, "evidence": cols[2]})
        if status in {"complete", "partial", "blocked", "pending", "out_of_scope"}:
            counts[status] += 1
    return counts, rows


def check_json_artifact(name: str, rel_path: str, checks: list[JsonCheck]) -> dict[str, Any]:
    path = json_path(rel_path)
    exists, detail = has_file(path)
    result: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": exists,
        "passed": False,
        "details": [detail],
    }
    if not exists:
        return result
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001 - audit should record parse failure.
        result["details"].append(f"json_parse_error={exc}")
        return result
    passed = True
    for check in checks:
        ok, msg = check(data)
        result["details"].append(msg)
        passed = passed and ok
    result["passed"] = passed
    return result


def check_file_artifact(name: str, rel_path: str) -> dict[str, Any]:
    path = json_path(rel_path)
    exists, detail = has_file(path)
    return {"name": name, "path": str(path), "exists": exists, "passed": exists, "details": [detail]}


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["name", "path", "exists", "passed", "details"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "path": row["path"],
                    "exists": row["exists"],
                    "passed": row["passed"],
                    "details": "; ".join(str(x) for x in row["details"]),
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    artifacts.extend(
        [
            check_file_artifact("local_inventory", "reproduction/docs/local_inventory.tsv"),
            check_file_artifact("released_figure_summary", "res/released_figures/released_figure_summary.tsv"),
            check_file_artifact("paper_panel_map", "reproduction/docs/paper_panel_map.tsv"),
            check_json_artifact(
                "bm_diffusion_env_audit",
                "res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["prefix_exists"], "bm_diffusion_prefix_exists"),
                    lambda d: (d["checks"]["lock_files_exist"], "bm_diffusion_lock_files_exist"),
                    lambda d: (
                        d["checks"]["base_numpy_scipy_yaml_tqdm_smoke_passes"],
                        "bm_diffusion_base_scientific_stack",
                    ),
                    lambda d: (d["checks"]["torch_cuda_smoke_passes"], "bm_diffusion_torch_cuda_smoke"),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_paper_results"],
                        "bm_diffusion_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "bm_diffusion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "takeover_audit",
                "res/takeover_audit/takeover_audit.json",
                [
                    lambda d: (
                        d.get("status") in {"ok", "ok_with_runtime_warnings"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["download_present"], "takeover_download_present"),
                    lambda d: (d["checks"]["other_backup_present"], "takeover_other_backup_present"),
                    lambda d: (d["checks"]["workspace_promoted"], "takeover_workspace_promoted"),
                    lambda d: (d["checks"]["old_root_text_paths_absent"], "takeover_old_root_paths_absent"),
                    lambda d: (d["checks"]["json_artifacts_readable"], "takeover_json_artifacts_readable"),
                    lambda d: (d["checks"]["smoke_scripts_compile"], "takeover_smoke_scripts_compile"),
                    lambda d: (d["checks"]["training_started"] is False, "takeover_no_training_started"),
                ],
            ),
            check_json_artifact(
                "env_import_probe",
                "res/setup/env_probe/env_import_probe.json",
                [
                    lambda d: (
                        d.get("status") in {"ok", "ok_with_live_kit_warning", "ok_with_runtime_warning", "partial_blocked"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["analysis_imports_ok"], "env_probe_analysis_imports"),
                    lambda d: (
                        d["checks"]["diffusion_torch_cuda_visible_devices_5_6_ok"],
                        "env_probe_diffusion_cuda_devices_5_6",
                    ),
                    lambda d: (d["checks"]["tracking_basic_imports_ok"], "env_probe_tracking_basic_imports"),
                    lambda d: (d["checks"]["isaaclab_import_ok"] is True, "env_probe_isaaclab_import_restored"),
                    lambda d: (
                        d["checks"].get("isaaclab_live_headless_gate_ok") is True,
                        "env_probe_live_kit_gate_currently_passed",
                    ),
                    lambda d: (d["checks"]["training_started"] is False, "env_probe_no_training_started"),
                ],
            ),
            check_json_artifact(
                "vulkan_runtime_probe",
                "res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json",
                [
                    lambda d: (d.get("status") in {"ok", "blocked"}, f"status={d.get('status')!r}"),
                    lambda d: (d["checks"]["nvidia_icd_json_exists"], "vulkan_probe_nvidia_icd_exists"),
                    lambda d: (d["checks"]["libglx_nvidia_resolves"], "vulkan_probe_libglx_resolves"),
                    lambda d: (d["checks"]["does_not_launch_kit_or_training"], "vulkan_probe_no_kit_training"),
                    lambda d: (
                        d["checks"]["does_not_claim_isaaclab_gate_passed"],
                        "vulkan_probe_no_gate_pass_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vulkan_probe_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "cuda_p2p_runtime_probe",
                "res/setup/cuda_p2p_runtime_probe/cuda_p2p_runtime_probe.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["nvidia_smi_ok"], "cuda_p2p_nvidia_smi_ok"),
                    lambda d: (d["checks"]["records_peer_access_results"], "cuda_p2p_records_results"),
                    lambda d: (d["checks"]["does_not_launch_kit_or_training"], "cuda_p2p_no_kit_training"),
                    lambda d: (
                        d["checks"]["does_not_claim_isaaclab_gate_passed"],
                        "cuda_p2p_no_gate_pass_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "cuda_p2p_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "isaaclab_gpu_foundation_settings_audit",
                "res/setup/isaaclab_gpu_foundation_settings_audit/isaaclab_gpu_foundation_settings_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["settings_surface_search_ran"], "gpu_foundation_settings_search_ran"),
                    lambda d: (
                        d["checks"]["project_egl_icd_removes_vulkan_error"],
                        "gpu_foundation_settings_egl_icd_effective",
                    ),
                    lambda d: (
                        d["checks"]["single_gpu_renderer_limits_active_gpu"],
                        "gpu_foundation_settings_single_gpu_effective",
                    ),
                    lambda d: (
                        d["checks"]["app_launcher_gate_clear_or_warning"],
                        "gpu_foundation_settings_gate_clear_or_warning_recorded",
                    ),
                    lambda d: (
                        d["checks"]["cuda_p2p_iommu_runtime_warning_retained"],
                        "gpu_foundation_settings_p2p_warning_retained",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "gpu_foundation_settings_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "isaaclab_live_gate_probe",
                "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json",
                [
                    lambda d: (
                        d.get("status") in {"ok", "blocked", "ok_with_runtime_warning"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["tracking_python_exists"], "isaaclab_live_tracking_python_exists"),
                    lambda d: (d["checks"]["package_import_probe_ok"], "isaaclab_live_package_import_ok"),
                    lambda d: (
                        d["checks"]["no_training_started"],
                        "isaaclab_live_no_training_started",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tracking_reproduction_complete"],
                        "isaaclab_live_no_tracking_completion_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "isaaclab_live_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "isaaclab_current_headless_gate",
                "res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["input_checks"]["tracking_python_exists"]
                        and d["input_checks"]["project_egl_icd_exists"]
                        and d["input_checks"]["gpu_foundation_deps_exists"],
                        "isaaclab_current_gate_inputs_exist",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] == 4
                        and d["config"]["device"] == "cuda:4"
                        and d["config"]["cuda_visible_devices"] == "",
                        "isaaclab_current_gate_uses_physical_gpu4_without_cuda_visible_devices",
                    ),
                    lambda d: (
                        d["run"]["attempted"]
                        and d["run"]["returncode"] == 0
                        and d["checks"]["app_launcher_headless_success_sentinel"]
                        and d["checks"]["payload_is_running"],
                        "isaaclab_current_gate_success_sentinel",
                    ),
                    lambda d: (d["checks"]["no_fatal_runtime_error"], "isaaclab_current_gate_no_fatal_runtime_error"),
                    lambda d: (d["checks"]["no_training_started"], "isaaclab_current_gate_no_training_started"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "isaaclab_current_gate_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_replay_preflight",
                "res/tracking/official_replay_preflight/tracking_official_replay_preflight.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["official_csv_to_npz_exists"], "official_replay_preflight_csv_to_npz"),
                    lambda d: (d["checks"]["official_replay_npz_exists"], "official_replay_preflight_replay_npz"),
                    lambda d: (d["checks"]["motion_csv_has_36_columns"], "official_replay_preflight_csv_width"),
                    lambda d: (d["checks"]["live_gate_allows_replay_preflight"], "official_replay_preflight_live_gate"),
                    lambda d: (
                        d["checks"]["does_not_execute_csv_to_npz_or_replay"],
                        "official_replay_preflight_no_replay_execution",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_replay_preflight_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_replay_conversion_audit",
                "res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json",
                [
                    lambda d: (
                        d.get("status") in {"ok", "ok_with_blocked_conversion"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["rsl_rl_env_import_ok"], "official_replay_conversion_rsl_rl_import"),
                    lambda d: (d["checks"]["tracking_pip_check_ok"], "official_replay_conversion_pip_check"),
                    lambda d: (d["checks"]["system_libglu_available"], "official_replay_conversion_libglu"),
                    lambda d: (d["checks"]["attempt_logs_present"], "official_replay_conversion_logs_present"),
                    lambda d: (d["checks"]["usd_save_blocker_recorded"], "official_replay_conversion_usd_blocker"),
                    lambda d: (
                        d["checks"]["urdf_converter_empty_usd_recorded"],
                        "official_replay_conversion_empty_usd_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_replay_success"],
                        "official_replay_conversion_no_false_success_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_replay_conversion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_replay_npz_entry_diagnostic",
                "res/tracking/official_replay_npz_entry_diagnostic/"
                "tracking_official_replay_npz_entry_diagnostic_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_official_replay_npz_entry_blocker",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d.get("latest_blocker") == "official_urdf_converter_layer_save_blocked",
                        f"latest_blocker={d.get('latest_blocker')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_replay_script_exists"],
                        "official_replay_npz_entry_script_exists",
                    ),
                    lambda d: (
                        d["checks"]["app_launcher_constructed"],
                        "official_replay_npz_entry_app_launcher",
                    ),
                    lambda d: (
                        d["run"]["markers"]["permission_to_save_false"],
                        "official_replay_npz_entry_save_permission_blocker",
                    ),
                    lambda d: (
                        d["run"]["markers"]["failed_to_save_layer"],
                        "official_replay_npz_entry_layer_save_blocker",
                    ),
                    lambda d: (
                        d["run"]["markers"]["empty_robot_after_converter"],
                        "official_replay_npz_entry_empty_robot_recorded",
                    ),
                    lambda d: (
                        Path(d["outputs"]["failed_log_copy"]).is_file(),
                        "official_replay_npz_entry_failed_log_retained",
                    ),
                    lambda d: (
                        Path(d["outputs"]["probe"]).is_file(),
                        "official_replay_npz_entry_probe_retained",
                    ),
                    lambda d: (
                        d["checks"]["fake_wandb_download_seen"] is False,
                        "official_replay_npz_entry_blocked_before_artifact_download",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_replay"],
                        "official_replay_npz_entry_no_paper_replay_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "official_replay_npz_entry_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_replay_npz_entry_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_replay_npz_loop_with_enriched_usd",
                "res/tracking/official_replay_npz_loop_with_enriched_usd/"
                "tracking_official_replay_npz_loop_with_enriched_usd_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_replay_loop_with_enriched_usd_patch",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_replay_script_exists"],
                        "official_replay_loop_script_exists",
                    ),
                    lambda d: (
                        d["checks"]["app_launcher_constructed"],
                        "official_replay_loop_app_launcher",
                    ),
                    lambda d: (
                        d["checks"]["g1_cfg_patched_to_enriched_usd"],
                        "official_replay_loop_uses_enriched_usd_patch",
                    ),
                    lambda d: (
                        d["checks"]["fake_wandb_download_seen"],
                        "official_replay_loop_fake_artifact_download",
                    ),
                    lambda d: (
                        d["checks"]["official_loop_call_299_seen"],
                        "official_replay_loop_call_299",
                    ),
                    lambda d: (
                        d["checks"]["official_loop_complete_seen"],
                        "official_replay_loop_complete_299",
                    ),
                    lambda d: (
                        d["checks"].get(
                            "process_returned_zero_or_forced_after_success_sentinel",
                            d["checks"].get("process_returned_zero"),
                        ),
                        "official_replay_loop_returned_or_forced_after_success_sentinel",
                    ),
                    lambda d: (d["checks"]["does_not_start_training"], "official_replay_loop_no_training"),
                    lambda d: (
                        d["checks"]["does_not_claim_resource_adjusted_asset_is_official_converter_output"],
                        "official_replay_loop_no_asset_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_resource_adjusted_motion_is_official_csv_to_npz_output"],
                        "official_replay_loop_no_motion_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_replay"],
                        "official_replay_loop_no_paper_replay_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_replay_loop_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_csv_to_npz_loop_with_enriched_usd",
                "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_to_npz_loop_with_enriched_usd_patch",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_to_npz_script_exists"],
                        "official_csv_loop_script_exists",
                    ),
                    lambda d: (
                        d["checks"]["app_launcher_constructed"],
                        "official_csv_loop_app_launcher",
                    ),
                    lambda d: (
                        d["checks"]["g1_cfg_patched_to_enriched_usd"],
                        "official_csv_loop_uses_enriched_usd_patch",
                    ),
                    lambda d: (
                        d["checks"]["motion_loaded"] and d["checks"]["motion_interpolated"],
                        "official_csv_loop_motion_loaded_and_interpolated",
                    ),
                    lambda d: (
                        d["checks"]["official_loop_call_299_seen"],
                        "official_csv_loop_call_299",
                    ),
                    lambda d: (
                        d["checks"]["official_loop_complete_seen"],
                        "official_csv_loop_complete_299",
                    ),
                    lambda d: (
                        d["checks"]["np_savez_redirect_seen"],
                        "official_csv_loop_project_output_redirect",
                    ),
                    lambda d: (
                        d["checks"]["fake_wandb_log_artifact_seen"],
                        "official_csv_loop_fake_wandb_log",
                    ),
                    lambda d: (
                        d["checks"]["motion_npz_written"],
                        "official_csv_loop_motion_npz_written",
                    ),
                    lambda d: (
                        d["checks"]["joint_pos_shape_299_29"],
                        "official_csv_loop_joint_shape",
                    ),
                    lambda d: (
                        d["checks"]["body_pos_shape_299_40_3"],
                        "official_csv_loop_body_shape",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_unpatched_official_asset_complete"],
                        "official_csv_loop_no_unpatched_asset_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "official_csv_loop_no_paper_replay_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "official_csv_loop_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_csv_loop_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd",
                "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_to_npz_loop_full_dataset_with_enriched_usd",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_40_csvs_selected"] and d["aggregate"]["row_count"] == 40,
                        "official_csv_full_dataset_40_motions",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_ok"] and d["aggregate"]["failed_count"] == 0,
                        "official_csv_full_dataset_all_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_joint_shapes_299_29"]
                        and d["checks"]["all_body_shapes_299_40"]
                        and d["aggregate"]["total_frames"] == 11960,
                        "official_csv_full_dataset_shapes",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_csv_to_npz_loop"]
                        and d["checks"]["uses_resource_adjusted_usd"],
                        "official_csv_full_dataset_source_scope",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_unpatched_official_asset_complete"]
                        and d["checks"]["does_not_claim_paper_level_replay"]
                        and d["checks"]["does_not_start_training"],
                        "official_csv_full_dataset_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_csv_full_dataset_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd",
                "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
                "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_replay_npz_loop_full_dataset_with_enriched_usd",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_40_csv_loop_outputs_selected"] and d["aggregate"]["row_count"] == 40,
                        "official_replay_full_dataset_40_motions",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_ok"] and d["aggregate"]["failed_count"] == 0,
                        "official_replay_full_dataset_all_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_reached_official_loop_299"]
                        and d["aggregate"]["total_replayed_steps"] == 11960,
                        "official_replay_full_dataset_steps",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_replay_npz_loop"]
                        and d["checks"]["uses_official_csv_loop_npz_inputs"]
                        and d["checks"]["uses_resource_adjusted_usd"],
                        "official_replay_full_dataset_source_scope",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_unpatched_official_asset_complete"]
                        and d["checks"]["does_not_claim_trained_policy_eval"]
                        and d["checks"]["does_not_claim_paper_level_replay"]
                        and d["checks"]["does_not_start_training"],
                        "official_replay_full_dataset_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_replay_full_dataset_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_csv_loop_full_dataset_task_eval",
                "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
                "tracking_g1_official_csv_loop_full_dataset_task_eval.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_dataset_task_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_40_motion_inputs_selected"] and d["aggregate"]["row_count"] == 40,
                        "official_csv_task_full_dataset_40_motions",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_ok"] and d["aggregate"]["failed_count"] == 0,
                        "official_csv_task_full_dataset_all_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_step_299"] and d["aggregate"]["total_steps"] == 11960,
                        "official_csv_task_full_dataset_steps",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_action_dim_29"]
                        and d["checks"]["all_rows_policy_obs_dim_160"]
                        and d["checks"]["all_rows_critic_obs_dim_286"],
                        "official_csv_task_full_dataset_obs_action_contract",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_reward_terms_9"]
                        and d["checks"]["all_rows_termination_terms_4"]
                        and d["checks"]["all_rows_robot_contract_29j_40b"],
                        "official_csv_task_full_dataset_reward_termination_robot_contract",
                    ),
                    lambda d: (
                        d["checks"]["csv_full_dataset_audit_passed"]
                        and d["checks"]["replay_full_dataset_audit_passed"]
                        and d["checks"]["uses_official_csv_loop_npz_inputs"]
                        and d["checks"]["uses_resource_adjusted_usd"],
                        "official_csv_task_full_dataset_source_scope",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_unpatched_official_asset_complete"]
                        and d["checks"]["does_not_claim_trained_policy_eval"]
                        and d["checks"]["does_not_start_training"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_csv_task_full_dataset_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False
                        and d["interpretation"]["paper_level_tracking_eval_complete"] is False,
                        "official_csv_task_full_dataset_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_task_smoke",
                "res/tracking/g1_official_importer_export_task_smoke/"
                "tracking_g1_official_importer_export_task_smoke.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_task_smoke",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["official_importer_usd_exists"], "official_importer_smoke_usd_exists"),
                    lambda d: (d["checks"]["export_structure_audit_passed"], "official_importer_smoke_structure_audit"),
                    lambda d: (d["checks"]["process_returned_zero"], "official_importer_smoke_process_zero"),
                    lambda d: (d["checks"]["env_created"] and d["checks"]["env_reset"], "official_importer_smoke_reset"),
                    lambda d: (d["checks"]["env_step_final"], "official_importer_smoke_steps"),
                    lambda d: (
                        d["checks"]["action_dim_29"]
                        and d["checks"]["policy_observation_dim_160"]
                        and d["checks"]["critic_observation_dim_286"],
                        "official_importer_smoke_obs_action_contract",
                    ),
                    lambda d: (
                        d["checks"]["reward_terms_9"]
                        and d["checks"]["termination_terms_4"]
                        and d["checks"]["robot_joint_count_29"]
                        and d["checks"]["robot_body_count_40"],
                        "official_importer_smoke_task_contract",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["does_not_claim_resource_adjusted_enriched_usd"],
                        "official_importer_smoke_uses_export_not_enriched_scaffold",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"]
                        and d["checks"]["does_not_start_training"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_smoke_no_overclaim",
                    ),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "official_importer_smoke_goal_open"),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_full_dataset_task_eval",
                "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
                "tracking_g1_official_importer_export_full_dataset_task_eval.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_full_dataset_task_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_40_motion_inputs_selected"] and d["aggregate"]["row_count"] == 40,
                        "official_importer_full_dataset_40_motions",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_ok"] and d["aggregate"]["failed_count"] == 0,
                        "official_importer_full_dataset_all_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_step_299"] and d["aggregate"]["total_steps"] == 11960,
                        "official_importer_full_dataset_steps",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_action_dim_29"]
                        and d["checks"]["all_rows_policy_obs_dim_160"]
                        and d["checks"]["all_rows_critic_obs_dim_286"],
                        "official_importer_full_dataset_obs_action_contract",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_reward_terms_9"]
                        and d["checks"]["all_rows_termination_terms_4"]
                        and d["checks"]["all_rows_robot_contract_29j_40b"],
                        "official_importer_full_dataset_task_contract",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_usd_exists"]
                        and d["checks"]["export_structure_audit_passed"]
                        and d["checks"]["all_rows_use_official_importer_export_usd"]
                        and d["checks"]["no_rows_use_resource_adjusted_enriched_usd"],
                        "official_importer_full_dataset_uses_export_not_enriched_scaffold",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_csv_loop_npz_inputs"]
                        and d["checks"]["does_not_claim_unpatched_official_asset_complete"],
                        "official_importer_full_dataset_source_scope",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_policy_eval"]
                        and d["checks"]["does_not_start_training"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_full_dataset_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False
                        and d["interpretation"]["paper_level_tracking_eval_complete"] is False,
                        "official_importer_full_dataset_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_tracking_eval_summary_assets",
                "res/report_assets/official_importer_export_tracking_eval_summary/"
                "official_importer_export_tracking_eval_summary_assets.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_tracking_eval_summary_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["task_eval_status_ok"] and d["checks"]["task_eval_40_of_40_ok"],
                        "tracking_eval_summary_task_eval_ok",
                    ),
                    lambda d: (
                        d["checks"]["scaled_ppo_eval_status_ok"]
                        and d["checks"]["scaled_ppo_uses_official_importer_export_usd"],
                        "tracking_eval_summary_scaled_ppo_eval_ok",
                    ),
                    lambda d: (
                        d["checks"]["policy_video_status_ok"] and d["checks"]["policy_video_exists"],
                        "tracking_eval_summary_policy_video_ok",
                    ),
                    lambda d: (d["checks"]["all_report_assets_exist"], "tracking_eval_summary_assets_exist"),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_tracking"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "tracking_eval_summary_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_import_config_variant_probe",
                "res/tracking/g1_urdf_import_config_variant_probe/"
                "tracking_g1_urdf_import_config_variant_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_import_config_surface_recorded_and_variants_blocked",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["method_app_reached"],
                        "g1_urdf_import_config_method_app_reached",
                    ),
                    lambda d: (
                        d["checks"]["import_config_methods_recorded"],
                        "g1_urdf_import_config_methods_recorded",
                    ),
                    lambda d: (
                        d["method_probe"]["payload"]["has_set_make_instanceable"] is False,
                        "g1_urdf_import_config_no_make_instanceable_setter",
                    ),
                    lambda d: (
                        d["method_probe"]["payload"]["has_set_instanceable_usd_path"] is False,
                        "g1_urdf_import_config_no_instanceable_usd_path_setter",
                    ),
                    lambda d: (
                        d["checks"]["variant_converter_attempted"],
                        "g1_urdf_import_config_converter_attempted",
                    ),
                    lambda d: (
                        d["variant_summary"]["variant_baseline_make_instanceable_false"]["usd"]["stage_open_ok"]
                        is True,
                        "g1_urdf_import_config_baseline_stage_opened",
                    ),
                    lambda d: (
                        d["variant_summary"]["variant_baseline_make_instanceable_false"]["usd"]["prim_count"] == 0,
                        "g1_urdf_import_config_baseline_empty_usd_recorded",
                    ),
                    lambda d: (
                        d["checks"]["no_valid_robotish_usd_claim"],
                        "g1_urdf_import_config_no_valid_robotish_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_urdf_import_config_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_enriched_usd_replay_preflight_audit",
                "res/tracking/g1_enriched_usd_replay_preflight/tracking_g1_enriched_usd_replay_preflight_audit.json",
                [
                    lambda d: (
                        d.get("status")
                        in {
                            "ok_resource_adjusted_preflight_passed",
                            "ok_resource_adjusted_step_gate_passed_with_explicit_exit",
                            "ok_with_resource_adjusted_step_gate_passed_shutdown_timeout",
                        },
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["enriched_usd_readback_ok"],
                        "g1_enriched_replay_preflight_enriched_usd_readback",
                    ),
                    lambda d: (
                        d["checks"]["sim_context_reached"],
                        "g1_enriched_replay_preflight_sim_context",
                    ),
                    lambda d: (
                        d["checks"]["scene_creation_reached"],
                        "g1_enriched_replay_preflight_scene_created",
                    ),
                    lambda d: (
                        d["checks"]["robot_contract_reached"],
                        "g1_enriched_replay_preflight_robot_contract",
                    ),
                    lambda d: (
                        d["checks"]["render_step_reached"],
                        "g1_enriched_replay_preflight_render_step",
                    ),
                    lambda d: (
                        d["checks"]["resource_adjusted_step_gate_passed"],
                        "g1_enriched_replay_preflight_step_gate",
                    ),
                    lambda d: (
                        d["checks"]["explicit_exit_after_success"] and not d["checks"]["clean_kit_shutdown_verified"]
                        or d["checks"]["clean_kit_shutdown_verified"],
                        "g1_enriched_replay_preflight_exit_boundary_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_replay_success"],
                        "g1_enriched_replay_preflight_no_official_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_enriched_replay_preflight_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_enriched_replay_preflight_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_enriched_usd_bounded_replay_metrics_audit",
                "res/tracking/g1_enriched_usd_bounded_replay_metrics/"
                "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_64step_metrics_gate",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["prior_4step_gate_passed"], "g1_enriched_metrics_prior_gate"),
                    lambda d: (d["checks"]["process_returned_zero"], "g1_enriched_metrics_returned_zero"),
                    lambda d: (d["checks"]["step_64_reached"], "g1_enriched_metrics_step_64"),
                    lambda d: (d["checks"]["metrics_file_written"], "g1_enriched_metrics_file_written"),
                    lambda d: (d["checks"]["robot_joint_count_29"], "g1_enriched_metrics_joint_count"),
                    lambda d: (d["checks"]["robot_body_count_40"], "g1_enriched_metrics_body_count"),
                    lambda d: (d["checks"]["executed_steps_64"], "g1_enriched_metrics_executed_steps"),
                    lambda d: (d["checks"]["joint_pos_error_recorded"], "g1_enriched_metrics_joint_error"),
                    lambda d: (d["checks"]["root_state_error_recorded"], "g1_enriched_metrics_root_error"),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_enriched_metrics_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_enriched_metrics_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_enriched_metrics_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_task_smoke_audit",
                "res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_tracking_task_smoke",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["bounded_replay_metrics_gate_passed"], "g1_task_smoke_prior_gate"),
                    lambda d: (d["checks"]["process_returned_zero"], "g1_task_smoke_returned_zero"),
                    lambda d: (d["checks"]["env_created"], "g1_task_smoke_env_created"),
                    lambda d: (d["checks"]["env_reset"], "g1_task_smoke_env_reset"),
                    lambda d: (d["checks"]["env_step_8"], "g1_task_smoke_step_8"),
                    lambda d: (d["checks"]["action_dim_29"], "g1_task_smoke_action_dim"),
                    lambda d: (d["checks"]["policy_observation_dim_160"], "g1_task_smoke_policy_obs"),
                    lambda d: (d["checks"]["critic_observation_dim_286"], "g1_task_smoke_critic_obs"),
                    lambda d: (d["checks"]["reward_terms_9"], "g1_task_smoke_reward_terms"),
                    lambda d: (d["checks"]["termination_terms_4"], "g1_task_smoke_termination_terms"),
                    lambda d: (d["checks"]["robot_joint_count_29"], "g1_task_smoke_joint_count"),
                    lambda d: (d["checks"]["robot_body_count_40"], "g1_task_smoke_body_count"),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_task_smoke_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_task_smoke_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "g1_task_smoke_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_task_smoke_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_multi_fixture_task_eval",
                "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
                "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_multi_fixture_task_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["prior_task_smoke_passed"],
                        "g1_multi_fixture_prior_task_smoke",
                    ),
                    lambda d: (
                        d["checks"]["all_fixture_processes_returned_zero"],
                        "g1_multi_fixture_all_processes_zero",
                    ),
                    lambda d: (
                        d["checks"]["no_fixture_stall_timeout"],
                        "g1_multi_fixture_no_stall_timeout",
                    ),
                    lambda d: (
                        d["checks"]["fixture_count_3"],
                        "g1_multi_fixture_count",
                    ),
                    lambda d: (
                        d["checks"]["fixture_step_299_count_3"],
                        "g1_multi_fixture_all_299_step_sentinels",
                    ),
                    lambda d: (
                        d["checks"]["total_steps_897"],
                        "g1_multi_fixture_total_steps",
                    ),
                    lambda d: (
                        d["checks"]["action_dim_all_29"],
                        "g1_multi_fixture_action_dim",
                    ),
                    lambda d: (
                        d["checks"]["policy_observation_dim_all_160"],
                        "g1_multi_fixture_policy_obs",
                    ),
                    lambda d: (
                        d["checks"]["critic_observation_dim_all_286"],
                        "g1_multi_fixture_critic_obs",
                    ),
                    lambda d: (
                        d["checks"]["reward_terms_all_9"],
                        "g1_multi_fixture_reward_terms",
                    ),
                    lambda d: (
                        d["checks"]["termination_terms_all_4"],
                        "g1_multi_fixture_termination_terms",
                    ),
                    lambda d: (
                        d["checks"]["robot_num_joints_all_29"],
                        "g1_multi_fixture_joint_count",
                    ),
                    lambda d: (
                        d["checks"]["robot_num_bodies_all_40"],
                        "g1_multi_fixture_body_count",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_multi_fixture_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_multi_fixture_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "g1_multi_fixture_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_multi_fixture_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_csv_conversion",
                "res/tracking/g1_resource_adjusted_csv_conversion/"
                "tracking_g1_resource_adjusted_csv_conversion_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_csv_conversion",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["prior_full_fixture_gate_passed"],
                        "g1_csv_conversion_prior_full_fixture_gate",
                    ),
                    lambda d: (
                        d["checks"]["process_returned_zero"],
                        "g1_csv_conversion_process_zero",
                    ),
                    lambda d: (
                        d["checks"]["step_299_reached"],
                        "g1_csv_conversion_step_299",
                    ),
                    lambda d: (
                        d["checks"]["input_frames_180"] and d["checks"]["input_columns_36"],
                        "g1_csv_conversion_input_contract",
                    ),
                    lambda d: (
                        d["checks"]["output_frames_299"],
                        "g1_csv_conversion_output_frames",
                    ),
                    lambda d: (
                        d["checks"]["robot_joints_29"] and d["checks"]["robot_bodies_40"],
                        "g1_csv_conversion_robot_contract",
                    ),
                    lambda d: (
                        d["checks"]["joint_pos_shape_299_29"],
                        "g1_csv_conversion_joint_shape",
                    ),
                    lambda d: (
                        d["checks"]["body_pos_shape_299_40_3"],
                        "g1_csv_conversion_body_shape",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_csv_conversion_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_csv_conversion_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "g1_csv_conversion_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_csv_conversion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_csv_full_replay",
                "res/tracking/g1_resource_adjusted_csv_full_replay/"
                "tracking_g1_resource_adjusted_csv_full_replay_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_csv_full_replay",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["prior_csv_conversion_passed"],
                        "g1_csv_full_replay_prior_conversion",
                    ),
                    lambda d: (
                        d["checks"]["process_returned_zero"],
                        "g1_csv_full_replay_process_zero",
                    ),
                    lambda d: (
                        d["checks"]["step_299_reached"],
                        "g1_csv_full_replay_step_299",
                    ),
                    lambda d: (
                        d["checks"]["executed_steps_299"] and d["checks"]["motion_total_steps_299"],
                        "g1_csv_full_replay_full_steps",
                    ),
                    lambda d: (
                        d["checks"]["robot_joint_count_29"] and d["checks"]["robot_body_count_40"],
                        "g1_csv_full_replay_robot_contract",
                    ),
                    lambda d: (
                        d["checks"]["joint_pos_shape_299_29"],
                        "g1_csv_full_replay_joint_shape",
                    ),
                    lambda d: (
                        d["checks"]["body_pos_shape_299_40_3"],
                        "g1_csv_full_replay_body_shape",
                    ),
                    lambda d: (
                        d["checks"]["joint_pos_error_recorded"] and d["checks"]["root_state_error_recorded"],
                        "g1_csv_full_replay_error_metrics",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_csv_full_replay_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_csv_full_replay_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "g1_csv_full_replay_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_csv_full_replay_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_csv_task_eval",
                "res/tracking/g1_resource_adjusted_csv_task_eval/"
                "tracking_g1_resource_adjusted_csv_task_eval_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_csv_task_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["prior_csv_full_replay_passed"],
                        "g1_csv_task_eval_prior_full_replay",
                    ),
                    lambda d: (
                        d["checks"]["process_returned_zero"],
                        "g1_csv_task_eval_process_zero",
                    ),
                    lambda d: (
                        d["checks"]["step_299_reached"],
                        "g1_csv_task_eval_step_299",
                    ),
                    lambda d: (
                        d["checks"]["step_count_299"],
                        "g1_csv_task_eval_step_count",
                    ),
                    lambda d: (
                        d["checks"]["action_dim_29"],
                        "g1_csv_task_eval_action_dim",
                    ),
                    lambda d: (
                        d["checks"]["policy_observation_dim_160"],
                        "g1_csv_task_eval_policy_obs",
                    ),
                    lambda d: (
                        d["checks"]["critic_observation_dim_286"],
                        "g1_csv_task_eval_critic_obs",
                    ),
                    lambda d: (
                        d["checks"]["reward_terms_9"],
                        "g1_csv_task_eval_reward_terms",
                    ),
                    lambda d: (
                        d["checks"]["termination_terms_4"],
                        "g1_csv_task_eval_termination_terms",
                    ),
                    lambda d: (
                        d["checks"]["robot_joint_count_29"] and d["checks"]["robot_body_count_40"],
                        "g1_csv_task_eval_robot_contract",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_source"],
                        "g1_csv_task_eval_official_csv_source",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_csv_task_eval_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_csv_task_eval_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_start_training"],
                        "g1_csv_task_eval_no_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_csv_task_eval_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_train_entry_diagnostic",
                "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
                "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_train_entry_diagnostic",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["prior_csv_task_eval_passed"],
                        "g1_train_entry_prior_csv_task_eval",
                    ),
                    lambda d: (
                        d["checks"]["process_returned_zero"],
                        "g1_train_entry_process_zero",
                    ),
                    lambda d: (
                        d["checks"]["env_created"] and d["checks"]["vec_env_wrapped"],
                        "g1_train_entry_env_and_vec_wrapper",
                    ),
                    lambda d: (
                        d["checks"]["runner_created"] and d["checks"]["runner_class_motion"],
                        "g1_train_entry_motion_runner",
                    ),
                    lambda d: (
                        d["checks"]["learn_completed"],
                        "g1_train_entry_learn_completed",
                    ),
                    lambda d: (
                        d["checks"]["runner_training_type_rl"],
                        "g1_train_entry_rl_mode",
                    ),
                    lambda d: (
                        d["checks"]["num_steps_per_env_4"] and d["checks"]["one_iteration_requested"],
                        "g1_train_entry_bounded_iteration",
                    ),
                    lambda d: (
                        d["checks"]["num_actions_29"]
                        and d["checks"]["num_obs_160"]
                        and d["checks"]["num_privileged_obs_286"],
                        "g1_train_entry_action_obs_contract",
                    ),
                    lambda d: (
                        d["checks"]["robot_joint_count_29"] and d["checks"]["robot_body_count_40"],
                        "g1_train_entry_robot_contract",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_source"],
                        "g1_train_entry_official_csv_source",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_csv_to_npz_output"],
                        "g1_train_entry_no_official_csv_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_rollout"],
                        "g1_train_entry_no_paper_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_formal_ppo_training"],
                        "g1_train_entry_no_formal_training_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_write_checkpoint"],
                        "g1_train_entry_no_checkpoint",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_train_entry_keeps_goal_incomplete",
                    ),
                    lambda d: (
                        d["interpretation"]["formal_gpu_experiment"] is False,
                        "g1_train_entry_not_formal_gpu_experiment",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_ppo_training_run",
                "res/tracking/g1_resource_adjusted_ppo_training_run/"
                "tracking_g1_resource_adjusted_ppo_training_run.json",
                [
                    lambda d: (
                        d.get("status")
                        in {
                            "ok_resource_adjusted_ppo_training_completed",
                            "ok_with_gpu_resource_unavailable_before_training",
                        },
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["train_entry_smoke_passed"],
                        "g1_resource_adjusted_ppo_train_entry_smoke_passed",
                    ),
                    lambda d: (
                        d["input_checks"]["enriched_usd_exists"] and d["input_checks"]["motion_npz_exists"],
                        "g1_resource_adjusted_ppo_inputs_exist",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 5, 6, 7]
                        and d["config"]["selected_physical_gpus"]
                        and d["config"]["world_size"] == len(d["config"]["selected_physical_gpus"])
                        and d["config"]["cuda_visible_devices"]
                        == ",".join(str(gpu) for gpu in d["config"]["selected_physical_gpus"]),
                        "g1_resource_adjusted_ppo_selected_gpu_config_recorded",
                    ),
                    lambda d: (
                        d["config"]["num_steps_per_env"] == 24
                        and d["config"]["max_iterations"] >= 100
                        and d["config"]["num_envs_per_rank"] >= 512
                        and d["config"]["total_num_envs"] >= 512,
                        "g1_resource_adjusted_ppo_resource_adjusted_training_config",
                    ),
                    lambda d: (
                        Path(d["outputs"]["worker_script"]).is_file(),
                        "g1_resource_adjusted_ppo_worker_script_retained",
                    ),
                    lambda d: (
                        (
                            d["status"] == "ok_resource_adjusted_ppo_training_completed"
                            and d["run"]["attempted_training"]
                            and d["run"]["returncode"] == 0
                            and d["run"]["checkpoint_count"] > 0
                            and len(d["run"].get("rank_metrics", [])) >= 1
                        )
                        or (
                            d["status"] == "ok_with_gpu_resource_unavailable_before_training"
                            and d["run"]["attempted_training"] is False
                            and d["gpu_preflight"]["resource_ready"] is False
                        ),
                        "g1_resource_adjusted_ppo_training_or_defer_consistent",
                    ),
                    lambda d: (
                        d["interpretation"]["official_ppo_training_complete"] is False
                        and d["interpretation"]["paper_level_tracking_training_complete"] is False,
                        "g1_resource_adjusted_ppo_no_paper_level_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_resource_adjusted_ppo_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_ppo_checkpoint_eval",
                "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
                "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["training_run_completed"] and d["input_checks"]["checkpoint_exists"],
                        "g1_resource_adjusted_ppo_eval_inputs_exist",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 7]
                        and d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["cuda_visible_devices"] == "4,7",
                        "g1_resource_adjusted_ppo_eval_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["config"]["num_envs"] >= 512
                        and d["config"]["eval_steps"] >= 299
                        and d["config"]["total_env_steps"] >= 153088,
                        "g1_resource_adjusted_ppo_eval_full_available_motion_steps",
                    ),
                    lambda d: (
                        d["run"]["attempted_eval"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["metrics_exists"]
                        and d["run"]["timeseries_exists"],
                        "g1_resource_adjusted_ppo_eval_outputs_exist",
                    ),
                    lambda d: (
                        d["run"]["metrics"]["loaded_iteration"] == 99
                        and d["run"]["metrics"]["num_envs"] == 512
                        and d["run"]["metrics"]["eval_steps"] == 299,
                        "g1_resource_adjusted_ppo_eval_checkpoint_loaded_and_rolled_out",
                    ),
                    lambda d: (
                        d["interpretation"]["official_tracking_eval_complete"] is False
                        and d["interpretation"]["paper_level_tracking_eval_complete"] is False,
                        "g1_resource_adjusted_ppo_eval_no_paper_level_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_resource_adjusted_ppo_eval_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_ppo_eval_report_assets",
                "res/report_assets/official_csv_loop_ppo_checkpoint_eval/"
                "official_csv_loop_ppo_checkpoint_eval_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["eval_status_ok"] and d["checks"]["timeseries_has_299_rows"],
                        "official_csv_loop_ppo_assets_source_eval_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["summary_csv_exists"]
                        and d["checks"]["gpu_summary_csv_exists"],
                        "official_csv_loop_ppo_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_unpatched_output"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_csv_loop_ppo_assets_no_overclaim",
                    ),
                    lambda d: (
                        d["metrics"]["total_env_steps"] == 153088
                        and d["metrics"]["done_count_total"] == 13127,
                        "official_csv_loop_ppo_assets_core_metrics_recorded",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_csv_loop_full_bundle_motion_npz",
                "res/tracking/official_csv_loop_full_bundle_motion_npz/"
                "tracking_g1_official_csv_loop_full_bundle_motion_npz.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_motion_npz",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_40_source_rows_used"]
                        and d["checks"]["total_frames_11960"]
                        and d["checks"]["joint_shape_11960_29"]
                        and d["checks"]["body_shape_11960_40_3"],
                        "full_bundle_motion_shapes_and_rows_ok",
                    ),
                    lambda d: (
                        d["bundle"]["motion_count"] == 40
                        and d["bundle"]["total_frames"] == 11960
                        and d["bundle"]["boundary_count"] == 39,
                        "full_bundle_motion_count_frames_boundaries_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_multimotion_sampler"]
                        and d["checks"]["does_not_claim_paper_level_training"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_motion_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_motion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_csv_loop_full_bundle_ppo_training_run",
                "res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/"
                "tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_ppo_training_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["full_bundle_has_40_motions"]
                        and d["input_checks"]["full_bundle_total_frames_11960"]
                        and d["input_checks"]["train_entry_smoke_passed"],
                        "full_bundle_ppo_training_inputs_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["world_size"] == 2
                        and d["config"]["total_num_envs"] == 1024
                        and d["config"]["max_iterations"] == 300,
                        "full_bundle_ppo_training_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["run"]["attempted_training"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["checkpoint_count"] >= 7
                        and len(d["run"].get("rank_metrics", [])) >= 1,
                        "full_bundle_ppo_training_completed_with_checkpoints",
                    ),
                    lambda d: (
                        d["interpretation"]["official_ppo_training_complete"] is False
                        and d["interpretation"]["paper_level_tracking_training_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_ppo_training_no_paper_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval",
                "res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/"
                "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_ppo_checkpoint_eval_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["full_bundle_motion_count_40"]
                        and d["input_checks"]["full_bundle_total_frames_11960"]
                        and d["input_checks"]["training_run_completed"]
                        and d["input_checks"]["checkpoint_exists"],
                        "full_bundle_ppo_eval_inputs_ok",
                    ),
                    lambda d: (
                        d["run"]["attempted_eval"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["metrics_exists"]
                        and d["run"]["timeseries_exists"],
                        "full_bundle_ppo_eval_outputs_exist",
                    ),
                    lambda d: (
                        d["run"]["metrics"]["loaded_iteration"] == 299
                        and d["run"]["metrics"]["num_envs"] == 512
                        and d["run"]["metrics"]["eval_steps"] == 299
                        and d["run"]["metrics"]["motion_count"] == 40
                        and d["run"]["metrics"]["total_motion_frames"] == 11960,
                        "full_bundle_ppo_eval_checkpoint_loaded_and_rolled_out",
                    ),
                    lambda d: (
                        d["interpretation"]["official_tracking_eval_complete"] is False
                        and d["interpretation"]["paper_level_tracking_eval_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_ppo_eval_no_paper_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_ppo_eval_report_assets",
                "res/report_assets/official_csv_loop_full_bundle_ppo_checkpoint_eval/"
                "official_csv_loop_full_bundle_ppo_checkpoint_eval_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["eval_status_ok"] and d["checks"]["timeseries_has_299_rows"],
                        "full_bundle_ppo_assets_source_eval_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["summary_csv_exists"]
                        and d["checks"]["gpu_summary_csv_exists"],
                        "full_bundle_ppo_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_unpatched_output"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_ppo_assets_no_overclaim",
                    ),
                    lambda d: (
                        d["metrics"]["motion_count"] == 40
                        and d["metrics"]["total_motion_frames"] == 11960
                        and d["metrics"]["total_env_steps"] == 153088,
                        "full_bundle_ppo_assets_core_metrics_recorded",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_full_bundle_ppo_training_run",
                "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
                "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_ppo_training_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["official_importer_usd_exists"]
                        and d["input_checks"]["official_importer_task_gate_passed"]
                        and d["input_checks"]["full_bundle_has_40_motions"]
                        and d["input_checks"]["full_bundle_total_frames_11960"],
                        "official_importer_ppo_training_inputs_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["world_size"] == 2
                        and d["config"]["total_num_envs"] == 1024
                        and d["config"]["max_iterations"] == 300,
                        "official_importer_ppo_training_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["run"]["attempted_training"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["checkpoint_count"] >= 7
                        and len(d["run"].get("rank_metrics", [])) >= 1,
                        "official_importer_ppo_training_completed_with_checkpoints",
                    ),
                    lambda d: (
                        d["interpretation"]["official_ppo_training_complete"] is False
                        and d["interpretation"]["paper_level_tracking_training_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "official_importer_ppo_training_no_paper_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval",
                "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
                "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_ppo_checkpoint_eval_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["official_importer_usd_exists"]
                        and d["input_checks"]["full_bundle_motion_count_40"]
                        and d["input_checks"]["full_bundle_total_frames_11960"]
                        and d["input_checks"]["training_run_completed"]
                        and d["input_checks"]["checkpoint_exists"],
                        "official_importer_ppo_eval_inputs_ok",
                    ),
                    lambda d: (
                        d["run"]["attempted_eval"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["metrics_exists"]
                        and d["run"]["timeseries_exists"],
                        "official_importer_ppo_eval_outputs_exist",
                    ),
                    lambda d: (
                        d["run"]["metrics"]["loaded_iteration"] == 299
                        and d["run"]["metrics"]["num_envs"] == 512
                        and d["run"]["metrics"]["eval_steps"] == 299
                        and d["run"]["metrics"]["motion_count"] == 40
                        and d["run"]["metrics"]["total_motion_frames"] == 11960
                        and d["run"]["metrics"]["uses_official_importer_export_usd"] is True,
                        "official_importer_ppo_eval_checkpoint_loaded_and_rolled_out",
                    ),
                    lambda d: (
                        d["interpretation"]["official_tracking_eval_complete"] is False
                        and d["interpretation"]["paper_level_tracking_eval_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "official_importer_ppo_eval_no_paper_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_ppo_eval_report_assets",
                "res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/"
                "official_importer_export_full_bundle_ppo_checkpoint_eval_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["eval_status_ok"]
                        and d["checks"]["timeseries_has_299_rows"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "official_importer_ppo_assets_source_eval_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["training_curve_assets_exist"]
                        and d["checks"]["summary_csv_exists"]
                        and d["checks"]["gpu_summary_csv_exists"],
                        "official_importer_ppo_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_ppo_assets_no_overclaim",
                    ),
                    lambda d: (
                        d["metrics"]["motion_count"] == 40
                        and d["metrics"]["total_motion_frames"] == 11960
                        and d["metrics"]["total_env_steps"] == 153088
                        and d["metrics"]["training_iteration_count"] >= 300,
                        "official_importer_ppo_assets_core_metrics_recorded",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset",
                "res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/"
                "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_teacher_rollout_dataset_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["official_importer_usd_exists"]
                        and d["input_checks"]["official_importer_training_completed"]
                        and d["input_checks"]["official_importer_checkpoint_eval_completed"]
                        and d["input_checks"]["full_bundle_motion_npz_exists"]
                        and d["input_checks"]["full_bundle_motion_count_40"]
                        and d["input_checks"]["full_bundle_total_frames_11960"],
                        "official_importer_teacher_rollout_inputs_ok",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 7]
                        and d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["cuda_visible_devices"] == "4,7"
                        and d["config"]["world_size"] == 2
                        and d["config"]["total_num_envs"] == 1024,
                        "official_importer_teacher_rollout_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["run"]["attempted_rollout"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["shard_count"] == 2
                        and d["aggregate_metrics"]["total_env_steps"] == 306176
                        and d["aggregate_metrics"]["motion_count"] == 40
                        and d["aggregate_metrics"]["total_motion_frames"] == 11960,
                        "official_importer_teacher_rollout_completed_with_expected_scope",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["run"]["shard_npz_paths"]),
                        "official_importer_teacher_rollout_npz_shards_retained",
                    ),
                    lambda d: (
                        Path(d["outputs"]["worker_script"]).is_file()
                        and Path(d["inputs"]["base_compatible_training_run_json"]).is_file(),
                        "official_importer_teacher_rollout_worker_and_status_shim_retained",
                    ),
                    lambda d: (
                        all(
                            bool(item.get("uses_official_importer_export_usd"))
                            for item in d["run"].get("shard_metrics", [])
                        ),
                        "official_importer_teacher_rollout_shards_use_official_importer_export_usd",
                    ),
                    lambda d: (
                        d["interpretation"]["official_dagger_dataset_complete"] is False
                        and d["interpretation"]["paper_level_teacher_rollout_dataset_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "official_importer_teacher_rollout_no_paper_level_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_teacher_rollout_report_assets",
                "res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/"
                "official_importer_export_full_bundle_teacher_rollout_report_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["source_rollout_status_ok"]
                        and d["checks"]["two_shards_loaded"]
                        and d["checks"]["motion_count_40"]
                        and d["checks"]["total_motion_frames_11960"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "official_importer_teacher_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["total_env_steps_match_source"]
                        and d["metrics"]["total_env_steps"] == 306176,
                        "official_importer_teacher_assets_total_env_steps",
                    ),
                    lambda d: (
                        d["checks"]["action_dim_29"] and d["checks"]["rollout_steps_299"],
                        "official_importer_teacher_assets_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "official_importer_teacher_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_teacher_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_checkpoint_completion_proxy",
                "res/report_assets/official_importer_export_scaled_ppo_checkpoint_completion_proxy/"
                "scaled_ppo_checkpoint_completion_proxy.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_checkpoint_completion_proxy",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_eval_status_ok"]
                        and d["checks"]["timeseries_has_299_rows"]
                        and d["metrics"]["attempted_env_steps"] == 612352
                        and d["metrics"]["total_env_steps_recorded"] == 612352,
                        "scaled_ppo_completion_proxy_source_scope",
                    ),
                    lambda d: (
                        d["config"]["num_envs"] == 2048
                        and d["config"]["eval_steps"] == 299
                        and d["config"]["motion_count"] == 40
                        and d["config"]["total_motion_frames"] == 11960,
                        "scaled_ppo_completion_proxy_eval_shape",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["does_not_use_resource_adjusted_usd"]
                        and d["config"]["uses_official_importer_export_usd"]
                        and not d["config"]["uses_resource_adjusted_usd"],
                        "scaled_ppo_completion_proxy_importer_export_scope",
                    ),
                    lambda d: (
                        d["metrics"]["done_count_total"] == 611642
                        and d["metrics"]["timeout_count_total"] == 0
                        and 0.0 <= d["metrics"]["local_completion_proxy_rate"] <= 1.0
                        and d["metrics"]["local_non_timeout_done_rate"] > 0.99,
                        "scaled_ppo_completion_proxy_rates_recorded",
                    ),
                    lambda d: (
                        d["checks"]["assets_exist"]
                        and all(Path(path).is_file() for path in d["assets"].values()),
                        "scaled_ppo_completion_proxy_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_success_or_fall"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_completion_proxy_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep",
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/"
                "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_checkpoint_sweep_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["metrics"]["checkpoint_count"] == 21
                        and d["metrics"]["ok_checkpoint_count"] == 21
                        and d["metrics"]["total_env_steps"] == 1607424,
                        "scaled_ppo_checkpoint_sweep_scope",
                    ),
                    lambda d: (
                        d["config"]["num_envs"] == 256
                        and d["config"]["eval_steps"] == 299
                        and d["checks"]["all_eval_steps_match_config"]
                        and d["checks"]["all_num_envs_match_config"],
                        "scaled_ppo_checkpoint_sweep_eval_shape",
                    ),
                    lambda d: (
                        d["checks"]["all_use_official_importer_export_usd"]
                        and d["checks"]["no_rows_use_resource_adjusted_usd"]
                        and d["checks"]["all_motion_count_40"]
                        and d["checks"]["all_total_motion_frames_11960"],
                        "scaled_ppo_checkpoint_sweep_asset_and_motion_scope",
                    ),
                    lambda d: (
                        d["metrics"]["best_iteration"] == 300
                        and d["checks"]["best_checkpoint_recorded"]
                        and d["metrics"]["best_local_non_timeout_done_rate"] == 1.0,
                        "scaled_ppo_checkpoint_sweep_best_checkpoint_recorded",
                    ),
                    lambda d: (
                        Path(d["outputs"]["rows_csv"]).is_file()
                        and Path(d["outputs"]["rows_tsv"]).is_file()
                        and all(Path(path).is_file() for path in d["report_assets"].values()),
                        "scaled_ppo_checkpoint_sweep_outputs_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_checkpoint_sweep_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval",
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval/"
                "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["best_eval_completed"]
                        and d["checks"]["num_envs_2048"]
                        and d["checks"]["eval_steps_299"]
                        and d["best_metrics"]["total_env_steps"] == 612352,
                        "scaled_ppo_best_checkpoint_confirmation_eval_shape",
                    ),
                    lambda d: (
                        d["best_metrics"]["loaded_iteration"] == 300
                        and d["checks"]["best_iteration_matches_sweep"]
                        and d["final_metrics"]["loaded_iteration"] == 999,
                        "scaled_ppo_best_checkpoint_confirmation_iterations",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["does_not_use_resource_adjusted_usd"]
                        and d["checks"]["motion_count_40"]
                        and d["checks"]["total_motion_frames_11960"],
                        "scaled_ppo_best_checkpoint_confirmation_asset_scope",
                    ),
                    lambda d: (
                        d["deltas"]["reward_mean"] < 0.0
                        and d["deltas"]["error_body_pos_mean"] > 0.0
                        and d["deltas"]["error_joint_pos_mean"] > 0.0,
                        "scaled_ppo_best_checkpoint_confirmation_final_not_beaten",
                    ),
                    lambda d: (
                        Path(d["outputs"]["comparison_csv"]).is_file()
                        and all(Path(path).is_file() for path in d["report_assets"].values()),
                        "scaled_ppo_best_checkpoint_confirmation_outputs_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_best_checkpoint_confirmation_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_reward_termination_diagnostic",
                "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/"
                "reward_termination_diagnostic.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_reward_termination_diagnostic",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["metrics"]["checkpoint_count"] == 2
                        and d["metrics"]["reward_component_count"] == 18
                        and d["metrics"]["termination_component_count"] == 8
                        and d["metrics"]["motion_metric_count"] == 26,
                        "scaled_ppo_reward_termination_diagnostic_scope",
                    ),
                    lambda d: (
                        d["checks"]["dominant_final_is_ee_body_pos"]
                        and d["checks"]["dominant_best_is_ee_body_pos"]
                        and d["checks"]["dominant_final_fraction_gt_0_99"]
                        and d["checks"]["dominant_best_fraction_gt_0_99"],
                        "scaled_ppo_reward_termination_diagnostic_dominant_ee_body_pos",
                    ),
                    lambda d: (
                        d["checks"]["reward_csv_exists"]
                        and d["checks"]["termination_csv_exists"]
                        and d["checks"]["motion_csv_exists"]
                        and d["checks"]["png_assets_exist"]
                        and all(Path(path).is_file() for path in d["assets"].values()),
                        "scaled_ppo_reward_termination_diagnostic_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_reward_termination_diagnostic_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit",
                "res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/"
                "ee_body_pos_termination_source_audit.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_files_exist"]
                        and d["checks"]["ee_body_pos_uses_z_only_function"]
                        and d["checks"]["z_only_source_indexes_last_coordinate"],
                        "scaled_ppo_ee_body_pos_source_function_scope",
                    ),
                    lambda d: (
                        d["checks"]["threshold_is_0_25_m"]
                        and d["checks"]["termination_body_names_are_four_distal_links"],
                        "scaled_ppo_ee_body_pos_source_threshold_and_bodies",
                    ),
                    lambda d: (
                        d["checks"]["motion_bundle_shape_matches_full_public_bundle"]
                        and d["motion_bundle"]["body_pos_w_shape"] == [11960, 40, 3],
                        "scaled_ppo_ee_body_pos_motion_bundle_scope",
                    ),
                    lambda d: (
                        d["checks"]["dominant_termination_is_ee_body_pos_both_checkpoints"]
                        and d["checks"]["ee_body_pos_fraction_gt_0_99_both_checkpoints"],
                        "scaled_ppo_ee_body_pos_dominant_high_fraction",
                    ),
                    lambda d: (
                        d["checks"]["assets_exist"]
                        and all(Path(path).is_file() for path in d["assets"].values()),
                        "scaled_ppo_ee_body_pos_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_ee_body_pos_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace",
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/"
                "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_endpoint_z_error_trace",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["process_returned_zero"]
                        and d["checks"]["metrics_status_ok"]
                        and d["checks"]["eval_shape_full_size"],
                        "scaled_ppo_endpoint_trace_process_and_shape",
                    ),
                    lambda d: (
                        d["config"]["num_envs"] == 2048
                        and d["config"]["eval_steps"] == 299
                        and d["config"]["total_env_steps"] == 612352,
                        "scaled_ppo_endpoint_trace_full_size",
                    ),
                    lambda d: (
                        d["checks"]["threshold_matches_source"]
                        and d["checks"]["termination_bodies_match_source"]
                        and d["checks"]["records_four_endpoint_bodies"],
                        "scaled_ppo_endpoint_trace_source_body_scope",
                    ),
                    lambda d: (
                        d["run"]["metrics"]["aggregate"]["exceed_rate"]["mean"] > 0.99
                        and d["run"]["metrics"]["body_rows"][0]["exceed_rate_mean_over_steps"] > 0.99
                        and d["run"]["metrics"]["body_rows"][1]["exceed_rate_mean_over_steps"] > 0.99,
                        "scaled_ppo_endpoint_trace_ankles_dominate",
                    ),
                    lambda d: (
                        d["checks"]["report_assets_exist"]
                        and all(Path(path).is_file() for path in d["outputs"]["report_assets"].values()),
                        "scaled_ppo_endpoint_trace_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_endpoint_trace_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval",
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval/"
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["metrics"]["seed_count"] == 3
                        and d["metrics"]["ok_seed_count"] == 3
                        and d["metrics"]["total_env_steps"] == 1837056,
                        "scaled_ppo_multiseed_scope",
                    ),
                    lambda d: (
                        d["checks"]["all_eval_steps_299"]
                        and d["checks"]["all_num_envs_match_config"]
                        and d["config"]["num_envs"] == 2048,
                        "scaled_ppo_multiseed_eval_shape",
                    ),
                    lambda d: (
                        d["checks"]["all_use_official_importer_export_usd"]
                        and d["checks"]["no_rows_use_resource_adjusted_usd"]
                        and d["checks"]["all_motion_count_40"]
                        and d["checks"]["all_total_motion_frames_11960"],
                        "scaled_ppo_multiseed_asset_and_motion_scope",
                    ),
                    lambda d: (
                        d["checks"]["rows_csv_exists"]
                        and d["checks"]["rows_tsv_exists"]
                        and Path(d["outputs"]["rows_csv"]).is_file()
                        and Path(d["outputs"]["rows_tsv"]).is_file(),
                        "scaled_ppo_multiseed_tables_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_multiseed_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_report_assets",
                "res/report_assets/official_importer_export_scaled_ppo_checkpoint_multiseed_eval/"
                "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["all_seeds_completed"]
                        and d["checks"]["all_use_official_importer_export_usd"],
                        "scaled_ppo_multiseed_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["summary_csv_exists"]
                        and d["checks"]["png_assets_exist"]
                        and d["checks"]["readme_exists"],
                        "scaled_ppo_multiseed_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_eval"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_multiseed_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset",
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["official_importer_usd_exists"]
                        and d["input_checks"]["scaled_training_completed"]
                        and d["input_checks"]["scaled_checkpoint_eval_completed"]
                        and d["input_checks"]["full_bundle_motion_npz_exists"]
                        and d["input_checks"]["full_bundle_motion_count_40"]
                        and d["input_checks"]["full_bundle_total_frames_11960"],
                        "scaled_importer_teacher_rollout_inputs_ok",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 7]
                        and d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["cuda_visible_devices"] == "4,7"
                        and d["config"]["world_size"] == 2
                        and d["config"]["total_num_envs"] == 4096,
                        "scaled_importer_teacher_rollout_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["run"]["attempted_rollout"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["shard_count"] == 2
                        and d["aggregate_metrics"]["total_env_steps"] == 1224704
                        and d["aggregate_metrics"]["motion_count"] == 40
                        and d["aggregate_metrics"]["total_motion_frames"] == 11960,
                        "scaled_importer_teacher_rollout_completed_with_expected_scope",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["run"]["shard_npz_paths"]),
                        "scaled_importer_teacher_rollout_npz_shards_retained",
                    ),
                    lambda d: (
                        all(
                            bool(item.get("uses_official_importer_export_usd"))
                            and item.get("loaded_iteration") == 999
                            for item in d["run"].get("shard_metrics", [])
                        ),
                        "scaled_importer_teacher_rollout_shards_use_iteration999_importer_checkpoint",
                    ),
                    lambda d: (
                        d["interpretation"]["official_dagger_dataset_complete"] is False
                        and d["interpretation"]["paper_level_teacher_rollout_dataset_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_importer_teacher_rollout_no_paper_level_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets",
                "res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
                "official_importer_export_full_bundle_teacher_rollout_report_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["source_rollout_status_ok"]
                        and d["checks"]["two_shards_loaded"]
                        and d["checks"]["motion_count_40"]
                        and d["checks"]["total_motion_frames_11960"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "scaled_importer_teacher_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["total_env_steps_match_source"]
                        and d["metrics"]["total_env_steps"] == 1224704
                        and d["metrics"]["total_envs"] == 4096,
                        "scaled_importer_teacher_assets_total_env_steps",
                    ),
                    lambda d: (
                        d["checks"]["action_dim_29"] and d["checks"]["rollout_steps_299"],
                        "scaled_importer_teacher_assets_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "scaled_importer_teacher_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "scaled_importer_teacher_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_latent_projection_report_assets",
                "res/report_assets/official_importer_export_full_bundle_latent_projection/"
                "official_importer_export_full_bundle_latent_projection_assets.json",
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_latent_projection_report_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["state_latent_status_ok"]
                        and d["checks"]["vae_status_ok"]
                        and d["checks"]["bundle_status_ok"]
                        and d["checks"]["two_latent_shards_loaded"],
                        "official_importer_latent_projection_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_teacher_rollout_samples"]
                        and d["metrics"]["total_latent_samples"] == 306176
                        and d["checks"]["latent_dim_32"]
                        and d["checks"]["motion_count_40"],
                        "official_importer_latent_projection_scope_ok",
                    ),
                    lambda d: (
                        d["checks"]["has_walk_and_run_labels"]
                        and d["checks"]["pca_variance_recorded"]
                        and d["checks"]["walk_run_trace_recorded"]
                        and d["metrics"]["walk_run_trace_rows"] > 0,
                        "official_importer_latent_projection_fig5d_proxy_metrics_ok",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "official_importer_latent_projection_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tsne_reproduction"]
                        and d["checks"]["does_not_claim_paper_fig5d"]
                        and d["checks"]["does_not_claim_official_beyondmimic_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "official_importer_latent_projection_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset",
                "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/"
                "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_teacher_rollout_dataset_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["full_bundle_training_completed"]
                        and d["input_checks"]["full_bundle_checkpoint_eval_completed"]
                        and d["input_checks"]["full_bundle_motion_npz_exists"]
                        and d["input_checks"]["full_bundle_motion_count_40"]
                        and d["input_checks"]["full_bundle_total_frames_11960"]
                        and d["input_checks"]["checkpoint_exists"],
                        "full_bundle_teacher_rollout_inputs_ok",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 7]
                        and d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["cuda_visible_devices"] == "4,7"
                        and d["config"]["world_size"] == 2
                        and d["config"]["total_num_envs"] == 1024,
                        "full_bundle_teacher_rollout_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["run"]["attempted_rollout"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["shard_count"] == 2
                        and d["aggregate_metrics"]["total_env_steps"] == 306176
                        and d["aggregate_metrics"]["motion_count"] == 40
                        and d["aggregate_metrics"]["total_motion_frames"] == 11960,
                        "full_bundle_teacher_rollout_completed_with_expected_scope",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["run"]["shard_npz_paths"]),
                        "full_bundle_teacher_rollout_npz_shards_retained",
                    ),
                    lambda d: (
                        Path(d["outputs"]["worker_script"]).is_file()
                        and Path(d["inputs"]["base_compatible_training_run_json"]).is_file(),
                        "full_bundle_teacher_rollout_worker_and_status_shim_retained",
                    ),
                    lambda d: (
                        d["interpretation"]["official_dagger_dataset_complete"] is False
                        and d["interpretation"]["paper_level_teacher_rollout_dataset_complete"] is False
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_teacher_rollout_no_paper_level_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_teacher_rollout_report_assets",
                "res/report_assets/official_csv_loop_full_bundle_teacher_rollout_dataset/"
                "official_csv_loop_full_bundle_teacher_rollout_report_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["source_rollout_status_ok"]
                        and d["checks"]["two_shards_loaded"]
                        and d["checks"]["motion_count_40"]
                        and d["checks"]["total_motion_frames_11960"],
                        "full_bundle_teacher_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["total_env_steps_match_source"]
                        and d["metrics"]["total_env_steps"] == 306176,
                        "full_bundle_teacher_assets_total_env_steps",
                    ),
                    lambda d: (
                        d["checks"]["action_dim_29"] and d["checks"]["rollout_steps_299"],
                        "full_bundle_teacher_assets_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "full_bundle_teacher_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_teacher_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_reference_replay_video_asset",
                "res/visualization/official_csv_loop_reference_replay/"
                "official_csv_loop_reference_replay_video_asset.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["source_motion_audit_ok"] and d["checks"]["body_shape_299_40_3"],
                        "reference_replay_video_source_motion_ok",
                    ),
                    lambda d: (
                        d["checks"]["video_exists_nonempty"] and d["checks"]["keyframes_exist_nonempty"],
                        "reference_replay_video_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"]
                        and d["checks"]["does_not_claim_paper_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "reference_replay_video_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "reference_replay_video_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_replay_full_dataset_report_assets",
                "res/report_assets/official_importer_export_replay_full_dataset/"
                "official_importer_export_replay_full_dataset_report_assets.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_replay_full_dataset_report_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_audit_ok"]
                        and d["checks"]["all_40_rows_ok"]
                        and d["aggregate"]["row_count"] == 40
                        and d["aggregate"]["ok_count"] == 40
                        and d["aggregate"]["failed_count"] == 0,
                        "official_importer_replay_report_source_40_of_40_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_reached_official_loop_299"]
                        and d["aggregate"]["total_replayed_steps"] == 11960,
                        "official_importer_replay_report_steps_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["does_not_claim_paper_level_replay"],
                        "official_importer_replay_report_claim_boundary_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_report_assets_exist"] and d["checks"]["reference_video_exists"],
                        "official_importer_replay_report_assets_exist",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False
                        and d["interpretation"]["paper_level_tracking_eval_complete"] is False,
                        "official_importer_replay_report_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_policy_rollout_video_capture",
                "res/visualization/official_csv_loop_policy_rollout/"
                "tracking_g1_official_csv_loop_policy_rollout_capture.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_policy_rollout_video_capture",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"] and d["checks"]["render_ok"],
                        "policy_rollout_capture_and_render_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] in {4, 7}
                        and d["config"]["rollout_steps"] == 299,
                        "policy_rollout_gpu_and_steps_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "policy_rollout_capture_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_policy_rollout_video_asset",
                "res/visualization/official_csv_loop_policy_rollout/official_csv_loop_policy_rollout_video_asset.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["capture_status_ok"]
                        and d["checks"]["frame_count_299"]
                        and d["checks"]["body_count_supported_14_or_40"]
                        and d["checks"]["target_body_count_14"],
                        "policy_rollout_video_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["video_exists_nonempty"] and d["checks"]["keyframes_exist_nonempty"],
                        "policy_rollout_video_assets_exist",
                    ),
                    lambda d: (
                        d["metrics"]["target_body_error_mean"] >= 0.0
                        and d["metrics"]["done_count_total"] >= 0,
                        "policy_rollout_video_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "policy_rollout_video_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_policy_rollout_video_capture",
                "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
                "tracking_g1_official_csv_loop_policy_rollout_capture.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_policy_rollout_video_capture",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"]
                        and d["checks"]["render_ok"]
                        and d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"],
                        "full_bundle_policy_rollout_capture_render_and_bundle_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] in {4, 7}
                        and d["config"]["rollout_steps"] == 299,
                        "full_bundle_policy_rollout_gpu_and_steps_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_policy_rollout_capture_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_policy_rollout_video_asset",
                "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
                "official_csv_loop_policy_rollout_video_asset.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["capture_status_ok"]
                        and d["checks"]["frame_count_299"]
                        and d["checks"]["body_count_supported_14_or_40"]
                        and d["checks"]["target_body_count_14"],
                        "full_bundle_policy_rollout_video_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["video_exists_nonempty"]
                        and d["checks"]["keyframes_exist_nonempty"]
                        and d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"],
                        "full_bundle_policy_rollout_video_assets_and_bundle_ok",
                    ),
                    lambda d: (
                        d["metrics"]["target_body_error_mean"] >= 0.0
                        and d["metrics"]["done_count_total"] >= 0,
                        "full_bundle_policy_rollout_video_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_policy_rollout_video_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_vae_closed_loop_rollout_video_capture",
                (
                    "res/visualization/official_csv_loop_vae_closed_loop_rollout/"
                    "tracking_g1_official_csv_loop_vae_closed_loop_rollout_capture.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_vae_closed_loop_rollout_video_capture",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"] and d["checks"]["render_ok"],
                        "vae_closed_loop_video_capture_and_render_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] in {4, 7}
                        and d["config"]["rollout_steps"] == 299,
                        "vae_closed_loop_video_gpu_and_steps_recorded",
                    ),
                    lambda d: (
                        d["run"]["capture_metrics"]["teacher_vae_action_mse"]["mean"] < 0.01
                        and d["run"]["capture_metrics"]["robot_body_pos_shape"] == [299, 14, 3],
                        "vae_closed_loop_video_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_autonomous_vae_policy"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "vae_closed_loop_video_capture_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_vae_closed_loop_rollout_video_asset",
                (
                    "res/visualization/official_csv_loop_vae_closed_loop_rollout/"
                    "official_csv_loop_vae_closed_loop_rollout_video_asset.json"
                ),
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["capture_status_ok"]
                        and d["checks"]["frame_count_299"]
                        and d["checks"]["body_count_supported_14_or_40"]
                        and d["checks"]["target_body_count_14"],
                        "vae_closed_loop_video_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["video_exists_nonempty"] and d["checks"]["keyframes_exist_nonempty"],
                        "vae_closed_loop_video_assets_exist",
                    ),
                    lambda d: (
                        d["metrics"]["teacher_vae_action_mse_mean"] < 0.01
                        and d["metrics"]["target_body_error_mean"] >= 0.0,
                        "vae_closed_loop_video_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_autonomous_vae_policy"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "vae_closed_loop_video_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_teacher_rollout_report_assets",
                "res/report_assets/official_csv_loop_teacher_rollout_dataset/"
                "official_csv_loop_teacher_rollout_report_assets.json",
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["source_rollout_status_ok"] and d["checks"]["two_shards_loaded"],
                        "teacher_rollout_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["total_env_steps_match_source"]
                        and d["metrics"]["total_env_steps"] == 306176,
                        "teacher_rollout_assets_total_env_steps",
                    ),
                    lambda d: (
                        d["checks"]["action_dim_29"] and d["checks"]["rollout_steps_299"],
                        "teacher_rollout_assets_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "teacher_rollout_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "teacher_rollout_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_action_guidance_rollout_eval",
                (
                    "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
                    "level_c_official_csv_loop_action_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_action_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"]
                        and d["checks"]["render_ok"]
                        and d["checks"]["three_variants_evaluated"]
                        and d["checks"]["rollout_steps_299"],
                        "action_guidance_rollout_capture_render_three_variants_299",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] in {4, 7}
                        and d["config"]["formal_gpu_experiment"] is False,
                        "action_guidance_rollout_gpu_and_scope_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_receding_horizon_latent_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "action_guidance_rollout_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "action_guidance_rollout_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_action_guidance_rollout_asset",
                (
                    "res/visualization/official_csv_loop_action_guidance_rollout/"
                    "official_csv_loop_action_guidance_rollout_asset.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["frame_count_299"]
                        and d["checks"]["target_body_count_14"]
                        and d["checks"]["body_count_supported_14_or_40"],
                        "action_guidance_rollout_asset_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["video_exists_nonempty"]
                        and d["checks"]["keyframes_exist_nonempty"]
                        and d["checks"]["metrics_plot_exists_nonempty"],
                        "action_guidance_rollout_visual_assets_exist",
                    ),
                    lambda d: (
                        set(d["variant_metrics"]) == {"teacher", "vae_base", "action_guided"},
                        "action_guidance_rollout_asset_three_variants",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_receding_horizon_latent_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "action_guidance_rollout_asset_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_teacher_rollout_dataset",
                "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
                "tracking_g1_resource_adjusted_teacher_rollout_dataset.json",
                [
                    lambda d: (
                        d.get("status") == "ok_resource_adjusted_teacher_rollout_dataset_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["training_run_completed"] and d["input_checks"]["checkpoint_exists"],
                        "g1_resource_adjusted_teacher_rollout_inputs_exist",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 7]
                        and d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["cuda_visible_devices"] == "4,7"
                        and d["config"]["world_size"] == 2,
                        "g1_resource_adjusted_teacher_rollout_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["config"]["num_envs_per_rank"] >= 512
                        and d["config"]["rollout_steps"] >= 299
                        and d["aggregate_metrics"]["total_env_steps"] >= 306176,
                        "g1_resource_adjusted_teacher_rollout_full_available_motion_steps",
                    ),
                    lambda d: (
                        d["run"]["attempted_rollout"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["shard_count"] == 2
                        and len(d["run"].get("shard_npz_paths", [])) == 2,
                        "g1_resource_adjusted_teacher_rollout_outputs_exist",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["run"]["shard_npz_paths"]),
                        "g1_resource_adjusted_teacher_rollout_npz_shards_retained",
                    ),
                    lambda d: (
                        Path(d["outputs"]["worker_script"]).is_file(),
                        "g1_resource_adjusted_teacher_rollout_worker_script_retained",
                    ),
                    lambda d: (
                        d["interpretation"]["official_dagger_dataset_complete"] is False
                        and d["interpretation"]["paper_level_teacher_rollout_dataset_complete"] is False,
                        "g1_resource_adjusted_teacher_rollout_no_paper_level_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_resource_adjusted_teacher_rollout_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_csv_loop_teacher_rollout_dataset",
                "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
                "tracking_g1_official_csv_loop_teacher_rollout_dataset.json",
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_teacher_rollout_dataset_completed",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["input_checks"]["training_run_completed"]
                        and d["input_checks"]["checkpoint_exists"]
                        and d["input_checks"]["official_csv_loop_training_completed"],
                        "g1_official_csv_loop_teacher_rollout_inputs_exist",
                    ),
                    lambda d: (
                        d["config"]["candidate_physical_gpus"] == [4, 7]
                        and d["config"]["selected_physical_gpus"] == [4, 7]
                        and d["config"]["cuda_visible_devices"] == "4,7"
                        and d["config"]["world_size"] == 2,
                        "g1_official_csv_loop_teacher_rollout_gpu47_config_recorded",
                    ),
                    lambda d: (
                        d["config"]["num_envs_per_rank"] >= 512
                        and d["config"]["rollout_steps"] >= 299
                        and d["aggregate_metrics"]["total_env_steps"] >= 306176,
                        "g1_official_csv_loop_teacher_rollout_full_available_motion_steps",
                    ),
                    lambda d: (
                        d["run"]["attempted_rollout"]
                        and d["run"]["returncode"] == 0
                        and d["run"]["shard_count"] == 2
                        and len(d["run"].get("shard_npz_paths", [])) == 2,
                        "g1_official_csv_loop_teacher_rollout_outputs_exist",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["run"]["shard_npz_paths"]),
                        "g1_official_csv_loop_teacher_rollout_npz_shards_retained",
                    ),
                    lambda d: (
                        Path(d["outputs"]["worker_script"]).is_file(),
                        "g1_official_csv_loop_teacher_rollout_worker_script_retained",
                    ),
                    lambda d: (
                        d["interpretation"]["official_dagger_dataset_complete"] is False
                        and d["interpretation"]["paper_level_teacher_rollout_dataset_complete"] is False,
                        "g1_official_csv_loop_teacher_rollout_no_paper_level_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_official_csv_loop_teacher_rollout_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_urdf_conversion_probe",
                "res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_urdf_usd_blocker",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["app_launcher_reached_payload"], "urdf_probe_app_payload"),
                    lambda d: (d["checks"]["app_launcher_closed"], "urdf_probe_app_closed"),
                    lambda d: (d["checks"]["libglu_missing_absent"], "urdf_probe_libglu_present"),
                    lambda d: (d["payload"]["stage_open_ok"], "urdf_probe_stage_open"),
                    lambda d: (d["payload"]["prim_count"] == 0, "urdf_probe_empty_usd_recorded"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "urdf_probe_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_urdf_path_tiny_probe",
                "res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_blocker_classified",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_g1_package_mesh_refs_resolve_statically"],
                        "urdf_path_tiny_g1_mesh_refs_resolve",
                    ),
                    lambda d: (d["checks"]["app_launcher_closed"] or d["markers"]["timed_out"], "urdf_path_tiny_kit_exit_recorded"),
                    lambda d: (d["checks"]["libglu_missing_absent"], "urdf_path_tiny_libglu_present"),
                    lambda d: (d["markers"]["usd_save_not_allowed"], "urdf_path_tiny_usd_save_blocker"),
                    lambda d: (d["markers"]["vulkan_device_lost"], "urdf_path_tiny_vulkan_device_lost_recorded"),
                    lambda d: (
                        d["current_blocker"]
                        == "usd_layer_save_forbidden_and_vulkan_device_lost_before_payload",
                        "urdf_path_tiny_current_blocker_classified",
                    ),
                    lambda d: (d["checks"]["does_not_claim_motion_npz"], "urdf_path_tiny_no_motion_npz_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "urdf_path_tiny_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_mjcf_stage_probe",
                "res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_blocker_classified",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_g1_mjcf_mesh_refs_resolve_statically"],
                        "mjcf_stage_g1_mesh_refs_resolve",
                    ),
                    lambda d: (
                        d["checks"]["app_launcher_reached_after_app"],
                        "mjcf_stage_app_launcher_after_app",
                    ),
                    lambda d: (d["checks"]["libglu_missing_absent"], "mjcf_stage_libglu_present"),
                    lambda d: (d["markers"]["usd_save_not_allowed"], "mjcf_stage_usd_save_blocker"),
                    lambda d: (d["markers"]["vulkan_device_lost"], "mjcf_stage_vulkan_device_lost_recorded"),
                    lambda d: (
                        d["checks"]["minimal_stage_save_success"] is False,
                        "mjcf_stage_minimal_stage_save_failed_recorded",
                    ),
                    lambda d: (
                        d["current_blocker"] == "mjcf_or_stage_usd_save_forbidden_and_vulkan_device_lost",
                        "mjcf_stage_current_blocker_classified",
                    ),
                    lambda d: (d["checks"]["does_not_claim_motion_npz"], "mjcf_stage_no_motion_npz_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "mjcf_stage_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_usd_save_policy_probe",
                "res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_blocker_classified",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["plain_python_records_absent_pxr"],
                        "usd_save_policy_plain_python_pxr_absent",
                    ),
                    lambda d: (
                        d["checks"]["app_launcher_reached_after_app"],
                        "usd_save_policy_app_after_app",
                    ),
                    lambda d: (
                        d["checks"]["pxr_import_ok_inside_app"],
                        "usd_save_policy_pxr_inside_app",
                    ),
                    lambda d: (d["checks"]["attempts_recorded"], "usd_save_policy_attempts_recorded"),
                    lambda d: (d["checks"]["permission_false_recorded"], "usd_save_policy_permission_false"),
                    lambda d: (
                        d["checks"]["all_attempts_failed_to_save"],
                        "usd_save_policy_all_save_attempts_failed",
                    ),
                    lambda d: (
                        d["checks"]["force_permission_attempts_failed"],
                        "usd_save_policy_force_permission_failed",
                    ),
                    lambda d: (d["checks"]["export_attempts_failed"], "usd_save_policy_export_failed"),
                    lambda d: (
                        d["current_blocker"] == "app_launcher_layers_permission_to_save_false",
                        "usd_save_policy_current_blocker_classified",
                    ),
                    lambda d: (d["checks"]["does_not_claim_motion_npz"], "usd_save_policy_no_motion_npz_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "usd_save_policy_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_simulationapp_save_policy_probe",
                "res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_blocker_classified",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["simulationapp_base_vulkan_device_lost_recorded"],
                        "simulationapp_save_policy_base_vulkan_crash_recorded",
                    ),
                    lambda d: (
                        d["checks"]["simulationapp_isaaclab_headless_reached_payload"],
                        "simulationapp_save_policy_isaaclab_payload",
                    ),
                    lambda d: (
                        d["checks"]["simulationapp_isaaclab_headless_permission_false"],
                        "simulationapp_save_policy_isaaclab_permission_false",
                    ),
                    lambda d: (
                        d["checks"]["simulationapp_isaaclab_headless_force_permission_failed"],
                        "simulationapp_save_policy_isaaclab_force_permission_failed",
                    ),
                    lambda d: (d["checks"]["applauncher_reached_payload"], "simulationapp_save_policy_app_payload"),
                    lambda d: (
                        d["checks"]["applauncher_permission_false"],
                        "simulationapp_save_policy_app_permission_false",
                    ),
                    lambda d: (
                        d["checks"]["applauncher_force_permission_failed"],
                        "simulationapp_save_policy_app_force_permission_failed",
                    ),
                    lambda d: (
                        d["current_blocker"]
                        == "isaaclab_headless_experience_layers_permission_to_save_false_with_isaacsim_base_vulkan_crash",
                        "simulationapp_save_policy_current_blocker_classified",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "simulationapp_save_policy_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "simulationapp_save_policy_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_usd_api_variant_probe",
                "res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_stage_export_workaround",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["app_reached_after_app"], "usd_api_variant_app_after_app"),
                    lambda d: (d["checks"]["payload_recorded"], "usd_api_variant_payload_recorded"),
                    lambda d: (d["checks"]["pxr_import_ok"], "usd_api_variant_pxr_import_ok"),
                    lambda d: (d["checks"]["create_new_blocked_by_permission_false"], "usd_api_variant_save_blocked"),
                    lambda d: (d["checks"]["stage_export_success"], "usd_api_variant_stage_export_success"),
                    lambda d: (
                        d["checks"]["in_memory_stage_export_success"],
                        "usd_api_variant_in_memory_export_success",
                    ),
                    lambda d: (d["checks"]["any_nonempty_usd_written"], "usd_api_variant_nonempty_usd_written"),
                    lambda d: (
                        "create_in_memory_stage_export" in d["successful_attempt_labels"],
                        "usd_api_variant_in_memory_label_success",
                    ),
                    lambda d: (
                        d["current_blocker"] == "layer_save_blocked_but_stage_export_succeeds",
                        "usd_api_variant_current_blocker_classified",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "usd_api_variant_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "usd_api_variant_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_stage_export_workaround_probe",
                "res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_importer_still_empty_after_stage_export_patch",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["app_reached_after_app"], "g1_stage_export_app_after_app"),
                    lambda d: (d["checks"]["payload_recorded"], "g1_stage_export_payload_recorded"),
                    lambda d: (d["checks"]["urdf_extension_enabled"], "g1_stage_export_urdf_extension"),
                    lambda d: (
                        d["checks"]["stage_create_new_save_routed_to_export"],
                        "g1_stage_export_patch_routed_save",
                    ),
                    lambda d: (d["checks"]["dest_stage_has_robot"] is False, "g1_stage_export_dest_empty_recorded"),
                    lambda d: (
                        d["checks"]["current_stage_has_robot"] is False,
                        "g1_stage_export_current_stage_empty_recorded",
                    ),
                    lambda d: (
                        d["current_blocker"] == "stage_export_patch_applied_but_importer_output_empty",
                        "g1_stage_export_current_blocker_classified",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_stage_export_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_stage_export_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_layer_save_workaround_probe",
                "res/tracking/g1_urdf_layer_save_workaround/tracking_g1_urdf_layer_save_workaround_probe.json",
                [
                    lambda d: (
                        d.get("status")
                        in {
                            "ok_with_layer_save_patch_unavailable",
                            "ok_with_importer_still_empty_after_layer_save_patch",
                            "ok_with_cpp_importer_save_path_not_intercepted",
                            "ok_with_valid_g1_usd",
                            "ok_with_current_stage_robot_not_dest",
                        },
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["app_reached_after_app"], "g1_layer_save_app_after_app"),
                    lambda d: (d["checks"]["payload_recorded"], "g1_layer_save_payload_recorded"),
                    lambda d: (d["checks"]["urdf_extension_enabled"], "g1_layer_save_urdf_extension"),
                    lambda d: (
                        "sdf_layer_save_patch_assignment_ok" in d["checks"],
                        "g1_layer_save_patch_assignment_recorded",
                    ),
                    lambda d: (
                        "configuration_layer_count" in d["checks"],
                        "g1_layer_save_configuration_layer_count_recorded",
                    ),
                    lambda d: (
                        "importer_configuration_layer_save_intercepted" in d["checks"],
                        "g1_layer_save_importer_interception_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_layer_save_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_layer_save_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_in_memory_import_probe",
                "res/tracking/g1_urdf_in_memory_import/tracking_g1_urdf_in_memory_import_probe.json",
                [
                    lambda d: (
                        d.get("status")
                        in {
                            "ok_with_valid_exported_current_stage_g1_usd",
                            "ok_with_current_stage_robot_but_export_invalid",
                            "ok_with_parse_success_but_current_stage_empty",
                            "ok_with_in_memory_import_failed",
                            "ok_with_vulkan_device_lost_before_payload",
                        },
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["app_reached_after_app"], "g1_in_memory_app_after_app"),
                    lambda d: (
                        "vulkan_device_lost_recorded" in d["checks"],
                        "g1_in_memory_vulkan_marker_recorded",
                    ),
                    lambda d: (
                        d["checks"]["dest_path_was_empty"],
                        "g1_in_memory_dest_path_empty",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_in_memory_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_in_memory_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_simulationapp_in_memory_import_probe",
                "res/tracking/g1_urdf_simulationapp_in_memory_import/tracking_g1_urdf_simulationapp_in_memory_import_probe.json",
                [
                    lambda d: (
                        d.get("status")
                        in {
                            "ok_with_valid_exported_current_stage_g1_usd",
                            "ok_with_current_stage_robot_but_export_invalid",
                            "ok_with_parse_success_but_current_stage_empty",
                            "ok_with_simulationapp_in_memory_import_failed",
                            "ok_with_vulkan_device_lost_before_payload",
                        },
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["app_reached_after_app"], "g1_simapp_in_memory_app_after_app"),
                    lambda d: (
                        d["checks"]["isaaclab_headless_experience_exists"],
                        "g1_simapp_in_memory_experience_exists",
                    ),
                    lambda d: (
                        d["checks"]["in_memory_stage_branch_recorded"],
                        "g1_simapp_in_memory_branch_recorded",
                    ),
                    lambda d: (
                        "vulkan_device_lost_recorded" in d["checks"],
                        "g1_simapp_in_memory_vulkan_marker_recorded",
                    ),
                    lambda d: (
                        d["checks"]["dest_path_was_empty"],
                        "g1_simapp_in_memory_dest_path_empty",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_simapp_in_memory_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_simapp_in_memory_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_in_memory_variant_matrix_probe",
                "res/tracking/g1_urdf_in_memory_variant_matrix/tracking_g1_urdf_in_memory_variant_matrix_probe.json",
                [
                    lambda d: (
                        d.get("status")
                        in {
                            "ok_with_valid_exported_current_stage_g1_usd",
                            "ok_with_current_stage_robot_but_export_invalid",
                            "ok_with_parse_success_but_current_stage_empty",
                            "ok_with_all_variants_vulkan_device_lost_before_payload",
                            "ok_with_no_valid_g1_usd",
                        },
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["case_count"] >= 3, "g1_variant_matrix_case_count"),
                    lambda d: (d["checks"]["gpu5_case_recorded"], "g1_variant_matrix_gpu5_recorded"),
                    lambda d: (d["checks"]["gpu6_case_recorded"], "g1_variant_matrix_gpu6_recorded"),
                    lambda d: (
                        d["checks"]["any_valid_exported_g1_usd"] is False
                        or d.get("status") == "ok_with_valid_exported_current_stage_g1_usd",
                        "g1_variant_matrix_valid_usd_status_consistent",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_variant_matrix_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_variant_matrix_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_in_memory_gpu4_probe",
                "res/tracking/g1_urdf_in_memory_gpu4_probe/tracking_g1_urdf_in_memory_gpu4_probe.json",
                [
                    lambda d: (
                        d.get("status") in {"ok_official_g1_in_memory_import_export", "ok_with_vulkan_device_lost_blocker"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["tracking_python_exists"], "g1_gpu4_tracking_python_exists"),
                    lambda d: (d["checks"]["g1_urdf_exists"], "g1_gpu4_urdf_exists"),
                    lambda d: (d["checks"]["project_egl_icd_exists"], "g1_gpu4_project_egl_icd_exists"),
                    lambda d: (d["checks"]["app_launcher_reached"], "g1_gpu4_app_launcher_reached"),
                    lambda d: (d["checks"]["in_memory_import_returned"], "g1_gpu4_import_returned"),
                    lambda d: (d["checks"]["export_exists"], "g1_gpu4_export_exists"),
                    lambda d: (d["markers"]["vulkan_device_lost"], "g1_gpu4_vulkan_device_lost_recorded"),
                    lambda d: ("latest_blocker" in d, "g1_gpu4_latest_blocker_recorded"),
                    lambda d: (d["checks"]["does_not_claim_motion_npz"], "g1_gpu4_no_motion_npz_claim"),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_replay"],
                        "g1_gpu4_no_paper_replay_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_gpu4_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_in_memory_gpu4_export_structure_audit",
                "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit/"
                "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_physics_usd_export_but_vulkan_device_lost",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["source_probe_records_import_returned"], "g1_export_import_returned"),
                    lambda d: (d["checks"]["source_probe_records_vulkan_device_lost"], "g1_export_vulkan_recorded"),
                    lambda d: (d["checks"]["export_exists"], "g1_export_exists"),
                    lambda d: (d["checks"]["export_nonempty_large"], "g1_export_nonempty_large"),
                    lambda d: (d["checks"]["default_prim_g1"], "g1_export_default_prim"),
                    lambda d: (d["checks"]["has_articulation_root_api"], "g1_export_articulation_root"),
                    lambda d: (d["checks"]["has_rigid_body_api"], "g1_export_rigid_bodies"),
                    lambda d: (d["checks"]["has_physics_revolute_joints"], "g1_export_revolute_joints"),
                    lambda d: (d["checks"]["all_29_action_joints_present"], "g1_export_all_action_joints"),
                    lambda d: (d["checks"]["payload_not_recorded"], "g1_export_no_payload"),
                    lambda d: (d["checks"]["does_not_claim_official_replay"], "g1_export_no_official_replay_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_export_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_preconverted_asset_audit",
                "res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json",
                [
                    lambda d: (
                        d.get("status") in {"ok_with_reference_usd_candidate", "ok_with_no_official_preconverted_g1_usd"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["candidate_count_positive"], "g1_preconverted_candidate_count_positive"),
                    lambda d: (d["checks"]["usd_candidate_count_positive"], "g1_preconverted_usd_candidate_count"),
                    lambda d: (d["checks"]["official_mesh_usd_present"], "g1_preconverted_official_mesh_usd_present"),
                    lambda d: (
                        d["checks"]["official_full_robot_preconverted_g1_usd_absent"],
                        "g1_preconverted_no_official_full_robot_usd",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_reference_usd_as_official"],
                        "g1_preconverted_reference_not_official",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_preconverted_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_preconverted_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_reference_usd_compatibility_audit",
                "res/tracking/g1_reference_usd_compatibility_audit/tracking_g1_reference_usd_compatibility_audit.json",
                [
                    lambda d: (
                        d.get("status")
                        in {"ok_with_resource_adjusted_usd_compatible", "ok_with_reference_usd_incompatible_or_partial"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["reference_stage_open_ok"], "g1_reference_usd_stage_open"),
                    lambda d: (
                        d["checks"]["all_target_bodies_in_reference_usd"],
                        "g1_reference_usd_target_bodies_present",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_asset"],
                        "g1_reference_usd_not_official_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_reference_usd_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_reference_usd_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_official_urdf_skeleton_usd_audit",
                "res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json",
                [
                    lambda d: (
                        d.get("status") in {"ok_with_minimal_29dof_skeleton_usd", "ok_with_skeleton_usd_partial"},
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["skeleton_contract_ok"], "g1_skeleton_usd_contract_ok"),
                    lambda d: (
                        d["checks"]["all_action_joints_revolute_in_skeleton"],
                        "g1_skeleton_usd_action_joints_present",
                    ),
                    lambda d: (
                        d["checks"]["all_target_bodies_in_skeleton"],
                        "g1_skeleton_usd_target_bodies_present",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_converter_success"],
                        "g1_skeleton_usd_no_converter_success_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_motion_npz"],
                        "g1_skeleton_usd_no_motion_npz_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_skeleton_usd_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_file_artifact(
                "tracking_g1_official_urdf_29dof_skeleton_usda",
                "res/tracking/g1_official_urdf_skeleton_usd/g1_official_urdf_29dof_skeleton.usda",
            ),
            check_json_artifact(
                "tracking_g1_urdf_physical_asset_contract_audit",
                "res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_physical_contract_ready_for_converter_scaffold",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["link_count_40"], "g1_physical_contract_link_count"),
                    lambda d: (d["checks"]["nonfixed_joint_count_29"], "g1_physical_contract_nonfixed_joints"),
                    lambda d: (d["checks"]["all_visual_mesh_files_exist"], "g1_physical_contract_meshes_exist"),
                    lambda d: (d["checks"]["collision_element_count_29"], "g1_physical_contract_collision_count"),
                    lambda d: (
                        d["checks"]["all_nonfixed_joints_have_action_drive_rows"],
                        "g1_physical_contract_drive_rows",
                    ),
                    lambda d: (
                        d["checks"]["missing_inertial_links_recorded"],
                        "g1_physical_contract_inertial_gaps_recorded",
                    ),
                    lambda d: (
                        d["checks"]["physical_fidelity_complete_for_replay"] is False,
                        "g1_physical_contract_no_replay_fidelity_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_replay_success"],
                        "g1_physical_contract_no_replay_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_physical_contract_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_urdf_source_equivalence_audit",
                "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_source_differences_recorded",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["download_and_reproduction_data_sha256_match"],
                        "g1_urdf_download_reprodata_sha_match",
                    ),
                    lambda d: (
                        d["checks"]["download_and_reproduction_data_structurally_identical"],
                        "g1_urdf_download_reprodata_structural_match",
                    ),
                    lambda d: (
                        d["checks"]["whole_body_tracking_has_same_29_nonfixed_action_joints"],
                        "g1_urdf_wbt_same_29_action_joints",
                    ),
                    lambda d: (
                        d["checks"]["whole_body_tracking_support_link_difference_recorded"],
                        "g1_urdf_wbt_support_link_diff_recorded",
                    ),
                    lambda d: (
                        d["checks"]["whole_body_tracking_support_joint_difference_recorded"],
                        "g1_urdf_wbt_support_joint_diff_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_urdf_sources_identical"],
                        "g1_urdf_no_identical_source_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_converter_success"],
                        "g1_urdf_no_converter_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_replay_success"],
                        "g1_urdf_no_replay_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_urdf_source_equivalence_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_enriched_usd_probe",
                "res/tracking/g1_resource_adjusted_enriched_usd/tracking_g1_resource_adjusted_enriched_usd_probe.json",
                [
                    lambda d: (
                        d.get("status") == "ok_with_resource_adjusted_enriched_usd_scaffold",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (d["checks"]["readback_link_count_40"], "g1_enriched_usd_link_count"),
                    lambda d: (d["checks"]["readback_mass_api_count_37"], "g1_enriched_usd_mass_api_count"),
                    lambda d: (d["checks"]["readback_visual_mesh_reference_count_35"], "g1_enriched_usd_visual_refs"),
                    lambda d: (d["checks"]["readback_collision_api_count_29"], "g1_enriched_usd_collision_api"),
                    lambda d: (d["checks"]["readback_joint_limit_count_29"], "g1_enriched_usd_joint_limits"),
                    lambda d: (
                        d["checks"]["readback_joint_drive_metadata_count_29"],
                        "g1_enriched_usd_joint_drives",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_converter_success"],
                        "g1_enriched_usd_no_official_converter_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_replay_success"],
                        "g1_enriched_usd_no_replay_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_enriched_usd_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_file_artifact(
                "tracking_g1_resource_adjusted_enriched_usda",
                "res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda",
            ),
            check_json_artifact(
                "gpu_resource_audit",
                "res/setup/gpu_resource_audit/gpu_resource_audit.json",
                [
                    status_ok,
                    lambda d: (d["gpu_count"] == 8, "gpu_count_8"),
                    lambda d: (d["rows_written"] >= 24, "gpu_metrics_rows_at_least_24"),
                    lambda d: (d["checks"]["gpu_metrics_csv_written"], "gpu_metrics_csv_written"),
                    lambda d: (
                        d["checks"]["row_count_matches_samples_x_gpus"],
                        "gpu_rows_match_samples_x_gpus",
                    ),
                    lambda d: (d["checks"]["has_goal_required_columns"], "gpu_goal_columns_present"),
                    lambda d: (d["checks"]["does_not_modify_power_or_clocks"], "gpu_no_power_clock_changes"),
                    lambda d: (d["checks"]["does_not_create_artificial_load"], "gpu_no_artificial_load"),
                    lambda d: (
                        d["checks"]["does_not_claim_training_utilization"],
                        "gpu_no_training_util_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "gpu_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "run_management_audit",
                "res/run_management_audit/run_management_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["run_id_matches_goal_pattern"], "run_id_matches_goal_pattern"),
                    lambda d: (d["checks"]["all_required_files_exist"], "run_required_files_exist"),
                    lambda d: (d["checks"]["all_required_dirs_exist"], "run_required_dirs_exist"),
                    lambda d: (d["checks"]["status_is_allowed"], "run_status_allowed"),
                    lambda d: (
                        d["checks"]["does_not_mark_success_without_training"],
                        "run_no_false_success",
                    ),
                    lambda d: (
                        d["checks"]["metrics_json_has_required_runtime_fields"],
                        "run_metrics_runtime_fields",
                    ),
                    lambda d: (d["checks"]["gpu_metrics_present"], "run_gpu_metrics_present"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "run_management_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "checkpoint_resume_smoke",
                "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["checkpoint_file_exists"], "resume_checkpoint_file_exists"),
                    lambda d: (
                        d["checks"]["checkpoint_contains_seed_step_state"],
                        "resume_checkpoint_contains_state",
                    ),
                    lambda d: (
                        d["checks"]["resume_step_matches_checkpoint"],
                        "resume_step_matches_checkpoint",
                    ),
                    lambda d: (
                        d["checks"]["resumed_matches_uninterrupted"],
                        "resume_matches_uninterrupted",
                    ),
                    lambda d: (
                        d["checks"]["status_success_but_not_training"],
                        "resume_success_but_not_training",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_model_checkpoint"],
                        "resume_no_model_checkpoint_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "resume_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "full_run_deliverable_gap_audit",
                "res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["run_directory_count"] >= 3, "full_run_gap_run_count_at_least_3"),
                    lambda d: (
                        d["metrics"]["schema_complete_run_count"] >= 3,
                        "full_run_gap_schema_complete_at_least_3",
                    ),
                    lambda d: (
                        d["metrics"]["valid_training_run_count"] == 0,
                        "full_run_gap_no_valid_training_runs",
                    ),
                    lambda d: (
                        d["metrics"]["diagnostic_or_debug_run_count"] >= 3,
                        "full_run_gap_debug_run_count_at_least_3",
                    ),
                    lambda d: (
                        d["metrics"]["run_with_nonempty_checkpoint_dir_count"] >= 2,
                        "full_run_gap_debug_checkpoints_recorded",
                    ),
                    lambda d: (
                        d["metrics"]["run_with_nonempty_figures_dir_count"] >= 1,
                        "full_run_gap_debug_figure_recorded",
                    ),
                    lambda d: (
                        d["metrics"]["valid_training_run_with_nonempty_videos_count"] == 0,
                        "full_run_gap_no_valid_training_videos",
                    ),
                    lambda d: (
                        d["metrics"]["run_with_training_endpoint_metrics_count"] >= 1,
                        "full_run_gap_debug_runtime_metrics_recorded",
                    ),
                    lambda d: (d["checks"]["no_valid_training_run_claimed"], "full_run_gap_no_valid_claim"),
                    lambda d: (
                        d["checks"]["all_diagnostic_or_debug_runs_not_marked_training"],
                        "full_run_gap_all_debug_not_training",
                    ),
                    lambda d: (
                        d["checks"]["output_gaps_or_debug_video_boundary_recorded"],
                        "full_run_gap_output_gaps_or_debug_video_boundary_recorded",
                    ),
                    lambda d: (
                        d["checks"]["at_least_one_training_endpoint_metric_gap_recorded"],
                        "full_run_gap_runtime_metric_gap_still_recorded",
                    ),
                    lambda d: (
                        d["checks"]["no_valid_training_run_videos"],
                        "full_run_gap_no_valid_training_run_videos",
                    ),
                    lambda d: (
                        d["checks"]["debug_videos_do_not_create_valid_training_run"],
                        "full_run_gap_debug_videos_not_valid",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_run_gap_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "failed_run_audit",
                "res/failed_runs/failed_run_audit/failed_run_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["all_required_failed_run_files_exist"],
                        "failed_run_required_files",
                    ),
                    lambda d: (d["checks"]["status_failed_not_success"], "failed_run_not_success"),
                    lambda d: (d["checks"]["run_id_recorded"], "failed_run_id_recorded"),
                    lambda d: (d["checks"]["config_recorded"], "failed_run_config_recorded"),
                    lambda d: (d["checks"]["checkpoint_absence_recorded"], "failed_checkpoint_absence_recorded"),
                    lambda d: (d["checks"]["last_log_records_errno28"], "failed_run_errno28_log"),
                    lambda d: (d["checks"]["gpu_status_recorded"], "failed_run_gpu_status"),
                    lambda d: (d["checks"]["failure_reason_recorded"], "failed_run_reason"),
                    lambda d: (d["checks"]["resolution_plan_recorded"], "failed_run_resolution"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "failed_run_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_train_entry_failed_run_audit",
                "res/failed_runs/official_train_entry_failed_run_audit/official_train_entry_failed_run_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["all_required_failed_run_files_exist"],
                        "official_train_failed_run_required_files",
                    ),
                    lambda d: (
                        d["checks"]["status_failed_not_success"],
                        "official_train_failed_run_not_success",
                    ),
                    lambda d: (
                        d["checks"]["run_id_recorded"],
                        "official_train_failed_run_id_recorded",
                    ),
                    lambda d: (
                        d["checks"]["retry_json_exists"],
                        "official_train_failed_run_retry_json_exists",
                    ),
                    lambda d: (
                        d["checks"]["retry_classified_blocked_inotify"],
                        "official_train_failed_run_blocked_inotify",
                    ),
                    lambda d: (
                        d["checks"]["config_records_command"],
                        "official_train_failed_run_config_records_command",
                    ),
                    lambda d: (
                        d["checks"]["checkpoint_absence_recorded"],
                        "official_train_failed_run_no_checkpoint",
                    ),
                    lambda d: (
                        d["checks"]["last_log_records_inotify_failure"],
                        "official_train_failed_run_inotify_log",
                    ),
                    lambda d: (
                        d["checks"]["gpu_status_recorded"],
                        "official_train_failed_run_gpu_status",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_training_success"],
                        "official_train_failed_run_no_success_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_train_failed_run_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "resolved_reproduction_config",
                "res/config/resolved_reproduction_config.json",
                [
                    status_ok,
                    lambda d: (d["tracking"]["ppo"]["max_iterations"] == 30000, "config_tracking_max_iterations"),
                    lambda d: (d["tracking"]["target_body_count"] == 14, "config_tracking_body_count_14"),
                    lambda d: (d["vae"]["latent_dim"] == 32, "config_vae_latent_dim_32"),
                    lambda d: (d["diffusion"]["training"]["batch_size"] == 512, "config_diffusion_batch_512"),
                    lambda d: (
                        d["diffusion"]["training"]["denoising_steps"] == 20
                        if "denoising_steps" in d["diffusion"]["training"]
                        else d["diffusion"]["denoising_steps"] == 20,
                        "config_diffusion_steps_20",
                    ),
                    lambda d: (d["checks"]["vae_contract_passes"], "config_vae_contract_passes"),
                    lambda d: (d["checks"]["diffusion_schedule_passes"], "config_diffusion_schedule_passes"),
                    lambda d: (d["checks"]["run_schema_available"], "config_run_schema_available"),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_paper_results"],
                        "config_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "config_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "artifact_manifest",
                "res/artifact_manifest/artifact_manifest.json",
                [
                    status_ok,
                    lambda d: (d["artifact_count"] >= 25, "artifact_manifest_count"),
                    lambda d: (d["missing_count"] == 0, "artifact_manifest_missing_zero"),
                    lambda d: (d["checks"]["all_manifest_artifacts_exist"], "artifact_manifest_all_exist"),
                    lambda d: (d["checks"]["raw_sources_hashed"], "artifact_manifest_raw_sources_hashed"),
                    lambda d: (d["checks"]["env_locks_hashed"], "artifact_manifest_env_locks_hashed"),
                    lambda d: (d["checks"]["final_report_hashed"], "artifact_manifest_final_report_hashed"),
                    lambda d: (
                        d["checks"]["tracking_local_smoke_scripts_hashed"],
                        "artifact_manifest_tracking_local_smoke_scripts_hashed",
                    ),
                    lambda d: (
                        d["checks"]["atomic_write_used"],
                        "artifact_manifest_atomic_write_used",
                    ),
                    lambda d: (d["checks"]["does_not_modify_raw_downloads"], "artifact_manifest_raw_readonly"),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "artifact_manifest_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "artifact_manifest_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "download_source_integrity_audit",
                "res/source_integrity/download_source_integrity/download_source_integrity_audit.json",
                [
                    status_ok,
                    lambda d: (d["file_count"] >= 6000, "download_integrity_manifest_rows_ge_6000"),
                    lambda d: (d["total_size_bytes"] > 6_000_000_000, "download_integrity_size_gt_6gb"),
                    lambda d: (d["required_hash_file_count"] >= 17, "download_integrity_required_hashes"),
                    lambda d: (d["reference_hash_file_count"] >= 8, "download_integrity_reference_hashes"),
                    lambda d: (d["checks"]["download_root_exists"], "download_integrity_root_exists"),
                    lambda d: (d["checks"]["downloaded_files_manifest_exists"], "download_integrity_manifest_exists"),
                    lambda d: (d["checks"]["required_paths_exist"], "download_integrity_required_paths_exist"),
                    lambda d: (d["checks"]["all_required_rows_have_sha256"], "download_integrity_required_sha256"),
                    lambda d: (d["checks"]["all_reference_rows_have_sha256"], "download_integrity_reference_sha256"),
                    lambda d: (d["checks"]["paper_pdf_and_source_present"], "download_integrity_paper_present"),
                    lambda d: (d["checks"]["official_dataset_and_code_present"], "download_integrity_official_present"),
                    lambda d: (d["checks"]["dependency_snapshots_present"], "download_integrity_deps_present"),
                    lambda d: (d["checks"]["download_manifests_present"], "download_integrity_manifests_present"),
                    lambda d: (
                        d["checks"]["does_not_modify_raw_downloads"],
                        "download_integrity_raw_readonly",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "download_integrity_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "run_log_config_catalog",
                "res/run_log_config_catalog/run_log_config_catalog.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["file_count"] >= 70, "run_log_catalog_file_count"),
                    lambda d: (d["metrics"]["log_file_count"] >= 20, "run_log_catalog_log_count"),
                    lambda d: (d["metrics"]["config_file_count"] >= 3, "run_log_catalog_config_count"),
                    lambda d: (d["metrics"]["valid_training_run_count"] == 0, "run_log_catalog_no_training_claim"),
                    lambda d: (d["checks"]["hashes_recorded_for_all_files"], "run_log_catalog_hashes"),
                    lambda d: (d["checks"]["run_status_files_are_indexed"], "run_log_catalog_status_files"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "run_log_catalog_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "experiment_protocol_audit",
                "res/docs/experiment_protocol_audit/experiment_protocol_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] >= 19, "experiment_protocol_rows"),
                    lambda d: (d["missing_count"] == 0, "experiment_protocol_missing_zero"),
                    lambda d: (d["checks"]["all_phases_present"], "experiment_protocol_all_phases"),
                    lambda d: (d["checks"]["run_contract_present"], "experiment_protocol_run_contract"),
                    lambda d: (d["checks"]["failure_handling_present"], "experiment_protocol_failure_handling"),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "experiment_protocol_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "experiment_protocol_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "readme_audit",
                "res/docs/readme_audit/readme_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] >= 19, "readme_rows"),
                    lambda d: (d["missing_count"] == 0, "readme_missing_zero"),
                    lambda d: (d["checks"]["points_to_final_report"], "readme_points_to_final_report"),
                    lambda d: (d["checks"]["points_to_experiment_protocol"], "readme_points_to_protocol"),
                    lambda d: (d["checks"]["points_to_master_audit"], "readme_points_to_master_audit"),
                    lambda d: (d["checks"]["documents_raw_download_readonly"], "readme_raw_download_readonly"),
                    lambda d: (
                        d["checks"]["documents_current_incomplete_boundary"],
                        "readme_incomplete_boundary",
                    ),
                    lambda d: (d["checks"]["documents_major_blockers"], "readme_major_blockers"),
                    lambda d: (d["checks"]["documents_no_fabrication_rule"], "readme_no_fabrication"),
                    lambda d: (d["checks"]["documents_failed_run_retention_rule"], "readme_failed_run_retention"),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "readme_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "readme_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "patch_inventory_audit",
                "res/code/patch_inventory_audit/patch_inventory_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["official_repo_count"] == 3, "patch_inventory_official_repos_3"),
                    lambda d: (d["checks"]["patch_directory_exists"], "patch_inventory_dir_exists"),
                    lambda d: (d["checks"]["patch_file_count_recorded"], "patch_inventory_patch_count_recorded"),
                    lambda d: (d["checks"]["official_heads_recorded"], "patch_inventory_heads_recorded"),
                    lambda d: (d["checks"]["status_timeouts_recorded"], "patch_inventory_timeouts_recorded"),
                    lambda d: (d["checks"]["tracked_changes_recorded"], "patch_inventory_changes_recorded"),
                    lambda d: (
                        d["checks"]["does_not_claim_patch_series_complete"],
                        "patch_inventory_no_patch_series_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "patch_inventory_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "patch_snapshot_audit",
                "res/code/patch_snapshot_audit/patch_snapshot_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["snapshot_row_count"] >= 1, "patch_snapshot_rows"),
                    lambda d: (d["checks"]["snapshot_dir_exists"], "patch_snapshot_dir_exists"),
                    lambda d: (d["checks"]["all_patch_files_exist"], "patch_snapshot_files_exist"),
                    lambda d: (d["checks"]["all_patch_files_hashed"], "patch_snapshot_files_hashed"),
                    lambda d: (d["checks"]["all_statuses_not_timed_out"], "patch_snapshot_status_not_timeout"),
                    lambda d: (
                        d["checks"]["does_not_modify_official_worktrees"],
                        "patch_snapshot_no_worktree_modification_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_functional_patch_series"],
                        "patch_snapshot_no_functional_series_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "patch_snapshot_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_config_audit",
                "res/tracking/smoke_config_audit/tracking_config_audit.json",
                [
                    lambda d: (d["g1_tracking_target_bodies"]["anchor_in_body_names"], "anchor_in_body_names"),
                    lambda d: (d["g1_tracking_target_bodies"]["body_count"] == 14, "target_body_count_14"),
                ],
            ),
            check_json_artifact(
                "tracking_official_source_contract",
                "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["target_body_count_14"], "tracking_contract_target_body_count_14"),
                    lambda d: (d["checks"]["anchor_torso_link"], "tracking_contract_anchor_torso_link"),
                    lambda d: (d["checks"]["g1_action_scale_assigned"], "tracking_contract_action_scale_assigned"),
                    lambda d: (d["checks"]["control_frequency_50hz"], "tracking_contract_control_frequency_50hz"),
                    lambda d: (d["checks"]["policy_obs_terms_8"], "tracking_contract_policy_obs_terms_8"),
                    lambda d: (d["checks"]["critic_obs_terms_10"], "tracking_contract_critic_obs_terms_10"),
                    lambda d: (d["checks"]["reward_terms_9"], "tracking_contract_reward_terms_9"),
                    lambda d: (d["checks"]["ppo_matches_official_key_values"], "tracking_contract_ppo_key_values"),
                    lambda d: (d["checks"]["urdf_meshes_all_present"], "tracking_contract_urdf_meshes_present"),
                    lambda d: (
                        d["checks"]["all_non_fixed_urdf_joints_covered_by_action_regex"],
                        "tracking_contract_action_regex_covers_urdf_joints",
                    ),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "tracking_contract_no_kit_or_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_contract_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_action_scale_audit",
                "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["urdf_nonfixed_joint_count_29"], "g1_action_scale_urdf_joint_count_29"),
                    lambda d: (d["checks"]["expanded_joint_rows_29"], "g1_action_scale_rows_29"),
                    lambda d: (d["checks"]["all_joints_matched_once"], "g1_action_scale_all_joints_matched_once"),
                    lambda d: (d["checks"]["actuator_group_count_5"], "g1_action_scale_group_count_5"),
                    lambda d: (d["checks"]["group_counts_expected"], "g1_action_scale_group_counts_expected"),
                    lambda d: (
                        d["checks"]["action_scale_formula_matches_official"],
                        "g1_action_scale_formula_matches_official",
                    ),
                    lambda d: (d["checks"]["all_action_scales_positive"], "g1_action_scales_positive"),
                    lambda d: (d["checks"]["natural_frequency_10hz_formula"], "g1_action_scale_10hz_formula"),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "g1_action_scale_no_kit_or_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "g1_action_scale_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_reward_formula_audit",
                "res/tracking/reward_formula_audit/tracking_reward_formula_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["reward_term_count_9"], "tracking_reward_terms_9"),
                    lambda d: (d["checks"]["motion_exp_reward_term_count_6"], "tracking_motion_exp_rewards_6"),
                    lambda d: (d["checks"]["regularizer_term_count_3"], "tracking_regularizer_rewards_3"),
                    lambda d: (
                        d["checks"]["expected_motion_functions_present"],
                        "tracking_reward_expected_motion_functions",
                    ),
                    lambda d: (
                        d["checks"]["source_uses_exp_negative_error_over_std_squared"],
                        "tracking_reward_exp_negative_error_over_std_squared",
                    ),
                    lambda d: (d["checks"]["numeric_rows_30"], "tracking_reward_numeric_rows_30"),
                    lambda d: (d["checks"]["reward_at_zero_is_one"], "tracking_reward_at_zero_is_one"),
                    lambda d: (
                        d["checks"]["weighted_at_zero_matches_weight"],
                        "tracking_reward_weighted_zero_matches_weight",
                    ),
                    lambda d: (
                        d["checks"]["rewards_monotone_nonincreasing"],
                        "tracking_reward_monotone_nonincreasing",
                    ),
                    lambda d: (d["checks"]["std_values_match_official"], "tracking_reward_std_values_official"),
                    lambda d: (d["checks"]["weight_values_match_official"], "tracking_reward_weight_values_official"),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "tracking_reward_no_kit_or_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_reward_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_observation_action_schema_audit",
                "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["policy_term_count_8"], "tracking_obs_policy_terms_8"),
                    lambda d: (d["checks"]["critic_term_count_10"], "tracking_obs_critic_terms_10"),
                    lambda d: (d["checks"]["policy_dimension_160"], "tracking_obs_policy_dim_160"),
                    lambda d: (d["checks"]["critic_dimension_286"], "tracking_obs_critic_dim_286"),
                    lambda d: (d["checks"]["action_dimension_29"], "tracking_obs_action_dim_29"),
                    lambda d: (d["checks"]["target_body_count_14"], "tracking_obs_target_bodies_14"),
                    lambda d: (d["checks"]["policy_noise_terms_6"], "tracking_obs_policy_noise_terms_6"),
                    lambda d: (d["checks"]["critic_noise_terms_0"], "tracking_obs_critic_noise_terms_0"),
                    lambda d: (
                        d["checks"]["observation_source_has_body_frame_transforms"],
                        "tracking_obs_body_frame_transforms",
                    ),
                    lambda d: (d["checks"]["observation_source_uses_rot6d"], "tracking_obs_rot6d"),
                    lambda d: (d["checks"]["fixture_count_3"], "tracking_obs_fixture_count_3"),
                    lambda d: (d["checks"]["fixture_joint_count_29"], "tracking_obs_fixture_joints_29"),
                    lambda d: (d["checks"]["fixture_fps_50"], "tracking_obs_fixture_fps_50"),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "tracking_obs_no_kit_or_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_obs_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_randomization_termination_audit",
                "res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["event_term_count_4"], "tracking_randomization_event_terms_4"),
                    lambda d: (d["checks"]["event_range_rows_13"], "tracking_randomization_event_rows_13"),
                    lambda d: (d["checks"]["termination_term_count_4"], "tracking_termination_terms_4"),
                    lambda d: (
                        d["checks"]["friction_ranges_match_official"],
                        "tracking_randomization_friction_ranges",
                    ),
                    lambda d: (
                        d["checks"]["joint_default_pos_range_match_official"],
                        "tracking_randomization_joint_default_pos_range",
                    ),
                    lambda d: (
                        d["checks"]["base_com_ranges_match_official"],
                        "tracking_randomization_base_com_ranges",
                    ),
                    lambda d: (
                        d["checks"]["push_velocity_ranges_match_official"],
                        "tracking_randomization_push_velocity_ranges",
                    ),
                    lambda d: (
                        d["checks"]["termination_thresholds_match_official"],
                        "tracking_termination_thresholds",
                    ),
                    lambda d: (
                        d["checks"]["ee_body_names_match_official"],
                        "tracking_termination_ee_body_names",
                    ),
                    lambda d: (
                        d["checks"]["termination_probe_strict_greater_than"],
                        "tracking_termination_strict_greater_than_probe",
                    ),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "tracking_randomization_no_kit_or_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_randomization_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_b_tracking_nonkit_suite",
                "res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["all_steps_pass"], "level_b_nonkit_suite_all_steps_pass"),
                    lambda d: (d["step_count"] == 13, "level_b_nonkit_suite_step_count_13"),
                    lambda d: (d["pass_count"] == 13, "level_b_nonkit_suite_pass_count_13"),
                    lambda d: (
                        d["checks"]["official_source_contract_ok"],
                        "level_b_nonkit_suite_official_source_contract",
                    ),
                    lambda d: (d["checks"]["g1_action_scale_rows_29"], "level_b_nonkit_suite_action_scale"),
                    lambda d: (d["checks"]["reward_formula_motion_terms_6"], "level_b_nonkit_suite_rewards"),
                    lambda d: (d["checks"]["observation_policy_dim_160"], "level_b_nonkit_suite_policy_dim"),
                    lambda d: (d["checks"]["observation_critic_dim_286"], "level_b_nonkit_suite_critic_dim"),
                    lambda d: (
                        d["checks"]["randomization_event_terms_4"],
                        "level_b_nonkit_suite_randomization",
                    ),
                    lambda d: (d["checks"]["fixture_count_3"], "level_b_nonkit_suite_fixture_count_3"),
                    lambda d: (
                        d["checks"]["tracking_local_smoke_preflight"],
                        "level_b_nonkit_suite_local_preflight",
                    )
                    if "tracking_local_smoke_preflight" in d["checks"]
                    else (d["checks"]["local_preflight_steps_6"], "level_b_nonkit_suite_local_preflight"),
                    lambda d: (
                        d["checks"]["adaptive_sampling_discrepancy_recorded"],
                        "level_b_nonkit_suite_adaptive_discrepancy",
                    ),
                    lambda d: (d["checks"]["onnx_contract_ok"], "level_b_nonkit_suite_onnx_contract"),
                    lambda d: (
                        d["checks"]["debug_onnx_export_ok"],
                        "level_b_nonkit_suite_debug_onnx_export",
                    ),
                    lambda d: (
                        d["checks"]["debug_onnx_file_written"],
                        "level_b_nonkit_suite_debug_onnx_file",
                    ),
                    lambda d: (
                        d["checks"]["debug_onnx_contract_match"],
                        "level_b_nonkit_suite_debug_onnx_contract",
                    ),
                    lambda d: (
                        d["checks"]["debug_onnx_inference_ok"],
                        "level_b_nonkit_suite_debug_onnx_inference",
                    ),
                    lambda d: (
                        d["checks"]["debug_onnx_reference_evaluator_loaded"],
                        "level_b_nonkit_suite_debug_onnx_reference_evaluator",
                    ),
                    lambda d: (
                        d["checks"]["debug_onnx_inference_outputs_match"],
                        "level_b_nonkit_suite_debug_onnx_inference_outputs",
                    ),
                    lambda d: (d["checks"]["mujoco_ros_contract_ok"], "level_b_nonkit_suite_mujoco_ros_contract"),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "level_b_nonkit_suite_no_kit_or_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "level_b_nonkit_suite_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_smoke_rerun_audit",
                "res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 3, "tracking_smoke_rerun_rows_3"),
                    lambda d: (d["failed_row_count"] == 0, "tracking_smoke_rerun_failures_zero"),
                    lambda d: (d["checks"]["nonkit_rerun_passed"], "tracking_smoke_nonkit_rerun_passed"),
                    lambda d: (d["checks"]["g1_urdf_checked"], "tracking_smoke_g1_urdf_checked"),
                    lambda d: (d["checks"]["g1_mesh_checked"], "tracking_smoke_g1_mesh_checked"),
                    lambda d: (
                        d["checks"]["kit_retry_log_records_errno28"],
                        "tracking_smoke_kit_errno28_recorded",
                    ),
                    lambda d: (
                        d["checks"]["current_inotify_below_retry_threshold"],
                        "tracking_smoke_inotify_below_threshold",
                    ),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "tracking_smoke_no_kit_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_smoke_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_official_train_entry_retry_audit",
                "res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["local_train_script_exists"],
                        "tracking_train_retry_local_train_exists",
                    ),
                    lambda d: (
                        d["checks"]["local_motion_npz_exists"],
                        "tracking_train_retry_motion_npz_exists",
                    ),
                    lambda d: (d["checks"]["log_written"], "tracking_train_retry_log_written"),
                    lambda d: (d["checks"]["tail_log_written"], "tracking_train_retry_tail_written"),
                    lambda d: (
                        d["checks"]["current_inotify_below_retry_target"],
                        "tracking_train_retry_inotify_below_target",
                    ),
                    lambda d: (
                        d["classification"]["retry_result"] == "blocked_inotify",
                        "tracking_train_retry_blocked_inotify",
                    ),
                    lambda d: (
                        d["classification"]["has_inotify_watch_failure"],
                        "tracking_train_retry_watch_failure_seen",
                    ),
                    lambda d: (
                        d["classification"]["has_no_space_left_on_device"],
                        "tracking_train_retry_no_space_signature_seen",
                    ),
                    lambda d: (
                        d["checks"]["retry_did_not_reach_valid_training_success"],
                        "tracking_train_retry_no_success_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_training_success"],
                        "tracking_train_retry_does_not_claim_success",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_train_retry_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "kit_inotify_budget_audit",
                "res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 5, "kit_inotify_budget_rows_5"),
                    lambda d: (d["failed_row_count"] == 0, "kit_inotify_budget_failures_zero"),
                    lambda d: (d["checks"]["retry_log_records_errno28"], "kit_inotify_errno28_recorded"),
                    lambda d: (
                        isinstance(d["checks"].get("current_inotify_below_retry_target"), bool),
                        "kit_inotify_current_limit_state_recorded",
                    ),
                    lambda d: (
                        isinstance(d["checks"].get("directory_pressure_exceeds_current_limit"), bool),
                        "kit_inotify_directory_pressure_state_recorded",
                    ),
                    lambda d: (
                        d["checks"]["filesystem_capacity_not_primary_no_space_cause"],
                        "kit_inotify_disk_capacity_not_primary",
                    ),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "kit_inotify_no_kit_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "kit_inotify_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "inotify_live_usage_audit",
                "res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["sysctl_limits_read"], "inotify_live_sysctl_read"),
                    lambda d: (d["checks"]["process_fdinfo_scanned"], "inotify_live_fdinfo_scanned"),
                    lambda d: (d["metrics"]["total_inotify_watch_count"] >= 0, "inotify_live_watches_recorded"),
                    lambda d: (d["metrics"]["watch_headroom"] is not None, "inotify_live_watch_headroom_recorded"),
                    lambda d: (
                        d["metrics"]["max_watch_process"] is not None,
                        "inotify_live_top_process_recorded",
                    ),
                    lambda d: (
                        "fileWatcher" in d["metrics"]["max_watch_process"]["command"],
                        "inotify_live_top_process_vscode_filewatcher",
                    ),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "inotify_live_no_kit_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "inotify_live_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "vscode_watcher_exclude_audit",
                "res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["settings_file_exists"], "vscode_watcher_settings_exists"),
                    lambda d: (
                        d["checks"]["all_required_watcher_excludes_present"],
                        "vscode_watcher_excludes_complete",
                    ),
                    lambda d: (
                        d["checks"]["all_required_search_excludes_present"],
                        "vscode_search_excludes_complete",
                    ),
                    lambda d: (d["missing_count"] == 0, "vscode_watcher_missing_zero"),
                    lambda d: (
                        isinstance(d["live_usage_snapshot"]["live_still_saturated_after_settings_write"], bool),
                        "vscode_watcher_live_saturation_state_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_kill_vscode_or_modify_sysctl"],
                        "vscode_watcher_no_kill_or_sysctl",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vscode_watcher_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "kit_watcher_config_surface_audit",
                "res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 6, "kit_watcher_surface_rows_6"),
                    lambda d: (d["failed_row_count"] == 0, "kit_watcher_surface_failures_zero"),
                    lambda d: (
                        d["checks"]["python_app_loads_extension_roots"],
                        "kit_watcher_python_app_extension_roots",
                    ),
                    lambda d: (
                        d["checks"]["retry_failures_overlap_app_extension_roots"],
                        "kit_watcher_retry_overlap_roots",
                    ),
                    lambda d: (
                        d["checks"]["fswatcher_config_surfaces_exist"],
                        "kit_watcher_fswatcher_surfaces_exist",
                    ),
                    lambda d: (
                        d["checks"]["kit_watched_config_component_present"],
                        "kit_watcher_component_present",
                    ),
                    lambda d: (
                        d["checks"]["no_documented_global_disable_found"],
                        "kit_watcher_no_global_disable_found",
                    ),
                    lambda d: (
                        d["checks"]["does_not_modify_or_launch_kit"],
                        "kit_watcher_no_modify_or_launch",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "kit_watcher_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_import_gate_audit",
                "res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 6, "tracking_import_gate_rows_6"),
                    lambda d: (d["failed_row_count"] == 0, "tracking_import_gate_failures_zero"),
                    lambda d: (
                        d["checks"]["isaaclab_plain_import_passes"],
                        "tracking_import_isaaclab_plain_import",
                    ),
                    lambda d: (
                        d["checks"]["isaacsim_core_extension_source_exists"],
                        "tracking_import_isaacsim_core_source_exists",
                    ),
                    lambda d: (
                        d["checks"]["plain_python_lacks_isaacsim_core_extension"],
                        "tracking_import_plain_python_lacks_core",
                    ),
                    lambda d: (
                        d["checks"].get("tracking_deep_imports_blocked_by_kit_namespace")
                        or d["checks"].get("tracking_deep_imports_blocked_by_isaacsim_core"),
                        "tracking_import_deep_imports_blocked",
                    ),
                    lambda d: (
                        d["checks"]["official_tracking_python_sources_present"],
                        "tracking_import_sources_present",
                    ),
                    lambda d: (
                        d["checks"]["does_not_launch_kit_or_training"],
                        "tracking_import_no_kit_training",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_import_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_extension_namespace_probe",
                "res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 4, "tracking_namespace_rows_4"),
                    lambda d: (d["failed_row_count"] == 0, "tracking_namespace_failures_zero"),
                    lambda d: (
                        d["checks"]["core_namespace_paths_present"],
                        "tracking_namespace_core_paths_present",
                    ),
                    lambda d: (
                        d["checks"]["manual_namespace_changes_failure_mode"],
                        "tracking_namespace_failure_mode_changed",
                    ),
                    lambda d: (
                        d["checks"]["kit_runtime_dependency_still_blocks_deep_import"],
                        "tracking_namespace_kit_runtime_still_blocks",
                    ),
                    lambda d: (
                        d["checks"]["does_not_modify_or_launch_kit"],
                        "tracking_namespace_no_modify_or_launch",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_namespace_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "adaptive_sampling_discrepancy_audit",
                "res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["paper_source_has_u_0_1_2"],
                        "adaptive_paper_source_u_0_1_2",
                    ),
                    lambda d: (
                        d["checks"]["official_code_default_kernel_size_1"],
                        "adaptive_official_default_kernel_1",
                    ),
                    lambda d: (
                        d["checks"]["no_runtime_override_to_kernel_size_3_found"],
                        "adaptive_no_kernel_3_override_found",
                    ),
                    lambda d: (
                        d["checks"]["distribution_difference_detected"],
                        "adaptive_distribution_difference_detected",
                    ),
                    lambda d: (
                        d["checks"]["paper_pre_failure_mass_exceeds_code"],
                        "adaptive_paper_prefailure_mass_exceeds_code",
                    ),
                    lambda d: (
                        d["checks"]["discrepancy_boundary_recorded"],
                        "adaptive_discrepancy_boundary_recorded",
                    ),
                ],
            ),
            check_json_artifact(
                "motion_preprocessing_contract_audit",
                "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["official_and_raw_csv_to_npz_normalized_text_match"],
                        "motion_preprocess_raw_and_repro_match",
                    ),
                    lambda d: (
                        d["checks"]["g1_joint_count_29"],
                        "motion_preprocess_g1_joint_count_29",
                    ),
                    lambda d: (
                        d["checks"]["all_g1_csv_column_counts_match"],
                        "motion_preprocess_g1_csv_columns",
                    ),
                    lambda d: (
                        d["checks"]["all_g1_csv_values_finite"],
                        "motion_preprocess_g1_csv_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_g1_csv_quaternions_unit"],
                        "motion_preprocess_g1_quaternions_unit",
                    ),
                    lambda d: (
                        d["checks"]["csv_to_npz_declares_required_output_keys"],
                        "motion_preprocess_producer_keys",
                    ),
                    lambda d: (
                        d["checks"]["motion_loader_consumes_required_output_keys"],
                        "motion_preprocess_consumer_keys",
                    ),
                    lambda d: (
                        d["checks"]["validator_matches_required_output_keys"],
                        "motion_preprocess_validator_keys",
                    ),
                    lambda d: (
                        d["checks"]["local_smoke_patches_prepared"],
                        "motion_preprocess_local_smoke_patches",
                    ),
                    lambda d: (
                        d["local_smoke_patch_checks"]["local_runner_shell_syntax_valid"],
                        "motion_preprocess_local_runner_shell_syntax",
                    ),
                    lambda d: (
                        d["local_smoke_patch_checks"]["local_runner_executable"],
                        "motion_preprocess_local_runner_executable",
                    ),
                    lambda d: (
                        d["local_smoke_patch_checks"]["local_generated_manifest_present"],
                        "motion_preprocess_local_generated_manifest",
                    ),
                    lambda d: (
                        d["local_smoke_patch_checks"]["local_generated_scripts_exist"],
                        "motion_preprocess_local_generated_scripts",
                    ),
                    lambda d: (
                        d["checks"]["kit_execution_boundary_recorded"],
                        "motion_preprocess_kit_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_motion_npz_fixture",
                "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["all_outputs_exist"], "tracking_motion_fixture_outputs_exist"),
                    lambda d: (
                        d["checks"]["all_validator_contracts_pass"],
                        "tracking_motion_fixture_validator_contracts_pass",
                    ),
                    lambda d: (d["checks"]["all_arrays_finite"], "tracking_motion_fixture_arrays_finite"),
                    lambda d: (d["checks"]["output_fps_50"], "tracking_motion_fixture_output_fps_50"),
                    lambda d: (d["checks"]["joint_count_29"], "tracking_motion_fixture_joint_count_29"),
                    lambda d: (d["checks"]["full_urdf_body_count_40"], "tracking_motion_fixture_full_body_count_40"),
                    lambda d: (
                        d["checks"]["tracking_body_names_present"],
                        "tracking_motion_fixture_tracking_bodies_present",
                    ),
                    lambda d: (
                        d["checks"]["selected_tracking_body_count_14"],
                        "tracking_motion_fixture_tracking_body_count_14",
                    ),
                    lambda d: (
                        d["checks"]["quaternions_unit_within_1e_minus_10"],
                        "tracking_motion_fixture_quaternions_unit",
                    ),
                    lambda d: (
                        d["checks"]["kit_execution_boundary_recorded"],
                        "tracking_motion_fixture_kit_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_local_smoke_preflight",
                "res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json",
                [
                    status_ok,
                    lambda d: (d["step_count"] == 6, "tracking_local_preflight_step_count_6"),
                    lambda d: (d["pass_count"] == 6, "tracking_local_preflight_pass_count_6"),
                    lambda d: (d["checks"]["prepare_script_executed"], "tracking_local_preflight_prepare_executed"),
                    lambda d: (d["checks"]["runner_bash_syntax_valid"], "tracking_local_preflight_runner_syntax"),
                    lambda d: (d["checks"]["generated_scripts_compile"], "tracking_local_preflight_scripts_compile"),
                    lambda d: (d["checks"]["fixture_count_3"], "tracking_local_preflight_fixture_count_3"),
                    lambda d: (d["checks"]["fixture_validators_pass"], "tracking_local_preflight_validators_pass"),
                    lambda d: (d["checks"]["does_not_launch_kit_or_training"], "tracking_local_preflight_no_kit"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tracking_local_preflight_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "motion_tracking_controller_audit",
                "res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.json",
                [
                    lambda d: (
                        d["code_patterns"]["MotionOnnxPolicy.cpp"]["patterns"]["time_step_input"],
                        "onnx_policy_time_step_input",
                    ),
                    lambda d: (
                        d["code_patterns"]["MotionOnnxPolicy.cpp"]["patterns"]["reference_outputs"],
                        "onnx_policy_reference_outputs",
                    ),
                ],
            ),
            check_json_artifact(
                "mujoco_ros_launch_contract_audit",
                "res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["readme_mentions_ros2_jazzy"],
                        "mujoco_ros_readme_mentions_jazzy",
                    ),
                    lambda d: (
                        d["checks"]["package_declares_required_deps"],
                        "mujoco_ros_required_deps_declared",
                    ),
                    lambda d: (
                        d["checks"]["plugin_declares_motion_tracking_controller"],
                        "mujoco_ros_plugin_declared",
                    ),
                    lambda d: (
                        d["checks"]["controllers_yaml_500hz_manager_50hz_walking"],
                        "mujoco_ros_controller_rates",
                    ),
                    lambda d: (
                        d["checks"]["standby_joint_impedance_counts_match_29"],
                        "mujoco_ros_standby_joint_counts",
                    ),
                    lambda d: (
                        d["checks"]["mujoco_launch_declares_required_args"],
                        "mujoco_launch_required_args",
                    ),
                    lambda d: (
                        d["checks"]["mujoco_launch_uses_sim_node_and_plugin"],
                        "mujoco_launch_sim_node_plugin",
                    ),
                    lambda d: (
                        d["checks"]["real_launch_declares_required_args"],
                        "real_launch_required_args",
                    ),
                    lambda d: (
                        d["checks"]["real_launch_records_mcap_with_exclusions"],
                        "real_launch_rosbag_mcap",
                    ),
                    lambda d: (
                        d["checks"]["host_runtime_gate_recorded"],
                        "mujoco_ros_host_gate_recorded",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_deployment_controller_semantics_audit",
                "res/tracking/deployment_controller_semantics_audit/tracking_deployment_controller_semantics_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["metrics"]["audited_source_file_count"] == 14,
                        "deployment_semantics_source_file_count_14",
                    ),
                    lambda d: (
                        d["metrics"]["standby_joint_count"] == 29,
                        "deployment_semantics_standby_joint_count_29",
                    ),
                    lambda d: (
                        d["metrics"]["motion_command_dim_for_29_joints"] == 58,
                        "deployment_semantics_motion_command_dim_58",
                    ),
                    lambda d: (
                        d["metrics"]["motion_observation_dim_for_14_target_bodies"] == 135,
                        "deployment_semantics_motion_obs_dim_135",
                    ),
                    lambda d: (
                        d["checks"]["package_and_plugin_match_motion_tracking_controller"],
                        "deployment_semantics_package_plugin_match",
                    ),
                    lambda d: (
                        d["checks"]["package_declares_core_controller_dependencies"],
                        "deployment_semantics_core_deps_declared",
                    ),
                    lambda d: (
                        d["checks"]["launch_references_unitree_description_and_robot_state_publisher"],
                        "deployment_semantics_launch_unitree_rsvp",
                    ),
                    lambda d: (
                        d["checks"]["controller_yaml_has_expected_rates_and_29_joint_standby"],
                        "deployment_semantics_rates_and_joint_counts",
                    ),
                    lambda d: (
                        d["checks"]["sim_launch_starts_walking_real_launch_starts_standby"],
                        "deployment_semantics_sim_real_controller_modes",
                    ),
                    lambda d: (
                        d["checks"]["policy_time_step_reference_output_metadata_contract"],
                        "deployment_semantics_policy_contract",
                    ),
                    lambda d: (
                        d["checks"]["controller_motion_command_and_observation_alias_contract"],
                        "deployment_semantics_motion_obs_aliases",
                    ),
                    lambda d: (
                        d["checks"]["motion_alignment_uses_anchor_yaw_and_local_frame_observations"],
                        "deployment_semantics_anchor_yaw_local_frame",
                    ),
                    lambda d: (
                        d["checks"]["readme_records_real_robot_risk_and_remote_switches"],
                        "deployment_semantics_risk_and_remote_switches",
                    ),
                    lambda d: (
                        d["checks"]["host_ros2_runtime_unavailable_or_not_jazzy_noble"],
                        "deployment_semantics_host_gate",
                    ),
                    lambda d: (
                        d["checks"]["does_not_execute_ros_mujoco_or_robot"],
                        "deployment_semantics_no_execution",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "deployment_semantics_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_motion_policy_onnx_contract_fixture",
                "res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["input_count"] == 2, "onnx_fixture_input_count_2"),
                    lambda d: (d["metrics"]["output_count"] == 7, "onnx_fixture_output_count_7"),
                    lambda d: (d["metrics"]["metadata_count"] == 11, "onnx_fixture_metadata_count_11"),
                    lambda d: (d["metrics"]["obs_dim"] == 160, "onnx_fixture_obs_dim_160"),
                    lambda d: (d["metrics"]["action_dim"] == 29, "onnx_fixture_action_dim_29"),
                    lambda d: (d["metrics"]["target_body_count"] == 14, "onnx_fixture_target_body_count_14"),
                    lambda d: (d["metrics"]["npz_size_bytes"] > 0, "onnx_fixture_npz_nonempty"),
                    lambda d: (len(d["metrics"]["npz_sha256"]) == 64, "onnx_fixture_npz_sha256"),
                    lambda d: (d["checks"]["all_required_inputs_present"], "onnx_fixture_inputs_present"),
                    lambda d: (d["checks"]["all_required_outputs_present"], "onnx_fixture_outputs_present"),
                    lambda d: (d["checks"]["all_required_metadata_present"], "onnx_fixture_metadata_present"),
                    lambda d: (d["checks"]["body_outputs_match_14_target_bodies"], "onnx_fixture_body_outputs"),
                    lambda d: (d["checks"]["metadata_action_scale_matches_official_audit"], "onnx_fixture_action_scale"),
                    lambda d: (d["checks"]["fixture_rows_cover_official_exporter_and_consumer"], "onnx_fixture_contract_rows"),
                    lambda d: (d["checks"]["does_not_write_real_onnx"], "onnx_fixture_no_real_onnx"),
                    lambda d: (d["checks"]["does_not_claim_trained_policy"], "onnx_fixture_no_trained_policy_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "onnx_fixture_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_debug_motion_policy_onnx_export",
                "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json",
                [
                    status_ok,
                    lambda d: (d["onnx_size_bytes"] > 0, "debug_onnx_file_nonempty"),
                    lambda d: (len(d["onnx_sha256"]) == 64, "debug_onnx_sha256"),
                    lambda d: (d["checks"]["contract_fixture_status_ok"], "debug_onnx_contract_fixture_ok"),
                    lambda d: (d["checks"]["onnx_checker_passed"], "debug_onnx_checker_passed"),
                    lambda d: (d["checks"]["all_inputs_match_contract"], "debug_onnx_inputs_match_contract"),
                    lambda d: (d["checks"]["all_outputs_match_contract"], "debug_onnx_outputs_match_contract"),
                    lambda d: (d["checks"]["all_required_metadata_present"], "debug_onnx_metadata_present"),
                    lambda d: (d["checks"]["uses_project_python"], "debug_onnx_project_python"),
                    lambda d: (d["checks"]["does_not_use_trained_checkpoint"], "debug_onnx_no_checkpoint"),
                    lambda d: (d["checks"]["does_not_claim_policy_performance"], "debug_onnx_no_performance_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "debug_onnx_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_debug_motion_policy_onnx_inference",
                "res/tracking/debug_motion_policy_onnx_inference/tracking_debug_motion_policy_onnx_inference_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["output_count"] == 7, "debug_onnx_inference_outputs_7"),
                    lambda d: (d["metrics"]["max_abs_error"] <= 1e-7, "debug_onnx_inference_max_error"),
                    lambda d: (len(d["onnx_sha256"]) == 64, "debug_onnx_inference_sha256"),
                    lambda d: (d["checks"]["export_status_ok"], "debug_onnx_inference_export_ok"),
                    lambda d: (
                        d["checks"]["onnx_sha256_matches_export"],
                        "debug_onnx_inference_sha256_matches_export",
                    ),
                    lambda d: (d["checks"]["onnx_checker_passed"], "debug_onnx_inference_checker"),
                    lambda d: (
                        d["checks"]["reference_evaluator_loaded"],
                        "debug_onnx_inference_reference_evaluator",
                    ),
                    lambda d: (
                        d["checks"]["all_expected_outputs_returned"],
                        "debug_onnx_inference_all_outputs_returned",
                    ),
                    lambda d: (
                        d["checks"]["all_outputs_match_contract_values"],
                        "debug_onnx_inference_outputs_match_contract",
                    ),
                    lambda d: (d["checks"]["uses_project_python"], "debug_onnx_inference_project_python"),
                    lambda d: (d["checks"]["does_not_use_onnxruntime"], "debug_onnx_inference_no_onnxruntime"),
                    lambda d: (
                        d["checks"]["does_not_use_trained_checkpoint"],
                        "debug_onnx_inference_no_checkpoint",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_policy_performance"],
                        "debug_onnx_inference_no_performance_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "debug_onnx_inference_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "unitree_reference_onnx_contract",
                "res/tracking/motion_tracking_controller_audit/unitree_g1_policy_onnx_contract.json",
                [
                    lambda d: (
                        bool(d.get("missing_motion_inputs") or d.get("missing_motion_outputs") or d.get("missing_motion_metadata")),
                        "missing_motion_contract_fields",
                    )
                ],
            ),
            check_json_artifact(
                "tracking_onnx_export_contract_audit",
                "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["exporter_provides_all_required_inputs_outputs_metadata"],
                        "tracking_onnx_exporter_contract_complete",
                    ),
                    lambda d: (d["checks"]["exporter_clamps_time_step"], "tracking_onnx_time_step_clamp"),
                    lambda d: (d["checks"]["exporter_attaches_metadata"], "tracking_onnx_metadata_attach"),
                    lambda d: (d["checks"]["consumer_uses_time_step"], "tracking_onnx_consumer_time_step"),
                    lambda d: (
                        d["checks"]["consumer_uses_motion_reference_outputs"],
                        "tracking_onnx_consumer_motion_outputs",
                    ),
                    lambda d: (
                        d["checks"]["consumer_reads_anchor_body_metadata"],
                        "tracking_onnx_consumer_anchor_metadata",
                    ),
                    lambda d: (
                        d["checks"]["reference_onnx_does_not_satisfy_motion_contract"],
                        "tracking_onnx_reference_rejected",
                    ),
                    lambda d: (
                        d["checks"]["no_beyondmimic_policy_onnx_export_claimed"],
                        "tracking_onnx_no_false_export_claim",
                    ),
                ],
            ),
            check_json_artifact("synthetic_smoke", "res/level_c/synthetic_smoke/level_c_synthetic_smoke.json", [status_ok]),
            check_json_artifact(
                "level_c_debug_suite",
                "res/level_c/debug_suite/level_c_debug_suite.json",
                [
                    status_ok,
                    lambda d: (d["step_count"] == 10, "level_c_debug_suite_step_count_10"),
                    lambda d: (d["pass_count"] == 10, "level_c_debug_suite_pass_count_10"),
                    lambda d: (d["checks"]["all_steps_pass"], "level_c_debug_suite_all_steps_pass"),
                    lambda d: (d["checks"]["vae_accumulation_matches_paper"], "level_c_debug_suite_vae_accum"),
                    lambda d: (
                        d["checks"]["diffusion_equation_denoising_steps_20"],
                        "level_c_debug_suite_diffusion_steps_20",
                    ),
                    lambda d: (d["checks"]["reverse_reduces_mse"], "level_c_debug_suite_reverse_reduces"),
                    lambda d: (
                        d["checks"]["paper_state_mask_reverse_reaches_zero"],
                        "level_c_debug_suite_mask_reverse_zero",
                    ),
                    lambda d: (
                        d["checks"]["guidance_formula_gradients_nonzero"],
                        "level_c_debug_suite_guidance_gradients",
                    ),
                    lambda d: (d["checks"]["guided_reverse_loop_valid"], "level_c_debug_suite_guided_reverse"),
                    lambda d: (
                        d["checks"]["diffusion_to_action_heldout_mse_below_0_02"],
                        "level_c_debug_suite_action_mse",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_deployment"],
                        "level_c_debug_suite_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "level_c_debug_suite_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_extended_debug_suite",
                "res/level_c/extended_debug_suite/level_c_extended_debug_suite.json",
                [
                    status_ok,
                    lambda d: (d["step_count"] == 10, "level_c_extended_suite_step_count_10"),
                    lambda d: (d["pass_count"] == 10, "level_c_extended_suite_pass_count_10"),
                    lambda d: (d["checks"]["all_steps_pass"], "level_c_extended_suite_all_steps_pass"),
                    lambda d: (d["checks"]["state_latent_rows_84"], "level_c_extended_state_latent_rows_84"),
                    lambda d: (
                        d["checks"]["state_latent_dims_99_32_131_29"],
                        "level_c_extended_state_latent_dims",
                    ),
                    lambda d: (d["checks"]["vae_debug_latents_nonzero"], "level_c_extended_vae_latents_nonzero"),
                    lambda d: (
                        d["checks"]["vae_motion_split_heldout_reduces_test_mse"],
                        "level_c_extended_vae_heldout_reduces_test",
                    ),
                    lambda d: (
                        d["checks"]["vae_latent_heldout_test_loss_decreases"],
                        "level_c_extended_diffusion_heldout_reduces_test",
                    ),
                    lambda d: (
                        d["checks"]["action_multiseed_reduces_test_mse"],
                        "level_c_extended_action_multiseed_reduces_test",
                    ),
                    lambda d: (
                        d["checks"]["smoothness_control_frequency_25hz"],
                        "level_c_extended_smoothness_25hz",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_training_or_deployment"],
                        "level_c_extended_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "level_c_extended_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "single_batch_overfit_probe",
                "res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["loss_decreases_vs_noisy_identity"],
                        "single_batch_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["loss_reduction_ratio_at_least_0_99"],
                        "single_batch_loss_reduction_99pct",
                    ),
                    lambda d: (
                        d["checks"]["final_loss_below_1e_minus_8"],
                        "single_batch_final_loss_tiny",
                    ),
                    lambda d: (
                        d["checks"]["uses_independent_state_latent_steps"],
                        "single_batch_independent_steps",
                    ),
                ],
            ),
            check_json_artifact(
                "single_motion_overfit_probe",
                "res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["all_motion_loss_decreases"],
                        "single_motion_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["all_motion_loss_reduction_ratio_at_least_0_99"],
                        "single_motion_loss_reduction_99pct",
                    ),
                    lambda d: (
                        d["checks"]["all_motion_final_loss_below_1e_minus_8"],
                        "single_motion_final_loss_tiny",
                    ),
                    lambda d: (
                        d["checks"]["overparameterized_memorization_basis_recorded"],
                        "single_motion_memorization_boundary",
                    ),
                    lambda d: (
                        d["checks"]["uses_all_fixture_windows"],
                        "single_motion_all_windows",
                    ),
                ],
            ),
            check_json_artifact(
                "small_dataset_overfit_probe",
                "res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["uses_multiple_motions"],
                        "small_dataset_uses_multiple_motions",
                    ),
                    lambda d: (
                        d["checks"]["small_dataset_loss_decreases"],
                        "small_dataset_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["small_dataset_loss_reduction_ratio_at_least_0_99"],
                        "small_dataset_loss_reduction_99pct",
                    ),
                    lambda d: (
                        d["checks"]["small_dataset_final_loss_below_1e_minus_8"],
                        "small_dataset_final_loss_tiny",
                    ),
                    lambda d: (
                        d["checks"]["overparameterized_memorization_basis_recorded"],
                        "small_dataset_memorization_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "small_dataset_split_manifest",
                "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["uses_multiple_motions"],
                        "small_dataset_split_uses_multiple_motions",
                    ),
                    lambda d: (
                        d["checks"]["motion_level_splits_nonzero"],
                        "motion_level_splits_nonzero",
                    ),
                    lambda d: (
                        d["checks"]["no_motion_crosses_splits"],
                        "no_motion_crosses_splits",
                    ),
                    lambda d: (
                        d["checks"]["latent_marked_missing"],
                        "small_dataset_latents_marked_missing",
                    ),
                    lambda d: (
                        d["checks"]["accept_reject_marked_debug"],
                        "small_dataset_accept_reject_debug",
                    ),
                ],
            ),
            check_json_artifact(
                "small_dataset_multiseed_audit",
                "res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "multiseed_at_least_three"),
                    lambda d: (d["checks"]["no_best_seed_only_reporting"], "no_best_seed_only_reporting"),
                    lambda d: (d["checks"]["all_seeds_reduce_loss"], "all_seeds_reduce_loss"),
                    lambda d: (
                        d["checks"]["all_seed_final_loss_below_1e_minus_8"],
                        "all_seed_final_loss_tiny",
                    ),
                    lambda d: (
                        d["checks"]["debug_fixture_boundary_recorded"],
                        "multiseed_debug_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "small_dataset_heldout_eval",
                "res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["heldout_splits_reported"], "heldout_splits_reported"),
                    lambda d: (
                        d["checks"]["does_not_use_token_identity_basis"],
                        "heldout_no_token_identity_basis",
                    ),
                    lambda d: (
                        d["checks"]["validation_loss_decreases_vs_noisy_baseline"],
                        "validation_loss_decreases",
                    ),
                    lambda d: (d["checks"]["test_loss_decreases_vs_noisy_baseline"], "test_loss_decreases"),
                    lambda d: (
                        d["checks"]["uses_motion_level_split_manifest"],
                        "heldout_uses_motion_level_split",
                    ),
                ],
            ),
            check_json_artifact(
                "small_dataset_heldout_multiseed_audit",
                (
                    "res/level_c/small_dataset_heldout_multiseed_audit/"
                    "level_c_small_dataset_heldout_multiseed_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "heldout_multiseed_at_least_three"),
                    lambda d: (
                        d["checks"]["no_best_seed_only_reporting"],
                        "heldout_multiseed_no_best_seed_only_reporting",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_validation_loss_decreases"],
                        "heldout_multiseed_validation_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_test_loss_decreases"],
                        "heldout_multiseed_test_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_do_not_use_token_identity_basis"],
                        "heldout_multiseed_no_token_identity_basis",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_motion_level_split_manifest"],
                        "heldout_multiseed_motion_level_split",
                    ),
                    lambda d: (
                        d["checks"]["debug_fixture_boundary_recorded"],
                        "heldout_multiseed_debug_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "motion_state_fixture",
                "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json",
                [status_ok],
            ),
            check_json_artifact(
                "motion_state_fixture_run2_subject1",
                "res/level_c/motion_state_fixture/run2_subject1_frames_1_180_state_fixture.json",
                [status_ok],
            ),
            check_json_artifact(
                "motion_state_fixture_jumps1_subject1",
                "res/level_c/motion_state_fixture/jumps1_subject1_frames_1_180_state_fixture.json",
                [status_ok],
            ),
            check_json_artifact(
                "level_c_dataset_collection_protocol_audit",
                "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["paper_protocol_source_indexed"],
                        "dataset_collection_paper_protocol_indexed",
                    ),
                    lambda d: (
                        d["checks"]["ou_and_symmetry_debug_evidence_ok"],
                        "dataset_collection_ou_symmetry_debug_ok",
                    ),
                    lambda d: (
                        d["checks"]["manifest_debug_evidence_ok"],
                        "dataset_collection_manifest_debug_ok",
                    ),
                    lambda d: (
                        d["checks"]["rollout_manifest_debug_evidence_ok"],
                        "dataset_collection_rollout_manifest_debug_ok",
                    ),
                    lambda d: (
                        d["checks"]["debug_boundaries_mark_missing_latent_and_accept_reject"],
                        "dataset_collection_missing_latent_accept_reject_marked",
                    ),
                    lambda d: (
                        d["checks"]["paper_state_windows_available"],
                        "dataset_collection_paper_state_windows_available",
                    ),
                    lambda d: (
                        d["checks"]["official_level_c_artifacts_absent_recorded"],
                        "dataset_collection_official_artifacts_absent",
                    ),
                    lambda d: (
                        d["checks"]["all_missing_paper_requirements_explicit"],
                        "dataset_collection_missing_requirements_explicit",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_dataset_collection"],
                        "dataset_collection_no_false_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "rollout_rejection_manifest_probe",
                "res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json",
                [
                    status_ok,
                    lambda d: (
                        d["metrics"]["motion_count"] == 3,
                        "rollout_manifest_motion_count_3",
                    ),
                    lambda d: (
                        d["metrics"]["episode_manifest_rows"] == 15000,
                        "rollout_manifest_rows_15000",
                    ),
                    lambda d: (
                        d["metrics"]["recorded_coverage_min_nonzero"] >= 100,
                        "rollout_manifest_recorded_coverage_min_100",
                    ),
                    lambda d: (
                        d["checks"]["all_windows_match_2_5s_and_5s"],
                        "rollout_manifest_windows_match",
                    ),
                    lambda d: (
                        d["checks"]["coverage_repeats_per_valid_start_100"],
                        "rollout_manifest_100_repeats",
                    ),
                    lambda d: (
                        d["checks"]["accept_reject_field_present_debug_only"],
                        "rollout_manifest_accept_reject_debug",
                    ),
                    lambda d: (
                        d["checks"]["failure_signal_marked_missing"],
                        "rollout_manifest_failure_signal_missing",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "rollout_manifest_no_true_vae_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "rollout_manifest_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "state_latent_schema_audit",
                "res/level_c/state_latent_schema_audit/state_latent_schema_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 84, "state_latent_schema_rows_84"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "state_latent_schema_evidence_exists"),
                    lambda d: (
                        d["checks"]["uses_package_state_latent_api"],
                        "state_latent_schema_uses_package_api",
                    ),
                    lambda d: (
                        d["checks"]["row_count_matches_paper_state_windows"],
                        "state_latent_schema_matches_paper_windows",
                    ),
                    lambda d: (
                        d["checks"]["split_counts_match_manifest"],
                        "state_latent_schema_split_counts_match",
                    ),
                    lambda d: (
                        d["checks"]["all_token_shapes_21x131"],
                        "state_latent_schema_token_shape",
                    ),
                    lambda d: (
                        d["checks"]["all_latents_marked_placeholder"],
                        "state_latent_schema_placeholder_latents",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_latents"],
                        "state_latent_schema_no_true_latent_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "state_latent_schema_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "dagger_schema_audit",
                "res/level_c/dagger_schema_audit/dagger_schema_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 30, "dagger_schema_rows_30"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "dagger_schema_evidence_exists"),
                    lambda d: (d["checks"]["uses_package_dagger_api"], "dagger_schema_uses_package_api"),
                    lambda d: (
                        d["checks"]["uses_package_evaluation_metric_api"],
                        "dagger_schema_uses_evaluation_metric_api",
                    ),
                    lambda d: (
                        d["checks"]["sample_count_matches_effective_batch"],
                        "dagger_schema_effective_batch_30",
                    ),
                    lambda d: (d["checks"]["micro_batch_layout_matches_accumulation"], "dagger_schema_micro_layout"),
                    lambda d: (d["checks"]["state_dim_163"], "dagger_schema_state_dim_163"),
                    lambda d: (d["checks"]["action_dim_29"], "dagger_schema_action_dim_29"),
                    lambda d: (d["checks"]["all_teacher_queried"], "dagger_schema_all_teacher_queried"),
                    lambda d: (d["checks"]["split_counts_nonzero"], "dagger_schema_split_counts_nonzero"),
                    lambda d: (d["checks"]["discrepancy_metrics_finite"], "dagger_schema_metrics_finite"),
                    lambda d: (
                        d["checks"]["manifest_marks_not_true_dagger_rollout"],
                        "dagger_schema_manifest_not_true_rollout",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_dagger_rollout"],
                        "dagger_schema_no_true_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "dagger_schema_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "trajectory_inverse_transform_audit",
                "res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["paper_formula_root_inverse_roundtrip"],
                        "paper_formula_root_inverse_roundtrip",
                    ),
                    lambda d: (
                        d["checks"]["paper_formula_root_rotation_roundtrip"],
                        "paper_formula_root_rotation_roundtrip",
                    ),
                    lambda d: (
                        d["checks"]["paper_formula_body_inverse_roundtrip"],
                        "paper_formula_body_inverse_roundtrip",
                    ),
                    lambda d: (
                        d["checks"]["existing_debug_fixture_not_full_paper_root_window_state"],
                        "debug_fixture_boundary_recorded",
                    ),
                ],
            ),
            check_json_artifact(
                "state_representation_source_audit",
                "res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["method_source_patterns_found"],
                        "state_rep_method_source_patterns_found",
                    ),
                    lambda d: (
                        d["checks"]["supplement_source_patterns_found"],
                        "state_rep_supplement_source_patterns_found",
                    ),
                    lambda d: (
                        d["checks"]["candidate_body_position_matches_paper_local_formula"],
                        "state_rep_body_position_local_formula",
                    ),
                    lambda d: (
                        d["checks"]["candidate_root_current_frame_difference_detected"],
                        "state_rep_root_current_frame_boundary",
                    ),
                    lambda d: (
                        d["checks"]["candidate_body_velocity_missing_root_velocity_subtraction_detected"],
                        "state_rep_body_velocity_boundary",
                    ),
                    lambda d: (
                        d["checks"]["candidate_full_root_relative_position_missing_detected"],
                        "state_rep_root_position_boundary",
                    ),
                    lambda d: (
                        d["checks"]["paper_exact_trainable_state_not_claimed"],
                        "state_rep_no_paper_exact_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_state_windows",
                "res/level_c/paper_state_windows/level_c_paper_state_windows.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["all_checks_pass"], "paper_state_windows_all_checks_pass"),
                    lambda d: (d["checks"]["all_state_dims_match_source_formula_terms"], "paper_state_dim_99"),
                    lambda d: (
                        d["checks"]["all_current_root_relative_positions_zero"],
                        "paper_state_current_root_pos_zero",
                    ),
                    lambda d: (
                        d["checks"]["all_current_root_relative_linear_velocities_zero"],
                        "paper_state_current_root_vel_zero",
                    ),
                    lambda d: (
                        d["checks"]["paper_exact_trainable_state_not_claimed"],
                        "paper_state_no_trainable_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "state_latent_dataset_consistency_audit",
                "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 84, "state_latent_consistency_rows_84"),
                    lambda d: (d["metrics"]["sequence_length"] == 21, "state_latent_consistency_seq_21"),
                    lambda d: (d["metrics"]["state_dim"] == 99, "state_latent_consistency_state_dim_99"),
                    lambda d: (d["metrics"]["latent_dim"] == 32, "state_latent_consistency_latent_dim_32"),
                    lambda d: (d["metrics"]["token_dim"] == 131, "state_latent_consistency_token_dim_131"),
                    lambda d: (d["metrics"]["action_dim"] == 29, "state_latent_consistency_action_dim_29"),
                    lambda d: (
                        d["metrics"]["per_split_counts"] == {"train": 28, "validation": 28, "test": 28},
                        "state_latent_consistency_split_counts",
                    ),
                    lambda d: (d["checks"]["paper_state_json_status_ok"], "state_latent_consistency_paper_json"),
                    lambda d: (d["checks"]["vae_debug_latents_json_status_ok"], "state_latent_consistency_vae_json"),
                    lambda d: (d["checks"]["diffusion_to_action_json_status_ok"], "state_latent_consistency_action_json"),
                    lambda d: (
                        d["checks"]["feature_slices_match_paper_state_builder"],
                        "state_latent_consistency_feature_slices",
                    ),
                    lambda d: (
                        d["checks"]["vae_states_equal_paper_state_windows"],
                        "state_latent_consistency_state_arrays",
                    ),
                    lambda d: (
                        d["checks"]["action_target_equals_decoded_action"],
                        "state_latent_consistency_action_arrays",
                    ),
                    lambda d: (
                        d["checks"]["action_npz_source_motion_order_matches_windows"],
                        "state_latent_consistency_action_motion_order",
                    ),
                    lambda d: (
                        d["checks"]["action_npz_split_labels_match_motion_split"],
                        "state_latent_consistency_action_splits",
                    ),
                    lambda d: (
                        d["checks"]["vae_json_start_timestep_matches_window_start_minus_history"],
                        "state_latent_consistency_start_timestep",
                    ),
                    lambda d: (
                        d["checks"]["current_root_position_zero"],
                        "state_latent_consistency_current_root_pos_zero",
                    ),
                    lambda d: (
                        d["checks"]["current_root_linear_velocity_zero"],
                        "state_latent_consistency_current_root_vel_zero",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_trainable_dataset"],
                        "state_latent_consistency_no_dataset_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_checkpoint"],
                        "state_latent_consistency_no_checkpoint_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "state_latent_consistency_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_state_latent_training_dataset_contract_audit",
                (
                    "res/level_c/state_latent_training_dataset_contract_audit/"
                    "level_c_state_latent_training_dataset_contract_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["metrics"]["contract_row_count"] == 12,
                        "state_latent_training_contract_rows_12",
                    ),
                    lambda d: (
                        d["metrics"]["paper_trainable_satisfied_count"] == 3,
                        "state_latent_training_contract_debug_satisfied_3",
                    ),
                    lambda d: (
                        d["metrics"]["missing_or_debug_only_count"] == 9,
                        "state_latent_training_contract_missing_9",
                    ),
                    lambda d: (
                        d["metrics"]["debug_npz_sample_count"] == 84,
                        "state_latent_training_contract_npz_samples_84",
                    ),
                    lambda d: (
                        d["checks"]["npz_shapes_match_21x99_and_21x32"],
                        "state_latent_training_contract_npz_shapes",
                    ),
                    lambda d: (
                        d["checks"]["debug_dataset_dimensions_match_contract"],
                        "state_latent_training_contract_dims",
                    ),
                    lambda d: (
                        d["checks"]["debug_dataset_split_counts_present"],
                        "state_latent_training_contract_splits",
                    ),
                    lambda d: (
                        d["checks"]["missing_training_requirements_recorded"],
                        "state_latent_training_contract_missing_recorded",
                    ),
                    lambda d: (
                        d["checks"]["no_current_artifact_qualifies_as_paper_trainable_dataset"],
                        "state_latent_training_contract_no_false_dataset",
                    ),
                    lambda d: (
                        d["interpretation"]["trainable_dataset_available"] is False,
                        "state_latent_training_contract_dataset_unavailable",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "state_latent_training_contract_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_state_overfit_probe",
                "res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["paper_state_windows_all_checks_pass"], "paper_state_overfit_input_ok"),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "paper_state_overfit_dim_99"),
                    lambda d: (d["checks"]["uses_multiple_motions"], "paper_state_overfit_multi_motion"),
                    lambda d: (d["checks"]["uses_independent_state_latent_steps"], "paper_state_overfit_steps"),
                    lambda d: (d["checks"]["loss_decreases"], "paper_state_overfit_loss_decreases"),
                    lambda d: (
                        d["checks"]["loss_reduction_ratio_at_least_0_99"],
                        "paper_state_overfit_loss_reduction_99pct",
                    ),
                    lambda d: (d["checks"]["final_loss_below_1e_minus_8"], "paper_state_overfit_tiny_loss"),
                    lambda d: (
                        d["checks"]["overparameterized_memorization_basis_recorded"],
                        "paper_state_overfit_memorization_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_latent_diffusion_overfit_probe",
                "res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["debug_vae_latent_artifact_status_ok"],
                        "vae_latent_diffusion_input_ok",
                    ),
                    lambda d: (
                        d["checks"]["debug_vae_latents_nonzero"],
                        "vae_latent_diffusion_latents_nonzero",
                    ),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "vae_latent_diffusion_state_dim_99"),
                    lambda d: (
                        d["checks"]["uses_debug_vae_latent_dim_32"],
                        "vae_latent_diffusion_latent_dim_32",
                    ),
                    lambda d: (d["checks"]["token_dim_131"], "vae_latent_diffusion_token_dim_131"),
                    lambda d: (d["checks"]["window_count_84"], "vae_latent_diffusion_windows_84"),
                    lambda d: (
                        d["checks"]["uses_independent_state_latent_steps"],
                        "vae_latent_diffusion_independent_steps",
                    ),
                    lambda d: (d["checks"]["loss_decreases"], "vae_latent_diffusion_loss_decreases"),
                    lambda d: (
                        d["checks"]["loss_reduction_ratio_at_least_0_99"],
                        "vae_latent_diffusion_reduction_99pct",
                    ),
                    lambda d: (
                        d["checks"]["final_loss_below_1e_minus_8"],
                        "vae_latent_diffusion_tiny_loss",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "vae_latent_diffusion_no_true_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_diffusion_checkpoint"],
                        "vae_latent_diffusion_no_checkpoint_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_state_heldout_eval",
                "res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["paper_state_windows_all_checks_pass"], "paper_state_heldout_input_ok"),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "paper_state_heldout_dim_99"),
                    lambda d: (
                        d["checks"]["heldout_splits_reported"],
                        "paper_state_heldout_splits_reported",
                    ),
                    lambda d: (
                        d["checks"]["does_not_use_token_identity_basis"],
                        "paper_state_heldout_no_token_identity",
                    ),
                    lambda d: (
                        d["checks"]["uses_motion_level_split_manifest"],
                        "paper_state_heldout_motion_level_split",
                    ),
                    lambda d: (
                        d["checks"]["validation_loss_decreases_vs_noisy_baseline"],
                        "paper_state_heldout_validation_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["test_loss_decreases_vs_noisy_baseline"],
                        "paper_state_heldout_test_loss_decreases",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_latent_heldout_eval",
                "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["debug_vae_latent_artifact_status_ok"],
                        "vae_latent_heldout_input_ok",
                    ),
                    lambda d: (
                        d["checks"]["debug_vae_latents_nonzero"],
                        "vae_latent_heldout_latents_nonzero",
                    ),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "vae_latent_heldout_state_dim_99"),
                    lambda d: (
                        d["checks"]["uses_debug_vae_latent_dim_32"],
                        "vae_latent_heldout_latent_dim_32",
                    ),
                    lambda d: (d["checks"]["token_dim_131"], "vae_latent_heldout_token_dim_131"),
                    lambda d: (d["checks"]["window_count_84"], "vae_latent_heldout_windows_84"),
                    lambda d: (
                        d["checks"]["heldout_splits_reported"],
                        "vae_latent_heldout_splits_reported",
                    ),
                    lambda d: (
                        d["checks"]["does_not_use_token_identity_basis"],
                        "vae_latent_heldout_no_token_identity",
                    ),
                    lambda d: (
                        d["checks"]["uses_motion_level_split_manifest"],
                        "vae_latent_heldout_motion_level_split",
                    ),
                    lambda d: (
                        d["checks"]["validation_loss_decreases_vs_noisy_baseline"],
                        "vae_latent_heldout_validation_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["test_loss_decreases_vs_noisy_baseline"],
                        "vae_latent_heldout_test_loss_decreases",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "vae_latent_heldout_no_true_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_diffusion_checkpoint"],
                        "vae_latent_heldout_no_checkpoint_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_state_heldout_multiseed_audit",
                "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "paper_state_heldout_multiseed_3_seeds"),
                    lambda d: (
                        d["checks"]["no_best_seed_only_reporting"],
                        "paper_state_heldout_multiseed_no_best_seed",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_validation_loss_decreases"],
                        "paper_state_heldout_multiseed_validation_decreases",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_test_loss_decreases"],
                        "paper_state_heldout_multiseed_test_decreases",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_do_not_use_token_identity_basis"],
                        "paper_state_heldout_multiseed_no_token_identity",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_motion_level_split_manifest"],
                        "paper_state_heldout_multiseed_motion_split",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_paper_state_dim_99"],
                        "paper_state_heldout_multiseed_dim_99",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_paper_state_windows_pass"],
                        "paper_state_heldout_multiseed_windows_pass",
                    ),
                    lambda d: (
                        d["checks"]["debug_fixture_boundary_recorded"],
                        "paper_state_heldout_multiseed_debug_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_latent_heldout_multiseed_audit",
                "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "vae_latent_heldout_multiseed_3_seeds"),
                    lambda d: (
                        d["checks"]["no_best_seed_only_reporting"],
                        "vae_latent_heldout_multiseed_no_best_seed",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_debug_vae_latent_artifact_ok"],
                        "vae_latent_heldout_multiseed_input_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_debug_vae_latents_nonzero"],
                        "vae_latent_heldout_multiseed_latents_nonzero",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_state_dim_99"],
                        "vae_latent_heldout_multiseed_state_dim_99",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_debug_vae_latent_dim_32"],
                        "vae_latent_heldout_multiseed_latent_dim_32",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_token_dim_131"],
                        "vae_latent_heldout_multiseed_token_dim_131",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_do_not_use_token_identity_basis"],
                        "vae_latent_heldout_multiseed_no_token_identity",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_motion_level_split_manifest"],
                        "vae_latent_heldout_multiseed_motion_split",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_validation_loss_decreases"],
                        "vae_latent_heldout_multiseed_validation_decreases",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_test_loss_decreases"],
                        "vae_latent_heldout_multiseed_test_decreases",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "vae_latent_heldout_multiseed_no_true_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_diffusion_checkpoint"],
                        "vae_latent_heldout_multiseed_no_checkpoint_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "fixture_split_manifest",
                "res/level_c/fixture_split_manifest/fixture_split_manifest_summary.json",
                [status_ok, lambda d: (d["checks"]["no_cross_split_overlap"], "no_cross_split_overlap")],
            ),
            check_json_artifact("augmentation_probe", "res/level_c/augmentation_probe/level_c_augmentation_probe.json", [status_ok]),
            check_json_artifact(
                "symmetry_mapping_audit",
                "res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["joint_count"] == 29, "symmetry_joint_count_29"),
                    lambda d: (d["metrics"]["covered_joint_count"] == 29, "symmetry_all_joints_covered"),
                    lambda d: (d["metrics"]["pair_count"] == 13, "symmetry_pair_count_13"),
                    lambda d: (d["metrics"]["center_joint_count"] == 3, "symmetry_center_joint_count_3"),
                    lambda d: (d["checks"]["paper_mentions_sagittal_symmetry"], "symmetry_paper_mentions"),
                    lambda d: (
                        d["checks"]["paper_does_not_publish_g1_sign_table"],
                        "symmetry_no_paper_sign_table",
                    ),
                    lambda d: (
                        d["checks"]["candidate_joint_names_match_controller_order"],
                        "symmetry_controller_order",
                    ),
                    lambda d: (d["checks"]["all_controller_joints_covered"], "symmetry_controller_coverage"),
                    lambda d: (d["checks"]["mapping_is_involution_by_rows"], "symmetry_involution"),
                    lambda d: (
                        d["checks"]["augmentation_probe_double_mirror_exact"],
                        "symmetry_aug_double_mirror",
                    ),
                    lambda d: (
                        d["checks"]["velocity_double_mirror_exact"],
                        "symmetry_velocity_double_mirror",
                    ),
                    lambda d: (d["checks"]["controller_joints_exist_in_urdf"], "symmetry_joints_in_urdf"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "symmetry_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guidance_formula_probe",
                "res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
                [status_ok, lambda d: (d["checks"]["composed_gradient_nonzero"], "composed_gradient_nonzero")],
            ),
            check_json_artifact(
                "guidance_cost_coverage_audit",
                "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 8, "guidance_coverage_rows_8"),
                    lambda d: (d["metrics"]["failed_row_count"] == 0, "guidance_coverage_failures_zero"),
                    lambda d: (
                        d["metrics"]["paper_explicit_row_count"] == 5,
                        "guidance_paper_explicit_rows_5",
                    ),
                    lambda d: (
                        d["metrics"]["formula_missing_row_count"] == 1,
                        "guidance_keyframe_formula_missing_recorded",
                    ),
                    lambda d: (
                        d["checks"]["all_paper_explicit_costs_have_source_and_gradients"],
                        "guidance_explicit_costs_source_and_gradients",
                    ),
                    lambda d: (
                        d["checks"]["keyframe_formula_recorded_missing"],
                        "guidance_keyframe_missing_formula",
                    ),
                    lambda d: (
                        d["checks"]["keyframe_candidate_has_gradient"],
                        "guidance_keyframe_candidate_gradient",
                    ),
                    lambda d: (d["checks"]["guided_reverse_loop_valid"], "guidance_reverse_loop_valid"),
                    lambda d: (d["checks"]["scale_sweep_valid"], "guidance_scale_sweep_valid"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_coverage_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "core_math_unit_tests",
                "res/tests/core_math_unit_tests/core_math_unit_tests.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 23, "core_math_rows_23"),
                    lambda d: (d["failed_row_count"] == 0, "core_math_failures_zero"),
                    lambda d: (
                        d["checks"]["all_core_math_tests_pass"],
                        "core_math_all_tests_pass",
                    ),
                    lambda d: (
                        d["checks"]["covers_goal_core_math_items"],
                        "core_math_goal_items_covered",
                    ),
                    lambda d: (
                        d["checks"]["pure_numpy_no_isaac_ros_dependency"],
                        "core_math_pure_numpy",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_deployment"],
                        "core_math_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "core_math_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "reimpl_package_api_tests",
                "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 8, "reimpl_api_test_rows_8"),
                    lambda d: (d["failed_row_count"] == 0, "reimpl_api_test_failures_zero"),
                    lambda d: (d["checks"]["all_package_api_tests_pass"], "reimpl_api_all_tests_pass"),
                    lambda d: (d["checks"]["covers_at_least_seven_modules"], "reimpl_api_modules_covered"),
                    lambda d: (d["checks"]["shape_error_paths_tested"], "reimpl_api_shape_errors"),
                    lambda d: (d["checks"]["metadata_paths_tested"], "reimpl_api_metadata"),
                    lambda d: (
                        d["checks"]["pure_numpy_no_isaac_ros_dependency"],
                        "reimpl_api_pure_numpy",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_deployment"],
                        "reimpl_api_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "reimpl_api_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "reimpl_test_suite",
                "res/tests/reimpl_test_suite/reimpl_test_suite.json",
                [
                    status_ok,
                    lambda d: (d["step_count"] == 5, "reimpl_suite_step_count_5"),
                    lambda d: (d["pass_count"] == 5, "reimpl_suite_pass_count_5"),
                    lambda d: (d["checks"]["all_steps_pass"], "reimpl_suite_all_steps_pass"),
                    lambda d: (d["checks"]["core_math_rows_23"], "reimpl_suite_core_rows_23"),
                    lambda d: (d["checks"]["api_rows_8"], "reimpl_suite_api_rows_8"),
                    lambda d: (d["checks"]["package_symbols_29"], "reimpl_suite_package_symbols_29"),
                    lambda d: (d["checks"]["runtime_window_count_84"], "reimpl_suite_runtime_windows_84"),
                    lambda d: (
                        d["checks"]["runtime_token_shape_84_21_131"],
                        "reimpl_suite_runtime_token_shape",
                    ),
                    lambda d: (d["checks"]["coverage_required_20"], "reimpl_suite_coverage_required_20"),
                    lambda d: (d["checks"]["pure_numpy_no_isaac_ros_dependency"], "reimpl_suite_pure_numpy"),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_deployment"],
                        "reimpl_suite_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "reimpl_suite_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "core_test_coverage_audit",
                "res/tests/core_test_coverage_audit/core_test_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["required_count"] == 20, "core_test_coverage_required_20"),
                    lambda d: (d["missing_count"] == 0, "core_test_coverage_missing_zero"),
                    lambda d: (d["core_test_row_count"] == 23, "core_test_coverage_core_rows_23"),
                    lambda d: (d["core_test_failed_row_count"] == 0, "core_test_coverage_core_failures_zero"),
                    lambda d: (
                        d["checks"]["all_20_required_items_have_test_evidence"],
                        "core_test_coverage_all_goal_items",
                    ),
                    lambda d: (
                        d["checks"]["core_math_tests_pass"],
                        "core_test_coverage_math_tests_pass",
                    ),
                    lambda d: (
                        d["checks"]["goal_metric_items_have_test_evidence"],
                        "core_test_coverage_goal_metric_items",
                    ),
                    lambda d: (
                        d["checks"]["pure_numpy_no_isaac_ros_dependency"],
                        "core_test_coverage_pure_numpy",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_training_or_deployment"],
                        "core_test_coverage_no_false_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "core_test_coverage_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "coding_requirements_audit",
                "res/code/coding_requirements_audit/coding_requirements_audit.json",
                [
                    status_ok,
                    lambda d: (d["requirement_row_count"] == 13, "coding_requirements_rows_13"),
                    lambda d: (d["failed_requirement_count"] == 0, "coding_requirements_failures_zero"),
                    lambda d: (d["checks"]["all_public_functions_typed"], "coding_requirements_typed"),
                    lambda d: (
                        d["checks"]["all_public_functions_have_docstrings"],
                        "coding_requirements_docstrings",
                    ),
                    lambda d: (
                        d["checks"]["all_public_functions_document_shape_or_frame"],
                        "coding_requirements_shape_frame_docs",
                    ),
                    lambda d: (
                        d["checks"]["nan_inf_guard_unit_test_passes"],
                        "coding_requirements_nan_inf_guard",
                    ),
                    lambda d: (
                        d["checks"]["cli_yaml_resolved_config_evidence_present"],
                        "coding_requirements_cli_yaml_config",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_full_training_or_deployment"],
                        "coding_requirements_no_false_completion",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "coding_requirements_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "reimpl_package_audit",
                "res/code/reimpl_package_audit/reimpl_package_audit.json",
                [
                    status_ok,
                    lambda d: (d["python_file_count"] >= 10, "reimpl_package_python_files"),
                    lambda d: (d["symbol_row_count"] == 29, "reimpl_package_symbols_29"),
                    lambda d: (d["checks"]["all_expected_modules_import"], "reimpl_modules_import"),
                    lambda d: (d["checks"]["all_expected_symbols_exist"], "reimpl_symbols_exist"),
                    lambda d: (
                        d["checks"]["core_math_tests_use_package_api"],
                        "reimpl_tests_use_package_api",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_training_implementation"],
                        "reimpl_no_false_official_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "reimpl_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "reimpl_runtime_integration_audit",
                "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["window_count"] == 84, "reimpl_runtime_windows_84"),
                    lambda d: (d["metrics"]["token_shape"] == [84, 21, 131], "reimpl_runtime_token_shape"),
                    lambda d: (
                        d["metrics"]["split_counts"] == {"train": 28, "validation": 28, "test": 28},
                        "reimpl_runtime_split_counts",
                    ),
                    lambda d: (d["checks"]["latents_npz_exists"], "reimpl_runtime_latents_npz"),
                    lambda d: (d["checks"]["action_npz_exists"], "reimpl_runtime_action_npz"),
                    lambda d: (d["checks"]["fixture_npz_exists"], "reimpl_runtime_fixture_npz"),
                    lambda d: (d["checks"]["state_dim_99"], "reimpl_runtime_state_dim_99"),
                    lambda d: (d["checks"]["latent_dim_32"], "reimpl_runtime_latent_dim_32"),
                    lambda d: (d["checks"]["action_dim_29"], "reimpl_runtime_action_dim_29"),
                    lambda d: (d["checks"]["all_metrics_finite"], "reimpl_runtime_metrics_finite"),
                    lambda d: (d["checks"]["projection_reconstructs_state"], "reimpl_runtime_projection"),
                    lambda d: (d["checks"]["diffusion_reverse_reduces_mse"], "reimpl_runtime_reverse_reduces"),
                    lambda d: (d["checks"]["observation_mask_clamps_exactly"], "reimpl_runtime_mask_clamps"),
                    lambda d: (
                        d["checks"]["decoded_teacher_action_mse_below_0_01"],
                        "reimpl_runtime_decoded_action_mse",
                    ),
                    lambda d: (d["checks"]["dagger_queries_recorded"], "reimpl_runtime_dagger_queries"),
                    lambda d: (
                        d["checks"]["vae_reparameterize_zero_eps_matches_mu"],
                        "reimpl_runtime_vae_zero_eps",
                    ),
                    lambda d: (d["checks"]["guidance_gradient_nonzero"], "reimpl_runtime_guidance_grad"),
                    lambda d: (d["checks"]["tracking_error_positive"], "reimpl_runtime_tracking_error"),
                    lambda d: (d["checks"]["survival_rate_in_unit_interval"], "reimpl_runtime_survival_rate"),
                    lambda d: (
                        d["checks"]["does_not_claim_official_training_code"],
                        "reimpl_runtime_no_official_training_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_checkpoint"],
                        "reimpl_runtime_no_checkpoint_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "reimpl_runtime_no_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "reimpl_runtime_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "timestep_mask_probe",
                "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
                [status_ok, lambda d: (d["checks"]["keyframe_inpainting_has_future_clean_state_only"], "future_keyframe_mask_ok")],
            ),
            check_json_artifact(
                "timestep_mask_coverage_audit",
                "res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 8, "timestep_mask_coverage_rows_8"),
                    lambda d: (d["metrics"]["failed_row_count"] == 0, "timestep_mask_coverage_failures_zero"),
                    lambda d: (
                        d["metrics"]["paper_state_debug_artifact_row_count"] == 1,
                        "timestep_mask_paper_state_debug_artifact_recorded",
                    ),
                    lambda d: (d["metrics"]["paper_state_mask_state_dim"] == 99, "timestep_mask_paper_state_dim_99"),
                    lambda d: (d["metrics"]["paper_state_mask_tau_dim"] == 131, "timestep_mask_paper_state_tau_dim_131"),
                    lambda d: (
                        d["checks"]["all_step_tensors_shape_21x2"],
                        "timestep_mask_steps_shape_21x2",
                    ),
                    lambda d: (
                        d["checks"]["training_independent_steps"],
                        "timestep_mask_training_independent_steps",
                    ),
                    lambda d: (
                        d["checks"]["history_and_keyframe_debug_masks_pass"],
                        "timestep_mask_history_keyframe_debug_masks",
                    ),
                    lambda d: (
                        d["checks"]["deployed_mask_policy_recorded_unspecified"],
                        "timestep_mask_deployed_policy_unspecified",
                    ),
                    lambda d: (
                        d["checks"]["paper_state_mask_reverse_debug_artifact_present"],
                        "timestep_mask_paper_state_reverse_debug_present",
                    ),
                    lambda d: (
                        d["checks"]["reverse_and_guided_loops_reach_zero"],
                        "timestep_mask_reverse_guided_loops_zero",
                    ),
                    lambda d: (
                        d["checks"]["paper_state_reverse_reaches_zero"],
                        "timestep_mask_paper_state_reverse_zero",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "timestep_mask_coverage_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_state_mask_reverse_probe",
                "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["paper_state_dim_99"], "paper_state_mask_state_dim_99"),
                    lambda d: (d["checks"]["tau_dim_131"], "paper_state_mask_tau_dim_131"),
                    lambda d: (
                        d["checks"]["all_step_tensors_shape_21x2"],
                        "paper_state_mask_step_shape_21x2",
                    ),
                    lambda d: (
                        d["checks"]["keyframe_inpainting_has_future_clean_state_only"],
                        "paper_state_mask_keyframe_future_state_only",
                    ),
                    lambda d: (
                        d["checks"]["reverse_steps_reach_zero"],
                        "paper_state_mask_reverse_zero",
                    ),
                    lambda d: (
                        d["checks"]["reverse_observed_tokens_clamped"],
                        "paper_state_mask_reverse_clamped",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "paper_state_mask_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "reverse_denoising_probe",
                "res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json",
                [status_ok, lambda d: (d["checks"]["all_steps_reach_zero"], "all_steps_reach_zero")],
            ),
            check_json_artifact(
                "diffusion_equation_audit",
                "res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["paper_forward_posterior_formula_found"],
                        "paper_forward_posterior_formula_found",
                    ),
                    lambda d: (
                        d["checks"]["paper_clean_prediction_loss_found"],
                        "paper_clean_prediction_loss_found",
                    ),
                    lambda d: (
                        d["checks"]["paper_reverse_alpha_gamma_sigma_form_found"],
                        "paper_reverse_alpha_gamma_sigma_form_found",
                    ),
                    lambda d: (d["checks"]["paper_denoising_steps_20_found"], "paper_denoising_steps_20"),
                    lambda d: (
                        d["checks"]["posterior_to_paper_form_equivalent"],
                        "posterior_to_paper_form_equivalent",
                    ),
                    lambda d: (
                        d["checks"]["public_source_exact_coefficient_schedule_missing"],
                        "public_source_exact_coefficient_schedule_missing",
                    ),
                ],
            ),
            check_json_artifact(
                "full_transformer_arch_probe",
                "res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.json",
                [status_ok, lambda d: (d["checks"]["uses_paper_embedding_dim"], "paper_embedding_dim")],
            ),
            check_json_artifact(
                "paper_state_transformer_arch_probe",
                "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "paper_state_transformer_state_dim_99"),
                    lambda d: (d["checks"]["uses_token_dim_131"], "paper_state_transformer_token_dim_131"),
                    lambda d: (d["checks"]["uses_paper_embedding_dim"], "paper_state_transformer_embedding_512"),
                    lambda d: (d["checks"]["uses_paper_attention_heads"], "paper_state_transformer_heads_8"),
                    lambda d: (d["checks"]["uses_paper_transformer_layers"], "paper_state_transformer_layers_6"),
                    lambda d: (d["checks"]["uses_paper_denoising_steps"], "paper_state_transformer_steps_20"),
                    lambda d: (
                        d["checks"]["prediction_shape_matches_clean_tau"],
                        "paper_state_transformer_prediction_shape",
                    ),
                    lambda d: (d["checks"]["loss_is_finite"], "paper_state_transformer_loss_finite"),
                    lambda d: (d["checks"]["grad_norm_is_positive"], "paper_state_transformer_grad_positive"),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "paper_state_transformer_goal_incomplete"),
                ],
            ),
            check_json_artifact(
                "vae_latent_transformer_arch_probe",
                "res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["debug_vae_latent_artifact_status_ok"],
                        "vae_latent_transformer_input_ok",
                    ),
                    lambda d: (
                        d["checks"]["debug_vae_latents_nonzero"],
                        "vae_latent_transformer_latents_nonzero",
                    ),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "vae_latent_transformer_state_dim_99"),
                    lambda d: (
                        d["checks"]["uses_debug_vae_latent_dim_32"],
                        "vae_latent_transformer_latent_dim_32",
                    ),
                    lambda d: (d["checks"]["uses_token_dim_131"], "vae_latent_transformer_token_dim_131"),
                    lambda d: (d["checks"]["uses_paper_embedding_dim"], "vae_latent_transformer_embedding_512"),
                    lambda d: (d["checks"]["uses_paper_attention_heads"], "vae_latent_transformer_heads_8"),
                    lambda d: (d["checks"]["uses_paper_transformer_layers"], "vae_latent_transformer_layers_6"),
                    lambda d: (d["checks"]["uses_paper_denoising_steps"], "vae_latent_transformer_steps_20"),
                    lambda d: (
                        d["checks"]["step_tensor_is_independent_state_latent"],
                        "vae_latent_transformer_independent_steps",
                    ),
                    lambda d: (
                        d["checks"]["prediction_shape_matches_clean_tau"],
                        "vae_latent_transformer_prediction_shape",
                    ),
                    lambda d: (d["checks"]["loss_is_finite"], "vae_latent_transformer_loss_finite"),
                    lambda d: (d["checks"]["grad_norm_is_positive"], "vae_latent_transformer_grad_positive"),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "vae_latent_transformer_no_true_rollout_claim",
                    ),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "vae_latent_transformer_goal_incomplete"),
                ],
            ),
            check_json_artifact(
                "transformer_parameter_count_audit",
                "res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["paper_parameter_count_statement_found"],
                        "transformer_param_count_paper_statement",
                    ),
                    lambda d: (
                        d["checks"]["all_local_architecture_checks_pass"],
                        "transformer_param_count_local_arch_checks",
                    ),
                    lambda d: (
                        d["checks"]["both_local_counts_within_5_percent_of_paper_approx"],
                        "transformer_param_count_within_5pct",
                    ),
                    lambda d: (
                        d["checks"]["local_counts_not_exact_paper_count"],
                        "transformer_param_count_not_exact",
                    ),
                    lambda d: (
                        d["checks"]["variant_delta_explained_by_token_io_projection"],
                        "transformer_param_count_io_delta",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_exact_paper_parameter_count"],
                        "transformer_param_count_no_exact_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "transformer_param_count_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guided_reverse_loop_probe",
                "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
                [status_ok, lambda d: (d["checks"]["guided_final_cost_below_unguided_final"], "guided_cost_improves")],
            ),
            check_json_artifact(
                "guidance_scale_sweep_probe",
                "res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json",
                [status_ok, lambda d: (d["checks"]["best_improves_over_zero_scale"], "best_scale_improves")],
            ),
            check_json_artifact(
                "guidance_task_scale_sweep",
                "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 40, "guidance_task_scale_rows_40"),
                    lambda d: (d["checks"]["five_tasks_swept"], "guidance_task_scale_five_tasks"),
                    lambda d: (d["checks"]["scale_count_per_task"], "guidance_task_scale_grid"),
                    lambda d: (d["checks"]["all_rows_finite"], "guidance_task_scale_finite"),
                    lambda d: (d["checks"]["all_gradients_nonzero"], "guidance_task_scale_gradients"),
                    lambda d: (
                        d["checks"]["all_tasks_have_improving_positive_scale"],
                        "guidance_task_scale_improves",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_rollout_or_video"],
                        "guidance_task_scale_no_rollout_video_claim",
                    ),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "guidance_task_scale_keeps_incomplete"),
                ],
            ),
            check_json_artifact(
                "guidance_debug_visualization",
                "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["writes_png_svg_pdf"], "guidance_debug_writes_static_figures"),
                    lambda d: (d["checks"]["writes_debug_gif"], "guidance_debug_writes_gif"),
                    lambda d: (d["checks"]["writes_npz"], "guidance_debug_writes_npz"),
                    lambda d: (
                        d["checks"]["five_debug_tasks_visualized"],
                        "guidance_debug_five_tasks",
                    ),
                    lambda d: (
                        d["checks"]["all_primary_metrics_improve"],
                        "guidance_debug_primary_metrics_improve",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_rollout"],
                        "guidance_debug_no_trained_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_video"],
                        "guidance_debug_no_paper_video_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_debug_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guidance_task_metric_audit",
                "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 5, "guidance_task_metric_rows_5"),
                    lambda d: (
                        d["checks"]["debug_visualization_status_ok"],
                        "guidance_task_metric_debug_source_ok",
                    ),
                    lambda d: (d["checks"]["scale_sweep_status_ok"], "guidance_task_metric_scale_source_ok"),
                    lambda d: (d["checks"]["five_tasks_recorded"], "guidance_task_metric_five_tasks"),
                    lambda d: (
                        d["checks"]["all_primary_metrics_finite"],
                        "guidance_task_metric_primary_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_primary_metrics_improve"],
                        "guidance_task_metric_primary_improve",
                    ),
                    lambda d: (
                        d["checks"]["all_tasks_have_scale_sweep_summary"],
                        "guidance_task_metric_scale_summaries",
                    ),
                    lambda d: (
                        d["checks"]["full_split_offline_links_present"],
                        "guidance_task_metric_full_split_offline_links",
                    ),
                    lambda d: (
                        d["checks"]["full_split_reverse_links_present"],
                        "guidance_task_metric_full_split_reverse_links",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "guidance_task_metric_no_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6_reproduction"],
                        "guidance_task_metric_no_fig56_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_task_metric_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guidance_full_split_result_table",
                "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["offline_status_ok"], "guidance_full_table_offline_ok"),
                    lambda d: (d["checks"]["reverse_status_ok"], "guidance_full_table_reverse_ok"),
                    lambda d: (d["checks"]["two_modes_recorded"], "guidance_full_table_two_modes"),
                    lambda d: (d["checks"]["five_tasks_per_mode"], "guidance_full_table_five_tasks"),
                    lambda d: (
                        d["checks"]["offline_row_count_matches_source"],
                        "guidance_full_table_offline_rows",
                    ),
                    lambda d: (
                        d["checks"]["reverse_row_count_matches_source"],
                        "guidance_full_table_reverse_rows",
                    ),
                    lambda d: (
                        d["checks"]["all_mean_cost_deltas_finite"],
                        "guidance_full_table_finite_deltas",
                    ),
                    lambda d: (
                        d["checks"]["all_primary_improved_counts_nonzero"],
                        "guidance_full_table_primary_counts",
                    ),
                    lambda d: (
                        d["checks"]["offline_records_all_cost_improve"],
                        "guidance_full_table_offline_all_improve",
                    ),
                    lambda d: (
                        d["checks"]["reverse_records_mixed_cost_outcome"],
                        "guidance_full_table_reverse_mixed",
                    ),
                    lambda d: (d["checks"]["figure_files_written"], "guidance_full_table_figures"),
                    lambda d: (d["metrics"]["offline_source_rows"] == 46200, "guidance_full_table_46200"),
                    lambda d: (d["metrics"]["reverse_source_rows"] == 33000, "guidance_full_table_33000"),
                    lambda d: (
                        d["metrics"]["reverse_min_after_reserve_used_mb"] >= 10000,
                        "guidance_full_table_reverse_10gb",
                    ),
                    lambda d: (
                        d["mode_summary"]["reverse"]["mean_best_cost_delta_by_task"]["joystick"] < 0.0,
                        "guidance_full_table_records_joystick_regression",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "guidance_full_table_no_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6_reproduction"],
                        "guidance_full_table_no_fig56_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_full_table_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guidance_checkpoint_visualization",
                "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["offline_status_ok"], "guidance_checkpoint_viz_offline_ok"),
                    lambda d: (d["checks"]["reverse_status_ok"], "guidance_checkpoint_viz_reverse_ok"),
                    lambda d: (
                        d["checks"]["representative_window_alignment"],
                        "guidance_checkpoint_viz_window_alignment",
                    ),
                    lambda d: (d["checks"]["five_tasks_visualized"], "guidance_checkpoint_viz_five_tasks"),
                    lambda d: (
                        d["checks"]["four_modes_per_task_recorded"],
                        "guidance_checkpoint_viz_four_modes",
                    ),
                    lambda d: (d["checks"]["all_rows_finite"], "guidance_checkpoint_viz_finite"),
                    lambda d: (
                        d["checks"]["all_visual_files_written"],
                        "guidance_checkpoint_viz_files_written",
                    ),
                    lambda d: (d["checks"]["gif_written"], "guidance_checkpoint_viz_gif"),
                    lambda d: (
                        d["checks"]["uses_full_split_checkpoint_sources"],
                        "guidance_checkpoint_viz_full_sources",
                    ),
                    lambda d: (d["metrics"]["task_count"] == 5, "guidance_checkpoint_viz_task_count"),
                    lambda d: (d["metrics"]["mode_count"] == 4, "guidance_checkpoint_viz_mode_count"),
                    lambda d: (d["metrics"]["visual_file_count"] == 16, "guidance_checkpoint_viz_file_count"),
                    lambda d: (d["metrics"]["offline_source_rows"] == 46200, "guidance_checkpoint_viz_46200"),
                    lambda d: (d["metrics"]["reverse_source_rows"] == 33000, "guidance_checkpoint_viz_33000"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "guidance_checkpoint_viz_no_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6_reproduction"],
                        "guidance_checkpoint_viz_no_fig56_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_checkpoint_viz_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guidance_visual_deliverables_audit",
                "res/level_c/guidance_visual_deliverables_audit/level_c_guidance_visual_deliverables_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["metrics"]["visual_file_count"] == 6,
                        "guidance_visual_deliverables_file_count_6",
                    ),
                    lambda d: (
                        d["metrics"]["guidance_task_count"] == 5,
                        "guidance_visual_deliverables_task_count_5",
                    ),
                    lambda d: (
                        d["metrics"]["improved_task_count"] == 5,
                        "guidance_visual_deliverables_all_tasks_improve",
                    ),
                    lambda d: (
                        d["metrics"]["scale_sweep_row_count"] == 40,
                        "guidance_visual_deliverables_scale_rows_40",
                    ),
                    lambda d: (
                        d["metrics"]["fig5_fig6_panel_count"] == 6
                        and d["metrics"]["blocked_fig_panel_count"] == 6,
                        "guidance_visual_deliverables_fig_panels_blocked",
                    ),
                    lambda d: (
                        d["metrics"]["blocked_video_requirement_count"] == 6,
                        "guidance_visual_deliverables_video_rows_blocked",
                    ),
                    lambda d: (
                        d["checks"]["png_svg_pdf_gif_npz_tsv_exist"],
                        "guidance_visual_deliverables_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["file_hashes_recorded"],
                        "guidance_visual_deliverables_hashes_recorded",
                    ),
                    lambda d: (
                        d["checks"]["five_task_metrics_improve"],
                        "guidance_visual_deliverables_task_metrics_improve",
                    ),
                    lambda d: (
                        d["checks"]["fig5_fig6_six_panels_recorded_blocked"],
                        "guidance_visual_deliverables_fig56_boundary",
                    ),
                    lambda d: (
                        d["checks"]["success_failure_video_requirements_recorded_blocked"],
                        "guidance_visual_deliverables_video_boundary",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_video_or_rollout"],
                        "guidance_visual_deliverables_no_video_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_visual_deliverables_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "visual_media_inventory_audit",
                "res/visual_media_inventory/visual_media_inventory_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] >= 80, "visual_media_rows_at_least_80"),
                    lambda d: (d["kind_counts"]["pdf"] >= 20, "visual_media_pdf_count"),
                    lambda d: (d["kind_counts"]["png"] >= 20, "visual_media_png_count"),
                    lambda d: (d["kind_counts"]["svg"] >= 20, "visual_media_svg_count"),
                    lambda d: (d["kind_counts"]["gif"] >= 3, "visual_media_gif_count"),
                    lambda d: (
                        d["category_counts"]["released_data_figure"] >= 60,
                        "visual_media_released_figures",
                    ),
                    lambda d: (
                        d["category_counts"]["debug_tiny_diffusion_preview"] >= 4,
                        "visual_media_tiny_debug_previews",
                    ),
                    lambda d: (d["checks"]["all_files_nonempty"], "visual_media_files_nonempty"),
                    lambda d: (d["checks"]["all_hashes_recorded"], "visual_media_hashes_recorded"),
                    lambda d: (
                        d["checks"]["no_paper_level_mp4_mov_mkv_reproduction_videos"],
                        "visual_media_no_paper_level_reproduction_videos",
                    ),
                    lambda d: (
                        d["checks"]["local_reference_video_allowed_and_labeled"],
                        "visual_media_reference_videos_labeled",
                    ),
                    lambda d: (
                        d["checks"]["paper_required_video_gaps_recorded"],
                        "visual_media_video_gaps_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_fig5_fig6_or_robot_video"],
                        "visual_media_no_false_video_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "visual_media_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "smoothness_latency_audit",
                "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["goal_metric_names_indexed"],
                        "smoothness_latency_goal_metrics_indexed",
                    ),
                    lambda d: (
                        d["checks"]["all_smoothness_metrics_finite"],
                        "smoothness_metrics_finite",
                    ),
                    lambda d: (d["checks"]["guidance_cost_decreases"], "smoothness_guidance_cost_decreases"),
                    lambda d: (d["checks"]["guided_mse_decreases"], "smoothness_guided_mse_decreases"),
                    lambda d: (d["checks"]["control_rate_25hz_recorded"], "smoothness_control_rate_25hz"),
                    lambda d: (
                        d["checks"]["paper_denoising_latency_within_control_period"],
                        "smoothness_denoising_latency_budget",
                    ),
                    lambda d: (
                        d["checks"]["paper_decoder_target_within_remaining_budget"],
                        "smoothness_decoder_latency_budget",
                    ),
                    lambda d: (
                        d["checks"]["schema_action_delta_finite_positive"],
                        "smoothness_action_delta_positive",
                    ),
                ],
            ),
            check_json_artifact(
                "deployment_protocol_audit",
                "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 9, "deployment_protocol_rows_9"),
                    lambda d: (d["metrics"]["failed_row_count"] == 0, "deployment_protocol_failures_zero"),
                    lambda d: (
                        d["checks"]["paper_deployment_protocol_source_found"],
                        "deployment_protocol_paper_sources_found",
                    ),
                    lambda d: (
                        d["checks"]["paper_latency_budget_recorded"],
                        "deployment_protocol_latency_budget",
                    ),
                    lambda d: (
                        d["checks"]["decoder_cpu_schema_probe_present"],
                        "deployment_protocol_cpu_decoder_schema",
                    ),
                    lambda d: (
                        d["checks"]["official_tensorrt_engine_absent_recorded"],
                        "deployment_protocol_tensorrt_absence_recorded",
                    ),
                    lambda d: (
                        d["checks"]["async_tensorrt_cppad_boundaries_recorded"],
                        "deployment_protocol_boundaries_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_deployment_reproduction"],
                        "deployment_protocol_no_false_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "training_schedule_probe",
                "res/level_c/training_schedule_probe/level_c_training_schedule_probe.json",
                [status_ok, lambda d: (d["checks"]["lr_peak_matches_paper_lr"], "lr_peak_matches")],
            ),
            check_json_artifact(
                "transformer_state_dict_manifest",
                "res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["uses_paper_state_token_dim_131"], "transformer_state_token_dim_131"),
                    lambda d: (
                        d["checks"]["uses_paper_transformer_hyperparameters"],
                        "transformer_state_paper_hyperparams",
                    ),
                    lambda d: (
                        d["checks"]["state_dict_hash_deterministic_for_same_seed"],
                        "transformer_state_hash_deterministic",
                    ),
                    lambda d: (
                        d["checks"]["state_dict_hash_changes_for_different_seed"],
                        "transformer_state_hash_seed_sensitive",
                    ),
                    lambda d: (
                        d["checks"]["parameter_count_matches_arch_probe"],
                        "transformer_state_matches_arch_probe",
                    ),
                    lambda d: (
                        d["checks"]["parameter_count_matches_parameter_audit"],
                        "transformer_state_matches_param_audit",
                    ),
                    lambda d: (d["checks"]["does_not_write_weight_checkpoint"], "transformer_state_no_weight_ckpt"),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "transformer_state_keeps_incomplete"),
                ],
            ),
            check_json_artifact(
                "transformer_ema_smoke",
                "res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["uses_paper_state_token_dim_131"], "transformer_ema_token_dim_131"),
                    lambda d: (
                        d["checks"]["uses_paper_transformer_hyperparameters"],
                        "transformer_ema_paper_hyperparams",
                    ),
                    lambda d: (
                        d["checks"]["uses_paper_optimizer_hyperparameters"],
                        "transformer_ema_optimizer_hyperparams",
                    ),
                    lambda d: (d["checks"]["learning_rate_schedule_applied"], "transformer_ema_lr_schedule"),
                    lambda d: (d["checks"]["ema_decay_schedule_applied"], "transformer_ema_decay_schedule"),
                    lambda d: (d["checks"]["all_grad_norms_positive"], "transformer_ema_grad_norms"),
                    lambda d: (d["checks"]["model_parameters_changed"], "transformer_ema_params_changed"),
                    lambda d: (
                        d["checks"]["ema_shadow_differs_from_final_model"],
                        "transformer_ema_shadow_differs",
                    ),
                    lambda d: (
                        d["checks"]["loss_after_not_worse_than_before_on_last_batch"],
                        "transformer_ema_loss_not_worse",
                    ),
                    lambda d: (d["checks"]["does_not_write_weight_checkpoint"], "transformer_ema_no_weight_ckpt"),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "transformer_ema_keeps_incomplete"),
                ],
            ),
            check_json_artifact(
                "vae_latent_transformer_ema_smoke",
                "res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["debug_vae_latent_artifact_status_ok"],
                        "vae_latent_ema_input_ok",
                    ),
                    lambda d: (
                        d["checks"]["debug_vae_latents_nonzero"],
                        "vae_latent_ema_latents_nonzero",
                    ),
                    lambda d: (d["checks"]["uses_paper_state_token_dim_131"], "vae_latent_ema_token_dim_131"),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "vae_latent_ema_state_dim_99"),
                    lambda d: (
                        d["checks"]["uses_debug_vae_latent_dim_32"],
                        "vae_latent_ema_latent_dim_32",
                    ),
                    lambda d: (
                        d["checks"]["uses_paper_transformer_hyperparameters"],
                        "vae_latent_ema_transformer_hyperparams",
                    ),
                    lambda d: (
                        d["checks"]["uses_paper_optimizer_hyperparameters"],
                        "vae_latent_ema_optimizer_hyperparams",
                    ),
                    lambda d: (d["checks"]["learning_rate_schedule_applied"], "vae_latent_ema_lr_schedule"),
                    lambda d: (d["checks"]["ema_decay_schedule_applied"], "vae_latent_ema_decay_schedule"),
                    lambda d: (d["checks"]["all_grad_norms_positive"], "vae_latent_ema_grad_norms"),
                    lambda d: (d["checks"]["model_parameters_changed"], "vae_latent_ema_params_changed"),
                    lambda d: (
                        d["checks"]["ema_shadow_differs_from_final_model"],
                        "vae_latent_ema_shadow_differs",
                    ),
                    lambda d: (
                        d["checks"]["loss_after_not_worse_than_before_on_last_batch"],
                        "vae_latent_ema_loss_not_worse",
                    ),
                    lambda d: (
                        d["checks"]["does_not_write_weight_checkpoint"],
                        "vae_latent_ema_no_weight_ckpt",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "vae_latent_ema_no_true_rollout_claim",
                    ),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "vae_latent_ema_keeps_incomplete"),
                ],
            ),
            check_json_artifact(
                "diffusion_checkpoint_smoke",
                "res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["checkpoint_file_exists"], "diffusion_checkpoint_file_exists"),
                    lambda d: (d["checks"]["checkpoint_has_required_keys"], "diffusion_checkpoint_required_keys"),
                    lambda d: (
                        d["checks"]["uses_paper_state_token_dim_131"],
                        "diffusion_checkpoint_token_dim_131",
                    ),
                    lambda d: (
                        d["checks"]["uses_paper_transformer_hyperparameters"],
                        "diffusion_checkpoint_transformer_hyperparams",
                    ),
                    lambda d: (
                        d["checks"]["uses_paper_optimizer_hyperparameters"],
                        "diffusion_checkpoint_optimizer_hyperparams",
                    ),
                    lambda d: (
                        d["checks"]["loaded_model_matches_uninterrupted_after_resume"],
                        "diffusion_checkpoint_model_resume_exact",
                    ),
                    lambda d: (
                        d["checks"]["loaded_ema_matches_uninterrupted_after_resume"],
                        "diffusion_checkpoint_ema_resume_exact",
                    ),
                    lambda d: (
                        d["checks"]["loaded_optimizer_matches_uninterrupted_after_resume"],
                        "diffusion_checkpoint_optimizer_resume_exact",
                    ),
                    lambda d: (
                        d["checks"]["loaded_eval_prediction_matches_uninterrupted"],
                        "diffusion_checkpoint_eval_resume_exact",
                    ),
                    lambda d: (
                        d["checks"]["marks_not_trained_paper_checkpoint"],
                        "diffusion_checkpoint_not_trained_paper",
                    ),
                    lambda d: (
                        d["checks"]["marks_not_ema_paper_checkpoint"],
                        "diffusion_checkpoint_not_ema_paper",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_rollout"],
                        "diffusion_checkpoint_no_true_vae_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "diffusion_checkpoint_keeps_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_bounded_debug_diffusion_training_run",
                "res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["run_dir_exists"], "bounded_debug_run_dir_exists"),
                    lambda d: (d["checks"]["all_required_files_exist"], "bounded_debug_required_files"),
                    lambda d: (d["checks"]["all_required_dirs_exist"], "bounded_debug_required_dirs"),
                    lambda d: (
                        d["checks"]["status_success_but_not_paper_training"],
                        "bounded_debug_success_not_paper_training",
                    ),
                    lambda d: (d["checks"]["checkpoint_file_exists"], "bounded_debug_checkpoint_exists"),
                    lambda d: (d["checks"]["figure_file_exists"], "bounded_debug_figure_exists"),
                    lambda d: (d["checks"]["videos_dir_empty"], "bounded_debug_videos_empty"),
                    lambda d: (
                        d["checks"]["metrics_have_runtime_fields"],
                        "bounded_debug_runtime_fields",
                    ),
                    lambda d: (d["metrics"]["debug_step_count"] == 3, "bounded_debug_step_count_3"),
                    lambda d: (d["metrics"]["debug_token_dim"] == 131, "bounded_debug_token_dim_131"),
                    lambda d: (d["checks"]["all_same_batch_losses_not_worse_after_step"], "bounded_debug_loss_steps"),
                    lambda d: (d["checks"]["model_parameters_changed"], "bounded_debug_params_changed"),
                    lambda d: (
                        d["checks"]["does_not_claim_full_training_run"],
                        "bounded_debug_no_full_training_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "bounded_debug_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_bounded_debug_diffusion_checkpoint_eval",
                "res/level_c/bounded_debug_diffusion_checkpoint_eval/level_c_bounded_debug_diffusion_checkpoint_eval.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["training_audit_status_ok"], "bounded_eval_training_audit_ok"),
                    lambda d: (d["checks"]["checkpoint_file_exists"], "bounded_eval_checkpoint_exists"),
                    lambda d: (
                        d["checks"]["checkpoint_sha_matches_training_audit"],
                        "bounded_eval_checkpoint_sha_matches",
                    ),
                    lambda d: (d["checks"]["checkpoint_marked_debug_only"], "bounded_eval_checkpoint_debug_only"),
                    lambda d: (d["checks"]["run_id_matches_training_run"], "bounded_eval_run_id_matches"),
                    lambda d: (d["checks"]["token_dim_131"], "bounded_eval_token_dim_131"),
                    lambda d: (d["checks"]["split_counts_28_each"], "bounded_eval_split_counts"),
                    lambda d: (d["checks"]["all_eval_losses_finite"], "bounded_eval_losses_finite"),
                    lambda d: (d["checks"]["model_differs_from_initial"], "bounded_eval_model_differs"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "bounded_eval_no_rollout_claim",
                    ),
                    lambda d: (d["checks"]["does_not_claim_paper_metrics"], "bounded_eval_no_paper_metrics"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "bounded_eval_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_bounded_debug_diffusion_action_eval",
                "res/level_c/bounded_debug_diffusion_action_eval/level_c_bounded_debug_diffusion_action_eval.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["training_audit_status_ok"], "bounded_action_training_audit_ok"),
                    lambda d: (d["checks"]["checkpoint_file_exists"], "bounded_action_checkpoint_exists"),
                    lambda d: (d["checks"]["checkpoint_marked_debug_only"], "bounded_action_checkpoint_debug"),
                    lambda d: (d["checks"]["vae_source_status_ok"], "bounded_action_vae_source_ok"),
                    lambda d: (d["checks"]["uses_motion_level_split_manifest"], "bounded_action_split_manifest"),
                    lambda d: (d["checks"]["token_dim_131"], "bounded_action_token_dim_131"),
                    lambda d: (d["checks"]["action_dim_29"], "bounded_action_dim_29"),
                    lambda d: (
                        d["checks"]["uses_train_split_only_for_decoder_surrogate"],
                        "bounded_action_train_decoder",
                    ),
                    lambda d: (d["checks"]["all_action_metrics_finite"], "bounded_action_metrics_finite"),
                    lambda d: (d["checks"]["split_counts_28_each"], "bounded_action_split_counts"),
                    lambda d: (
                        d["checks"]["checkpoint_action_eval_recorded_even_if_poor"],
                        "bounded_action_eval_recorded",
                    ),
                    lambda d: (d["checks"]["npz_written"], "bounded_action_npz_written"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "bounded_action_no_rollout_claim",
                    ),
                    lambda d: (d["checks"]["does_not_claim_paper_metrics"], "bounded_action_no_paper_metrics"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "bounded_action_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_training_run",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
                    "level_c_resource_adjusted_tiny_diffusion_training_run.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["checkpoint_file_exists"], "tiny_denoiser_checkpoint_exists"),
                    lambda d: (d["checks"]["figures_exist"], "tiny_denoiser_figures_exist"),
                    lambda d: (d["checks"]["npz_written"], "tiny_denoiser_npz_written"),
                    lambda d: (d["checks"]["split_counts_28_each"], "tiny_denoiser_split_counts"),
                    lambda d: (
                        d["checks"]["train_token_mse_improves_vs_noisy"],
                        "tiny_denoiser_train_improves",
                    ),
                    lambda d: (d["checks"]["validation_token_mse_finite"], "tiny_denoiser_val_finite"),
                    lambda d: (d["checks"]["test_token_mse_finite"], "tiny_denoiser_test_finite"),
                    lambda d: (d["checks"]["all_action_metrics_finite"], "tiny_denoiser_action_finite"),
                    lambda d: (
                        d["checks"]["status_success_but_not_paper_training"],
                        "tiny_denoiser_not_paper_training",
                    ),
                    lambda d: (
                        d["checks"].get("videos_are_not_required_full_run_videos")
                        or d["checks"].get("videos_dir_empty"),
                        "tiny_denoiser_no_required_video_claim",
                    ),
                    lambda d: (d["metrics"]["parameter_count"] == 143491, "tiny_denoiser_param_count"),
                    lambda d: (d["metrics"]["epochs"] >= 100, "tiny_denoiser_epochs"),
                    lambda d: (
                        d["metrics"]["validation_pred_token_mse"] < d["eval_rows"][1]["noisy_token_mse"],
                        "tiny_denoiser_val_better_than_noisy",
                    ),
                    lambda d: (
                        d["metrics"]["test_pred_token_mse"] < d["eval_rows"][2]["noisy_token_mse"],
                        "tiny_denoiser_test_better_than_noisy",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_denoiser_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_suite",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_suite/"
                    "level_c_resource_adjusted_tiny_diffusion_suite.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["all_steps_pass"], "tiny_suite_all_steps_pass"),
                    lambda d: (d["step_count"] == 6, "tiny_suite_step_count_6"),
                    lambda d: (d["pass_count"] == 6, "tiny_suite_pass_count_6"),
                    lambda d: (d["checks"]["training_checkpoint_exists"], "tiny_suite_checkpoint_exists"),
                    lambda d: (
                        d["checks"]["training_heldout_token_mse_better_than_noisy"],
                        "tiny_suite_heldout_token_mse_improves",
                    ),
                    lambda d: (d["checks"]["multiseed_three_seeds"], "tiny_suite_multiseed_three"),
                    lambda d: (
                        d["checks"]["multiseed_heldout_improves"],
                        "tiny_suite_multiseed_improves",
                    ),
                    lambda d: (
                        d["checks"]["checkpoint_eval_reproduces_training_metrics"],
                        "tiny_suite_checkpoint_reproduces",
                    ),
                    lambda d: (d["checks"]["onnx_matches_torch"], "tiny_suite_onnx_matches_torch"),
                    lambda d: (d["checks"]["latency_under_debug_budget"], "tiny_suite_latency_budget"),
                    lambda d: (d["checks"]["video_previews_written"], "tiny_suite_video_previews"),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_training_or_deployment"],
                        "tiny_suite_no_paper_claim",
                    ),
                    lambda d: (d["metrics"]["parameter_count"] == 143491, "tiny_suite_param_count"),
                    lambda d: (d["metrics"]["onnx_reference_cpu_p95_ms"] < 20.0, "tiny_suite_latency_p95"),
                    lambda d: (d["metrics"]["video_preview_count"] == 2, "tiny_suite_video_count_2"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_suite_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_vae_diffusion_training",
                (
                    "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
                    "lafan1_paper_arch_vae_diffusion_training.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["uses_all_available_or_requested_public_g1_motions"],
                        "lafan1_paper_arch_uses_all_40_motions",
                    ),
                    lambda d: (d["checks"]["paper_vae_architecture_used"], "lafan1_paper_arch_vae"),
                    lambda d: (
                        d["checks"]["paper_diffusion_architecture_used"],
                        "lafan1_paper_arch_diffusion",
                    ),
                    lambda d: (
                        d["checks"]["run_schema_checkpoint_metrics_figures_written"],
                        "lafan1_paper_arch_outputs_written",
                    ),
                    lambda d: (d["checks"]["uses_multi_gpu_when_available"], "lafan1_paper_arch_multigpu"),
                    lambda d: (d["metrics"]["public_lafan1_motion_count"] == 40, "lafan1_paper_arch_motion_count_40"),
                    lambda d: (d["metrics"]["window_count"] >= 2000, "lafan1_paper_arch_window_count"),
                    lambda d: (d["settings"]["diffusion_epochs"] >= 1000, "lafan1_paper_arch_diffusion_epochs"),
                    lambda d: (d["settings"]["vae_epochs"] >= 24, "lafan1_paper_arch_vae_epochs"),
                    lambda d: (d["metrics"]["diffusion_parameter_count"] >= 19000000, "lafan1_paper_arch_param_count"),
                    lambda d: (d["metrics"]["checkpoint_size_bytes"] > 250_000_000, "lafan1_paper_arch_checkpoint_size"),
                    lambda d: (
                        d["metrics"]["data_parallel"] is True and len(d["metrics"]["gpu_device_ids"]) == 8,
                        "lafan1_paper_arch_8gpu_dataparallel",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_unavailable_teacher_rollout_dataset"],
                        "lafan1_paper_arch_no_teacher_rollout_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_real_robot_or_fig5_fig6"],
                        "lafan1_paper_arch_no_robot_fig_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_paper_arch_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_onnx_latency_audit",
                (
                    "res/level_c/lafan1_paper_arch_onnx_latency/"
                    "level_c_lafan1_paper_arch_onnx_latency_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_onnx_source_training_ok"),
                    lambda d: (d["checks"]["source_checkpoint_hash_matches"], "lafan1_onnx_checkpoint_sha"),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_onnx_paper_arch_checkpoint"),
                    lambda d: (
                        d["checks"]["public_lafan1_dataset_boundary_recorded"],
                        "lafan1_onnx_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["vae_onnx_written"], "lafan1_onnx_vae_written"),
                    lambda d: (d["checks"]["diffusion_onnx_written"], "lafan1_onnx_diffusion_written"),
                    lambda d: (d["checks"]["onnx_checker_passed"], "lafan1_onnx_checker"),
                    lambda d: (d["checks"]["reference_evaluator_loaded"], "lafan1_onnx_reference_evaluator"),
                    lambda d: (d["checks"]["vae_onnx_matches_torch"], "lafan1_onnx_vae_matches_torch"),
                    lambda d: (d["checks"]["diffusion_onnx_matches_torch"], "lafan1_onnx_diffusion_matches_torch"),
                    lambda d: (d["checks"]["latency_rows_written"], "lafan1_onnx_latency_rows"),
                    lambda d: (d["checks"]["io_fixture_written"], "lafan1_onnx_io_fixture"),
                    lambda d: (d["checks"]["metadata_marks_boundary"], "lafan1_onnx_metadata_boundary"),
                    lambda d: (d["checks"]["does_not_claim_tensorrt"], "lafan1_onnx_no_tensorrt_claim"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_or_robot"],
                        "lafan1_onnx_no_closed_loop_robot_claim",
                    ),
                    lambda d: (d["metrics"]["diffusion_parameter_count"] >= 19000000, "lafan1_onnx_param_count"),
                    lambda d: (d["metrics"]["diffusion_onnx_size_bytes"] > 70_000_000, "lafan1_onnx_size"),
                    lambda d: (
                        d["metrics"]["diffusion_max_abs_onnx_vs_torch"] <= 1e-4,
                        "lafan1_onnx_diffusion_max_abs",
                    ),
                    lambda d: (
                        d["metrics"]["vae_max_abs_onnx_vs_torch"] <= 1e-5,
                        "lafan1_onnx_vae_max_abs",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_onnx_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_onnx_latency_audit",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
                    "level_c_lafan1_paper_arch_onnx_latency_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_symonnx_source_training_ok"),
                    lambda d: (d["checks"]["source_checkpoint_hash_matches"], "lafan1_symonnx_checkpoint_sha"),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_symonnx_paper_arch_checkpoint"),
                    lambda d: (
                        d["checks"]["public_lafan1_dataset_boundary_recorded"],
                        "lafan1_symonnx_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["vae_onnx_written"], "lafan1_symonnx_vae_written"),
                    lambda d: (d["checks"]["diffusion_onnx_written"], "lafan1_symonnx_diffusion_written"),
                    lambda d: (d["checks"]["onnx_checker_passed"], "lafan1_symonnx_checker"),
                    lambda d: (d["checks"]["reference_evaluator_loaded"], "lafan1_symonnx_reference_evaluator"),
                    lambda d: (d["checks"]["vae_onnx_matches_torch"], "lafan1_symonnx_vae_matches_torch"),
                    lambda d: (d["checks"]["diffusion_onnx_matches_torch"], "lafan1_symonnx_diffusion_matches_torch"),
                    lambda d: (d["checks"]["latency_rows_written"], "lafan1_symonnx_latency_rows"),
                    lambda d: (d["checks"]["io_fixture_written"], "lafan1_symonnx_io_fixture"),
                    lambda d: (d["checks"]["metadata_marks_boundary"], "lafan1_symonnx_metadata_boundary"),
                    lambda d: (d["checks"]["does_not_claim_tensorrt"], "lafan1_symonnx_no_tensorrt_claim"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_or_robot"],
                        "lafan1_symonnx_no_closed_loop_robot_claim",
                    ),
                    lambda d: (d["metrics"]["diffusion_parameter_count"] >= 19000000, "lafan1_symonnx_param_count"),
                    lambda d: (d["metrics"]["diffusion_onnx_size_bytes"] > 70_000_000, "lafan1_symonnx_size"),
                    lambda d: (
                        d["metrics"]["diffusion_max_abs_onnx_vs_torch"] <= 1e-4,
                        "lafan1_symonnx_diffusion_max_abs",
                    ),
                    lambda d: (
                        d["metrics"]["vae_max_abs_onnx_vs_torch"] <= 1e-5,
                        "lafan1_symonnx_vae_max_abs",
                    ),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_training" in d["settings"]["training_json"],
                        "lafan1_symonnx_training_source",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_symonnx_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_offline_metrics_audit",
                (
                    "res/level_c/lafan1_paper_arch_offline_metrics/"
                    "level_c_lafan1_paper_arch_offline_metrics_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_metrics_source_training_ok"),
                    lambda d: (d["checks"]["source_checkpoint_hash_matches"], "lafan1_metrics_checkpoint_sha"),
                    lambda d: (d["checks"]["onnx_latency_status_ok"], "lafan1_metrics_onnx_latency_ok"),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_metrics_paper_arch"),
                    lambda d: (
                        d["checks"]["public_dataset_boundary_recorded"],
                        "lafan1_metrics_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["all_splits_present"], "lafan1_metrics_all_splits"),
                    lambda d: (d["checks"]["all_metrics_finite"], "lafan1_metrics_finite"),
                    lambda d: (
                        d["checks"]["diffusion_improves_vs_noisy_all_splits"],
                        "lafan1_metrics_diffusion_improves",
                    ),
                    lambda d: (
                        d["checks"]["validation_and_test_action_metrics_present"],
                        "lafan1_metrics_action_metrics",
                    ),
                    lambda d: (d["checks"]["smoothness_metrics_present"], "lafan1_metrics_smoothness"),
                    lambda d: (d["checks"]["npz_written"], "lafan1_metrics_npz"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_success"],
                        "lafan1_metrics_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt_or_robot"],
                        "lafan1_metrics_no_tensorrt_robot_claim",
                    ),
                    lambda d: (
                        d["metrics"]["validation_diffusion_pred_tau_mse"] < 0.02,
                        "lafan1_metrics_validation_tau_mse",
                    ),
                    lambda d: (
                        d["metrics"]["test_diffusion_pred_tau_mse"] < 0.02,
                        "lafan1_metrics_test_tau_mse",
                    ),
                    lambda d: (
                        d["metrics"]["validation_decoded_pred_current_action_mse"] < 0.05,
                        "lafan1_metrics_validation_action_mse",
                    ),
                    lambda d: (
                        d["metrics"]["test_decoded_pred_current_action_mse"] < 0.05,
                        "lafan1_metrics_test_action_mse",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_metrics_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_offline_metrics_audit",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/"
                    "level_c_lafan1_paper_arch_offline_metrics_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_symmetrics_source_training_ok"),
                    lambda d: (d["checks"]["source_checkpoint_hash_matches"], "lafan1_symmetrics_checkpoint_sha"),
                    lambda d: (d["checks"]["onnx_latency_status_ok"], "lafan1_symmetrics_onnx_latency_ok"),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_symmetrics_paper_arch"),
                    lambda d: (
                        d["checks"]["public_dataset_boundary_recorded"],
                        "lafan1_symmetrics_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["all_splits_present"], "lafan1_symmetrics_all_splits"),
                    lambda d: (d["checks"]["all_metrics_finite"], "lafan1_symmetrics_finite"),
                    lambda d: (
                        d["checks"]["diffusion_improves_vs_noisy_all_splits"],
                        "lafan1_symmetrics_diffusion_improves",
                    ),
                    lambda d: (
                        d["checks"]["validation_and_test_action_metrics_present"],
                        "lafan1_symmetrics_action_metrics",
                    ),
                    lambda d: (d["checks"]["smoothness_metrics_present"], "lafan1_symmetrics_smoothness"),
                    lambda d: (d["checks"]["npz_written"], "lafan1_symmetrics_npz"),
                    lambda d: (
                        d["settings"]["window_count"] == 4400,
                        "lafan1_symmetrics_window_count_4400",
                    ),
                    lambda d: (
                        d["settings"]["token_count"] == 92400,
                        "lafan1_symmetrics_token_count_92400",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_success"],
                        "lafan1_symmetrics_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt_or_robot"],
                        "lafan1_symmetrics_no_tensorrt_robot_claim",
                    ),
                    lambda d: (
                        d["metrics"]["validation_diffusion_pred_tau_mse"] < 0.02,
                        "lafan1_symmetrics_validation_tau_mse",
                    ),
                    lambda d: (
                        d["metrics"]["test_diffusion_pred_tau_mse"] < 0.02,
                        "lafan1_symmetrics_test_tau_mse",
                    ),
                    lambda d: (
                        d["metrics"]["validation_decoded_pred_current_action_mse"] < 0.05,
                        "lafan1_symmetrics_validation_action_mse",
                    ),
                    lambda d: (
                        d["metrics"]["test_decoded_pred_current_action_mse"] < 0.05,
                        "lafan1_symmetrics_test_action_mse",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_symmetrics_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_guidance_eval",
                "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_guidance_source_training_ok"),
                    lambda d: (d["checks"]["offline_metrics_status_ok"], "lafan1_guidance_offline_metrics_ok"),
                    lambda d: (d["checks"]["source_checkpoint_hash_matches"], "lafan1_guidance_checkpoint_sha"),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_guidance_paper_arch"),
                    lambda d: (
                        d["checks"]["public_dataset_boundary_recorded"],
                        "lafan1_guidance_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["five_tasks_evaluated"], "lafan1_guidance_five_tasks"),
                    lambda d: (d["checks"]["scale_grid_evaluated"], "lafan1_guidance_scale_grid"),
                    lambda d: (d["checks"]["all_rows_finite"], "lafan1_guidance_finite"),
                    lambda d: (
                        d["checks"]["all_task_best_costs_improve"],
                        "lafan1_guidance_best_costs_improve",
                    ),
                    lambda d: (d["checks"]["gradients_nonzero"], "lafan1_guidance_gradients_nonzero"),
                    lambda d: (d["checks"]["npz_written"], "lafan1_guidance_npz"),
                    lambda d: (d["row_count"] == 175, "lafan1_guidance_rows_175"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "lafan1_guidance_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_success_failure_videos"],
                        "lafan1_guidance_no_video_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/"
                    "level_c_lafan1_paper_arch_guidance_eval.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_symguidance_source_training_ok"),
                    lambda d: (d["checks"]["offline_metrics_status_ok"], "lafan1_symguidance_offline_metrics_ok"),
                    lambda d: (d["checks"]["source_checkpoint_hash_matches"], "lafan1_symguidance_checkpoint_sha"),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_symguidance_paper_arch"),
                    lambda d: (
                        d["checks"]["public_dataset_boundary_recorded"],
                        "lafan1_symguidance_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["five_tasks_evaluated"], "lafan1_symguidance_five_tasks"),
                    lambda d: (d["checks"]["scale_grid_evaluated"], "lafan1_symguidance_scale_grid"),
                    lambda d: (d["checks"]["all_rows_finite"], "lafan1_symguidance_finite"),
                    lambda d: (
                        d["checks"]["all_task_best_costs_improve"],
                        "lafan1_symguidance_best_costs_improve",
                    ),
                    lambda d: (d["checks"]["gradients_nonzero"], "lafan1_symguidance_gradients_nonzero"),
                    lambda d: (d["checks"]["npz_written"], "lafan1_symguidance_npz"),
                    lambda d: (d["row_count"] == 175, "lafan1_symguidance_rows_175"),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_training" in d["settings"]["training_json"],
                        "lafan1_symguidance_training_source",
                    ),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_offline_metrics"
                        in d["settings"]["offline_metrics_json"],
                        "lafan1_symguidance_offline_source",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "lafan1_symguidance_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_success_failure_videos"],
                        "lafan1_symguidance_no_video_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_symguidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
                    "level_c_lafan1_paper_arch_guidance_eval.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_fullsymguidance_source_ok"),
                    lambda d: (d["checks"]["offline_metrics_status_ok"], "lafan1_fullsymguidance_metrics_ok"),
                    lambda d: (
                        d["checks"]["source_checkpoint_hash_matches"],
                        "lafan1_fullsymguidance_checkpoint_sha",
                    ),
                    lambda d: (d["checks"]["paper_architecture_checkpoint"], "lafan1_fullsymguidance_paper_arch"),
                    lambda d: (
                        d["checks"]["public_dataset_boundary_recorded"],
                        "lafan1_fullsymguidance_public_dataset_boundary",
                    ),
                    lambda d: (d["checks"]["five_tasks_evaluated"], "lafan1_fullsymguidance_five_tasks"),
                    lambda d: (d["checks"]["scale_grid_evaluated"], "lafan1_fullsymguidance_scale_grid"),
                    lambda d: (
                        d["checks"]["requested_splits_evaluated"],
                        "lafan1_fullsymguidance_requested_splits",
                    ),
                    lambda d: (
                        d["checks"]["selected_window_count_matches_split_counts"],
                        "lafan1_fullsymguidance_selected_count",
                    ),
                    lambda d: (d["checks"]["all_rows_finite"], "lafan1_fullsymguidance_finite"),
                    lambda d: (
                        d["checks"]["all_task_best_costs_improve"],
                        "lafan1_fullsymguidance_best_costs_improve",
                    ),
                    lambda d: (d["checks"]["gradients_nonzero"], "lafan1_fullsymguidance_gradients"),
                    lambda d: (d["checks"]["npz_written"], "lafan1_fullsymguidance_npz"),
                    lambda d: (d["row_count"] == 46200, "lafan1_fullsymguidance_rows_46200"),
                    lambda d: (
                        d["settings"]["split_window_counts"] == {"test": 660, "validation": 660},
                        "lafan1_fullsymguidance_split_counts",
                    ),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_training" in d["settings"]["training_json"],
                        "lafan1_fullsymguidance_training_source",
                    ),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_offline_metrics"
                        in d["settings"]["offline_metrics_json"],
                        "lafan1_fullsymguidance_offline_source",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "lafan1_fullsymguidance_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_success_failure_videos"],
                        "lafan1_fullsymguidance_no_video_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_fullsymguidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_reverse_guidance",
                (
                    "res/level_c/lafan1_paper_arch_reverse_guidance/"
                    "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["source_training_status_ok"],
                        "lafan1_reverse_guidance_source_training_ok",
                    ),
                    lambda d: (
                        d["checks"]["offline_guidance_status_ok"],
                        "lafan1_reverse_guidance_offline_guidance_ok",
                    ),
                    lambda d: (
                        d["checks"]["source_checkpoint_hash_matches"],
                        "lafan1_reverse_guidance_checkpoint_sha",
                    ),
                    lambda d: (
                        d["checks"]["paper_architecture_checkpoint"],
                        "lafan1_reverse_guidance_paper_arch",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_equation_reverse_form_audited"],
                        "lafan1_reverse_guidance_formula_audited",
                    ),
                    lambda d: (
                        d["checks"]["uses_twenty_reverse_steps"],
                        "lafan1_reverse_guidance_20_steps",
                    ),
                    lambda d: (d["checks"]["five_tasks_evaluated"], "lafan1_reverse_guidance_five_tasks"),
                    lambda d: (
                        d["checks"]["scale_grid_includes_unguided"],
                        "lafan1_reverse_guidance_scale_grid",
                    ),
                    lambda d: (d["checks"]["all_rows_finite"], "lafan1_reverse_guidance_finite"),
                    lambda d: (
                        d["checks"]["at_least_three_tasks_best_costs_improve"],
                        "lafan1_reverse_guidance_cost_improves",
                    ),
                    lambda d: (
                        d["checks"]["all_tasks_have_some_primary_improvement"],
                        "lafan1_reverse_guidance_primary_improves",
                    ),
                    lambda d: (
                        d["checks"]["guidance_gradients_nonzero_all_reverse_steps"],
                        "lafan1_reverse_guidance_gradients",
                    ),
                    lambda d: (d["checks"]["npz_written"], "lafan1_reverse_guidance_npz"),
                    lambda d: (d["row_count"] == 75, "lafan1_reverse_guidance_rows_75"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "lafan1_reverse_guidance_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt_or_robot"],
                        "lafan1_reverse_guidance_no_tensorrt_robot_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_coefficient_schedule"],
                        "lafan1_reverse_guidance_no_schedule_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_reverse_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
                    "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["source_training_status_ok"],
                        "lafan1_fullsymreverse_source_training_ok",
                    ),
                    lambda d: (
                        d["checks"]["offline_guidance_status_ok"],
                        "lafan1_fullsymreverse_offline_guidance_ok",
                    ),
                    lambda d: (
                        d["checks"]["source_checkpoint_hash_matches"],
                        "lafan1_fullsymreverse_checkpoint_sha",
                    ),
                    lambda d: (
                        d["checks"]["paper_architecture_checkpoint"],
                        "lafan1_fullsymreverse_paper_arch",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_equation_reverse_form_audited"],
                        "lafan1_fullsymreverse_formula_audited",
                    ),
                    lambda d: (
                        d["checks"]["uses_twenty_reverse_steps"],
                        "lafan1_fullsymreverse_20_steps",
                    ),
                    lambda d: (d["checks"]["five_tasks_evaluated"], "lafan1_fullsymreverse_five_tasks"),
                    lambda d: (
                        d["checks"]["scale_grid_includes_unguided"],
                        "lafan1_fullsymreverse_scale_grid",
                    ),
                    lambda d: (
                        d["checks"]["requested_splits_evaluated"],
                        "lafan1_fullsymreverse_requested_splits",
                    ),
                    lambda d: (
                        d["checks"]["selected_window_count_matches_split_counts"],
                        "lafan1_fullsymreverse_selected_count",
                    ),
                    lambda d: (
                        d["checks"]["row_count_matches_windows_tasks_scales"],
                        "lafan1_fullsymreverse_row_count_formula",
                    ),
                    lambda d: (d["checks"]["all_rows_finite"], "lafan1_fullsymreverse_finite"),
                    lambda d: (
                        d["checks"]["task_cost_improvement_statistics_recorded"],
                        "lafan1_fullsymreverse_stats_recorded",
                    ),
                    lambda d: (
                        d["checks"]["all_tasks_have_some_primary_improvement"],
                        "lafan1_fullsymreverse_primary_improves",
                    ),
                    lambda d: (
                        d["checks"]["guidance_gradients_nonzero_all_reverse_steps"],
                        "lafan1_fullsymreverse_gradients",
                    ),
                    lambda d: (
                        d["checks"]["eight_cuda_gpus_visible"],
                        "lafan1_fullsymreverse_8_gpus_visible",
                    ),
                    lambda d: (
                        d["checks"]["uses_dataparallel_8_gpus"],
                        "lafan1_fullsymreverse_dataparallel_8",
                    ),
                    lambda d: (
                        d["checks"]["all_gpus_reach_target_memory_after_reserve"],
                        "lafan1_fullsymreverse_10gb_all_gpus",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_memory_reserve_as_reverse_batch_memory"],
                        "lafan1_fullsymreverse_no_reserve_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_require_all_tasks_to_improve_for_audit_pass"],
                        "lafan1_fullsymreverse_no_improvement_overclaim",
                    ),
                    lambda d: (d["checks"]["npz_written"], "lafan1_fullsymreverse_npz"),
                    lambda d: (d["metrics"]["row_count"] == 33000, "lafan1_fullsymreverse_rows_33000"),
                    lambda d: (
                        d["settings"]["split_window_counts"] == {"test": 660, "validation": 660},
                        "lafan1_fullsymreverse_split_counts",
                    ),
                    lambda d: (
                        d["settings"]["selected_window_count"] == 1320,
                        "lafan1_fullsymreverse_window_count",
                    ),
                    lambda d: (
                        d["metrics"]["min_after_reserve_used_mb"] >= 10000,
                        "lafan1_fullsymreverse_min_memory_10gb",
                    ),
                    lambda d: (
                        d["improvement_summary"]["tasks_with_all_best_costs_improved"] == 2,
                        "lafan1_fullsymreverse_records_two_cost_improved_tasks",
                    ),
                    lambda d: (
                        d["improvement_summary"]["tasks_with_some_primary_metric_improvement"] == 5,
                        "lafan1_fullsymreverse_records_five_primary_improved_tasks",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "lafan1_fullsymreverse_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt_or_robot"],
                        "lafan1_fullsymreverse_no_tensorrt_robot_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_fullsymreverse_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/"
                    "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["source_training_status_ok"],
                        "lafan1_symreverse_source_training_ok",
                    ),
                    lambda d: (
                        d["checks"]["offline_guidance_status_ok"],
                        "lafan1_symreverse_offline_guidance_ok",
                    ),
                    lambda d: (
                        d["checks"]["source_checkpoint_hash_matches"],
                        "lafan1_symreverse_checkpoint_sha",
                    ),
                    lambda d: (
                        d["checks"]["paper_architecture_checkpoint"],
                        "lafan1_symreverse_paper_arch",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_equation_reverse_form_audited"],
                        "lafan1_symreverse_formula_audited",
                    ),
                    lambda d: (
                        d["checks"]["uses_twenty_reverse_steps"],
                        "lafan1_symreverse_20_steps",
                    ),
                    lambda d: (d["checks"]["five_tasks_evaluated"], "lafan1_symreverse_five_tasks"),
                    lambda d: (
                        d["checks"]["scale_grid_includes_unguided"],
                        "lafan1_symreverse_scale_grid",
                    ),
                    lambda d: (d["checks"]["all_rows_finite"], "lafan1_symreverse_finite"),
                    lambda d: (
                        d["checks"]["at_least_three_tasks_best_costs_improve"],
                        "lafan1_symreverse_cost_improves",
                    ),
                    lambda d: (
                        d["checks"]["all_tasks_have_some_primary_improvement"],
                        "lafan1_symreverse_primary_improves",
                    ),
                    lambda d: (
                        d["checks"]["guidance_gradients_nonzero_all_reverse_steps"],
                        "lafan1_symreverse_gradients",
                    ),
                    lambda d: (d["checks"]["npz_written"], "lafan1_symreverse_npz"),
                    lambda d: (d["row_count"] == 75, "lafan1_symreverse_rows_75"),
                    lambda d: (
                        d["improvement_summary"]["tasks_with_all_best_costs_improved"] >= 3,
                        "lafan1_symreverse_cost_task_count",
                    ),
                    lambda d: (
                        d["improvement_summary"]["tasks_with_some_primary_metric_improvement"] == 5,
                        "lafan1_symreverse_primary_task_count",
                    ),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_training" in d["settings"]["training_json"],
                        "lafan1_symreverse_training_source",
                    ),
                    lambda d: (
                        "lafan1_paper_arch_symmetry_augmented_guidance_eval"
                        in d["settings"]["offline_guidance_json"],
                        "lafan1_symreverse_guidance_source",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "lafan1_symreverse_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt_or_robot"],
                        "lafan1_symreverse_no_tensorrt_robot_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_coefficient_schedule"],
                        "lafan1_symreverse_no_schedule_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_symreverse_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_multiseed_audit",
                (
                    "res/level_c/lafan1_paper_arch_multiseed_audit/"
                    "level_c_lafan1_paper_arch_multiseed_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "lafan1_multiseed_at_least_3"),
                    lambda d: (d["checks"]["seeds_unique"], "lafan1_multiseed_unique"),
                    lambda d: (d["settings"]["seed_count"] == 3, "lafan1_multiseed_count_3"),
                    lambda d: (
                        d["checks"]["all_source_training_status_ok"],
                        "lafan1_multiseed_sources_ok",
                    ),
                    lambda d: (d["checks"]["all_checkpoints_exist"], "lafan1_multiseed_checkpoints"),
                    lambda d: (d["checks"]["all_metrics_finite"], "lafan1_multiseed_finite"),
                    lambda d: (
                        d["checks"]["all_runs_use_same_public_lafan1_motion_count"],
                        "lafan1_multiseed_motion_count",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_same_window_count"],
                        "lafan1_multiseed_window_count",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_paper_vae_and_diffusion_architecture"],
                        "lafan1_multiseed_paper_arch",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_paper_training_epochs"],
                        "lafan1_multiseed_epochs",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_multigpu_dataparallel"],
                        "lafan1_multiseed_8gpu",
                    ),
                    lambda d: (
                        d["checks"]["statistics_include_mean_std_min_max"],
                        "lafan1_multiseed_stats",
                    ),
                    lambda d: (
                        d["checks"]["no_best_seed_only_reporting"],
                        "lafan1_multiseed_no_best_seed",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_teacher_rollout_dataset"],
                        "lafan1_multiseed_no_teacher_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_or_robot_or_fig_videos"],
                        "lafan1_multiseed_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["statistics"]["final_validation_pred_tau_mse"]["mean"] < 0.02,
                        "lafan1_multiseed_validation_tau_mean",
                    ),
                    lambda d: (
                        d["statistics"]["final_test_pred_tau_mse"]["mean"] < 0.02,
                        "lafan1_multiseed_test_tau_mean",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/"
                    "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "lafan1_sym_multiseed_at_least_3"),
                    lambda d: (d["checks"]["seeds_unique"], "lafan1_sym_multiseed_unique"),
                    lambda d: (d["settings"]["seed_count"] == 3, "lafan1_sym_multiseed_count_3"),
                    lambda d: (
                        d["checks"]["all_source_training_status_ok"],
                        "lafan1_sym_multiseed_sources_ok",
                    ),
                    lambda d: (d["checks"]["all_checkpoints_exist"], "lafan1_sym_multiseed_checkpoints"),
                    lambda d: (d["checks"]["all_metrics_finite"], "lafan1_sym_multiseed_finite"),
                    lambda d: (
                        d["checks"]["all_runs_use_symmetry_augmented_dataset"],
                        "lafan1_sym_multiseed_dataset",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_same_public_lafan1_unique_motion_count"],
                        "lafan1_sym_multiseed_public_motions",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_same_augmented_motion_count"],
                        "lafan1_sym_multiseed_aug_motions",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_same_window_count"],
                        "lafan1_sym_multiseed_window_count",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_same_token_count"],
                        "lafan1_sym_multiseed_token_count",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_paper_vae_and_diffusion_architecture"],
                        "lafan1_sym_multiseed_paper_arch",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_paper_training_epochs"],
                        "lafan1_sym_multiseed_epochs",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_multigpu_dataparallel"],
                        "lafan1_sym_multiseed_8gpu",
                    ),
                    lambda d: (
                        d["checks"]["statistics_include_mean_std_min_max"],
                        "lafan1_sym_multiseed_stats",
                    ),
                    lambda d: (
                        d["checks"]["no_best_seed_only_reporting"],
                        "lafan1_sym_multiseed_no_best_seed",
                    ),
                    lambda d: (
                        d["statistics"]["final_validation_pred_tau_mse"]["mean"] < 0.02,
                        "lafan1_sym_multiseed_validation_tau_mean",
                    ),
                    lambda d: (
                        d["statistics"]["final_test_pred_tau_mse"]["mean"] < 0.02,
                        "lafan1_sym_multiseed_test_tau_mean",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_teacher_rollout_dataset"],
                        "lafan1_sym_multiseed_no_teacher_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_or_robot_or_fig_videos"],
                        "lafan1_sym_multiseed_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_sym_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_high_memory_batch_audit",
                (
                    "res/level_c/lafan1_paper_arch_high_memory_batch_audit/"
                    "level_c_lafan1_paper_arch_high_memory_batch_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_highmem_source_ok"),
                    lambda d: (d["checks"]["source_checkpoint_exists"], "lafan1_highmem_checkpoint_exists"),
                    lambda d: (d["checks"]["dataset_npz_exists"], "lafan1_highmem_dataset_exists"),
                    lambda d: (d["checks"]["eight_cuda_gpus_visible"], "lafan1_highmem_8gpus_visible"),
                    lambda d: (d["checks"]["uses_dataparallel_8_gpus"], "lafan1_highmem_dataparallel_8"),
                    lambda d: (
                        d["checks"]["paper_diffusion_architecture_used"],
                        "lafan1_highmem_paper_diffusion_arch",
                    ),
                    lambda d: (
                        d["checks"]["paper_clean_trajectory_loss_executed"],
                        "lafan1_highmem_clean_loss",
                    ),
                    lambda d: (d["checks"]["forward_backward_executed"], "lafan1_highmem_backward"),
                    lambda d: (
                        d["checks"]["all_gpus_reach_target_memory_after_reserve"],
                        "lafan1_highmem_20gb_all_gpus",
                    ),
                    lambda d: (
                        d["metrics"]["min_after_reserve_used_mb"] >= d["settings"]["target_memory_mb"],
                        "lafan1_highmem_min_memory",
                    ),
                    lambda d: (
                        d["metrics"]["min_batch_peak_allocated_mb"] > 4000,
                        "lafan1_highmem_real_batch_peak_recorded",
                    ),
                    lambda d: (d["checks"]["reserve_tensors_recorded"], "lafan1_highmem_reserve_recorded"),
                    lambda d: (
                        d["checks"]["does_not_claim_memory_reserve_as_training_batch_memory"],
                        "lafan1_highmem_no_reserve_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_write_new_checkpoint"],
                        "lafan1_highmem_no_checkpoint_write",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_highmem_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_dataset_audit",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_dataset/"
                    "lafan1_paper_arch_symmetry_dataset_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "lafan1_symmetry_source_training_ok"),
                    lambda d: (d["checks"]["symmetry_mapping_status_ok"], "lafan1_symmetry_mapping_ok"),
                    lambda d: (
                        d["checks"]["source_window_count_matches_training"],
                        "lafan1_symmetry_source_count_matches",
                    ),
                    lambda d: (d["checks"]["all_joint_mirror_rows_covered"], "lafan1_symmetry_joint_coverage"),
                    lambda d: (d["checks"]["all_body_mirror_rows_covered"], "lafan1_symmetry_body_coverage"),
                    lambda d: (
                        d["checks"]["augmented_window_count_doubles_source"],
                        "lafan1_symmetry_window_count_doubled",
                    ),
                    lambda d: (d["checks"]["split_counts_doubled"], "lafan1_symmetry_split_counts"),
                    lambda d: (
                        d["checks"]["double_mirror_state_within_tolerance"],
                        "lafan1_symmetry_state_double_mirror",
                    ),
                    lambda d: (
                        d["checks"]["double_mirror_action_exact_float32"],
                        "lafan1_symmetry_action_double_mirror",
                    ),
                    lambda d: (
                        d["checks"]["mirrored_projection_matches_recomputed"],
                        "lafan1_symmetry_projection_recompute",
                    ),
                    lambda d: (
                        d["checks"]["all_augmented_arrays_finite"],
                        "lafan1_symmetry_augmented_finite",
                    ),
                    lambda d: (d["checks"]["npz_written"], "lafan1_symmetry_npz_written"),
                    lambda d: (d["checks"]["tsv_written"], "lafan1_symmetry_tsv_written"),
                    lambda d: (d["metrics"]["source_window_count"] == 2200, "lafan1_symmetry_source_2200"),
                    lambda d: (d["metrics"]["augmented_window_count"] == 4400, "lafan1_symmetry_aug_4400"),
                    lambda d: (d["metrics"]["augmented_token_count"] == 92400, "lafan1_symmetry_tokens_92400"),
                    lambda d: (
                        d["metrics"]["double_mirror_state_max_abs_error"] < 2e-5,
                        "lafan1_symmetry_state_error",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_sign_table"],
                        "lafan1_symmetry_no_official_sign_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_retrained_checkpoint"],
                        "lafan1_symmetry_no_retrain_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_symmetry_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_augmented_training",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
                    "lafan1_paper_arch_vae_diffusion_training.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["external_dataset_npz_exists_when_requested"],
                        "lafan1_symtrain_source_npz_exists",
                    ),
                    lambda d: (d["checks"]["paper_state_action_shapes"], "lafan1_symtrain_shapes"),
                    lambda d: (d["checks"]["paper_vae_architecture_used"], "lafan1_symtrain_vae_arch"),
                    lambda d: (d["checks"]["paper_diffusion_architecture_used"], "lafan1_symtrain_diff_arch"),
                    lambda d: (
                        d["checks"]["run_schema_checkpoint_metrics_figures_written"],
                        "lafan1_symtrain_outputs",
                    ),
                    lambda d: (d["checks"]["uses_multi_gpu_when_available"], "lafan1_symtrain_8gpu"),
                    lambda d: (
                        d["metrics"]["public_lafan1_unique_motion_label_count"] == 40,
                        "lafan1_symtrain_unique_motions_40",
                    ),
                    lambda d: (d["metrics"]["augmented_motion_label_count"] == 80, "lafan1_symtrain_aug_labels_80"),
                    lambda d: (d["metrics"]["window_count"] == 4400, "lafan1_symtrain_windows_4400"),
                    lambda d: (d["metrics"]["token_count"] == 92400, "lafan1_symtrain_tokens_92400"),
                    lambda d: (d["settings"]["vae_epochs"] >= 24, "lafan1_symtrain_vae_epochs"),
                    lambda d: (d["settings"]["diffusion_epochs"] >= 1000, "lafan1_symtrain_diffusion_epochs"),
                    lambda d: (d["metrics"]["diffusion_parameter_count"] >= 19000000, "lafan1_symtrain_params"),
                    lambda d: (d["metrics"]["checkpoint_size_bytes"] > 250_000_000, "lafan1_symtrain_checkpoint"),
                    lambda d: (
                        d["metrics"]["data_parallel"] is True and len(d["metrics"]["gpu_device_ids"]) == 8,
                        "lafan1_symtrain_dataparallel_8",
                    ),
                    lambda d: (
                        d["metrics"]["final_validation_decoded_action_mse"] < 0.05,
                        "lafan1_symtrain_val_action",
                    ),
                    lambda d: (
                        d["metrics"]["final_test_decoded_action_mse"] < 0.05,
                        "lafan1_symtrain_test_action",
                    ),
                    lambda d: (
                        d["metrics"]["final_validation_pred_tau_mse"] < 0.02,
                        "lafan1_symtrain_val_tau",
                    ),
                    lambda d: (d["metrics"]["final_test_pred_tau_mse"] < 0.02, "lafan1_symtrain_test_tau"),
                    lambda d: (
                        d["checks"]["does_not_claim_unavailable_teacher_rollout_dataset"],
                        "lafan1_symtrain_no_teacher_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_real_robot_or_fig5_fig6"],
                        "lafan1_symtrain_no_robot_fig_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "lafan1_symtrain_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_teacher_rollout_state_latent_dataset",
                (
                    "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
                    "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["uses_full_teacher_rollout_samples"],
                        "state_latent_dataset_full_teacher_samples",
                    ),
                    lambda d: (d["checks"]["has_full_window_index"], "state_latent_dataset_full_window_index"),
                    lambda d: (
                        d["checks"]["has_train_validation_test_splits"],
                        "state_latent_dataset_has_splits",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696,
                        "state_latent_dataset_window_count_285696",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["token_dim"] == 192,
                        "state_latent_dataset_token_dim_192",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"],
                        "state_latent_dataset_no_official_dagger_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_state_latent_dataset"],
                        "state_latent_dataset_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "state_latent_dataset_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_teacher_rollout_vae_training",
                (
                    "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
                    "level_c_official_csv_loop_teacher_rollout_vae_training.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_teacher_rollout_vae_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_teacher_rollout_source"],
                        "official_loop_vae_source_rollout_ok",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["sample_count"] == 306176,
                        "official_loop_vae_full_teacher_samples",
                    ),
                    lambda d: (
                        d["worker_summary"]["splits"] == {"train": 244940, "validation": 30618, "test": 30618},
                        "official_loop_vae_splits",
                    ),
                    lambda d: (
                        d["worker_summary"]["torch_cuda_device_count"] >= 2
                        and d["worker_summary"]["data_parallel_used"] is True,
                        "official_loop_vae_dataparallel",
                    ),
                    lambda d: (
                        d["worker_summary"]["evaluation"]["test"]["action_mse"] < 0.01,
                        "official_loop_vae_test_action_mse",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_closed_loop_eval"],
                        "official_loop_vae_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_loop_vae_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_teacher_rollout_state_latent_dataset",
                (
                    "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
                    "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_teacher_rollout_state_latent_dataset",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_teacher_rollout_source"],
                        "official_loop_state_latent_rollout_source",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_vae_source"],
                        "official_loop_state_latent_vae_source",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_teacher_rollout_samples"],
                        "official_loop_state_latent_full_samples",
                    ),
                    lambda d: (
                        d["checks"]["has_full_window_index"],
                        "official_loop_state_latent_full_window_index",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696,
                        "official_loop_state_latent_window_count_285696",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["token_dim"] == 192,
                        "official_loop_state_latent_token_dim_192",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"],
                        "official_loop_state_latent_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_loop_state_latent_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_state_latent_diffusion_training",
                (
                    "res/level_c/official_csv_loop_state_latent_diffusion_training/"
                    "level_c_official_csv_loop_state_latent_diffusion_training.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_state_latent_diffusion_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_state_latent_dataset_source"],
                        "official_loop_diffusion_state_latent_source",
                    ),
                    lambda d: (d["checks"]["uses_full_window_dataset"], "official_loop_diffusion_full_windows"),
                    lambda d: (d["checks"]["uses_two_visible_gpus"], "official_loop_diffusion_two_visible_gpus"),
                    lambda d: (d["checks"]["data_parallel_used"], "official_loop_diffusion_data_parallel"),
                    lambda d: (
                        d["checks"]["test_denoising_improves_over_noisy"],
                        "official_loop_diffusion_test_improves",
                    ),
                    lambda d: (
                        d["checks"]["checkpoint_written_to_ignored_runs_dir"],
                        "official_loop_diffusion_checkpoint_ignored_run_dir",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696,
                        "official_loop_diffusion_window_count_285696",
                    ),
                    lambda d: (
                        d["worker_summary"]["training"]["epochs"] >= 30,
                        "official_loop_diffusion_epochs_ge_30",
                    ),
                    lambda d: (
                        d["worker_summary"]["evaluation"]["test"]["pred_token_mse"]
                        < d["worker_summary"]["evaluation"]["test"]["noisy_token_mse"],
                        "official_loop_diffusion_test_pred_better_than_noisy",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_diffusion_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"],
                        "official_loop_diffusion_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_loop_diffusion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training",
                (
                    "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
                    "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_teacher_rollout_vae_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_full_bundle_teacher_rollout_source"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["full_bundle_total_motion_frames_11960"],
                        "full_bundle_vae_source_rollout_ok",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["sample_count"] == 306176
                        and d["worker_summary"]["dataset"]["motion_time_step_max"] == 11959,
                        "full_bundle_vae_samples_and_motion_coverage",
                    ),
                    lambda d: (
                        d["worker_summary"]["torch_cuda_device_count"] >= 2
                        and d["worker_summary"]["data_parallel_used"] is True,
                        "full_bundle_vae_dataparallel",
                    ),
                    lambda d: (
                        d["worker_summary"]["evaluation"]["test"]["action_mse"] < 0.01,
                        "full_bundle_vae_test_action_mse",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_closed_loop_eval"],
                        "full_bundle_vae_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_vae_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training",
                (
                    "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
                    "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_teacher_rollout_vae_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_full_bundle_teacher_rollout_source"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["full_bundle_total_motion_frames_11960"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "official_importer_vae_source_rollout_ok",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["sample_count"] == 306176
                        and d["worker_summary"]["dataset"]["motion_time_step_max"] == 11959,
                        "official_importer_vae_samples_and_motion_coverage",
                    ),
                    lambda d: (
                        d["worker_summary"]["splits"] == {"train": 244940, "validation": 30618, "test": 30618},
                        "official_importer_vae_split_counts",
                    ),
                    lambda d: (
                        d["worker_summary"]["torch_cuda_device_count"] >= 2
                        and d["worker_summary"]["data_parallel_used"] is True
                        and d["settings"]["visible_gpus"] == [4, 7],
                        "official_importer_vae_dataparallel_gpu47",
                    ),
                    lambda d: (
                        d["worker_summary"]["evaluation"]["test"]["action_mse"] < 0.01,
                        "official_importer_vae_test_action_mse",
                    ),
                    lambda d: (
                        d["checks"]["checkpoint_written_to_ignored_runs_dir"],
                        "official_importer_vae_checkpoint_ignored_run_dir",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_closed_loop_eval"]
                        and d["checks"]["does_not_claim_goal_complete"],
                        "official_importer_vae_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_importer_vae_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_vae_report_assets",
                (
                    "res/report_assets/official_importer_export_full_bundle_vae_training/"
                    "official_importer_export_full_bundle_vae_training_assets.json"
                ),
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["vae_status_ok"]
                        and d["checks"]["sample_count_306176"]
                        and d["checks"]["action_dim_29"]
                        and d["checks"]["obs_dim_160"],
                        "official_importer_vae_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["test_action_mse_below_0_01"],
                        "official_importer_vae_assets_mse_threshold",
                    ),
                    lambda d: (
                        d["checks"]["curve_exists"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "official_importer_vae_assets_files_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_vae_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_vae_closed_loop_rollout_eval",
                (
                    "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
                    "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["rollout_success"]
                        and d["checks"]["two_shards_completed"]
                        and d["checks"]["rollout_steps_299"],
                        "official_importer_vae_closed_loop_success_two_shards_299_steps",
                    ),
                    lambda d: (
                        d["run"]["aggregate_metrics"]["total_num_envs"] == 3072
                        and d["run"]["aggregate_metrics"]["total_env_steps"] == 918528,
                        "official_importer_vae_closed_loop_full_env_steps",
                    ),
                    lambda d: (
                        d["checks"]["uses_gpus_4_7"] and d["checks"]["uses_official_importer_export_usd"],
                        "official_importer_vae_closed_loop_gpu47_and_usd",
                    ),
                    lambda d: (
                        d["run"]["aggregate_metrics"]["teacher_vae_action_mse"]["mean"] < 0.001
                        and d["run"]["aggregate_metrics"]["teacher_vae_action_abs_error"]["mean"] < 0.01,
                        "official_importer_vae_closed_loop_low_action_error",
                    ),
                    lambda d: (
                        d["run"]["aggregate_metrics"]["done_count_total"]
                        == d["run"]["aggregate_metrics"]["total_env_steps"],
                        "official_importer_vae_closed_loop_done_boundary_recorded",
                    ),
                    lambda d: (
                        d["checks"]["peak_memory_each_gpu_at_least_10gb"] is False
                        and all(
                            item["peak_memory_used_mb"] < 10240
                            for item in d["run"]["gpu_metrics_summary"]["per_gpu"].values()
                        ),
                        "official_importer_vae_closed_loop_low_memory_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_autonomous_vae_policy"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_vae_closed_loop_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_importer_vae_closed_loop_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_vae_closed_loop_rollout_assets",
                (
                    "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
                    "official_importer_export_full_bundle_vae_closed_loop_rollout_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["two_shards_completed"]
                        and d["checks"]["rollout_steps_299"],
                        "official_importer_vae_closed_loop_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"] and d["checks"]["csv_assets_exist"],
                        "official_importer_vae_closed_loop_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["total_env_steps"] == 918528
                        and d["metrics"]["teacher_vae_action_mse_mean"] < 0.001,
                        "official_importer_vae_closed_loop_assets_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_vae_closed_loop_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture",
                (
                    "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
                    "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_capture.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"] and d["checks"]["render_ok"],
                        "official_importer_vae_video_capture_and_render_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] in {4, 7}
                        and d["config"]["rollout_steps"] == 299,
                        "official_importer_vae_video_gpu_and_steps_recorded",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and not d["run"]["capture_metrics"]["uses_resource_adjusted_usd"],
                        "official_importer_vae_video_uses_importer_usd",
                    ),
                    lambda d: (
                        d["run"]["capture_metrics"]["teacher_vae_action_mse"]["mean"] < 0.001
                        and d["run"]["capture_metrics"]["robot_body_pos_shape"] == [299, 14, 3],
                        "official_importer_vae_video_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_autonomous_vae_policy"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_vae_video_capture_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_importer_vae_video_capture_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset",
                (
                    "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
                    "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_status_ok"]
                        and d["checks"]["frame_count_299"]
                        and d["frame_count"] == 299
                        and d["target_body_count"] == 14,
                        "official_importer_vae_video_shape_contract",
                    ),
                    lambda d: (
                        d["checks"]["video_exists_nonempty"] and d["checks"]["keyframes_exist_nonempty"],
                        "official_importer_vae_video_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["metrics"]["teacher_vae_action_mse_mean"] < 0.001
                        and d["metrics"]["target_body_error_mean"] >= 0.0,
                        "official_importer_vae_video_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_autonomous_vae_policy"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_vae_video_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_importer_vae_video_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset",
                (
                    "res/level_c/official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset/"
                    "level_c_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_full_bundle_teacher_rollout_source"]
                        and d["checks"]["official_csv_loop_full_bundle_vae_source"],
                        "full_bundle_state_latent_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_teacher_rollout_samples"]
                        and d["checks"]["has_full_window_index"],
                        "full_bundle_state_latent_full_samples_windows",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696
                        and d["worker_summary"]["dataset"]["token_dim"] == 192,
                        "full_bundle_state_latent_window_count_token_dim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"],
                        "full_bundle_state_latent_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_state_latent_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training",
                (
                    "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
                    "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_state_latent_diffusion_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_full_bundle_state_latent_dataset_source"],
                        "full_bundle_diffusion_state_latent_source",
                    ),
                    lambda d: (d["checks"]["uses_full_window_dataset"], "full_bundle_diffusion_full_windows"),
                    lambda d: (d["checks"]["uses_two_visible_gpus"], "full_bundle_diffusion_two_visible_gpus"),
                    lambda d: (d["checks"]["data_parallel_used"], "full_bundle_diffusion_data_parallel"),
                    lambda d: (
                        d["checks"]["test_denoising_improves_over_noisy"]
                        and d["worker_summary"]["evaluation"]["test"]["pred_token_mse"]
                        < d["worker_summary"]["evaluation"]["test"]["noisy_token_mse"],
                        "full_bundle_diffusion_test_improves",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696
                        and d["worker_summary"]["training"]["epochs"] >= 30,
                        "full_bundle_diffusion_window_count_epochs",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_diffusion_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"],
                        "full_bundle_diffusion_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_diffusion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_downstream_report_assets",
                (
                    "res/report_assets/official_csv_loop_full_bundle_downstream/"
                    "official_csv_loop_full_bundle_downstream_report_assets.json"
                ),
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["vae_status_ok"]
                        and d["checks"]["state_latent_status_ok"]
                        and d["checks"]["diffusion_status_ok"],
                        "full_bundle_downstream_assets_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["vae_curve_exists"]
                        and d["checks"]["diffusion_curve_exists"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "full_bundle_downstream_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["vae_sample_count"] == 306176
                        and d["metrics"]["state_latent_window_count"] == 285696
                        and d["metrics"]["diffusion_test_denoising_improvement_ratio"] > 0.0,
                        "full_bundle_downstream_assets_metrics",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_downstream_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset",
                (
                    "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
                    "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_full_bundle_teacher_rollout_source"]
                        and d["checks"]["official_importer_export_full_bundle_vae_source"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "official_importer_state_latent_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_teacher_rollout_samples"]
                        and d["checks"]["has_full_window_index"],
                        "official_importer_state_latent_full_samples_windows",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696
                        and d["worker_summary"]["dataset"]["token_dim"] == 192,
                        "official_importer_state_latent_window_count_token_dim",
                    ),
                    lambda d: (
                        d["settings"]["selected_physical_gpus"] == [5, 6]
                        and d["settings"]["cuda_visible_devices"] == "5,6",
                        "official_importer_state_latent_uses_gpu56",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"],
                        "official_importer_state_latent_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_importer_state_latent_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_full_bundle_state_latent_diffusion_training",
                (
                    "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
                    "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_state_latent_diffusion_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_full_bundle_state_latent_dataset_source"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "official_importer_diffusion_state_latent_source",
                    ),
                    lambda d: (d["checks"]["uses_full_window_dataset"], "official_importer_diffusion_full_windows"),
                    lambda d: (d["checks"]["uses_two_visible_gpus"], "official_importer_diffusion_two_visible_gpus"),
                    lambda d: (d["checks"]["data_parallel_used"], "official_importer_diffusion_data_parallel"),
                    lambda d: (
                        d["settings"]["selected_physical_gpus"] == [5, 6]
                        and d["settings"]["cuda_visible_devices"] == "5,6",
                        "official_importer_diffusion_uses_gpu56",
                    ),
                    lambda d: (
                        d["checks"]["test_denoising_improves_over_noisy"]
                        and d["worker_summary"]["evaluation"]["test"]["pred_token_mse"]
                        < d["worker_summary"]["evaluation"]["test"]["noisy_token_mse"],
                        "official_importer_diffusion_test_improves",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696
                        and d["worker_summary"]["training"]["epochs"] >= 30,
                        "official_importer_diffusion_window_count_epochs",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_diffusion_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_diffusion_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_importer_diffusion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_downstream_report_assets",
                (
                    "res/report_assets/official_importer_export_full_bundle_downstream/"
                    "official_importer_export_full_bundle_downstream_report_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_full_bundle_downstream_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["vae_status_ok"]
                        and d["checks"]["state_latent_status_ok"]
                        and d["checks"]["diffusion_status_ok"],
                        "official_importer_downstream_assets_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["vae_curve_exists"]
                        and d["checks"]["diffusion_curve_exists"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "official_importer_downstream_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["vae_sample_count"] == 306176
                        and d["metrics"]["state_latent_window_count"] == 285696
                        and d["metrics"]["diffusion_test_denoising_improvement_ratio"] > 0.0,
                        "official_importer_downstream_assets_metrics",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "official_importer_downstream_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training",
                (
                    "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
                    "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_teacher_rollout_vae_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_teacher_rollout_source"]
                        and d["checks"]["scaled_teacher_total_env_steps_1224704"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["full_bundle_total_motion_frames_11960"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "scaled_importer_vae_source_rollout_ok",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["sample_count"] == 1224704
                        and d["worker_summary"]["dataset"]["motion_time_step_max"] == 11959,
                        "scaled_importer_vae_samples_and_motion_coverage",
                    ),
                    lambda d: (
                        d["worker_summary"]["splits"]
                        == {"train": 979763, "validation": 122470, "test": 122471},
                        "scaled_importer_vae_split_counts",
                    ),
                    lambda d: (
                        d["worker_summary"]["torch_cuda_device_count"] >= 2
                        and d["worker_summary"]["data_parallel_used"] is True
                        and d["settings"]["visible_gpus"] == [4, 7],
                        "scaled_importer_vae_dataparallel_gpu47",
                    ),
                    lambda d: (
                        d["worker_summary"]["evaluation"]["test"]["action_mse"] < 0.001,
                        "scaled_importer_vae_test_action_mse",
                    ),
                    lambda d: (
                        d["checks"]["checkpoint_written_to_ignored_runs_dir"],
                        "scaled_importer_vae_checkpoint_ignored_run_dir",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_closed_loop_eval"]
                        and d["checks"]["does_not_claim_goal_complete"],
                        "scaled_importer_vae_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "scaled_importer_vae_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset",
                (
                    "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
                    "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_teacher_rollout_source"]
                        and d["checks"]["official_importer_export_scaled_ppo_vae_source"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "scaled_importer_state_latent_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_teacher_rollout_samples"]
                        and d["checks"]["has_full_window_index"],
                        "scaled_importer_state_latent_full_samples_windows",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["sample_count"] == 1224704
                        and d["worker_summary"]["dataset"]["window_count"] == 1142784
                        and d["worker_summary"]["dataset"]["token_dim"] == 192,
                        "scaled_importer_state_latent_window_count_token_dim",
                    ),
                    lambda d: (
                        d["settings"]["selected_physical_gpus"] == [4, 7]
                        and d["settings"]["cuda_visible_devices"] == "4,7",
                        "scaled_importer_state_latent_uses_gpu47",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"],
                        "scaled_importer_state_latent_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "scaled_importer_state_latent_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training",
                (
                    "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
                    "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_state_latent_diffusion_training",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_state_latent_dataset_source"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "scaled_importer_diffusion_state_latent_source",
                    ),
                    lambda d: (d["checks"]["uses_full_window_dataset"], "scaled_importer_diffusion_full_windows"),
                    lambda d: (d["checks"]["uses_two_visible_gpus"], "scaled_importer_diffusion_two_visible_gpus"),
                    lambda d: (d["checks"]["data_parallel_used"], "scaled_importer_diffusion_data_parallel"),
                    lambda d: (
                        d["settings"]["selected_physical_gpus"] == [4, 7]
                        and d["settings"]["cuda_visible_devices"] == "4,7",
                        "scaled_importer_diffusion_uses_gpu47",
                    ),
                    lambda d: (
                        d["checks"]["test_denoising_improves_over_noisy"]
                        and d["worker_summary"]["evaluation"]["test"]["pred_token_mse"]
                        < d["worker_summary"]["evaluation"]["test"]["noisy_token_mse"],
                        "scaled_importer_diffusion_test_improves",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 1142784
                        and d["worker_summary"]["training"]["epochs"] >= 30,
                        "scaled_importer_diffusion_window_count_epochs",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_diffusion_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "scaled_importer_diffusion_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "scaled_importer_diffusion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_downstream_report_assets",
                (
                    "res/report_assets/official_importer_export_scaled_ppo_downstream/"
                    "official_importer_export_full_bundle_downstream_report_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_downstream_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["vae_status_ok"]
                        and d["checks"]["state_latent_status_ok"]
                        and d["checks"]["diffusion_status_ok"],
                        "scaled_importer_downstream_assets_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["vae_curve_exists"]
                        and d["checks"]["diffusion_curve_exists"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "scaled_importer_downstream_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["vae_sample_count"] == 1224704
                        and d["metrics"]["state_latent_window_count"] == 1142784
                        and d["metrics"]["diffusion_test_denoising_improvement_ratio"] > 0.0,
                        "scaled_importer_downstream_assets_metrics",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "scaled_importer_downstream_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_state_latent_diffusion_training",
                (
                    "res/level_c/resource_adjusted_state_latent_diffusion_training/"
                    "level_c_resource_adjusted_state_latent_diffusion_training.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["uses_full_window_dataset"], "state_latent_diffusion_full_windows"),
                    lambda d: (d["checks"]["uses_two_visible_gpus"], "state_latent_diffusion_two_visible_gpus"),
                    lambda d: (d["checks"]["data_parallel_used"], "state_latent_diffusion_data_parallel"),
                    lambda d: (
                        d["checks"]["test_denoising_improves_over_noisy"],
                        "state_latent_diffusion_test_improves",
                    ),
                    lambda d: (
                        d["checks"]["checkpoint_written_to_ignored_runs_dir"],
                        "state_latent_diffusion_checkpoint_ignored_run_dir",
                    ),
                    lambda d: (
                        d["worker_summary"]["dataset"]["window_count"] == 285696,
                        "state_latent_diffusion_window_count_285696",
                    ),
                    lambda d: (
                        d["worker_summary"]["training"]["epochs"] >= 30,
                        "state_latent_diffusion_epochs_ge_30",
                    ),
                    lambda d: (
                        d["worker_summary"]["evaluation"]["test"]["pred_token_mse"]
                        < d["worker_summary"]["evaluation"]["test"]["noisy_token_mse"],
                        "state_latent_diffusion_test_pred_better_than_noisy",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_dagger"],
                        "state_latent_diffusion_no_official_dagger_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_diffusion"],
                        "state_latent_diffusion_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "state_latent_diffusion_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_state_latent_guidance_eval",
                (
                    "res/level_c/official_csv_loop_state_latent_guidance_eval/"
                    "level_c_official_csv_loop_state_latent_guidance_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_state_latent_guidance_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_diffusion_source"],
                        "official_loop_guidance_diffusion_source",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_state_latent_dataset_source"],
                        "official_loop_guidance_dataset_source",
                    ),
                    lambda d: (
                        d["checks"]["evaluates_full_validation_test_splits"],
                        "official_loop_guidance_full_validation_test",
                    ),
                    lambda d: (
                        d["checks"]["all_tasks_evaluated"],
                        "official_loop_guidance_all_tasks",
                    ),
                    lambda d: (
                        d["checks"]["all_best_costs_improve"],
                        "official_loop_guidance_best_costs_improve",
                    ),
                    lambda d: (
                        d["checks"]["all_best_guidance_gradients_nonzero"],
                        "official_loop_guidance_nonzero_gradients",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["total_selected_windows"] == 57140,
                        "official_loop_guidance_selected_windows_57140",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["row_count"] == 48,
                        "official_loop_guidance_row_count_48",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"]
                        and d["checks"]["does_not_claim_closed_loop_guidance"],
                        "official_loop_guidance_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6_reproduction"]
                        and d["checks"]["does_not_claim_fig5_fig6"],
                        "official_loop_guidance_no_fig56_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"],
                        "official_loop_guidance_no_paper_guidance_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_loop_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval",
                (
                    "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
                    "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_state_latent_guidance_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_full_bundle_diffusion_source"],
                        "full_bundle_guidance_diffusion_source",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_full_bundle_state_latent_dataset_source"],
                        "full_bundle_guidance_dataset_source",
                    ),
                    lambda d: (
                        d["checks"]["evaluates_full_validation_test_splits"],
                        "full_bundle_guidance_full_validation_test",
                    ),
                    lambda d: (d["checks"]["all_tasks_evaluated"], "full_bundle_guidance_all_tasks"),
                    lambda d: (
                        d["checks"]["all_best_costs_improve"],
                        "full_bundle_guidance_best_costs_improve",
                    ),
                    lambda d: (
                        d["checks"]["all_best_guidance_gradients_nonzero"],
                        "full_bundle_guidance_nonzero_gradients",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["total_selected_windows"] == 57139,
                        "full_bundle_guidance_selected_windows_57139",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["row_count"] == 48,
                        "full_bundle_guidance_row_count_48",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_paper_level_guidance"],
                        "full_bundle_guidance_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_guidance_report_assets",
                (
                    "res/report_assets/official_csv_loop_full_bundle_guidance/"
                    "official_csv_loop_full_bundle_guidance_report_assets.json"
                ),
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["guidance_status_ok"]
                        and d["checks"]["full_split_evaluated"]
                        and d["checks"]["all_tasks_improve"],
                        "full_bundle_guidance_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "full_bundle_guidance_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["total_selected_windows"] == 57139
                        and d["metrics"]["tasks_with_all_best_costs_improve"] == 4,
                        "full_bundle_guidance_assets_metrics",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_guidance_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval",
                (
                    "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/"
                    "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_state_latent_guidance_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_diffusion_source"],
                        "scaled_ppo_guidance_diffusion_source",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_state_latent_dataset_source"],
                        "scaled_ppo_guidance_dataset_source",
                    ),
                    lambda d: (
                        d["checks"]["evaluates_full_validation_test_splits"],
                        "scaled_ppo_guidance_full_validation_test",
                    ),
                    lambda d: (
                        d["worker_summary"]["settings"]["selected_split_counts"]
                        == {"validation": 114279, "test": 114278},
                        "scaled_ppo_guidance_split_counts",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["total_selected_windows"] == 228557
                        and d["worker_summary"]["metrics"]["row_count"] == 48,
                        "scaled_ppo_guidance_window_row_counts",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["tasks_with_all_best_costs_improve"] == 4
                        and d["worker_summary"]["metrics"]["tasks_with_nonzero_best_gradients"] == 4,
                        "scaled_ppo_guidance_all_tasks_improve",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "scaled_ppo_guidance_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_guidance_report_assets",
                (
                    "res/report_assets/official_importer_export_scaled_ppo_guidance/"
                    "official_importer_export_scaled_ppo_guidance_report_assets.json"
                ),
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["guidance_status_ok"]
                        and d["checks"]["full_split_evaluated"]
                        and d["checks"]["all_tasks_improve"],
                        "scaled_ppo_guidance_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"]
                        and d["checks"]["csv_assets_exist"]
                        and d["checks"]["summary_md_exists"],
                        "scaled_ppo_guidance_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["total_selected_windows"] == 228557
                        and d["metrics"]["tasks_with_all_best_costs_improve"] == 4,
                        "scaled_ppo_guidance_assets_metrics",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "scaled_ppo_guidance_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_official_csv_loop_guidance_vae_action_decode_eval",
                (
                    "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
                    "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_guidance_vae_action_decode_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_csv_loop_guidance"],
                        "official_loop_decode_guidance_source",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_csv_loop_diffusion"],
                        "official_loop_decode_diffusion_source",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_csv_loop_vae"],
                        "official_loop_decode_vae_source",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_validation_test_windows"],
                        "official_loop_decode_full_validation_test",
                    ),
                    lambda d: (
                        d["checks"]["all_decoded_actions_finite"],
                        "official_loop_decode_actions_finite",
                    ),
                    lambda d: (
                        d["checks"]["decoded_action_dim_29"],
                        "official_loop_decode_action_dim_29",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["total_windows"] == 57140,
                        "official_loop_decode_windows_57140",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["tasks_with_finite_actions"] == 4,
                        "official_loop_decode_tasks_finite_4",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"]
                        and d["checks"]["does_not_claim_fig5_fig6_reproduction"]
                        and d["checks"]["does_not_claim_paper_level_guidance"],
                        "official_loop_decode_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "official_loop_decode_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_guidance_vae_action_decode_assets",
                (
                    "res/report_assets/official_csv_loop_guidance_vae_action_decode/"
                    "official_csv_loop_guidance_vae_action_decode_assets.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["eval_status_ok"], "decode_assets_eval_status_ok"),
                    lambda d: (d["checks"]["png_assets_exist"], "decode_assets_png_exist"),
                    lambda d: (d["checks"]["metrics_csv_exists"], "decode_assets_metrics_csv_exists"),
                    lambda d: (d["checks"]["summary_md_exists"], "decode_assets_summary_md_exists"),
                    lambda d: (d["checks"]["does_not_claim_closed_loop"], "decode_assets_no_closed_loop_claim"),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_guided_action_rollout_probe",
                (
                    "res/level_c/official_csv_loop_guided_action_rollout_probe/"
                    "tracking_g1_official_csv_loop_guided_action_rollout_probe.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_guided_action_rollout_probe",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"] and d["checks"]["plot_ok"],
                        "guided_action_probe_capture_and_plot_ok",
                    ),
                    lambda d: (
                        d["config"]["selected_physical_gpu"] in {4, 7}
                        and d["metrics"]["rollout_steps"] == 21,
                        "guided_action_probe_gpu_and_steps_recorded",
                    ),
                    lambda d: (
                        d["checks"]["base_guided_actions_almost_identical"],
                        "guided_action_probe_negative_action_delta_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_closed_loop_receding_horizon_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "guided_action_probe_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guided_action_probe_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_guided_action_rollout_probe_assets",
                (
                    "res/level_c/official_csv_loop_guided_action_rollout_probe/"
                    "official_csv_loop_guided_action_rollout_probe_assets.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["timeseries_csv_exists"] and d["checks"]["plot_png_exists"],
                        "guided_action_probe_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["base_guided_actions_almost_identical"],
                        "guided_action_probe_asset_negative_delta_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "guided_action_probe_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_action_guidance_rollout_eval",
                (
                    "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
                    "level_c_official_csv_loop_action_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_action_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"]
                        and d["checks"]["render_ok"]
                        and d["checks"]["three_variants_evaluated"]
                        and d["checks"]["rollout_steps_299"],
                        "action_guidance_rollout_capture_render_three_variants_299",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_csv_loop_motion"]
                        and d["checks"]["uses_resource_adjusted_usd"],
                        "action_guidance_rollout_source_scope_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_receding_horizon_latent_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "action_guidance_rollout_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_action_guidance_rollout_asset",
                (
                    "res/visualization/official_csv_loop_action_guidance_rollout/"
                    "official_csv_loop_action_guidance_rollout_asset.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["video_exists_nonempty"]
                        and d["checks"]["keyframes_exist_nonempty"]
                        and d["checks"]["metrics_plot_exists_nonempty"],
                        "action_guidance_rollout_visual_assets_exist",
                    ),
                    lambda d: (
                        set(d["variant_metrics"]) == {"teacher", "vae_base", "action_guided"},
                        "action_guidance_rollout_asset_three_variants",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_receding_horizon_latent_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "action_guidance_rollout_asset_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval",
                (
                    "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
                    "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_task_conditioned_latent_guidance_multiseed_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["three_seed_groups"]
                        and d["checks"]["four_tasks_per_seed_group"]
                        and d["checks"]["all_rows_ok"],
                        "task_conditioned_guidance_multiseed_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rollouts_299_steps"]
                        and d["metrics"]["row_count"] == 12
                        and d["metrics"]["total_rollout_variant_steps"] == 14352,
                        "task_conditioned_guidance_multiseed_rollout_steps",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_have_mp4_paths"],
                        "task_conditioned_guidance_multiseed_mp4_paths_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "task_conditioned_guidance_multiseed_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "task_conditioned_guidance_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_task_conditioned_guidance_multiseed_assets",
                (
                    "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
                    "official_csv_loop_task_conditioned_guidance_multiseed_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_task_conditioned_guidance_multiseed_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["summary_status_ok"]
                        and d["checks"]["all_rows_ok"]
                        and d["checks"]["three_seed_groups"]
                        and d["checks"]["four_tasks_per_seed_group"],
                        "task_conditioned_guidance_multiseed_assets_source_ok",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "task_conditioned_guidance_multiseed_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval",
                (
                    "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
                    "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["capture_ok"]
                        and d["checks"]["render_ok"]
                        and d["checks"]["four_variants_evaluated"]
                        and d["checks"]["rollout_steps_299"],
                        "full_bundle_receding_guidance_rollout_success",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"],
                        "full_bundle_receding_guidance_uses_40_motion_bundle",
                    ),
                    lambda d: (
                        Path(d["outputs"]["assets"]["mp4"]).is_file()
                        and Path(d["outputs"]["assets"]["mp4"]).stat().st_size > 0
                        and Path(d["outputs"]["assets"]["metrics_png"]).is_file()
                        and Path(d["outputs"]["assets"]["keyframes_png"]).is_file(),
                        "full_bundle_receding_guidance_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_receding_guidance_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_receding_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval",
                (
                    "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
                    "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["four_tasks_attempted"]
                        and d["checks"]["all_tasks_ok"]
                        and len(d["rows"]) == 4,
                        "full_bundle_task_conditioned_guidance_four_tasks_ok",
                    ),
                    lambda d: (
                        all(row.get("rollout_steps") == 299 for row in d["rows"])
                        and all(row.get("mp4") for row in d["rows"])
                        and d["checks"]["all_tasks_have_mp4_paths"],
                        "full_bundle_task_conditioned_guidance_steps_and_videos_recorded",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["bundle"]["motion_count"] == 40,
                        "full_bundle_task_conditioned_guidance_uses_40_motion_bundle",
                    ),
                    lambda d: (
                        all(Path(row["asset_json"]).is_file() for row in d["rows"])
                        and all(Path(row["mp4"]).is_file() and Path(row["mp4"]).stat().st_size > 0 for row in d["rows"]),
                        "full_bundle_task_conditioned_guidance_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_task_conditioned_guidance_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_task_conditioned_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_task_conditioned_guidance_summary_assets",
                (
                    "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
                    "official_csv_loop_task_conditioned_guidance_summary_assets.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["variant"] == "full_bundle"
                        and d["checks"]["four_tasks_recorded"]
                        and d["checks"]["all_guided_cost_deltas_present"],
                        "full_bundle_task_conditioned_guidance_assets_source_ok",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "full_bundle_task_conditioned_guidance_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_task_conditioned_guidance_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
                (
                    "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
                    "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"].get("seed_group_count_at_least_3", d["checks"].get("three_seed_groups", False))
                        and d["checks"]["four_tasks_per_seed_group"]
                        and d["checks"]["all_rows_ok"]
                        and d["metrics"]["row_count"]
                        == d["metrics"]["seed_group_count"] * d["metrics"]["task_count"],
                        "full_bundle_task_conditioned_guidance_multiseed_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rollouts_299_steps"]
                        and d["metrics"]["total_rollout_variant_steps"] == d["metrics"]["row_count"] * 299 * 4
                        and d["checks"]["all_rows_have_mp4_paths"],
                        "full_bundle_task_conditioned_guidance_multiseed_steps_and_videos_recorded",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["bundle"]["motion_count"] == 40,
                        "full_bundle_task_conditioned_guidance_multiseed_uses_40_motion_bundle",
                    ),
                    lambda d: (
                        all(Path(row["asset_json"]).is_file() for row in d["rows"])
                        and all(Path(row["mp4"]).is_file() and Path(row["mp4"]).stat().st_size > 0 for row in d["rows"]),
                        "full_bundle_task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "full_bundle_task_conditioned_guidance_multiseed_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "full_bundle_task_conditioned_guidance_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets",
                (
                    "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
                    "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["summary_status_ok"]
                        and d["checks"]["all_rows_ok"]
                        and d["checks"].get("seed_group_count_at_least_3", d["checks"].get("three_seed_groups", False))
                        and d["checks"]["four_tasks_per_seed_group"],
                        "full_bundle_task_conditioned_guidance_multiseed_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"],
                        "full_bundle_task_conditioned_guidance_multiseed_assets_bundle_ok",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "full_bundle_task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_task_conditioned_guidance_multiseed_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
                (
                    "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
                    "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["seed_group_count_at_least_5"]
                        and d["checks"]["four_tasks_per_seed_group"]
                        and d["checks"]["all_rows_ok"]
                        and d["metrics"]["row_count"]
                        == d["metrics"]["seed_group_count"] * d["metrics"]["task_count"],
                        "importer_export_task_conditioned_guidance_multiseed_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rollouts_299_steps"]
                        and d["metrics"]["total_rollout_variant_steps"] == d["metrics"]["row_count"] * 299 * 4
                        and d["checks"]["all_rows_have_mp4_paths"],
                        "importer_export_task_conditioned_guidance_multiseed_steps_and_videos_recorded",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["uses_official_importer_export_usd"]
                        and d["bundle"]["motion_count"] == 40,
                        "importer_export_task_conditioned_guidance_multiseed_uses_importer_export_bundle",
                    ),
                    lambda d: (
                        all(Path(row["asset_json"]).is_file() for row in d["rows"])
                        and all(Path(row["mp4"]).is_file() and Path(row["mp4"]).stat().st_size > 0 for row in d["rows"]),
                        "importer_export_task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "importer_export_task_conditioned_guidance_multiseed_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "importer_export_task_conditioned_guidance_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval",
                (
                    "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
                    "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["four_tasks_attempted"]
                        and d["checks"]["all_tasks_ok"]
                        and all(row["rollout_steps"] == 299 for row in d["rows"]),
                        "scaled_ppo_task_conditioned_guidance_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_scaled_ppo_training_run"]
                        and d["checks"]["uses_scaled_ppo_checkpoint_eval"]
                        and d["checks"]["uses_scaled_ppo_vae"]
                        and d["checks"]["uses_scaled_ppo_denoiser"]
                        and d["checks"]["uses_scaled_ppo_offline_guidance"],
                        "scaled_ppo_task_conditioned_guidance_uses_scaled_chain",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["uses_full_public_motion_bundle"]
                        and d["bundle"]["motion_count"] == 40,
                        "scaled_ppo_task_conditioned_guidance_uses_importer_export_bundle",
                    ),
                    lambda d: (
                        all(Path(row["asset_json"]).is_file() for row in d["rows"])
                        and all(Path(row["mp4"]).is_file() and Path(row["mp4"]).stat().st_size > 0 for row in d["rows"]),
                        "scaled_ppo_task_conditioned_guidance_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_task_conditioned_guidance_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_task_conditioned_guidance_summary_assets",
                (
                    "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_summary/"
                    "official_csv_loop_task_conditioned_guidance_summary_assets.json"
                ),
                [
                    lambda d: (d.get("status") == "ok", f"status={d.get('status')!r}"),
                    lambda d: (
                        d["checks"]["four_tasks_recorded"]
                        and d["checks"]["all_guided_cost_deltas_present"]
                        and d["checks"]["all_assets_nonempty"],
                        "scaled_ppo_task_conditioned_guidance_summary_assets_ok",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_task_conditioned_guidance_summary_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets",
                (
                    "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
                    "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["summary_status_ok"]
                        and d["checks"]["all_rows_ok"]
                        and d["checks"]["seed_group_count_at_least_5"]
                        and d["checks"]["four_tasks_per_seed_group"],
                        "importer_export_task_conditioned_guidance_multiseed_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "importer_export_task_conditioned_guidance_multiseed_assets_bundle_ok",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "importer_export_task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "importer_export_task_conditioned_guidance_multiseed_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval",
                (
                    "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
                    "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["seed_group_count_at_least_5"]
                        and d["checks"]["four_tasks_per_seed_group"]
                        and d["checks"]["all_rows_ok"]
                        and d["metrics"]["row_count"]
                        == d["metrics"]["seed_group_count"] * d["metrics"]["task_count"],
                        "scaled_ppo_task_conditioned_guidance_multiseed_rows_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rollouts_299_steps"]
                        and d["metrics"]["total_rollout_variant_steps"] == d["metrics"]["row_count"] * 299 * 4
                        and d["checks"]["all_rows_have_mp4_paths"],
                        "scaled_ppo_task_conditioned_guidance_multiseed_steps_and_videos_recorded",
                    ),
                    lambda d: (
                        d["checks"]["uses_scaled_ppo_training_run"]
                        and d["checks"]["uses_scaled_ppo_checkpoint_eval"]
                        and d["checks"]["uses_scaled_ppo_vae"]
                        and d["checks"]["uses_scaled_ppo_denoiser"]
                        and d["checks"]["uses_scaled_ppo_offline_guidance"],
                        "scaled_ppo_task_conditioned_guidance_multiseed_uses_scaled_chain",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["uses_official_importer_export_usd"]
                        and d["bundle"]["motion_count"] == 40,
                        "scaled_ppo_task_conditioned_guidance_multiseed_uses_importer_export_bundle",
                    ),
                    lambda d: (
                        all(Path(row["asset_json"]).is_file() for row in d["rows"])
                        and all(Path(row["mp4"]).is_file() and Path(row["mp4"]).stat().st_size > 0 for row in d["rows"]),
                        "scaled_ppo_task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "scaled_ppo_task_conditioned_guidance_multiseed_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_task_conditioned_guidance_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets",
                (
                    "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/"
                    "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["summary_status_ok"]
                        and d["checks"]["all_rows_ok"]
                        and d["checks"]["seed_group_count_at_least_5"]
                        and d["checks"]["four_tasks_per_seed_group"],
                        "scaled_ppo_task_conditioned_guidance_multiseed_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_scaled_ppo_chain"]
                        and d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "scaled_ppo_task_conditioned_guidance_multiseed_assets_chain_and_bundle_ok",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "scaled_ppo_task_conditioned_guidance_multiseed_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_ppo_task_conditioned_guidance_multiseed_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_task_conditioned_guidance_success_boundary",
                (
                    "res/report_assets/"
                    "official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
                    "local_proxy_success_boundary.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_task_conditioned_guidance_success_boundary",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["row_count_matches_summary"]
                        and d["checks"]["seed_group_count_matches_summary"]
                        and d["checks"]["seed_group_count_at_least_5"]
                        and d["checks"]["four_tasks"],
                        "importer_export_guidance_success_boundary_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["full_bundle_motion_count_40"]
                        and d["checks"]["uses_official_importer_export_usd"],
                        "importer_export_guidance_success_boundary_bundle_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_completed_299"]
                        and d["checks"]["all_guidance_signals_positive"]
                        and d["checks"]["all_actions_changed_above_threshold"]
                        and d["checks"]["all_rows_have_mp4_paths"],
                        "importer_export_guidance_success_boundary_rollouts_and_videos",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "importer_export_guidance_success_boundary_assets_exist",
                    ),
                    lambda d: (
                        d["metrics"]["overall_completion_rate_299"] == 1.0
                        and d["metrics"]["overall_guidance_signal_positive_rate"] == 1.0
                        and d["metrics"]["overall_action_changed_rate"] == 1.0,
                        "importer_export_guidance_success_boundary_rates_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "importer_export_guidance_success_boundary_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary",
                (
                    "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/"
                    "local_proxy_success_boundary.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["row_count_20"]
                        and d["checks"]["five_seed_groups"]
                        and d["checks"]["four_tasks"],
                        "full_bundle_guidance_success_boundary_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_completed_299"]
                        and d["checks"]["all_guidance_signals_positive"]
                        and d["checks"]["all_rows_have_mp4_paths"],
                        "full_bundle_guidance_success_boundary_rollouts_and_videos",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "full_bundle_guidance_success_boundary_assets_exist",
                    ),
                    lambda d: (
                        d["metrics"]["overall_completion_rate_299"] == 1.0
                        and d["metrics"]["overall_guidance_signal_positive_rate"] == 1.0,
                        "full_bundle_guidance_success_boundary_rates_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_guidance_success_boundary_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_guidance_video_contact_sheet",
                (
                    "res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/"
                    "full_bundle_guidance_video_index.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_guidance_video_contact_sheet",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["row_count_20"]
                        and d["checks"]["seed_group_count_5"]
                        and d["checks"]["task_count_4"],
                        "full_bundle_guidance_video_contact_sheet_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_mp4_exist"]
                        and d["checks"]["all_keyframes_exist"]
                        and d["checks"]["contact_sheet_exists"],
                        "full_bundle_guidance_video_contact_sheet_assets_exist",
                    ),
                    lambda d: (
                        d["metrics"]["video_count"] == 20
                        and d["metrics"]["row_count"] == 20
                        and d["metrics"]["total_mp4_size_bytes"] > 0,
                        "full_bundle_guidance_video_contact_sheet_metrics_recorded",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "full_bundle_guidance_video_contact_sheet_report_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_guidance_video_contact_sheet_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_guidance_video_contact_sheet",
                (
                    "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/"
                    "importer_export_guidance_video_index.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_full_bundle_guidance_video_contact_sheet",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["row_count_matches_summary"]
                        and d["checks"]["seed_group_count_matches_summary"]
                        and d["checks"]["seed_group_count_at_least_5"]
                        and d["checks"]["task_count_4"],
                        "importer_export_guidance_video_contact_sheet_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["all_mp4_exist"]
                        and d["checks"]["all_keyframes_exist"]
                        and d["checks"]["contact_sheet_exists"],
                        "importer_export_guidance_video_contact_sheet_assets_exist",
                    ),
                    lambda d: (
                        d["metrics"]["video_count"] == d["metrics"]["row_count"]
                        and d["metrics"]["row_count"] >= 20
                        and d["metrics"]["seed_group_count"] >= 5
                        and d["metrics"]["total_mp4_size_bytes"] > 0,
                        "importer_export_guidance_video_contact_sheet_metrics_recorded",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "importer_export_guidance_video_contact_sheet_report_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "importer_export_guidance_video_contact_sheet_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_fig5_fig6_proxy_protocol_matrix",
                (
                    "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
                    "fig5_fig6_proxy_protocol_matrix.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_fig5_fig6_proxy_protocol_matrix",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["all_six_paper_panels_mapped"]
                        and d["metrics"]["panel_count"] == 6
                        and d["metrics"]["paper_level_reproduced_panel_count"] == 0,
                        "fig5_fig6_proxy_matrix_panel_accounting",
                    ),
                    lambda d: (
                        d["checks"]["has_importer_export_proxy_closed_loop_rows"]
                        and d["metrics"]["importer_export_closed_loop_rollout_rows_referenced"] >= 12
                        and d["metrics"]["closed_loop_proxy_panel_count"] >= 3,
                        "fig5_fig6_proxy_matrix_local_closed_loop_evidence",
                    ),
                    lambda d: (
                        d["checks"]["has_inpainting_offline_or_debug_evidence"]
                        and d["checks"]["has_inpainting_importer_export_proxy_closed_loop"]
                        and d["checks"]["records_inpainting_guided_delta"]
                        and d["metrics"]["offline_or_debug_panel_count"] >= 4,
                        "fig5_fig6_proxy_matrix_inpainting_boundary",
                    ),
                    lambda d: (
                        d["checks"]["source_latent_projection_status_ok"]
                        and d["checks"]["has_fig5d_latent_projection_proxy"]
                        and d["metrics"]["latent_projection_proxy_panel_count"] == 1,
                        "fig5_fig6_proxy_matrix_latent_projection_boundary",
                    ),
                    lambda d: (
                        d["checks"]["source_transition_proxy_status_ok"]
                        and d["checks"]["has_fig5b_transition_importer_export_proxy_closed_loop"]
                        and d["checks"]["records_transition_guided_speed_metrics"]
                        and d["metrics"]["transition_proxy_rollout_rows_referenced"] == 1,
                        "fig5_fig6_proxy_matrix_transition_boundary",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "fig5_fig6_proxy_matrix_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_not_paper_level"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "fig5_fig6_proxy_matrix_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_fig5_fig6_task_protocol_proxy",
                (
                    "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                    "fig5_fig6_task_protocol_proxy.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_fig5_fig6_task_protocol_proxy",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["metrics"]["row_count"] == 20
                        and d["metrics"]["task_count"] == 4
                        and d["metrics"]["seed_group_count"] == 5,
                        "fig5_fig6_task_protocol_proxy_source_and_shape",
                    ),
                    lambda d: (
                        d["checks"]["all_trace_npz_exist"]
                        and d["checks"]["all_mp4_paths_exist"]
                        and d["metrics"]["trace_npz_count"] == 20
                        and d["metrics"]["mp4_count"] == 20,
                        "fig5_fig6_task_protocol_proxy_trace_and_video_paths",
                    ),
                    lambda d: (
                        d["checks"]["all_records_have_299_steps"]
                        and d["checks"]["all_guidance_cost_delta_positive"]
                        and d["checks"]["local_proxy_pass_rate_recorded"],
                        "fig5_fig6_task_protocol_proxy_metrics_recorded",
                    ),
                    lambda d: (
                        d["metrics"]["paper_level_reproduced_panel_count"] == 0
                        and d["checks"]["does_not_claim_fig5_fig6_paper_level"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "fig5_fig6_task_protocol_proxy_no_overclaim",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "fig5_fig6_task_protocol_proxy_assets_exist",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy",
                (
                    "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                    "fig5_fig6_task_protocol_proxy.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["metrics"]["row_count"] == 20
                        and d["metrics"]["task_count"] == 4
                        and d["metrics"]["seed_group_count"] == 5,
                        "scaled_fig5_fig6_task_protocol_proxy_source_and_shape",
                    ),
                    lambda d: (
                        d["checks"]["all_trace_npz_exist"]
                        and d["checks"]["all_mp4_paths_exist"]
                        and d["metrics"]["trace_npz_count"] == 20
                        and d["metrics"]["mp4_count"] == 20,
                        "scaled_fig5_fig6_task_protocol_proxy_trace_and_video_paths",
                    ),
                    lambda d: (
                        d["checks"]["all_records_have_299_steps"]
                        and d["checks"]["all_guidance_cost_delta_positive"]
                        and d["checks"]["local_proxy_pass_rate_recorded"]
                        and d["metrics"]["overall_local_task_protocol_proxy_pass_rate"] >= 0.0,
                        "scaled_fig5_fig6_task_protocol_proxy_metrics_recorded",
                    ),
                    lambda d: (
                        d["metrics"]["paper_level_reproduced_panel_count"] == 0
                        and d["checks"]["does_not_claim_fig5_fig6_paper_level"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_fig5_fig6_task_protocol_proxy_no_overclaim",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "scaled_fig5_fig6_task_protocol_proxy_assets_exist",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy",
                (
                    "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
                    "success_fall_collision_proxy.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["metrics"]["row_count"] == 20
                        and d["metrics"]["task_count"] == 4
                        and d["metrics"]["seed_group_count"] == 5,
                        "scaled_fig5_fig6_success_fall_collision_proxy_source_and_shape",
                    ),
                    lambda d: (
                        d["checks"]["all_trace_npz_exist"]
                        and d["checks"]["all_mp4_paths_exist"]
                        and d["metrics"]["trace_npz_count"] == 20
                        and d["metrics"]["mp4_count"] == 20,
                        "scaled_fig5_fig6_success_fall_collision_proxy_trace_and_video_paths",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_completed_299"]
                        and d["checks"]["success_proxy_rate_recorded"]
                        and d["checks"]["fall_proxy_rate_recorded"]
                        and d["checks"]["collision_contact_signal_unavailable_recorded"],
                        "scaled_fig5_fig6_success_fall_collision_proxy_metrics_recorded",
                    ),
                    lambda d: (
                        d["metrics"]["paper_level_success_rate_available"] is False
                        and d["metrics"]["paper_level_fall_rate_available"] is False
                        and d["metrics"]["paper_level_collision_rate_available"] is False
                        and d["checks"]["does_not_claim_paper_success_rate"]
                        and d["checks"]["does_not_claim_paper_fall_rate"]
                        and d["checks"]["does_not_claim_paper_collision_rate"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_fig5_fig6_success_fall_collision_proxy_no_overclaim",
                    ),
                    lambda d: (
                        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in d["assets"].values()),
                        "scaled_fig5_fig6_success_fall_collision_proxy_assets_exist",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_transition_guidance_rollout_eval",
                (
                    "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
                    "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_transition_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["underlying_returncode_zero"]
                        and d["checks"]["underlying_status_ok"]
                        and d["checks"]["single_transition_task_attempted"]
                        and d["checks"]["task_row_status_ok"],
                        "transition_guidance_underlying_run_ok",
                    ),
                    lambda d: (
                        d["checks"]["rollout_299_steps"]
                        and d["rows"][0]["rollout_steps"] == 299
                        and d["rows"][0]["task"] == "transition",
                        "transition_guidance_rollout_steps",
                    ),
                    lambda d: (
                        d["checks"]["capture_npz_exists"]
                        and d["checks"]["mp4_path_recorded"]
                        and Path(d["rows"][0]["mp4"]).is_file()
                        and Path(d["rows"][0]["mp4"]).stat().st_size > 0,
                        "transition_guidance_video_recorded",
                    ),
                    lambda d: (
                        d["checks"]["transition_metrics_recorded"]
                        and d["rows"][0]["guided_late_minus_early_speed_mps"] is not None
                        and d["rows"][0]["guided_target_speed_rmse_mps"] is not None,
                        "transition_guidance_speed_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["report_assets_exist"]
                        and all(
                            Path(path).is_file() and Path(path).stat().st_size > 0
                            for key, path in d["outputs"]["assets"].items()
                            if key not in {"json", "mp4"} and path
                        ),
                        "transition_guidance_report_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["uses_full_public_motion_bundle"],
                        "transition_guidance_importer_bundle_boundary",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5b_paper_protocol"]
                        and d["checks"]["does_not_claim_fig5d_tsne"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "transition_guidance_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_inpainting_guidance_rollout_eval",
                (
                    "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
                    "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_inpainting_guidance_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["subprocess_returncode_zero"]
                        and d["checks"]["underlying_status_ok"]
                        and d["checks"]["single_inpainting_task_attempted"]
                        and d["checks"]["task_row_status_ok"],
                        "inpainting_guidance_underlying_run_ok",
                    ),
                    lambda d: (
                        d["checks"]["rollout_299_steps"]
                        and d["rows"][0]["rollout_steps"] == 299
                        and d["rows"][0]["task"] == "inpainting",
                        "inpainting_guidance_rollout_steps",
                    ),
                    lambda d: (
                        d["checks"]["capture_npz_exists"]
                        and d["checks"]["mp4_exists"]
                        and Path(d["outputs"]["capture_npz"]).is_file()
                        and Path(d["outputs"]["mp4"]).is_file()
                        and Path(d["outputs"]["mp4"]).stat().st_size > 0,
                        "inpainting_guidance_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["keyframe_proxy_metrics_recorded"]
                        and d["rows"][0]["guided_keyframe_error_mean"] is not None
                        and d["rows"][0]["denoised_keyframe_error_mean"] is not None
                        and d["rows"][0]["guided_keyframe_error_delta_vs_denoised"] is not None,
                        "inpainting_guidance_keyframe_proxy_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["uses_official_importer_export_usd"]
                        and d["checks"]["uses_full_public_motion_bundle"]
                        and d["checks"]["used_fallback_guidance_scale"],
                        "inpainting_guidance_inputs_and_scale_boundary",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig6a_paper_protocol"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"]
                        and d["interpretation"]["goal_complete"] is False,
                        "inpainting_guidance_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit",
                (
                    "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
                    "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["vae_training_source_ok"]
                        and d["checks"]["denoiser_training_source_ok"]
                        and d["settings"]["variant"] == "full_bundle",
                        "full_bundle_onnx_async_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["onnx_files_written"]
                        and d["checks"]["onnx_checker_passed"]
                        and d["checks"]["vae_encoder_matches_torch"]
                        and d["checks"]["vae_decoder_matches_torch"]
                        and d["checks"]["denoiser_matches_torch"],
                        "full_bundle_onnx_async_exports_match_torch",
                    ),
                    lambda d: (
                        d["checks"]["onnxruntime_cpu_available"]
                        and d["checks"]["onnxruntime_cuda_unavailable_recorded"]
                        and d["checks"]["onnxruntime_tensorrt_unavailable_recorded"],
                        "full_bundle_onnx_async_provider_boundary_recorded",
                    ),
                    lambda d: (
                        d["checks"]["async_pipeline_completed"]
                        and d["async_summary"]["throughput_speedup_vs_sequential_mean"] > 1.0,
                        "full_bundle_onnx_async_threadpool_completed",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt"]
                        and d["checks"]["does_not_claim_paper_latency"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "full_bundle_onnx_async_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_full_bundle_vae_denoiser_onnx_async_audit",
                (
                    "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
                    "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["vae_training_source_ok"]
                        and d["checks"]["denoiser_training_source_ok"]
                        and d["settings"]["variant"] == "importer_export_full_bundle",
                        "importer_export_onnx_async_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["onnx_files_written"]
                        and d["checks"]["onnx_checker_passed"]
                        and d["checks"]["vae_encoder_matches_torch"]
                        and d["checks"]["vae_decoder_matches_torch"]
                        and d["checks"]["denoiser_matches_torch"],
                        "importer_export_onnx_async_exports_match_torch",
                    ),
                    lambda d: (
                        d["checks"]["onnxruntime_cpu_available"]
                        and d["checks"]["onnxruntime_cuda_unavailable_recorded"]
                        and d["checks"]["onnxruntime_tensorrt_unavailable_recorded"],
                        "importer_export_onnx_async_provider_boundary_recorded",
                    ),
                    lambda d: (
                        d["checks"]["async_pipeline_completed"]
                        and d["async_summary"]["throughput_speedup_vs_sequential_mean"] > 1.0,
                        "importer_export_onnx_async_threadpool_completed",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt"]
                        and d["checks"]["does_not_claim_paper_latency"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "importer_export_onnx_async_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit",
                (
                    "res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/"
                    "level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.json"
                ),
                [
                    lambda d: (
                        d.get("status")
                        == "ok_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["vae_training_source_ok"]
                        and d["checks"]["denoiser_training_source_ok"]
                        and d["settings"]["variant"] == "importer_export_scaled_ppo",
                        "scaled_importer_export_onnx_async_sources_ok",
                    ),
                    lambda d: (
                        d["checks"]["onnx_files_written"]
                        and d["checks"]["onnx_checker_passed"]
                        and d["checks"]["vae_encoder_matches_torch"]
                        and d["checks"]["vae_decoder_matches_torch"]
                        and d["checks"]["denoiser_matches_torch"],
                        "scaled_importer_export_onnx_async_exports_match_torch",
                    ),
                    lambda d: (
                        d["checks"]["onnxruntime_cpu_available"]
                        and d["checks"]["onnxruntime_cuda_unavailable_recorded"]
                        and d["checks"]["onnxruntime_tensorrt_unavailable_recorded"],
                        "scaled_importer_export_onnx_async_provider_boundary_recorded",
                    ),
                    lambda d: (
                        d["checks"]["async_pipeline_completed"]
                        and d["async_summary"]["throughput_speedup_vs_sequential_mean"] > 1.0,
                        "scaled_importer_export_onnx_async_threadpool_completed",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_tensorrt"]
                        and d["checks"]["does_not_claim_paper_latency"]
                        and d["checks"]["does_not_claim_official_checkpoint"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["interpretation"]["goal_complete"] is False,
                        "scaled_importer_export_onnx_async_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_vae_closed_loop_rollout_eval",
                (
                    "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
                    "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"
                ),
                [
                    lambda d: (
                        d.get("status") == "ok_official_csv_loop_vae_closed_loop_rollout_eval",
                        f"status={d.get('status')!r}",
                    ),
                    lambda d: (
                        d["checks"]["rollout_success"]
                        and d["checks"]["two_shards_completed"]
                        and d["checks"]["rollout_steps_299"],
                        "vae_closed_loop_success_two_shards_299_steps",
                    ),
                    lambda d: (
                        d["run"]["aggregate_metrics"]["total_num_envs"] == 2048
                        and d["run"]["aggregate_metrics"]["total_env_steps"] == 612352,
                        "vae_closed_loop_full_env_steps",
                    ),
                    lambda d: (
                        d["checks"]["uses_gpus_4_7"]
                        and d["run"]["gpu_metrics_summary"]["per_gpu"]["4"]["peak_memory_used_mb"] >= 10240,
                        "vae_closed_loop_gpu4_formal_memory_recorded",
                    ),
                    lambda d: (
                        d["checks"]["peak_memory_each_gpu_at_least_10gb"] is False
                        and d["run"]["gpu_metrics_summary"]["per_gpu"]["7"]["peak_memory_used_mb"] < 10240,
                        "vae_closed_loop_gpu7_below_10gb_recorded",
                    ),
                    lambda d: (
                        d["run"]["aggregate_metrics"]["teacher_vae_action_mse"]["mean"] < 0.01,
                        "vae_closed_loop_action_reconstruction_mse_lt_0p01",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_autonomous_vae_policy"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "vae_closed_loop_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vae_closed_loop_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "official_csv_loop_vae_closed_loop_rollout_assets",
                (
                    "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
                    "official_csv_loop_vae_closed_loop_rollout_assets.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["source_status_ok"]
                        and d["checks"]["two_shards_completed"]
                        and d["checks"]["rollout_steps_299"],
                        "vae_closed_loop_assets_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["png_assets_exist"] and d["checks"]["csv_assets_exist"],
                        "vae_closed_loop_assets_files_exist",
                    ),
                    lambda d: (
                        d["metrics"]["total_env_steps"] == 612352
                        and d["metrics"]["teacher_vae_action_mse_mean"] < 0.01,
                        "vae_closed_loop_assets_metrics_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_beyondmimic_vae"]
                        and d["checks"]["does_not_claim_guided_diffusion"]
                        and d["checks"]["does_not_claim_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"],
                        "vae_closed_loop_assets_no_overclaim",
                    ),
                ],
            ),
            check_json_artifact(
                "visual_evidence_index",
                "res/report_assets/visual_evidence_index/visual_evidence_index.json",
                [
                    status_ok,
                    lambda d: (
                        d["metrics"]["asset_json_count"] >= 20
                        and d["metrics"]["report_ready_video_count"] >= 9
                        and d["metrics"]["report_ready_png_count"] >= 40,
                        "visual_evidence_index_counts",
                    ),
                    lambda d: (
                        d["checks"]["all_indexed_assets_exist"]
                        and d["checks"]["has_report_ready_videos"]
                        and d["checks"]["has_report_ready_pngs"]
                        and d["checks"]["has_metric_tables_or_readmes"],
                        "visual_evidence_index_assets_exist",
                    ),
                    lambda d: (
                        d["checks"]["all_video_rows_marked_do_not_commit_large_video"],
                        "visual_evidence_index_videos_not_for_github_commit",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_avoid_paper_level_overclaim"]
                        and d["checks"]["all_rows_avoid_real_robot_overclaim"]
                        and d["checks"]["all_rows_keep_goal_incomplete"],
                        "visual_evidence_index_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "visual_evidence_index_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guided_vs_unguided_closed_loop_matrix",
                (
                    "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
                    "guided_vs_unguided_closed_loop_matrix.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["metrics"]["row_count"] >= 23
                        and d["metrics"]["aggregate_row_count"] >= 8
                        and d["metrics"]["multiseed_row_count"] >= 12,
                        "guided_matrix_counts",
                    ),
                    lambda d: (
                        d["checks"]["has_multiseed_rows"]
                        and d["checks"]["has_four_task_aggregate"]
                        and d["metrics"].get("full_bundle_task_conditioned_row_count", 0) == 4
                        and d["metrics"].get("full_bundle_task_conditioned_multiseed_row_count", 0) >= 12
                        and d["metrics"].get("video_row_count", 0) == d["metrics"]["row_count"]
                        and d["checks"]["all_video_paths_exist_when_recorded"],
                        "guided_matrix_rows_and_videos",
                    ),
                    lambda d: (
                        d["checks"]["all_rows_have_claim_level"]
                        and d["checks"]["all_rows_keep_local_or_limited_claim"],
                        "guided_matrix_claim_levels",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_fig5_fig6"]
                        and d["checks"]["does_not_claim_real_robot"]
                        and d["checks"]["does_not_claim_goal_complete"],
                        "guided_matrix_no_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guided_matrix_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_state_latent_guidance_eval",
                (
                    "res/level_c/resource_adjusted_state_latent_guidance_eval/"
                    "level_c_resource_adjusted_state_latent_guidance_eval.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["all_tasks_evaluated"], "state_latent_guidance_all_tasks"),
                    lambda d: (
                        d["checks"]["all_best_costs_improve"],
                        "state_latent_guidance_best_costs_improve",
                    ),
                    lambda d: (
                        d["checks"]["all_best_guidance_gradients_nonzero"],
                        "state_latent_guidance_nonzero_gradients",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["total_selected_windows"] == 8192,
                        "state_latent_guidance_selected_windows_8192",
                    ),
                    lambda d: (
                        d["worker_summary"]["metrics"]["row_count"] == 48,
                        "state_latent_guidance_row_count_48",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "state_latent_guidance_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6_reproduction"],
                        "state_latent_guidance_no_fig56_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_level_guidance"],
                        "state_latent_guidance_no_paper_guidance_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "state_latent_guidance_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_multiseed_audit",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/"
                    "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["seed_count_3"], "tiny_multiseed_seed_count_3"),
                    lambda d: (
                        d["checks"]["all_validation_token_mse_finite"],
                        "tiny_multiseed_val_token_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_test_token_mse_finite"],
                        "tiny_multiseed_test_token_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_validation_action_mse_finite"],
                        "tiny_multiseed_val_action_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_test_action_mse_finite"],
                        "tiny_multiseed_test_action_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_validation_token_improves_vs_noisy"],
                        "tiny_multiseed_val_improves",
                    ),
                    lambda d: (
                        d["checks"]["all_test_token_improves_vs_noisy"],
                        "tiny_multiseed_test_improves",
                    ),
                    lambda d: (d["checks"]["npz_written"], "tiny_multiseed_npz_written"),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_multiseed"],
                        "tiny_multiseed_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_multiseed_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/"
                    "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["source_training_status_ok"],
                        "tiny_checkpoint_eval_source_ok",
                    ),
                    lambda d: (d["checks"]["checkpoint_exists"], "tiny_checkpoint_eval_checkpoint_exists"),
                    lambda d: (
                        d["checks"]["checkpoint_sha_matches_training_json"],
                        "tiny_checkpoint_eval_sha_matches",
                    ),
                    lambda d: (
                        d["checks"]["payload_marks_not_paper_checkpoint"],
                        "tiny_checkpoint_eval_not_paper_checkpoint",
                    ),
                    lambda d: (d["checks"]["three_split_rows"], "tiny_checkpoint_eval_three_splits"),
                    lambda d: (d["checks"]["all_split_counts_28"], "tiny_checkpoint_eval_split_counts"),
                    lambda d: (
                        d["checks"]["all_prediction_metrics_finite"],
                        "tiny_checkpoint_eval_metrics_finite",
                    ),
                    lambda d: (
                        d["checks"]["max_token_mse_delta_below_1e_minus_12"],
                        "tiny_checkpoint_eval_token_delta_zero",
                    ),
                    lambda d: (
                        d["checks"]["max_action_mse_delta_below_1e_minus_12"],
                        "tiny_checkpoint_eval_action_delta_zero",
                    ),
                    lambda d: (
                        d["checks"]["validation_and_test_better_than_noisy"],
                        "tiny_checkpoint_eval_heldout_improves",
                    ),
                    lambda d: (d["checks"]["npz_written"], "tiny_checkpoint_eval_npz_written"),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_checkpoint"],
                        "tiny_checkpoint_eval_no_paper_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_checkpoint_eval_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
                    "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_training_status_ok"], "tiny_onnx_source_training_ok"),
                    lambda d: (d["checks"]["checkpoint_exists"], "tiny_onnx_checkpoint_exists"),
                    lambda d: (
                        d["checks"]["checkpoint_sha_matches_training_json"],
                        "tiny_onnx_checkpoint_sha_matches",
                    ),
                    lambda d: (
                        d["checks"]["payload_marks_not_paper_checkpoint"],
                        "tiny_onnx_payload_not_paper",
                    ),
                    lambda d: (d["checks"]["onnx_file_written"], "tiny_onnx_file_written"),
                    lambda d: (d["checks"]["onnx_checker_passed"], "tiny_onnx_checker"),
                    lambda d: (d["checks"]["reference_evaluator_loaded"], "tiny_onnx_reference_evaluator"),
                    lambda d: (d["checks"]["onnx_matches_torch"], "tiny_onnx_matches_torch"),
                    lambda d: (d["checks"]["npz_written"], "tiny_onnx_npz_written"),
                    lambda d: (
                        d["checks"]["metadata_marks_debug_not_paper"],
                        "tiny_onnx_metadata_not_paper",
                    ),
                    lambda d: (d["checks"]["does_not_claim_tensorrt"], "tiny_onnx_no_tensorrt_claim"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "tiny_onnx_no_rollout_claim",
                    ),
                    lambda d: (d["metrics"]["parameter_count"] == 143491, "tiny_onnx_param_count"),
                    lambda d: (d["metrics"]["onnx_size_bytes"] > 0, "tiny_onnx_nonempty"),
                    lambda d: (len(d["metrics"]["onnx_sha256"]) == 64, "tiny_onnx_sha256"),
                    lambda d: (
                        d["metrics"]["max_abs_onnx_vs_torch"] <= 1e-5,
                        "tiny_onnx_max_abs_matches_torch",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_onnx_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_latency_audit",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/"
                    "level_c_resource_adjusted_tiny_diffusion_latency_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["source_onnx_export_status_ok"], "tiny_latency_source_onnx_ok"),
                    lambda d: (d["checks"]["training_status_ok"], "tiny_latency_training_ok"),
                    lambda d: (d["checks"]["onnx_file_exists"], "tiny_latency_onnx_exists"),
                    lambda d: (d["checks"]["io_fixture_exists"], "tiny_latency_io_fixture_exists"),
                    lambda d: (d["checks"]["onnx_checker_passed"], "tiny_latency_onnx_checker"),
                    lambda d: (d["checks"]["onnx_matches_torch"], "tiny_latency_onnx_matches_torch"),
                    lambda d: (
                        d["checks"]["torch_latency_finite_positive"],
                        "tiny_latency_torch_finite_positive",
                    ),
                    lambda d: (
                        d["checks"]["onnx_reference_latency_finite_positive"],
                        "tiny_latency_onnx_finite_positive",
                    ),
                    lambda d: (
                        d["checks"]["tiny_onnx_reference_p95_under_paper_20ms_debug_budget"],
                        "tiny_latency_onnx_under_20ms_debug_budget",
                    ),
                    lambda d: (
                        d["checks"]["tiny_torch_p95_under_paper_20ms_debug_budget"],
                        "tiny_latency_torch_under_20ms_debug_budget",
                    ),
                    lambda d: (d["checks"]["npz_written"], "tiny_latency_npz_written"),
                    lambda d: (d["checks"]["does_not_claim_tensorrt"], "tiny_latency_no_tensorrt_claim"),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_model_latency"],
                        "tiny_latency_no_paper_latency_claim",
                    ),
                    lambda d: (
                        d["metrics"]["max_abs_onnx_vs_torch"] <= 1e-5,
                        "tiny_latency_output_match",
                    ),
                    lambda d: (d["metrics"]["onnx_reference_cpu_p95_ms"] < 20.0, "tiny_latency_onnx_p95"),
                    lambda d: (d["metrics"]["torch_cpu_p95_ms"] < 20.0, "tiny_latency_torch_p95"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_latency_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_resource_adjusted_tiny_diffusion_video_preview",
                (
                    "res/level_c/resource_adjusted_tiny_diffusion_video_preview/"
                    "level_c_resource_adjusted_tiny_diffusion_video_preview.json"
                ),
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["source_training_status_ok"],
                        "tiny_video_source_training_status_ok",
                    ),
                    lambda d: (d["checks"]["two_debug_gifs_written"], "tiny_video_two_gifs"),
                    lambda d: (d["checks"]["posters_written"], "tiny_video_posters"),
                    lambda d: (d["checks"]["all_gifs_nonempty"], "tiny_video_gifs_nonempty"),
                    lambda d: (
                        d["checks"]["all_previews_have_21_frames"],
                        "tiny_video_21_frames",
                    ),
                    lambda d: (
                        d["checks"]["all_pred_improves_vs_noisy"],
                        "tiny_video_pred_improves",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_video"],
                        "tiny_video_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_figures"],
                        "tiny_video_no_paper_figure_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "tiny_video_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "emphasis_projection_audit",
                "res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["paper_source_projection_patterns_found"],
                        "emphasis_projection_source_patterns",
                    ),
                    lambda d: (d["checks"]["uses_paper_state_dim_99"], "emphasis_projection_state_dim_99"),
                    lambda d: (d["checks"]["emphasis_coefficient_c_6"], "emphasis_projection_c_6"),
                    lambda d: (
                        d["checks"]["projection_full_column_rank"],
                        "emphasis_projection_full_column_rank",
                    ),
                    lambda d: (
                        d["checks"]["pseudoinverse_roundtrip_below_1e_minus_10"],
                        "emphasis_projection_pinv_roundtrip",
                    ),
                    lambda d: (
                        d["checks"]["pinv_projection_identity_below_1e_minus_10"],
                        "emphasis_projection_pinv_identity",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_accumulation_probe",
                "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
                [status_ok, lambda d: (d["checks"]["gradient_accumulation_matches_paper"], "vae_accumulation_15")],
            ),
            check_json_artifact(
                "vae_checkpoint_smoke",
                "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["checkpoint_file_exists"], "vae_checkpoint_file_exists"),
                    lambda d: (d["checks"]["checkpoint_has_required_keys"], "vae_checkpoint_required_keys"),
                    lambda d: (d["checks"]["paper_dimensions_match"], "vae_checkpoint_paper_dims"),
                    lambda d: (d["checks"]["gradient_accumulation_matches_paper"], "vae_checkpoint_accumulation_15"),
                    lambda d: (d["checks"]["loaded_eval_action_matches_saved_model"], "vae_checkpoint_load_consistency"),
                    lambda d: (d["checks"]["optimizer_state_restored"], "vae_checkpoint_optimizer_restored"),
                    lambda d: (d["checks"]["marks_not_trained_paper_checkpoint"], "vae_checkpoint_not_paper_trained"),
                    lambda d: (d["interpretation"]["goal_complete"] is False, "vae_checkpoint_keeps_goal_incomplete"),
                ],
            ),
            check_json_artifact(
                "vae_debug_overfit_latent_artifact",
                "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["row_count_84"], "vae_debug_latent_rows_84"),
                    lambda d: (
                        d["checks"]["split_counts_match_manifest"],
                        "vae_debug_latent_split_counts",
                    ),
                    lambda d: (
                        d["checks"]["all_tokens_shape_21x131"],
                        "vae_debug_latent_token_shape",
                    ),
                    lambda d: (d["checks"]["latent_dim_32"], "vae_debug_latent_dim_32"),
                    lambda d: (d["checks"]["all_latents_nonzero"], "vae_debug_latents_nonzero"),
                    lambda d: (d["checks"]["reconstruction_loss_decreases"], "vae_debug_recon_decreases"),
                    lambda d: (
                        d["checks"]["loss_reduction_ratio_above_half"],
                        "vae_debug_recon_reduction",
                    ),
                    lambda d: (d["checks"]["debug_only_boundary_recorded"], "vae_debug_boundary"),
                    lambda d: (
                        d["checks"]["does_not_claim_true_dagger_rollout"],
                        "vae_debug_no_true_dagger_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_paper_checkpoint"],
                        "vae_debug_no_paper_checkpoint_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vae_debug_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_motion_split_heldout_eval",
                "res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json",
                [
                    status_ok,
                    lambda d: (
                        d["checks"]["uses_motion_level_train_validation_test_split"],
                        "vae_motion_split_uses_split",
                    ),
                    lambda d: (
                        d["checks"]["no_motion_crosses_splits"],
                        "vae_motion_split_no_motion_leakage",
                    ),
                    lambda d: (
                        d["checks"]["optimizer_trains_only_train_split"],
                        "vae_motion_split_train_only",
                    ),
                    lambda d: (
                        d["checks"]["uses_package_action_mse_metric"],
                        "vae_motion_split_uses_action_mse",
                    ),
                    lambda d: (
                        d["checks"]["uses_package_vae_kl_metric"],
                        "vae_motion_split_uses_kl_metric",
                    ),
                    lambda d: (d["checks"]["latent_dim_32"], "vae_motion_split_latent_dim_32"),
                    lambda d: (d["checks"]["action_dim_29"], "vae_motion_split_action_dim_29"),
                    lambda d: (
                        d["checks"]["validation_action_mse_decreases"],
                        "vae_motion_split_validation_decreases",
                    ),
                    lambda d: (d["checks"]["test_action_mse_decreases"], "vae_motion_split_test_decreases"),
                    lambda d: (
                        d["checks"]["validation_reduction_above_half"],
                        "vae_motion_split_validation_reduction",
                    ),
                    lambda d: (
                        d["checks"]["test_reduction_above_half"],
                        "vae_motion_split_test_reduction",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_dagger_rollout"],
                        "vae_motion_split_no_true_dagger_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_paper_checkpoint"],
                        "vae_motion_split_no_paper_checkpoint_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_vae_eval"],
                        "vae_motion_split_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vae_motion_split_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_receding_horizon_rollout_smoke",
                "res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["source_json_status_ok"], "vae_rh_source_json_ok"),
                    lambda d: (d["checks"]["source_npz_exists"], "vae_rh_source_npz_exists"),
                    lambda d: (d["checks"]["row_count_84"], "vae_rh_rows_84"),
                    lambda d: (d["checks"]["motion_count_3"], "vae_rh_motion_count_3"),
                    lambda d: (d["checks"]["split_counts_match_source"], "vae_rh_split_counts"),
                    lambda d: (d["checks"]["sequence_length_21"], "vae_rh_sequence_21"),
                    lambda d: (d["checks"]["state_dim_99"], "vae_rh_state_dim_99"),
                    lambda d: (d["checks"]["latent_dim_32"], "vae_rh_latent_dim_32"),
                    lambda d: (d["checks"]["action_dim_29"], "vae_rh_action_dim_29"),
                    lambda d: (d["checks"]["current_index_is_history"], "vae_rh_current_index_history"),
                    lambda d: (d["checks"]["all_decoded_actions_finite"], "vae_rh_actions_finite"),
                    lambda d: (d["checks"]["current_action_mse_below_0_01"], "vae_rh_current_mse"),
                    lambda d: (d["checks"]["full_window_mse_below_0_01"], "vae_rh_full_mse"),
                    lambda d: (d["checks"]["next_latent_changes_action"], "vae_rh_next_latent_changes_action"),
                    lambda d: (d["checks"]["does_not_claim_true_dagger_rollout"], "vae_rh_no_true_dagger_claim"),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_paper_checkpoint"],
                        "vae_rh_no_paper_checkpoint_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_simulation"],
                        "vae_rh_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vae_rh_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "diffusion_to_vae_action_smoke",
                "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["vae_source_status_ok"], "diffusion_to_action_vae_source_ok"),
                    lambda d: (
                        d["checks"]["diffusion_source_status_ok"],
                        "diffusion_to_action_diffusion_source_ok",
                    ),
                    lambda d: (
                        d["checks"]["uses_train_split_only_for_decoder_surrogate"],
                        "diffusion_to_action_train_split_only",
                    ),
                    lambda d: (d["checks"]["window_count_84"], "diffusion_to_action_windows_84"),
                    lambda d: (d["checks"]["sequence_length_21"], "diffusion_to_action_sequence_21"),
                    lambda d: (d["checks"]["token_dim_131"], "diffusion_to_action_token_dim_131"),
                    lambda d: (d["checks"]["state_dim_99"], "diffusion_to_action_state_dim_99"),
                    lambda d: (d["checks"]["latent_dim_32"], "diffusion_to_action_latent_dim_32"),
                    lambda d: (d["checks"]["action_dim_29"], "diffusion_to_action_action_dim_29"),
                    lambda d: (
                        d["checks"]["no_token_identity_basis_in_action_decoder"],
                        "diffusion_to_action_no_token_identity",
                    ),
                    lambda d: (d["checks"]["all_actions_finite"], "diffusion_to_action_finite"),
                    lambda d: (
                        d["checks"]["validation_prediction_improves_current_vs_noisy"],
                        "diffusion_to_action_validation_current_improves",
                    ),
                    lambda d: (
                        d["checks"]["test_prediction_improves_current_vs_noisy"],
                        "diffusion_to_action_test_current_improves",
                    ),
                    lambda d: (
                        d["checks"]["heldout_current_action_mse_below_0_02"],
                        "diffusion_to_action_heldout_current_mse",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_decoder"],
                        "diffusion_to_action_no_true_vae_decoder_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_diffusion_checkpoint"],
                        "diffusion_to_action_no_trained_diffusion_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "diffusion_to_action_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "diffusion_to_action_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "diffusion_to_vae_action_multiseed_audit",
                "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "diffusion_to_action_ms_3_seeds"),
                    lambda d: (d["checks"]["no_best_seed_only_reporting"], "diffusion_to_action_ms_no_best_seed"),
                    lambda d: (d["checks"]["all_metrics_finite"], "diffusion_to_action_ms_finite"),
                    lambda d: (
                        d["checks"]["all_seed_validation_prediction_improves_current_vs_noisy"],
                        "diffusion_to_action_ms_validation_current_improves",
                    ),
                    lambda d: (
                        d["checks"]["all_seed_test_prediction_improves_current_vs_noisy"],
                        "diffusion_to_action_ms_test_current_improves",
                    ),
                    lambda d: (
                        d["checks"]["validation_current_mse_mean_below_0_02"],
                        "diffusion_to_action_ms_validation_mse_mean",
                    ),
                    lambda d: (
                        d["checks"]["test_current_mse_mean_below_0_02"],
                        "diffusion_to_action_ms_test_mse_mean",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_train_split_only_for_decoder_surrogate"],
                        "diffusion_to_action_ms_train_split_only",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_do_not_use_token_identity_basis"],
                        "diffusion_to_action_ms_no_token_identity",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_use_independent_state_latent_steps"],
                        "diffusion_to_action_ms_independent_steps",
                    ),
                    lambda d: (
                        d["checks"]["all_runs_debug_vae_latents_nonzero"],
                        "diffusion_to_action_ms_latents_nonzero",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_vae_decoder"],
                        "diffusion_to_action_ms_no_true_vae_decoder_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_trained_diffusion_checkpoint"],
                        "diffusion_to_action_ms_no_trained_diffusion_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_rollout"],
                        "diffusion_to_action_ms_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "diffusion_to_action_ms_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "diffusion_to_vae_action_smoothness_audit",
                "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["source_smoke_status_ok"], "diffusion_to_action_smooth_source_ok"),
                    lambda d: (
                        d["checks"]["multiseed_source_status_ok"],
                        "diffusion_to_action_smooth_multiseed_source_ok",
                    ),
                    lambda d: (d["checks"]["row_count_3_splits"], "diffusion_to_action_smooth_3_splits"),
                    lambda d: (d["checks"]["action_dim_29"], "diffusion_to_action_smooth_action_dim"),
                    lambda d: (d["checks"]["sequence_length_21"], "diffusion_to_action_smooth_sequence_21"),
                    lambda d: (d["checks"]["control_frequency_25hz"], "diffusion_to_action_smooth_25hz"),
                    lambda d: (d["checks"]["all_metrics_finite"], "diffusion_to_action_smooth_finite"),
                    lambda d: (
                        d["checks"]["validation_predicted_smoother_than_noisy"],
                        "diffusion_to_action_smooth_validation_smoother",
                    ),
                    lambda d: (
                        d["checks"]["test_predicted_smoother_than_noisy"],
                        "diffusion_to_action_smooth_test_smoother",
                    ),
                    lambda d: (
                        d["checks"]["validation_predicted_action_rate_below_noisy"],
                        "diffusion_to_action_smooth_validation_rate",
                    ),
                    lambda d: (
                        d["checks"]["test_predicted_action_rate_below_noisy"],
                        "diffusion_to_action_smooth_test_rate",
                    ),
                    lambda d: (
                        d["checks"]["validation_predicted_acceleration_below_noisy"],
                        "diffusion_to_action_smooth_validation_acceleration",
                    ),
                    lambda d: (
                        d["checks"]["test_predicted_acceleration_below_noisy"],
                        "diffusion_to_action_smooth_test_acceleration",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_action_smoothness"],
                        "diffusion_to_action_smooth_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "diffusion_to_action_smooth_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "direct_vs_latent_action_ablation_audit",
                "res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 3, "direct_vs_latent_rows_3"),
                    lambda d: (d["metrics"]["window_count"] == 84, "direct_vs_latent_windows_84"),
                    lambda d: (d["metrics"]["state_dim"] == 99, "direct_vs_latent_state_dim_99"),
                    lambda d: (d["metrics"]["latent_dim"] == 32, "direct_vs_latent_latent_dim_32"),
                    lambda d: (d["metrics"]["token_dim"] == 131, "direct_vs_latent_token_dim_131"),
                    lambda d: (d["metrics"]["action_dim"] == 29, "direct_vs_latent_action_dim_29"),
                    lambda d: (d["checks"]["vae_source_status_ok"], "direct_vs_latent_vae_source_ok"),
                    lambda d: (d["checks"]["diffusion_source_status_ok"], "direct_vs_latent_diffusion_source_ok"),
                    lambda d: (d["checks"]["latent_action_source_status_ok"], "direct_vs_latent_action_source_ok"),
                    lambda d: (
                        d["checks"]["latent_action_target_matches_vae_target"],
                        "direct_vs_latent_target_matches",
                    ),
                    lambda d: (
                        d["checks"]["latent_action_order_matches_vae_rows"],
                        "direct_vs_latent_order_matches",
                    ),
                    lambda d: (
                        d["checks"]["uses_train_split_only_for_direct_decoder"],
                        "direct_vs_latent_train_split_only",
                    ),
                    lambda d: (
                        d["checks"]["direct_decoder_uses_state_only"],
                        "direct_vs_latent_direct_state_only",
                    ),
                    lambda d: (
                        d["checks"]["direct_decoder_no_token_identity_basis"],
                        "direct_vs_latent_no_identity_basis",
                    ),
                    lambda d: (
                        d["checks"]["latent_better_than_direct_on_validation_current"],
                        "direct_vs_latent_validation_latent_better",
                    ),
                    lambda d: (
                        d["checks"]["latent_better_than_direct_on_test_current"],
                        "direct_vs_latent_test_latent_better",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_paper_direct_diffusion_success"],
                        "direct_vs_latent_no_paper_success_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "direct_vs_latent_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_contract_audit",
                "res/level_c/vae_contract_audit/level_c_vae_contract_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 35, "vae_contract_rows_35"),
                    lambda d: (d["metrics"]["failed_row_count"] == 0, "vae_contract_failures_zero"),
                    lambda d: (d["metrics"]["vae_table_value_rows"] == 8, "vae_table_rows_8"),
                    lambda d: (d["checks"]["paper_source_formulas_present"], "vae_contract_paper_formulas"),
                    lambda d: (d["checks"]["all_vae_table_values_match"], "vae_contract_table_values"),
                    lambda d: (d["checks"]["all_dimension_contracts_match"], "vae_contract_dimensions"),
                    lambda d: (d["checks"]["runtime_shapes_match"], "vae_contract_runtime_shapes"),
                    lambda d: (d["checks"]["latent_math_contract_passes"], "vae_contract_latent_math"),
                    lambda d: (
                        d["checks"]["training_contract_passes_debug_only"],
                        "vae_contract_training_debug_only",
                    ),
                    lambda d: (d["checks"]["debug_only_boundary_recorded"], "vae_contract_boundary"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "vae_contract_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_dagger_vae_pipeline_audit",
                "res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["stage_count"] == 9, "dagger_vae_pipeline_stage_count_9"),
                    lambda d: (d["metrics"]["ok_stage_count"] == 9, "dagger_vae_pipeline_ok_stage_count_9"),
                    lambda d: (
                        d["metrics"]["dagger_teacher_queries"] == 288,
                        "dagger_vae_pipeline_teacher_queries_288",
                    ),
                    lambda d: (
                        d["metrics"]["vae_effective_batch_size"] == 30,
                        "dagger_vae_pipeline_effective_batch_30",
                    ),
                    lambda d: (
                        d["metrics"]["state_latent_rows"] == 84,
                        "dagger_vae_pipeline_state_latent_rows_84",
                    ),
                    lambda d: (
                        d["metrics"]["state_dim"] == 99
                        and d["metrics"]["latent_dim"] == 32
                        and d["metrics"]["token_dim"] == 131
                        and d["metrics"]["action_dim"] == 29,
                        "dagger_vae_pipeline_dims_99_32_131_29",
                    ),
                    lambda d: (
                        d["checks"]["dagger_schema_not_true_rollout"],
                        "dagger_vae_pipeline_schema_not_true_rollout",
                    ),
                    lambda d: (
                        d["checks"]["dagger_iteration_three_iters_and_288_queries"],
                        "dagger_vae_pipeline_dagger_iteration_queries",
                    ),
                    lambda d: (
                        d["checks"]["vae_contract_zero_failed_rows"],
                        "dagger_vae_pipeline_vae_contract_passes",
                    ),
                    lambda d: (
                        d["checks"]["vae_checkpoint_roundtrip_debug_only"],
                        "dagger_vae_pipeline_checkpoint_roundtrip",
                    ),
                    lambda d: (
                        d["checks"]["vae_debug_latents_nonzero_and_84_rows"],
                        "dagger_vae_pipeline_nonzero_latents",
                    ),
                    lambda d: (
                        d["checks"]["vae_motion_split_heldout_test_improves"],
                        "dagger_vae_pipeline_motion_split_test_improves",
                    ),
                    lambda d: (
                        d["checks"]["receding_horizon_84_rows_current_action_low_mse"],
                        "dagger_vae_pipeline_receding_horizon_mse",
                    ),
                    lambda d: (
                        d["checks"]["all_boundaries_preserve_goal_incomplete"],
                        "dagger_vae_pipeline_boundary_incomplete",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "dagger_vae_pipeline_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "vae_latent_probe",
                "res/level_c/vae_latent_probe/level_c_vae_latent_probe.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["at_least_three_seeds"], "vae_latent_probe_3_seeds"),
                    lambda d: (d["checks"]["latent_dim_matches_paper"], "vae_latent_dim_32"),
                    lambda d: (d["checks"]["encoder_dims_match_paper"], "vae_encoder_dims"),
                    lambda d: (d["checks"]["decoder_dims_match_paper"], "vae_decoder_dims"),
                    lambda d: (d["checks"]["all_reparameterization_exact"], "vae_reparameterization_exact"),
                    lambda d: (d["checks"]["all_kl_formula_matches_manual"], "vae_kl_formula"),
                    lambda d: (
                        d["checks"]["all_interpolation_endpoints_match"],
                        "vae_interpolation_endpoints",
                    ),
                    lambda d: (
                        d["checks"]["all_interpolation_actions_finite"],
                        "vae_interpolation_actions_finite",
                    ),
                    lambda d: (d["checks"]["debug_only_boundary_recorded"], "vae_latent_debug_boundary"),
                ],
            ),
            check_json_artifact(
                "receding_horizon_decoder_probe",
                "res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json",
                [status_ok, lambda d: (d["checks"]["uses_current_latent_only_for_action"], "current_latent_action")],
            ),
            check_json_artifact(
                "official_artifact_audit",
                "res/level_c/official_artifact_audit/level_c_official_artifact_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["conclusion"]["official_beyondmimic_vae_diffusion_code_found"] is False,
                        "official_level_c_code_absent",
                    ),
                ],
            ),
            check_json_artifact(
                "fig5_fig6_feasibility_audit",
                "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
                [
                    status_ok,
                    lambda d: (
                        d["conclusion"]["fig5_fig6_paper_reproduction_possible_from_current_local_artifacts"] is False,
                        "fig5_fig6_blocked",
                    ),
                ],
            ),
            check_json_artifact(
                "blocked_gate_audit",
                "res/blocked_gates/blocked_gate_audit.json",
                [
                    status_ok,
                    lambda d: (d["gate_count"] >= 7, "reproduction_gates_audited"),
                    lambda d: (d["gate_status_counts"].get("blocked", 0) >= 3, "external_blockers_recorded"),
                    lambda d: (
                        any(
                            gate["gate_id"] == "isaaclab_kit_inotify"
                            and gate["status"] == "clear_with_historical_failure"
                            for gate in d["gates"]
                        ),
                        "inotify_gate_cleared_with_history",
                    ),
                    lambda d: (
                        any(
                            gate["gate_id"] == "official_g1_usd_conversion_replay"
                            and gate["status"] == "blocked"
                            for gate in d["gates"]
                        ),
                        "official_g1_conversion_gate_blocked",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "goal_not_complete_due_to_blocked_gates",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_source_coverage_audit",
                "res/paper_source_coverage/paper_source_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["counts"]["parsed_figures"] == 10, "ten_latex_figures_indexed"),
                    lambda d: (d["counts"]["parsed_tables"] == 8, "eight_latex_tables_indexed"),
                    lambda d: (d["counts"]["method_claims"] >= 9, "method_claims_indexed"),
                    lambda d: (d["counts"]["missing_expected_labels"] == 0, "no_missing_expected_labels"),
                    lambda d: (d["counts"]["unmapped_rows"] == 0, "no_unmapped_paper_rows"),
                    lambda d: (d["counts"]["missing_evidence_rows"] == 0, "all_coverage_evidence_paths_exist"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "goal_not_complete_due_to_partial_blocked_paper_items",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_latex_inventory_audit",
                "res/paper_latex_inventory/paper_latex_inventory_audit.json",
                [
                    status_ok,
                    lambda d: (d["counts"]["tex_file_count"] == 5, "paper_latex_tex_files_5"),
                    lambda d: (d["counts"]["equation_count"] == 8, "paper_latex_equations_8"),
                    lambda d: (d["counts"]["figure_count"] == 10, "paper_latex_figures_10"),
                    lambda d: (d["counts"]["table_count"] == 8, "paper_latex_tables_8"),
                    lambda d: (
                        d["counts"]["experiment_setting_count"] == 14,
                        "paper_latex_experiment_settings_14",
                    ),
                    lambda d: (
                        d["counts"]["missing_expected_setting_count"] == 0,
                        "paper_latex_no_missing_settings",
                    ),
                    lambda d: (d["checks"]["all_tex_files_hashed"], "paper_latex_tex_files_hashed"),
                    lambda d: (d["checks"]["source_coverage_audit_ok"], "paper_latex_source_coverage_ok"),
                    lambda d: (d["checks"]["table_value_audit_ok"], "paper_latex_table_values_ok"),
                    lambda d: (d["checks"]["table_value_mismatch_zero"], "paper_latex_table_mismatches_zero"),
                    lambda d: (d["checks"]["paper_source_unmapped_zero"], "paper_latex_source_unmapped_zero"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "paper_latex_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_formula_code_trace_audit",
                "res/paper_formula_code_trace/paper_formula_code_trace_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 11, "paper_formula_trace_rows_11"),
                    lambda d: (d["missing_evidence_row_count"] == 0, "paper_formula_trace_no_missing_evidence"),
                    lambda d: (d["source_counts"]["latex_equation_count"] == 8, "paper_formula_trace_equations_8"),
                    lambda d: (
                        d["source_counts"]["latex_experiment_setting_count"] == 14,
                        "paper_formula_trace_settings_14",
                    ),
                    lambda d: (d["source_counts"]["paper_table_value_rows"] == 58, "paper_formula_trace_table_rows_58"),
                    lambda d: (
                        d["source_counts"]["paper_table_value_mismatch_rows"] == 0,
                        "paper_formula_trace_no_table_mismatch",
                    ),
                    lambda d: (
                        d["source_counts"]["core_math_test_row_count"] == 23,
                        "paper_formula_trace_core_math_rows_23",
                    ),
                    lambda d: (
                        d["source_counts"]["reimpl_symbol_row_count"] == 29,
                        "paper_formula_trace_reimpl_symbols_29",
                    ),
                    lambda d: (
                        d["source_counts"]["api_test_row_count"] == 8,
                        "paper_formula_trace_api_rows_8",
                    ),
                    lambda d: (d["checks"]["latex_inventory_ok"], "paper_formula_trace_latex_ok"),
                    lambda d: (d["checks"]["table_value_audit_ok"], "paper_formula_trace_table_ok"),
                    lambda d: (d["checks"]["core_test_coverage_ok"], "paper_formula_trace_core_tests_ok"),
                    lambda d: (
                        d["checks"]["core_math_metric_tests_present"],
                        "paper_formula_trace_metric_core_tests",
                    ),
                    lambda d: (d["checks"]["reimpl_package_audit_ok"], "paper_formula_trace_reimpl_ok"),
                    lambda d: (
                        d["checks"]["reimpl_package_metric_symbols_present"],
                        "paper_formula_trace_metric_symbols",
                    ),
                    lambda d: (d["checks"]["api_tests_ok"], "paper_formula_trace_api_tests_ok"),
                    lambda d: (
                        d["checks"]["api_goal_metric_contracts_present"],
                        "paper_formula_trace_goal_metric_api_contracts",
                    ),
                    lambda d: (d["checks"]["all_evidence_paths_exist"], "paper_formula_trace_paths_exist"),
                    lambda d: (
                        d["checks"]["records_debug_or_blocked_boundaries"],
                        "paper_formula_trace_boundaries_recorded",
                    ),
                    lambda d: (
                        d["checks"]["guidance_formulas_link_full_split_public_data"],
                        "paper_formula_trace_guidance_full_split_public_data",
                    ),
                    lambda d: (
                        d["checks"]["guidance_formula_boundaries_do_not_claim_closed_loop"],
                        "paper_formula_trace_guidance_no_closed_loop_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "paper_formula_trace_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_pdf_source_consistency_audit",
                "res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["pdf_page_count"] == 59, "paper_pdf_pages_59"),
                    lambda d: (
                        d["metrics"]["pdf_text_page_count"] == 59,
                        "paper_pdf_text_pages_59",
                    ),
                    lambda d: (
                        d["metrics"]["pdf_anchor_present_count"] == 20,
                        "paper_pdf_anchors_20",
                    ),
                    lambda d: (
                        d["metrics"]["source_tar_expected_member_count"] == 19,
                        "paper_source_tar_expected_members_19",
                    ),
                    lambda d: (
                        d["metrics"]["source_tar_present_member_count"] == 19,
                        "paper_source_tar_present_members_19",
                    ),
                    lambda d: (
                        d["metrics"]["source_tar_extracted_member_count"] == 19,
                        "paper_source_tar_extracted_members_19",
                    ),
                    lambda d: (
                        d["metrics"]["unexpected_tar_file_member_count"] == 0,
                        "paper_source_tar_no_extra_files",
                    ),
                    lambda d: (d["checks"]["all_pdf_pages_text_extracted"], "paper_pdf_text_extracted"),
                    lambda d: (d["checks"]["all_expected_pdf_anchors_present"], "paper_pdf_anchors_present"),
                    lambda d: (
                        d["checks"]["all_expected_source_tar_members_present"],
                        "paper_source_tar_members_present",
                    ),
                    lambda d: (d["checks"]["latex_inventory_ok"], "paper_pdf_latex_inventory_ok"),
                    lambda d: (d["checks"]["source_coverage_ok"], "paper_pdf_source_coverage_ok"),
                    lambda d: (d["checks"]["table_values_ok"], "paper_pdf_table_values_ok"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "paper_pdf_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_table_value_audit",
                "res/paper_table_values/paper_table_value_audit.json",
                [
                    status_ok,
                    lambda d: (d["counts"]["total_rows"] == 58, "fifty_eight_table_values_audited"),
                    lambda d: (d["counts"]["mismatch_rows"] == 0, "no_table_value_mismatches"),
                    lambda d: (d["counts"]["tables"].get("tab:ppo_hyperparameters") == 14, "ppo_table_values"),
                    lambda d: (d["counts"]["tables"].get("tab:vae_hyperparameters") == 8, "vae_table_values"),
                    lambda d: (d["counts"]["tables"].get("tab:diffusion_hyperparameters") == 14, "diffusion_table_values"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "goal_not_complete_due_to_debug_table_matches",
                    ),
                ],
            ),
            check_json_artifact(
                "skill_success_table_data_audit",
                "res/paper_skill_success_table_audit/skill_success_table_data_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["paper_table_source_found"], "skill_table_source_found"),
                    lambda d: (d["checks"]["short_sequence_rows_parsed"], "skill_table_short_rows_7"),
                    lambda d: (d["checks"]["lafan_rows_parsed"], "skill_table_lafan_rows_29"),
                    lambda d: (
                        d["checks"]["local_data_mismatches_recorded"],
                        "skill_table_local_mismatches_recorded",
                    ),
                    lambda d: (
                        d["checks"]["available_lafan_csvs_have_36_columns"],
                        "skill_table_available_csv_columns",
                    ),
                    lambda d: (
                        d["checks"]["available_lafan_csv_values_finite"],
                        "skill_table_available_csv_finite",
                    ),
                    lambda d: (
                        d["metrics"]["missing_lafan_csv_count"] == 1,
                        "skill_table_missing_lafan_csv_count_1",
                    ),
                    lambda d: (
                        d["metrics"]["segment_out_of_range_row_count"] == 2,
                        "skill_table_segment_out_of_range_count_2",
                    ),
                    lambda d: (
                        d["checks"]["sim_real_success_not_claimed_reproduced"],
                        "skill_table_no_success_reproduction_claim",
                    ),
                ],
            ),
            check_json_artifact(
                "dagger_iteration_smoke",
                "res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["uses_package_dagger_api"], "dagger_iteration_uses_package_api"),
                    lambda d: (
                        d["checks"]["uses_package_evaluation_metric_api"],
                        "dagger_iteration_uses_eval_metric_api",
                    ),
                    lambda d: (d["checks"]["three_dagger_iterations"], "dagger_iteration_three_iters"),
                    lambda d: (
                        d["checks"]["all_iterations_query_teacher"],
                        "dagger_iteration_teacher_queries",
                    ),
                    lambda d: (
                        d["checks"]["aggregate_dataset_grows_each_iteration"],
                        "dagger_iteration_dataset_grows",
                    ),
                    lambda d: (d["checks"]["state_dim_163"], "dagger_iteration_state_dim_163"),
                    lambda d: (d["checks"]["action_dim_29"], "dagger_iteration_action_dim_29"),
                    lambda d: (
                        d["checks"]["heldout_discrepancy_decreases"],
                        "dagger_iteration_heldout_decreases",
                    ),
                    lambda d: (
                        d["checks"]["final_heldout_reduction_at_least_0_5"],
                        "dagger_iteration_reduction_threshold",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_true_dagger_rollout"],
                        "dagger_iteration_no_true_rollout_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "dagger_iteration_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "released_panel_mapping_audit",
                "res/released_panel_mapping_audit/released_panel_mapping_audit.json",
                [
                    status_ok,
                    lambda d: (d["checks"]["raw_zip_exists"], "released_panel_raw_zip_exists"),
                    lambda d: (d["checks"]["extracted_dataset_exists"], "released_panel_extracted_dataset_exists"),
                    lambda d: (d["metrics"]["released_panel_rows"] == 15, "released_panel_rows_15"),
                    lambda d: (d["metrics"]["released_summary_rows"] == 13, "released_summary_rows_13"),
                    lambda d: (d["metrics"]["released_panel_fail_count"] == 0, "released_panel_failures_zero"),
                    lambda d: (
                        d["checks"]["all_expected_released_figure_ids_present"],
                        "released_panel_expected_figure_ids_present",
                    ),
                    lambda d: (
                        d["checks"]["all_summary_figure_ids_mapped_to_paper_panels"],
                        "released_panel_summary_ids_mapped",
                    ),
                    lambda d: (
                        d["checks"]["non_released_paper_items_not_claimed"],
                        "released_panel_no_fig5_fig6_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "released_panel_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "paper_vs_reproduction_comparison",
                "res/comparison/paper_vs_reproduction.json",
                [
                    status_ok,
                    lambda d: (d["total_rows"] >= 100, "comparison_rows_at_least_100"),
                    lambda d: (len(d["missing_required_field_rows"]) == 0, "comparison_required_fields_present"),
                    lambda d: (len(d["invalid_comparison_type_rows"]) == 0, "comparison_types_valid"),
                    lambda d: (len(d["missing_goal_checkpoint_rows"]) == 0, "goal_checkpoint_metrics_present"),
                    lambda d: (
                        len(d.get("missing_guidance_full_split_rows", [])) == 0,
                        "guidance_full_split_comparison_rows_present",
                    ),
                    lambda d: (
                        d["checks"]["guidance_full_split_rows_remain_qualitative_only"],
                        "guidance_full_split_not_overclaimed",
                    ),
                    lambda d: (
                        d["checks"]["guidance_full_split_rows_do_not_claim_paper_rollout"],
                        "guidance_full_split_rollout_boundary",
                    ),
                    lambda d: (
                        d["checks"]["goal_checkpoint_rows_have_formula_evidence"],
                        "goal_checkpoint_formula_evidence_present",
                    ),
                    lambda d: (
                        d["checks"]["goal_checkpoint_rows_remain_not_reproduced"],
                        "goal_checkpoint_not_overclaimed",
                    ),
                    lambda d: (d["comparison_type_counts"].get("exactly_comparable") == 58, "exact_table_rows_58"),
                    lambda d: (
                        d["comparison_type_counts"].get("not_publicly_reproducible", 0) >= 6,
                        "blocked_paper_results_marked",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "comparison_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "results_claims_audit",
                "res/results_claims_audit/results_claims_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 14, "results_claims_rows_14"),
                    lambda d: (d["metrics"]["failed_row_count"] == 0, "results_claims_failures_zero"),
                    lambda d: (
                        d["checks"]["released_panel_mapping_passes"],
                        "results_claims_released_panel_mapping_passes",
                    ),
                    lambda d: (
                        d["checks"]["released_statistical_audit_passes"],
                        "results_claims_released_statistical_audit_passes",
                    ),
                    lambda d: (
                        d["checks"]["fig5_fig6_blocked_boundary_reused"],
                        "results_claims_fig5_fig6_boundary",
                    ),
                    lambda d: (
                        "res/tables/released_data_statistical_audit/released_data_statistical_audit.json"
                        in next(row for row in d["rows"] if row["claim_id"] == "fig3b_imu_dynamics_released")[
                            "evidence"
                        ],
                        "results_claims_imu_uses_released_stats",
                    ),
                    lambda d: (
                        "res/tables/released_data_statistical_audit/released_grf_confidence_intervals.csv"
                        in next(row for row in d["rows"] if row["claim_id"] == "fig4c_grf_released")[
                            "evidence"
                        ],
                        "results_claims_grf_uses_released_stats",
                    ),
                    lambda d: (
                        d["checks"]["official_level_c_artifacts_absent_recorded"],
                        "results_claims_official_level_c_absent",
                    ),
                    lambda d: (
                        d["checks"]["goal_checkpoint_rows_present_in_comparison"],
                        "results_claims_goal_checkpoints_present",
                    ),
                    lambda d: (
                        d["metrics"]["formula_api_linked_paper_metric_claim_rows"] == 2,
                        "results_claims_formula_api_metric_claim_rows",
                    ),
                    lambda d: (
                        d["checks"]["paper_metric_claims_have_formula_api_evidence"],
                        "results_claims_formula_api_evidence_linked",
                    ),
                    lambda d: (
                        d["checks"]["formula_api_evidence_not_overclaimed_as_paper_results"],
                        "results_claims_formula_api_no_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_fig5_fig6_reproduction"],
                        "results_claims_no_fig5_fig6_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "results_claims_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "final_reproduction_report",
                "res/final_report/final_reproduction_report.json",
                [
                    status_ok,
                    lambda d: (d["goal_complete"] is False, "final_report_goal_not_complete"),
                    lambda d: (d["checks"]["atomic_write_used"], "final_report_atomic_write_used"),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "final_report_no_goal_complete_claim_check",
                    ),
                    lambda d: (d["master"]["artifact_count"] >= 80, "final_report_embeds_master_summary"),
                    lambda d: (d["level_a_released_data"]["released_figure_rows"] == 13, "released_figure_rows_13"),
                    lambda d: (
                        d["level_c_diffusion"]["official_level_c_code_found"] is False,
                        "final_report_level_c_official_code_absent",
                    ),
                    lambda d: (
                        d["blocked_gates"]["gate_status_counts"].get("blocked", 0) >= 3,
                        "final_report_blocked_gates_recorded",
                    ),
                    lambda d: (
                        any(
                            gate["gate_id"] == "official_g1_usd_conversion_replay"
                            and gate["status"] == "blocked"
                            for gate in d["blocked_gates"]["gates"]
                        ),
                        "final_report_official_g1_conversion_blocked",
                    ),
                    lambda d: (
                        d["english_reading_report"]["doc_exists"]
                        and d["english_reading_report"]["final_exists"],
                        "final_report_english_reading_report_exists",
                    ),
                    lambda d: (
                        d["english_reading_report"]["word_count"] >= 1500,
                        "final_report_english_reading_report_word_count",
                    ),
                    lambda d: (
                        d["english_reading_report"]["contains_no_full_reproduction_claim"],
                        "final_report_english_reading_report_no_full_claim",
                    ),
                    lambda d: (
                        d["english_reading_report"]["mentions_official_loop_virtual_chain"],
                        "final_report_english_reading_report_official_loop_chain",
                    ),
                ],
            ),
            check_file_artifact("goal_final_report_markdown", "res/final_report/reproduction_report.md"),
            check_file_artifact("english_reading_report_doc", "reproduction/docs/english_reading_report.md"),
            check_file_artifact("english_reading_report_final", "res/final_report/english_reading_report.md"),
            check_json_artifact(
                "final_report_requirement_audit",
                "res/final_report/final_report_requirement_audit/final_report_requirement_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 12, "final_report_requirements_12"),
                    lambda d: (d["missing_count"] == 0, "final_report_requirements_missing_zero"),
                    lambda d: (d["checks"]["markdown_reports_identical"], "final_report_markdowns_identical"),
                    lambda d: (d["checks"]["summary_points_to_goal_report"], "final_report_summary_goal_report"),
                    lambda d: (
                        d["checks"]["summary_atomic_write_used"],
                        "final_report_requirements_summary_atomic_write",
                    ),
                    lambda d: (
                        d["checks"]["atomic_write_used"],
                        "final_report_requirements_atomic_write_used",
                    ),
                    lambda d: (
                        d["checks"]["all_12_goal_report_requirements_present"],
                        "final_report_all_12_requirements",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "final_report_requirements_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "final_report_requirements_keep_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "final_deliverables_audit",
                "res/final_deliverables_audit/final_deliverables_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 38, "final_deliverables_rows_38"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "final_deliverables_evidence_exists"),
                    lambda d: (
                        d["checks"]["environment_deliverables_present"],
                        "final_deliverables_environment_present",
                    ),
                    lambda d: (
                        d["checks"]["documentation_deliverables_present"],
                        "final_deliverables_documentation_present",
                    ),
                    lambda d: (
                        d["checks"]["patch_inventory_recorded"],
                        "final_deliverables_patch_inventory",
                    ),
                    lambda d: (
                        d["checks"]["experiment_deliverables_record_missing_checkpoints_and_videos"],
                        "final_deliverables_records_missing_checkpoints_videos",
                    ),
                    lambda d: (
                        d["checks"]["visual_media_inventory_status_ok"],
                        "final_deliverables_visual_inventory_status_ok",
                    ),
                    lambda d: (
                        d["checks"]["visual_media_inventory_hashes_recorded"],
                        "final_deliverables_visual_inventory_hashes",
                    ),
                    lambda d: (
                        d["checks"]["visual_media_inventory_video_gaps_recorded"],
                        "final_deliverables_visual_inventory_video_gaps",
                    ),
                    lambda d: (
                        d["checks"]["visual_media_inventory_no_paper_level_reproduction_videos"],
                        "final_deliverables_visual_inventory_no_paper_level_videos",
                    ),
                    lambda d: (
                        d["checks"]["visual_media_inventory_reference_videos_labeled"],
                        "final_deliverables_visual_inventory_reference_videos_labeled",
                    ),
                    lambda d: (
                        d["checks"]["visual_media_inventory_counts_match_deliverable_rows"],
                        "final_deliverables_visual_inventory_counts",
                    ),
                    lambda d: (
                        d["checks"]["metrics_deliverable_records_formula_api_evidence"],
                        "final_deliverables_formula_api_metrics",
                    ),
                    lambda d: (
                        d["checks"]["tests_deliverable_records_core_coverage"],
                        "final_deliverables_core_test_coverage",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "final_deliverables_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "verification_command_coverage_audit",
                "res/verification_command_coverage/verification_command_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["command_count"] >= 80, "verification_command_count_at_least_80"),
                    lambda d: (d["smoke_command_count"] == 10, "verification_smoke_command_count_10"),
                    lambda d: (d["smoke_pass_count"] == 10, "verification_smoke_pass_count_10"),
                    lambda d: (d["checks"]["no_duplicate_commands"], "verification_no_duplicates"),
                    lambda d: (d["checks"]["all_detected_scripts_exist"], "verification_scripts_exist"),
                    lambda d: (
                        d["checks"]["smoke_commands_listed_in_final_report"],
                        "verification_smoke_commands_listed",
                    ),
                    lambda d: (d["checks"]["smoke_commands_pass"], "verification_smoke_commands_pass"),
                    lambda d: (
                        d["checks"]["mapped_smoke_outputs_exist"],
                        "verification_mapped_outputs_exist",
                    ),
                    lambda d: (d["checks"]["atomic_write_used"], "verification_atomic_write_used"),
                    lambda d: (
                        d["checks"]["does_not_execute_heavy_or_env_specific_commands"],
                        "verification_no_heavy_execution",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "verification_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "verification_command_syntax_audit",
                "res/verification_command_syntax/verification_command_syntax_audit.json",
                [
                    status_ok,
                    lambda d: (d["python_script_count"] >= 100, "verification_syntax_python_scripts_at_least_100"),
                    lambda d: (d["failed_count"] == 0, "verification_syntax_failed_zero"),
                    lambda d: (d["checks"]["all_scripts_exist"], "verification_syntax_scripts_exist"),
                    lambda d: (d["checks"]["all_scripts_compile"], "verification_syntax_scripts_compile"),
                    lambda d: (d["checks"]["atomic_write_used"], "verification_syntax_atomic_write_used"),
                    lambda d: (d["checks"]["does_not_execute_commands"], "verification_syntax_static_only"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "verification_syntax_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "verification_command_script_manifest",
                "res/verification_command_script_manifest/verification_command_script_manifest.json",
                [
                    status_ok,
                    lambda d: (
                        d["python_script_count"] >= 100,
                        "verification_script_manifest_scripts_at_least_100",
                    ),
                    lambda d: (
                        d["checks"]["script_count_matches_syntax_audit"],
                        "verification_script_manifest_matches_syntax",
                    ),
                    lambda d: (d["checks"]["no_missing_scripts"], "verification_script_manifest_no_missing"),
                    lambda d: (d["checks"]["all_hashes_present"], "verification_script_manifest_hashes_present"),
                    lambda d: (d["checks"]["atomic_write_used"], "verification_script_manifest_atomic_write_used"),
                    lambda d: (d["checks"]["does_not_execute_commands"], "verification_script_manifest_static_only"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "verification_script_manifest_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "required_artifact_absence_audit",
                "res/required_artifact_absence/required_artifact_absence_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 32, "required_artifact_rows_32_with_debug_reference_exclusion"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "required_artifact_evidence_exists"),
                    lambda d: (
                        d["status_counts"]["missing_required_artifact"] == 12,
                        "required_artifact_missing_count_12",
                    ),
                    lambda d: (
                        d["checks"]["reference_download_models_separated"],
                        "required_artifact_reference_models_separated",
                    ),
                    lambda d: (
                        d["checks"]["no_beyondmimic_named_model_in_download"],
                        "required_artifact_no_beyondmimic_model_download",
                    ),
                    lambda d: (
                        d["checks"]["public_lafan1_paper_arch_checkpoint_present"],
                        "required_artifact_public_lafan1_checkpoint_present",
                    ),
                    lambda d: (
                        d["checks"]["no_unclassified_local_reproduction_model_checkpoint"]
                        or (
                            d["checks"]["official_csv_loop_tracking_checkpoint_excluded"]
                            and d["checks"]["official_importer_export_tracking_checkpoint_excluded"]
                            and d["checks"]["resource_adjusted_tracking_checkpoint_excluded"]
                            and d["status_counts"]["missing_required_artifact"] == 12
                        ),
                        "required_artifact_local_model_checkpoints_classified_as_non_paper",
                    ),
                    lambda d: (
                        d["checks"]["no_local_paper_level_reproduction_video"],
                        "required_artifact_no_paper_level_local_video",
                    ),
                    lambda d: (
                        d["checks"]["diagnostic_checkpoint_excluded"],
                        "required_artifact_diagnostic_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["resource_adjusted_tracking_checkpoint_excluded"],
                        "required_artifact_resource_adjusted_tracking_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_tracking_checkpoint_excluded"],
                        "required_artifact_official_csv_loop_tracking_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_tracking_checkpoint_excluded"],
                        "required_artifact_official_importer_export_tracking_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["resource_adjusted_teacher_rollout_vae_checkpoint_excluded"],
                        "required_artifact_resource_adjusted_teacher_rollout_vae_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_teacher_rollout_vae_checkpoint_excluded"],
                        "required_artifact_official_csv_loop_teacher_rollout_vae_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_teacher_rollout_vae_checkpoint_excluded"],
                        "required_artifact_official_importer_export_teacher_rollout_vae_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_teacher_rollout_vae_checkpoint_excluded"],
                        "required_artifact_official_importer_export_scaled_ppo_teacher_rollout_vae_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["resource_adjusted_state_latent_diffusion_checkpoint_excluded"],
                        "required_artifact_resource_adjusted_state_latent_diffusion_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_state_latent_diffusion_checkpoint_excluded"],
                        "required_artifact_official_csv_loop_state_latent_diffusion_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_state_latent_diffusion_checkpoint_excluded"],
                        "required_artifact_official_importer_export_state_latent_diffusion_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_state_latent_diffusion_checkpoint_excluded"],
                        "required_artifact_official_importer_export_scaled_ppo_state_latent_diffusion_checkpoint_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_vae_denoiser_onnx_exports_excluded"],
                        "required_artifact_official_csv_loop_vae_denoiser_onnx_exports_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_csv_loop_teacher_rollout_dataset_excluded"],
                        "required_artifact_official_csv_loop_teacher_rollout_dataset_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_importer_export_scaled_ppo_teacher_rollout_dataset_excluded"],
                        "required_artifact_official_importer_export_scaled_ppo_teacher_rollout_dataset_excluded",
                    ),
                    lambda d: (
                        d["checks"]["debug_preview_videos_excluded"],
                        "required_artifact_debug_preview_videos_excluded",
                    ),
                    lambda d: (
                        d["checks"]["official_reference_doc_videos_excluded"],
                        "required_artifact_official_reference_videos_excluded",
                    ),
                    lambda d: (
                        d["checks"]["local_reference_video_excluded"],
                        "required_artifact_local_reference_video_excluded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "required_artifact_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "required_artifact_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "evaluation_metrics_coverage_audit",
                "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 44, "evaluation_metrics_rows_44"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "evaluation_metrics_evidence_exists"),
                    lambda d: (
                        d["checks"]["all_44_goal_metrics_mapped"],
                        "evaluation_metrics_all_goal_items_mapped",
                    ),
                    lambda d: (
                        d["checks"]["section_counts_match_goal_md"],
                        "evaluation_metrics_section_counts_match",
                    ),
                    lambda d: (
                        d["checks"]["blocked_metrics_not_claimed_complete"],
                        "evaluation_metrics_blocked_not_overclaimed",
                    ),
                    lambda d: (
                        d["checks"]["released_tracking_metrics_use_numeric_summary"],
                        "evaluation_metrics_tracking_numeric_summary",
                    ),
                    lambda d: (
                        d["checks"]["statistics_three_seed_boundary_recorded"],
                        "evaluation_metrics_three_seed_boundary",
                    ),
                    lambda d: (
                        d["checks"]["trial_failure_accounting_linked"],
                        "evaluation_metrics_trial_failure_accounting",
                    ),
                    lambda d: (
                        d["status_counts"]["formula_api_only"] == 5,
                        "evaluation_metrics_formula_api_rows_5",
                    ),
                    lambda d: (
                        d["status_counts"]["public_data_checkpoint"] == 18,
                        "evaluation_metrics_public_data_checkpoint_rows_18",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_velocity_error_links_guidance_full_split_metrics"],
                        "evaluation_metrics_velocity_guidance_full_split",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_goal_distance_links_guidance_full_split_metrics"],
                        "evaluation_metrics_goal_distance_guidance_full_split",
                    ),
                    lambda d: (
                        d["checks"]["goal_metric_formula_api_rows_linked"],
                        "evaluation_metrics_formula_api_linked",
                    ),
                    lambda d: (
                        d["checks"]["formula_api_metrics_not_claimed_rollout_results"],
                        "evaluation_metrics_formula_api_no_rollout_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "evaluation_metrics_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "evaluation_metrics_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "tracking_g1_resource_adjusted_csv_task_eval_gpu47_failed_rerun",
                "res/failed_runs/tracking_g1_resource_adjusted_csv_task_eval_gpu47_20260619_124125/status.json",
                [
                    lambda d: (d.get("status") == "failed_retained", f"status={d.get('status')!r}"),
                    lambda d: (d.get("returncode") == -9, "gpu47_task_eval_returncode_sigkill"),
                    lambda d: (
                        d["markers"]["after_app"] and d["markers"]["env_created"] and d["markers"]["env_reset"],
                        "gpu47_task_eval_reached_env_reset_before_sigkill",
                    ),
                    lambda d: (
                        d["markers"]["step_299"] is False and d["markers"]["metrics_file"] is False,
                        "gpu47_task_eval_no_false_success_metrics",
                    ),
                    lambda d: (
                        "does not invalidate earlier successful GPU6" in d["claim_level"],
                        "gpu47_task_eval_preserves_canonical_success_boundary",
                    ),
                ],
            ),
            check_json_artifact(
                "trial_failure_accounting_audit",
                "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["row_count"] == 14, "trial_failure_rows_14"),
                    lambda d: (d["checks"]["all_evidence_paths_exist"], "trial_failure_evidence_exists"),
                    lambda d: (d["checks"]["skill_success_rows_accounted"], "trial_failure_skill_rows"),
                    lambda d: (d["checks"]["released_metric_rows_accounted"], "trial_failure_released_rows"),
                    lambda d: (d["checks"]["debug_seed_runs_accounted"], "trial_failure_debug_seed_runs"),
                    lambda d: (d["checks"]["retained_failed_run_count_recorded"], "trial_failure_failed_run"),
                    lambda d: (d["checks"]["valid_training_runs_zero_recorded"], "trial_failure_no_training_runs"),
                    lambda d: (
                        d["checks"]["missing_paper_rollout_trials_recorded"],
                        "trial_failure_missing_rollouts_recorded",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_rollout_failure_counts"],
                        "trial_failure_no_rollout_overclaim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "trial_failure_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "ablation_coverage_audit",
                "res/ablation_coverage/ablation_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 15, "ablation_coverage_rows_15"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "ablation_coverage_evidence_exists"),
                    lambda d: (
                        d["checks"]["all_15_goal_ablation_items_mapped"],
                        "ablation_coverage_all_goal_items_mapped",
                    ),
                    lambda d: (
                        d["checks"]["motion_tracking_six_items_mapped"],
                        "ablation_coverage_tracking_items",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_nine_items_mapped"],
                        "ablation_coverage_diffusion_items",
                    ),
                    lambda d: (
                        d["checks"]["released_tracking_ablation_panels_present"],
                        "ablation_coverage_tracking_released_panels",
                    ),
                    lambda d: (
                        d["checks"]["diffusion_training_ablations_not_overclaimed"],
                        "ablation_coverage_diffusion_not_overclaimed",
                    ),
                    lambda d: (
                        d["checks"]["guidance_scale_sensitivity_links_full_split_metrics"],
                        "ablation_guidance_scale_full_split_metrics",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "ablation_coverage_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "ablation_coverage_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_c_lafan1_paper_arch_symmetry_training_comparison",
                (
                    "res/level_c/lafan1_paper_arch_symmetry_training_comparison/"
                    "level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json"
                ),
                [
                    status_ok,
                    lambda d: (d["checks"]["base_training_status_ok"], "symcomp_base_training_ok"),
                    lambda d: (d["checks"]["symmetry_dataset_status_ok"], "symcomp_dataset_ok"),
                    lambda d: (d["checks"]["symmetry_training_status_ok"], "symcomp_training_ok"),
                    lambda d: (d["checks"]["same_paper_vae_architecture"], "symcomp_same_vae_arch"),
                    lambda d: (d["checks"]["same_paper_diffusion_architecture"], "symcomp_same_diff_arch"),
                    lambda d: (d["checks"]["same_projection_matrix_sha256"], "symcomp_same_projection"),
                    lambda d: (d["checks"]["window_count_doubled"], "symcomp_window_count_doubled"),
                    lambda d: (d["checks"]["token_count_doubled"], "symcomp_token_count_doubled"),
                    lambda d: (d["checks"]["both_runs_use_8gpu_dataparallel"], "symcomp_8gpu"),
                    lambda d: (
                        d["checks"]["validation_action_mse_improved_with_symmetry"],
                        "symcomp_val_action_improved",
                    ),
                    lambda d: (
                        d["checks"]["test_action_mse_improved_with_symmetry"],
                        "symcomp_test_action_improved",
                    ),
                    lambda d: (d["checks"]["diffusion_tau_metrics_finite"], "symcomp_tau_finite"),
                    lambda d: (
                        d["metrics"]["window_count_ratio"] == 2.0 and d["metrics"]["token_count_ratio"] == 2.0,
                        "symcomp_ratios_2",
                    ),
                    lambda d: (d["checks"]["comparison_tsv_written"], "symcomp_tsv"),
                    lambda d: (d["checks"]["comparison_npz_written"], "symcomp_npz"),
                    lambda d: (
                        d["checks"]["does_not_claim_closed_loop_ablation"],
                        "symcomp_no_closed_loop_claim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_official_teacher_rollout_dataset"],
                        "symcomp_no_teacher_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "symcomp_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "released_data_metrics_summary",
                "res/tables/released_data_metrics_summary/released_data_metrics_summary.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["source_csv_count"] == 10, "released_metrics_source_csvs_10"),
                    lambda d: (d["metrics"]["ablation_row_count"] == 30, "released_metrics_ablation_rows_30"),
                    lambda d: (d["metrics"]["grf_row_count"] == 12, "released_metrics_grf_rows_12"),
                    lambda d: (d["metrics"]["imu_row_count"] == 10, "released_metrics_imu_rows_10"),
                    lambda d: (d["checks"]["all_expected_source_csvs_present"], "released_metrics_sources_exist"),
                    lambda d: (d["checks"]["source_hashes_recorded"], "released_metrics_hashes_recorded"),
                    lambda d: (
                        d["checks"]["ablation_rows_cover_five_groups"],
                        "released_metrics_five_ablation_groups",
                    ),
                    lambda d: (d["checks"]["grf_rows_cover_four_groups"], "released_metrics_four_grf_groups"),
                    lambda d: (
                        d["checks"]["imu_rows_cover_orientation_accel_angular_velocity"],
                        "released_metrics_imu_signals",
                    ),
                    lambda d: (d["checks"]["metrics_are_finite"], "released_metrics_finite"),
                    lambda d: (d["checks"]["does_not_claim_training"], "released_metrics_no_training_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "released_metrics_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "released_data_statistical_audit",
                "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["source_csv_count"] == 10, "released_stats_source_csvs_10"),
                    lambda d: (
                        d["metrics"]["ablation_comparison_rows"] == 30,
                        "released_stats_ablation_rows_30",
                    ),
                    lambda d: (d["metrics"]["grf_ci_rows"] == 12, "released_stats_grf_rows_12"),
                    lambda d: (d["metrics"]["imu_ci_rows"] == 11, "released_stats_imu_rows_11"),
                    lambda d: (
                        d["checks"]["all_expected_source_csvs_present"],
                        "released_stats_sources_exist",
                    ),
                    lambda d: (d["checks"]["source_hashes_recorded"], "released_stats_hashes_recorded"),
                    lambda d: (
                        d["checks"]["all_confidence_intervals_finite"],
                        "released_stats_ci_finite",
                    ),
                    lambda d: (
                        d["checks"]["all_ablation_effect_sizes_finite"],
                        "released_stats_effect_sizes_finite",
                    ),
                    lambda d: (
                        d["checks"]["imu_norm_claim_metrics_present"],
                        "released_stats_imu_claim_metrics",
                    ),
                    lambda d: (d["checks"]["does_not_claim_training"], "released_stats_no_training_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "released_stats_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "level_a_released_data_suite",
                "res/level_a/released_data_suite/level_a_released_data_suite.json",
                [
                    status_ok,
                    lambda d: (d["step_count"] == 6, "level_a_suite_step_count_6"),
                    lambda d: (d["pass_count"] == 6, "level_a_suite_pass_count_6"),
                    lambda d: (d["checks"]["all_steps_pass"], "level_a_suite_all_steps_pass"),
                    lambda d: (d["checks"]["paper_table_rows_58"], "level_a_suite_table_rows_58"),
                    lambda d: (d["checks"]["paper_table_mismatch_zero"], "level_a_suite_table_mismatch_zero"),
                    lambda d: (d["checks"]["released_panel_rows_15"], "level_a_suite_panel_rows_15"),
                    lambda d: (d["checks"]["released_panel_failures_zero"], "level_a_suite_panel_failures_zero"),
                    lambda d: (
                        d["checks"]["released_metrics_source_csv_count_10"],
                        "level_a_suite_source_csv_count_10",
                    ),
                    lambda d: (d["checks"]["released_metrics_finite"], "level_a_suite_metrics_finite"),
                    lambda d: (d["checks"]["released_stats_ci_rows_present"], "level_a_suite_stats_ci_rows"),
                    lambda d: (
                        d["checks"]["comparison_does_not_claim_goal_complete"],
                        "level_a_suite_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "level_a_suite_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "metrics_catalog",
                "res/metrics/metrics_catalog/metrics_catalog.json",
                [
                    status_ok,
                    lambda d: (d["metrics"]["source_count"] == 23, "metrics_catalog_sources_23"),
                    lambda d: (d["metrics"]["released_data_source_count"] >= 4, "metrics_catalog_released_sources"),
                    lambda d: (d["metrics"]["formula_api_source_count"] >= 2, "metrics_catalog_formula_api_sources"),
                    lambda d: (d["metrics"]["debug_only_source_count"] >= 5, "metrics_catalog_debug_sources"),
                    lambda d: (
                        d["metrics"]["blocked_boundary_source_count"] >= 2,
                        "metrics_catalog_blocked_sources",
                    ),
                    lambda d: (d["checks"]["all_metric_sources_exist"], "metrics_catalog_sources_exist"),
                    lambda d: (d["checks"]["all_metric_sources_hashed"], "metrics_catalog_sources_hashed"),
                    lambda d: (d["checks"]["comparison_source_present"], "metrics_catalog_comparison_source"),
                    lambda d: (d["checks"]["coverage_sources_present"], "metrics_catalog_coverage_sources"),
                    lambda d: (d["checks"]["formula_api_sources_present"], "metrics_catalog_formula_api_check"),
                    lambda d: (d["checks"]["does_not_claim_training"], "metrics_catalog_no_training_claim"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "metrics_catalog_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "guidance_task_coverage_audit",
                "res/guidance_task_coverage/guidance_task_coverage_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 30, "guidance_task_rows_30"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "guidance_task_evidence_exists"),
                    lambda d: (d["checks"]["six_tasks_mapped"], "guidance_task_six_tasks"),
                    lambda d: (d["checks"]["five_requirements_per_task"], "guidance_task_five_requirements"),
                    lambda d: (
                        d["checks"]["joystick_has_with_without_and_scale_sweep_offline_evidence"],
                        "guidance_task_joystick_offline_coverage",
                    ),
                    lambda d: (d["checks"]["inpainting_mask_metrics_recorded"], "guidance_task_inpainting_metrics"),
                    lambda d: (d["checks"]["offline_guidance_eval_linked"], "guidance_task_offline_eval_linked"),
                    lambda d: (
                        d["checks"]["task_metric_audit_linked_to_all_guided_quantitative_rows"],
                        "guidance_task_metric_full_split_linked",
                    ),
                    lambda d: (
                        d["checks"]["full_split_result_table_linked_to_all_guided_quantitative_rows"],
                        "guidance_task_result_table_full_split_linked",
                    ),
                    lambda d: (d["checks"]["all_video_requirements_recorded_blocked"], "guidance_task_videos_blocked"),
                    lambda d: (
                        d["checks"]["closed_loop_rollouts_not_overclaimed"],
                        "guidance_task_no_rollout_overclaim",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "guidance_task_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "guidance_task_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "progress_report_audit",
                "res/progress_report_audit/progress_report_audit.json",
                [
                    status_ok,
                    lambda d: (d["required_field_count"] == 21, "progress_required_fields_21"),
                    lambda d: (d["missing_count"] == 0, "progress_missing_zero"),
                    lambda d: (
                        d["checks"]["all_required_fields_present_and_nonempty"],
                        "progress_required_fields_nonempty",
                    ),
                    lambda d: (
                        d["checks"]["all_key_progress_markers_present"],
                        "progress_key_markers_present",
                    ),
                    lambda d: (
                        d["checks"]["master_audit_progression_recorded"],
                        "progress_master_audit_progression",
                    ),
                    lambda d: (
                        d["checks"]["records_incomplete_boundary"],
                        "progress_records_incomplete_boundary",
                    ),
                    lambda d: (
                        d["checks"]["records_missing_checkpoints_and_videos"],
                        "progress_records_missing_checkpoints_videos",
                    ),
                    lambda d: (
                        d["checks"]["atomic_write_used"],
                        "progress_atomic_write_used",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "progress_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "completion_matrix_status_audit",
                "res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] >= 110, "completion_matrix_rows_ge_110"),
                    lambda d: (d["invalid_status_count"] == 0, "completion_matrix_invalid_status_zero"),
                    lambda d: (d["invalid_row_count"] == 0, "completion_matrix_invalid_rows_zero"),
                    lambda d: (
                        d["checks"]["all_requirement_rows_have_three_columns"],
                        "completion_matrix_three_columns",
                    ),
                    lambda d: (
                        d["checks"]["all_statuses_are_allowed_enum"],
                        "completion_matrix_allowed_statuses",
                    ),
                    lambda d: (
                        d["checks"]["row_count_matches_status_counts"],
                        "completion_matrix_status_counts_cover_rows",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "completion_matrix_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "completion_matrix_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "project_boundary_audit",
                "res/project_boundary_audit/project_boundary_audit.json",
                [
                    status_ok,
                    lambda d: (d["row_count"] == 8, "project_boundary_rows_8"),
                    lambda d: (d["failed_count"] == 0, "project_boundary_failures_zero"),
                    lambda d: (
                        d["checks"]["download_toplevel_allowlist_passes"],
                        "project_boundary_download_allowlist",
                    ),
                    lambda d: (
                        d["checks"]["supplemental_manifest_exists"],
                        "project_boundary_supplemental_manifest",
                    ),
                    lambda d: (
                        d["checks"]["project_env_cache_tmp_redirects"],
                        "project_boundary_cache_tmp_redirects",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "project_boundary_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "goal_traceability_audit",
                "res/goal_traceability/goal_traceability_audit.json",
                [
                    status_ok,
                    lambda d: (d["goal_line_count"] >= 1950, "goal_line_count_indexed"),
                    lambda d: (d["heading_count"] >= 80, "goal_headings_indexed"),
                    lambda d: (d["trace_row_count"] >= 24, "goal_trace_rows_present"),
                    lambda d: (len(d["missing_evidence_rows"]) == 0, "goal_trace_evidence_paths_exist"),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "goal_trace_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "goal_directive_index_audit",
                "res/goal_directive_index/goal_directive_index_audit.json",
                [
                    status_ok,
                    lambda d: (d["line_count"] == 1951, "goal_directive_line_count_1951"),
                    lambda d: (d["heading_count"] == 80, "goal_directive_heading_count_80"),
                    lambda d: (d["directive_row_count"] >= 250, "goal_directive_rows_ge_250"),
                    lambda d: (d["tag_counts"]["prohibition"] >= 40, "goal_directive_prohibition_rows"),
                    lambda d: (d["tag_counts"]["deliverable"] >= 70, "goal_directive_deliverable_rows"),
                    lambda d: (d["tag_counts"]["execution"] >= 120, "goal_directive_execution_rows"),
                    lambda d: (
                        d["checks"]["line_count_matches_goal_traceability"],
                        "goal_directive_line_count_matches_trace",
                    ),
                    lambda d: (
                        d["checks"]["heading_count_matches_goal_traceability"],
                        "goal_directive_heading_count_matches_trace",
                    ),
                    lambda d: (
                        d["checks"]["traceability_has_no_missing_evidence"],
                        "goal_directive_trace_no_missing_evidence",
                    ),
                    lambda d: (
                        d["checks"]["requirement_matrix_has_no_missing_evidence"],
                        "goal_directive_matrix_no_missing_evidence",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "goal_directive_keeps_goal_incomplete",
                    ),
                ],
            ),
            check_json_artifact(
                "goal_requirement_matrix_audit",
                "res/goal_requirement_matrix/goal_requirement_matrix_audit.json",
                [
                    status_ok,
                    lambda d: (d["goal_line_count"] >= 1950, "goal_matrix_goal_line_count"),
                    lambda d: (d["requirement_row_count"] >= 25, "goal_matrix_requirement_rows"),
                    lambda d: (d["checks"]["all_evidence_paths_exist"], "goal_matrix_evidence_paths_exist"),
                    lambda d: (
                        d["checks"]["partial_or_blocked_items_remain"],
                        "goal_matrix_partial_blocked_remain",
                    ),
                    lambda d: (
                        d["checks"]["evaluation_requirement_links_formula_api_metric_contracts"],
                        "goal_matrix_evaluation_formula_api_links",
                    ),
                    lambda d: (
                        d["checks"]["evaluation_requirement_references_goal_checkpoint_values"],
                        "goal_matrix_evaluation_goal_checkpoint_values",
                    ),
                    lambda d: (
                        d["checks"]["formula_api_metric_contracts_pass"],
                        "goal_matrix_formula_api_metric_contracts_pass",
                    ),
                    lambda d: (
                        d["checks"]["result_claim_formula_api_links_preserved"],
                        "goal_matrix_results_claim_formula_links",
                    ),
                    lambda d: (
                        d["checks"]["evaluation_requirement_still_partial"],
                        "goal_matrix_evaluation_still_partial",
                    ),
                    lambda d: (
                        d["checks"]["does_not_claim_goal_complete"],
                        "goal_matrix_no_goal_complete_claim",
                    ),
                    lambda d: (
                        d["interpretation"]["goal_complete"] is False,
                        "goal_matrix_keeps_goal_incomplete",
                    ),
                ],
            ),
        ]
    )

    counts, matrix_rows = matrix_counts()
    json_path = OUT / "reproduction_master_audit.json"
    tsv_path = OUT / "reproduction_master_audit.tsv"
    passed = [row for row in artifacts if row["passed"]]
    failed = [row for row in artifacts if not row["passed"]]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "audit",
        "scope": "master audit for current BeyondMimic reproduction evidence",
        "completion_matrix_counts": dict(sorted(counts.items())),
        "completion_matrix_rows": matrix_rows,
        "artifact_count": len(artifacts),
        "artifact_pass_count": len(passed),
        "artifact_fail_count": len(failed),
        "failed_artifacts": failed,
        "artifacts": artifacts,
        "interpretation": {
            "pending_items": counts.get("pending", 0),
            "blocked_items": counts.get("blocked", 0),
            "partial_items": counts.get("partial", 0),
            "goal_complete": False,
            "why_not_complete": (
                "The evidence set is internally consistent, but completion matrix still contains partial/blocked/"
                "out_of_scope items for live Kit tracking, teacher rollouts, true DAgger, trained Level C checkpoints, "
                "Fig. 5/6 paper reproduction, and real robot deployment."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, artifacts)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
