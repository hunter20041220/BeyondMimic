#!/usr/bin/env python3
"""Build a machine-readable resolved reproduction config manifest.

The manifest consolidates paper/source hyperparameters and local execution
schema from existing audits. It is a configuration contract, not a claim that
long training or paper-level evaluation has completed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/config"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    def emit(obj: Any, indent: int = 0) -> list[str]:
        pad = " " * indent
        if isinstance(obj, dict):
            lines: list[str] = []
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(emit(value, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {yaml_scalar(value)}")
            return lines
        if isinstance(obj, list):
            lines = []
            for value in obj:
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(emit(value, indent + 2))
                else:
                    lines.append(f"{pad}- {yaml_scalar(value)}")
            return lines
        return [f"{pad}{yaml_scalar(obj)}"]

    path.write_text("\n".join(emit(data)) + "\n", encoding="utf-8")


def flatten(prefix: str, obj: Any, rows: list[dict[str, str]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), value, rows)
    elif isinstance(obj, list):
        rows.append({"key": prefix, "value": json.dumps(obj, sort_keys=True), "type": "list"})
    else:
        rows.append({"key": prefix, "value": "" if obj is None else str(obj), "type": type(obj).__name__})


def build_manifest() -> dict[str, Any]:
    tracking = load_json("res/tracking/smoke_config_audit/tracking_config_audit.json")
    diffusion_schedule = load_json("res/level_c/training_schedule_probe/level_c_training_schedule_probe.json")
    vae_contract = load_json("res/level_c/vae_contract_audit/level_c_vae_contract_audit.json")
    gpu_resource = load_json("res/setup/gpu_resource_audit/gpu_resource_audit.json")
    run_management = load_json("res/run_management_audit/run_management_audit.json")
    reimpl = load_json("res/code/reimpl_package_audit/reimpl_package_audit.json")
    table_values = load_json("res/paper_table_values/paper_table_value_audit.json")

    manifest = {
        "status": "ok",
        "experiment_type": "resolved_config_manifest",
        "scope": "paper/source-aligned config contract for current BeyondMimic reproduction state",
        "tracking": {
            "control_frequency_hz": tracking["tracking_env"]["control_frequency_hz"],
            "sim_dt": tracking["tracking_env"]["sim_dt"],
            "decimation": tracking["tracking_env"]["decimation"],
            "num_envs": tracking["tracking_env"]["num_envs"],
            "episode_length_s": tracking["tracking_env"]["episode_length_s"],
            "actor_hidden_dims": tracking["ppo"]["actor_hidden_dims"],
            "critic_hidden_dims": tracking["ppo"]["critic_hidden_dims"],
            "activation": tracking["ppo"]["activation"],
            "ppo": {
                "num_steps_per_env": tracking["ppo"]["num_steps_per_env"],
                "max_iterations": tracking["ppo"]["max_iterations"],
                "learning_rate": tracking["ppo"]["learning_rate"],
                "clip_param": tracking["ppo"]["clip_param"],
                "entropy_coef": tracking["ppo"]["entropy_coef"],
                "value_loss_coef": tracking["ppo"]["value_loss_coef"],
                "gamma": tracking["ppo"]["gamma"],
                "gae_lambda": tracking["ppo"]["lam"],
                "desired_kl": tracking["ppo"]["desired_kl"],
                "num_learning_epochs": tracking["ppo"]["num_learning_epochs"],
                "num_mini_batches": tracking["ppo"]["num_mini_batches"],
                "schedule": tracking["ppo"]["schedule"],
            },
            "anchor_body": tracking["g1_tracking_target_bodies"]["anchor_body_name"],
            "target_body_count": tracking["g1_tracking_target_bodies"]["body_count"],
            "target_bodies": tracking["g1_tracking_target_bodies"]["body_names"],
            "termination_bodies": tracking["termination_bodies"]["termination_body_names"],
            "adaptive_sampling": tracking["adaptive_sampling_code"],
        },
        "vae": {
            "latent_dim": 32,
            "student_hidden_dims": [2048, 1024, 512],
            "teacher_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "learning_rate": 5e-4,
            "gradient_accumulation_steps": 15,
            "kl_weight": 0.01,
            "contract_rows": vae_contract["metrics"]["row_count"],
            "contract_failed_rows": vae_contract["metrics"]["failed_row_count"],
            "debug_only": True,
        },
        "diffusion": {
            "history_steps": 4,
            "future_horizon_steps": 16,
            "sequence_length": 21,
            "transformer_embed_dim": 512,
            "attention_heads": 8,
            "transformer_layers": 6,
            "denoising_steps": 20,
            "training": diffusion_schedule["settings"],
            "debug_only": True,
        },
        "gpu_resource_policy": {
            "target_memory_gib_per_gpu": "18-22",
            "target_gpu_util_percent": ">=90 during real training steady state",
            "current_snapshot_rows": gpu_resource["rows_written"],
            "current_gpu_count": gpu_resource["gpu_count"],
            "gpu_metrics_csv": gpu_resource["outputs"]["gpu_metrics_csv"],
            "no_artificial_load": gpu_resource["checks"]["does_not_create_artificial_load"],
            "no_power_or_clock_changes": gpu_resource["checks"]["does_not_modify_power_or_clocks"],
        },
        "run_management": {
            "run_id_pattern": "{stage}_{method}_{motion}_{config}_{seed}_{YYYYMMDD_HHMMSS}",
            "allowed_statuses": ["QUEUED", "RUNNING", "SUCCESS", "FAILED", "FAILED_OOM", "INTERRUPTED", "INVALID"],
            "diagnostic_run_id": run_management["run_id"],
            "diagnostic_run_status": "INVALID",
            "required_files_and_dirs_pass": run_management["checks"]["all_required_files_exist"]
            and run_management["checks"]["all_required_dirs_exist"],
        },
        "code": {
            "reimplementation_package_root": reimpl["source_root"],
            "python_file_count": reimpl["python_file_count"],
            "checked_formula_api_symbols": reimpl["symbol_row_count"],
        },
        "source_evidence": {
            "paper_parameter_map": str(ROOT / "reproduction/docs/paper_parameter_map.md"),
            "paper_table_value_rows": table_values["counts"]["total_rows"],
            "paper_table_mismatch_rows": table_values["counts"]["mismatch_rows"],
        },
        "checks": {
            "tracking_ppo_values_present": tracking["ppo"]["max_iterations"] == 30000
            and tracking["ppo"]["num_steps_per_env"] == 24,
            "tracking_target_body_count_14": tracking["g1_tracking_target_bodies"]["body_count"] == 14,
            "vae_contract_passes": vae_contract["checks"]["all_contract_rows_pass"],
            "diffusion_schedule_passes": all(diffusion_schedule["checks"].values()),
            "gpu_schema_available": gpu_resource["checks"]["gpu_metrics_csv_written"],
            "run_schema_available": run_management["checks"]["all_required_files_exist"]
            and run_management["checks"]["does_not_mark_success_without_training"],
            "reimpl_package_api_available": reimpl["checks"]["all_expected_symbols_exist"],
            "paper_table_values_have_no_mismatches": table_values["counts"]["mismatch_rows"] == 0,
            "does_not_claim_training_or_paper_results": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The resolved config manifest consolidates current paper/source-aligned settings, but full Kit "
                "training, DAgger rollouts, trained checkpoints, Fig. 5/6 evaluation, and deployment remain incomplete."
            ),
        },
        "outputs": {
            "json": str(OUT / "resolved_reproduction_config.json"),
            "yaml": str(OUT / "resolved_reproduction_config.yaml"),
            "csv": str(OUT / "resolved_reproduction_config.csv"),
        },
    }
    return manifest


def write_outputs(manifest: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resolved_reproduction_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_yaml(OUT / "resolved_reproduction_config.yaml", manifest)
    rows: list[dict[str, str]] = []
    flatten("", manifest, rows)
    with (OUT / "resolved_reproduction_config.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "value", "type"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = build_manifest()
    write_outputs(manifest)
    print(json.dumps({"status": manifest["status"], "json": manifest["outputs"]["json"]}))


if __name__ == "__main__":
    main()
