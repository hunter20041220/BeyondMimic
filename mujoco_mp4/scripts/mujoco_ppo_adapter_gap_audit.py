#!/usr/bin/env python3
"""Audit the gap between IsaacLab PPO checkpoints and native MuJoCo control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, ROOT, sha256, utc_now, write_json, write_tsv


CHECKPOINT = (
    ROOT
    / "res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training/"
    "resource_adjusted_ppo_20260621_121940_seed20260720/rank_0/model_999.pt"
)
SCHEMA = ROOT / "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json"
TASK_SMOKE = ROOT / "res/tracking/g1_official_importer_export_task_smoke/tracking_g1_official_importer_export_task_smoke.json"


def main() -> None:
    import torch

    out_dir = PKG / "res/adapter_gap"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["model_state_dict"]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    task_smoke = json.loads(TASK_SMOKE.read_text(encoding="utf-8"))
    actor_shapes = {key: list(value.shape) for key, value in state.items() if key.startswith("actor") or key == "std"}
    rows = [
        {
            "item": "checkpoint_actor_input_dim",
            "evidence": str(actor_shapes.get("actor.0.weight")),
            "status": "known",
            "notes": "Actor first layer is [512, 160].",
        },
        {
            "item": "checkpoint_action_dim",
            "evidence": str(actor_shapes.get("actor.6.weight")),
            "status": "known",
            "notes": "Actor output layer is [29, 128].",
        },
        {
            "item": "official_schema_policy_terms",
            "evidence": ",".join(row["term"] for row in schema["observation_rows"] if row["group"] == "policy"),
            "status": "known",
            "notes": "Static audit derives the IsaacLab policy observation contract.",
        },
        {
            "item": "native_mujoco_observation_manager",
            "evidence": "missing",
            "status": "gap",
            "notes": "MuJoCo package does not yet implement IsaacLab command manager, corruption/noise, normalization, reset and termination semantics.",
        },
        {
            "item": "native_mujoco_ppo_closed_loop_claim",
            "evidence": "not claimed",
            "status": "blocked_by_adapter_gap",
            "notes": "Current videos render existing IsaacLab closed-loop traces as MuJoCo mesh videos.",
        },
    ]
    checks = {
        "checkpoint_exists": CHECKPOINT.is_file(),
        "checkpoint_has_model_state_dict": "model_state_dict" in ckpt,
        "actor_input_dim_160": actor_shapes.get("actor.0.weight") == [512, 160],
        "actor_output_dim_29": actor_shapes.get("actor.6.weight") == [29, 128],
        "obs_norm_present": "obs_norm_state_dict" in ckpt,
        "schema_policy_dim_160": schema["metrics"]["policy_dimension"] == 160,
        "schema_action_dim_29": schema["metrics"]["action_dimension"] == 29,
        "task_smoke_policy_dim_160": task_smoke["metrics"]["policy_observation_dim"] == 160,
        "task_smoke_action_dim_29": task_smoke["metrics"]["action_dim"] == 29,
        "native_mujoco_adapter_complete": False,
        "does_not_claim_native_mujoco_ppo_closed_loop": True,
    }
    payload = {
        "status": "adapter_gap_documented",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_ppo_adapter_gap_audit",
        "claim_level": "Audit of why native MuJoCo PPO closed-loop is not claimed",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "schema_audit": str(SCHEMA),
        "task_smoke": str(TASK_SMOKE),
        "checkpoint_iter": int(ckpt.get("iter", -1)),
        "actor_shapes": actor_shapes,
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "native_mujoco_ppo_closed_loop_complete": False,
            "current_video_evidence": "MuJoCo mesh rendering of existing IsaacLab closed-loop traces",
            "next_step_for_native_control": "Implement and validate the IsaacLab command/observation/action/normalization adapter in MuJoCo before claiming native PPO control.",
        },
    }
    write_json(out_dir / "mujoco_ppo_adapter_gap_audit.json", payload)
    write_tsv(out_dir / "mujoco_ppo_adapter_gap_audit.tsv", rows, ["item", "evidence", "status", "notes"])
    print(json.dumps({"status": payload["status"], "json": str(out_dir / "mujoco_ppo_adapter_gap_audit.json")}))


if __name__ == "__main__":
    main()
