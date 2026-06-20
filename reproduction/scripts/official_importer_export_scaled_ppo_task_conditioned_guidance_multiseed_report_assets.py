#!/usr/bin/env python3
"""Create report assets for scaled-PPO importer-export multi-seed guidance."""

from __future__ import annotations

import json
import os
from pathlib import Path

import official_importer_export_full_bundle_task_conditioned_guidance_multiseed_report_assets as base


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SUMMARY_JSON = (
    ROOT
    / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
    "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed"
ASSET_JSON = OUT / "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    base.SUMMARY_JSON = SUMMARY_JSON
    base.OUT = OUT
    base.main()
    source = load_json(OUT / "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json")
    source["status"] = "ok_official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets"
    source["source_summary"] = str(SUMMARY_JSON)
    source["claim_level"] = "local_virtual_official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets"
    source["checks"]["summary_status_ok"] = (
        load_json(SUMMARY_JSON)["status"]
        == "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
    )
    source["checks"]["uses_scaled_ppo_chain"] = all(
        load_json(SUMMARY_JSON)["checks"].get(key, False)
        for key in [
            "uses_scaled_ppo_training_run",
            "uses_scaled_ppo_checkpoint_eval",
            "uses_scaled_ppo_vae",
            "uses_scaled_ppo_denoiser",
            "uses_scaled_ppo_offline_guidance",
        ]
    )
    source["interpretation"] = {
        "goal_complete": False,
        "claim_level": source["claim_level"],
        "why_not_paper_level": (
            "The source rollouts use local scaled PPO/VAE/denoiser checkpoints and proxy task costs. "
            "They are useful report/PPT evidence but not official BeyondMimic Fig. 5/Fig. 6 metrics, "
            "TensorRT deployment, or real-robot validation."
        ),
    }
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official importer-export scaled PPO task-conditioned guidance multiseed assets",
                "",
                "These assets summarize local virtual multi-seed task-conditioned receding latent-guidance ",
                "rollouts over the 40-motion public bundle using the official-importer-export G1 USDA path ",
                "and the local iteration-999 scaled PPO/VAE/denoiser chain.",
                "",
                "Claim level: local virtual closed-loop guidance evidence only. Not paper Fig. 5/Fig. 6 ",
                "reproduction, not official BeyondMimic checkpoints, not TensorRT deployment, and not ",
                "real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source["assets"]["readme"] = str(readme)
    write_json(ASSET_JSON, source)
    if os.environ.get("BM_KEEP_COMPAT_ASSET_JSON", "1") == "1":
        write_json(OUT / "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json", source)
    print(json.dumps({"status": source["status"], "json": str(ASSET_JSON)}, sort_keys=True))


if __name__ == "__main__":
    main()
