#!/usr/bin/env python3
"""Create report assets for scaled official-importer-export downstream VAE/diffusion runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/official_importer_export_full_bundle_downstream_report_assets.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_importer_downstream_assets_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report asset base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_base_module()
    module.VAE_JSON = (
        ROOT
        / "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json"
    )
    module.STATE_LATENT_JSON = (
        ROOT
        / "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json"
    )
    module.DIFFUSION_JSON = (
        ROOT
        / "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
        "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
    )
    module.OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_downstream"
    module.STATUS = "ok_official_importer_export_scaled_ppo_downstream_assets"
    module.CLAIM_LEVEL = "local_virtual_official_importer_export_scaled_ppo_downstream_report_asset"
    module.README_TITLE = "Official-Importer-Export Scaled PPO Downstream Training Assets"
    module.README_DESCRIPTION = (
        "These assets summarize the local scaled official-importer-export teacher-rollout VAE, state-latent "
        "dataset, and state-latent denoiser training chain."
    )
    module.EXPECTED_VAE_STATUS = "ok_official_importer_export_scaled_ppo_teacher_rollout_vae_training"
    module.EXPECTED_STATE_LATENT_STATUS = (
        "ok_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset"
    )
    module.EXPECTED_DIFFUSION_STATUS = "ok_official_importer_export_scaled_ppo_state_latent_diffusion_training"
    module.main()


if __name__ == "__main__":
    main()
