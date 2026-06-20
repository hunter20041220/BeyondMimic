#!/usr/bin/env python3
"""Run the official-importer-export full-bundle VAE/denoiser ONNX async audit."""

from __future__ import annotations

import os


os.environ.setdefault("BM_OFFICIAL_CSV_LOOP_ONNX_VARIANT", "importer_export_full_bundle")

from level_c_official_csv_loop_vae_denoiser_onnx_async_audit import main  # noqa: E402


if __name__ == "__main__":
    main()
