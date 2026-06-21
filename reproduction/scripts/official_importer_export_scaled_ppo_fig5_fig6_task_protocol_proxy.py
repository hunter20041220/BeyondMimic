#!/usr/bin/env python3
"""Build scaled PPO local Fig. 5/6 task-protocol proxy metrics."""

from __future__ import annotations

import os


os.environ.setdefault("BM_FIG56_TASK_PROTOCOL_VARIANT", "scaled_ppo")

from official_importer_export_fig5_fig6_task_protocol_proxy import main  # noqa: E402


if __name__ == "__main__":
    main()
