#!/usr/bin/env python3
"""Headless Isaac Sim / Isaac Lab smoke test."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    """Launch Isaac Sim headless, import core modules, and close cleanly."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaacsim.core.api  # noqa: F401
    import isaaclab  # noqa: F401
    import isaaclab_assets  # noqa: F401
    import isaaclab_rl  # noqa: F401

    print("headless_app_started", simulation_app.is_running())
    simulation_app.close()
    print("headless_app_closed")


if __name__ == "__main__":
    main()
