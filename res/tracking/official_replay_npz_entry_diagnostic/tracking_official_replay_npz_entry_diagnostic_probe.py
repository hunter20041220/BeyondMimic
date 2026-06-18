
import os
import runpy
import sys
import traceback
import types
from pathlib import Path

OFFICIAL_REPLAY = Path(os.environ["BM_OFFICIAL_REPLAY"])
FAKE_ARTIFACT_DIR = Path(os.environ["BM_FAKE_ARTIFACT_DIR"])
REGISTRY_NAME = os.environ["BM_FAKE_REGISTRY_NAME"]
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])


class _FakeArtifact:
    def download(self):
        print(f"BM_SENTINEL:fake_wandb_download={FAKE_ARTIFACT_DIR}", flush=True)
        return str(FAKE_ARTIFACT_DIR)


class _FakeApi:
    def artifact(self, name):
        print(f"BM_SENTINEL:fake_wandb_artifact_request={name}", flush=True)
        if name != REGISTRY_NAME:
            raise RuntimeError(f"Unexpected fake registry name: {name!r} != {REGISTRY_NAME!r}")
        return _FakeArtifact()


fake_wandb = types.ModuleType("wandb")
fake_wandb.Api = _FakeApi
sys.modules["wandb"] = fake_wandb

import isaaclab.app as isaaclab_app

_RealAppLauncher = isaaclab_app.AppLauncher


class _BoundedSimulationApp:
    def __init__(self, app):
        self._app = app
        self._calls = 0

    def is_running(self):
        self._calls += 1
        if self._calls <= MAX_STEPS:
            if self._calls == 1 or self._calls % 50 == 0 or self._calls == MAX_STEPS:
                print(f"BM_SENTINEL:bounded_is_running_step={self._calls}", flush=True)
            return self._app.is_running()
        print(f"BM_SENTINEL:bounded_loop_complete={MAX_STEPS}", flush=True)
        return False

    def close(self, *args, **kwargs):
        print("BM_SENTINEL:simulation_app_close_called", flush=True)
        return self._app.close(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._app, name)


class _BoundedAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        print("BM_SENTINEL:add_app_launcher_args", flush=True)
        return _RealAppLauncher.add_app_launcher_args(parser)

    def __init__(self, args):
        print("BM_SENTINEL:before_real_app_launcher", flush=True)
        self._real = _RealAppLauncher(args)
        self.app = _BoundedSimulationApp(self._real.app)
        print("BM_SENTINEL:after_real_app_launcher", flush=True)

    def __getattr__(self, name):
        return getattr(self._real, name)


isaaclab_app.AppLauncher = _BoundedAppLauncher
sys.argv = [
    str(OFFICIAL_REPLAY),
    "--registry_name",
    REGISTRY_NAME.removesuffix(":latest"),
    "--headless",
    "--device",
    os.environ["BM_DEVICE"],
]

print(f"BM_SENTINEL:before_runpy={OFFICIAL_REPLAY}", flush=True)
try:
    runpy.run_path(str(OFFICIAL_REPLAY), run_name="__main__")
    print("BM_SENTINEL:after_runpy", flush=True)
except BaseException as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    traceback.print_exc()
    raise
