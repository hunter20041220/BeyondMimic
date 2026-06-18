
import os
import runpy
import signal
import sys
import traceback
import types
from pathlib import Path

OFFICIAL_CSV_TO_NPZ = Path(os.environ["BM_OFFICIAL_CSV_TO_NPZ"])
OUTPUT_NPZ = Path(os.environ["BM_OUTPUT_NPZ"])
OUTPUT_NAME = os.environ["BM_OUTPUT_NAME"]
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])
ENRICHED_USD = Path(os.environ["BM_ENRICHED_USD"])


def _sigterm_handler(signum, frame):
    print(f"BM_SENTINEL:received_signal={signum}", flush=True)
    traceback.print_stack(frame)
    print("BM_SENTINEL:received_signal_ignored_for_bounded_probe", flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)

import numpy as _np

_real_savez = _np.savez


def _redirected_savez(file, *args, **kwargs):
    if str(file) == "/tmp/motion.npz":
        OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
        print(f"BM_SENTINEL:np_savez_redirect=/tmp/motion.npz->{OUTPUT_NPZ}", flush=True)
        return _real_savez(str(OUTPUT_NPZ), *args, **kwargs)
    return _real_savez(file, *args, **kwargs)


_np.savez = _redirected_savez


class _FakeRun:
    def __init__(self, project, name):
        self.project = project
        self.name = name
        print(f"BM_SENTINEL:fake_wandb_init=project:{project},name:{name}", flush=True)

    def log_artifact(self, artifact_or_path, name, type):
        print(
            f"BM_SENTINEL:fake_wandb_log_artifact=artifact:{artifact_or_path},name:{name},type:{type}",
            flush=True,
        )
        if name != OUTPUT_NAME:
            raise RuntimeError(f"Unexpected artifact name {name!r} != {OUTPUT_NAME!r}")
        if not OUTPUT_NPZ.is_file():
            raise RuntimeError(f"Redirected output NPZ missing before fake log_artifact: {OUTPUT_NPZ}")
        return {"artifact_or_path": str(OUTPUT_NPZ), "name": name, "type": type}

    def link_artifact(self, artifact, target_path):
        print(f"BM_SENTINEL:fake_wandb_link_artifact=target:{target_path}", flush=True)
        return None


def _fake_init(project, name):
    return _FakeRun(project=project, name=name)


fake_wandb = types.ModuleType("wandb")
fake_wandb.init = _fake_init
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
                print(f"BM_SENTINEL:official_csv_loop_is_running_call={self._calls}", flush=True)
            return self._app.is_running()
        print(f"BM_SENTINEL:official_csv_loop_complete={MAX_STEPS}", flush=True)
        return False

    def close(self, *args, **kwargs):
        print("BM_SENTINEL:simulation_app_close_called", flush=True)
        return self._app.close(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._app, name)


def patch_g1_cfg_to_enriched_usd():
    import isaaclab.sim as sim_utils
    import whole_body_tracking.robots.g1 as g1_module

    cfg = g1_module.G1_CYLINDER_CFG.copy()
    cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ENRICHED_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )
    g1_module.G1_CYLINDER_CFG = cfg
    print(f"BM_SENTINEL:g1_cfg_patched_to_enriched_usd={ENRICHED_USD}", flush=True)


class _BoundedAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        print("BM_SENTINEL:add_app_launcher_args", flush=True)
        return _RealAppLauncher.add_app_launcher_args(parser)

    def __init__(self, args):
        print("BM_SENTINEL:before_real_app_launcher", flush=True)
        args.headless = True
        args.enable_cameras = False
        args.device = os.environ["BM_DEVICE"]
        args.multi_gpu = False
        args.kit_args = os.environ.get("BM_KIT_ARGS", "")
        self._real = _RealAppLauncher(args)
        self.app = _BoundedSimulationApp(self._real.app)
        print("BM_SENTINEL:after_real_app_launcher", flush=True)
        patch_g1_cfg_to_enriched_usd()

    def __getattr__(self, name):
        return getattr(self._real, name)


isaaclab_app.AppLauncher = _BoundedAppLauncher
sys.argv = [
    str(OFFICIAL_CSV_TO_NPZ),
    "--input_file",
    os.environ["BM_INPUT_CSV"],
    "--input_fps",
    "30",
    "--frame_range",
    "1",
    "180",
    "--output_name",
    OUTPUT_NAME,
    "--output_fps",
    "50",
    "--headless",
    "--device",
    os.environ["BM_DEVICE"],
]

print(f"BM_SENTINEL:before_runpy={OFFICIAL_CSV_TO_NPZ}", flush=True)
try:
    runpy.run_path(str(OFFICIAL_CSV_TO_NPZ), run_name="__main__")
    print("BM_SENTINEL:after_runpy", flush=True)
except BaseException as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    traceback.print_exc()
    raise
