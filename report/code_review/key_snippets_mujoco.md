# MuJoCo Code Snippets

## MuJoCo action-to-PD rendering

File: `mujoco_mp4/scripts/mujoco_pd_control_video.py`
Function/Class: `def .*render|def .*step|def .*action`
Purpose: Loads G1 model, maps action to joint setpoints, steps MuJoCo, and renders MP4.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Local simulation visualization path; not official Isaac rendering.

```python
0071:     "guided_latent_control": {
0072:         "target_source": "ik_from_guided_latent_body_trace",
0073:         "trace_spec": "guided_latent",
0074:         "claim": "MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab guided-latent body trace; not native MuJoCo guided controller",
0075:     },
0076: }
0077: 
0078: 
0079: def load_action_rows() -> list[dict[str, Any]]:
0080:     payload = json.loads(ACTION_SCALE_AUDIT.read_text(encoding="utf-8"))
0081:     rows = payload.get("joint_rows", [])
0082:     if len(rows) != 29:
0083:         raise ValueError(f"Expected 29 action-scale rows, got {len(rows)}")
0084:     return rows
0085: 
0086: 
0087: def add_or_update_option(root: ET.Element) -> None:
0088:     option = root.find("option")
0089:     if option is None:
0090:         option = ET.Element("option")
0091:         root.insert(1, option)
0092:     option.set("timestep", os.environ.get("BM_MUJOCO_PD_TIMESTEP", "0.005"))
0093:     option.set("gravity", "0 0 -9.81")
0094:     option.set("integrator", os.environ.get("BM_MUJOCO_PD_INTEGRATOR", "implicitfast"))
0095: 
0096: 
0097: def add_fixed_camera(root: ET.Element) -> None:
0098:     for cam in root.findall(".//camera"):
0099:         if cam.attrib.get("name") == PD_CAMERA:
0100:             break
0101:     else:
0102:         world = root.find("worldbody")
0103:         if world is None:
0104:             raise ValueError("Cannot add camera: no worldbody in MuJoCo XML")
0105:         cam = ET.SubElement(world, "camera", {"name": PD_CAMERA})
0106:     cam.set("mode", "fixed")
0107:     cam.set("pos", os.environ.get("BM_MUJOCO_PD_CAMERA_POS", "-0.35 -4.80 1.75"))
0108:     cam.set("xyaxes", os.environ.get("BM_MUJOCO_PD_CAMERA_XYAXES", "1 0 0 0 0.32 0.947"))
0109:     cam.set("fovy", os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY", "48"))
```

## Continuous 5/6 video suite wrapper

File: `reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py`
Function/Class: `def patch_artifact_bindings`
Purpose: Binds the latest multi-source teacher/VAE/denoiser artifacts and filters continuous segments.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Prevents reset-spliced video artifacts and records honest claim level.

```python
0066:     return datetime.now(timezone.utc).isoformat()
0067: 
0068: 
0069: def write_json(path: Path, payload: dict[str, Any]) -> None:
0070:     path.parent.mkdir(parents=True, exist_ok=True)
0071:     path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
0072: 
0073: 
0074: def patch_artifact_bindings() -> None:
0075:     # Rebind globals used directly inside the continuous implementation.
0076:     base.OUT_ROOT = OUT_ROOT
0077:     base.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
0078:     base.BEST_TEACHER_SWEEP_JSON = BEST_TEACHER_SWEEP_JSON
0079:     base.MOTION_BUNDLE = MOTION_BUNDLE_NPZ
0080:     base.VAE_CKPT = VAE_CKPT
0081:     base.DENOISER_CKPT = DENOISER_CKPT
0082:     base.OLD_FAILURE_AUDIT = FRESH_AUDIT
0083: 
0084:     # Rebind globals used by imported helper functions whose __globals__ live
0085:     # in the paper-contract module.
0086:     paper_base.OUT_ROOT = OUT_ROOT
0087:     paper_base.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
0088:     paper_base.BEST_TEACHER_SWEEP_JSON = BEST_TEACHER_SWEEP_JSON
0089:     paper_base.MOTION_BUNDLE = MOTION_BUNDLE_NPZ
0090:     paper_base.VAE_CKPT = VAE_CKPT
0091: 
0092: 
0093: def source_motion_for_segment(segment: dict[str, Any]) -> dict[str, Any] | None:
0094:     audit = json.loads(MOTION_BUNDLE_AUDIT.read_text(encoding="utf-8"))
0095:     start = int(segment["motion_time_step_start"])
0096:     end = int(segment["motion_time_step_end"])
0097:     for row in audit.get("rows", []):
0098:         if int(row["start_frame"]) <= start and end < int(row["end_frame_exclusive"]):
0099:             return {
0100:                 "motion": row.get("motion"),
0101:                 "source_family": row.get("source_family"),
0102:                 "source_kind": row.get("source_kind"),
0103:                 "source_path": row.get("source_path"),
0104:                 "start_frame": row.get("start_frame"),
```
