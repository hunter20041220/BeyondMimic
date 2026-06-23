# Full Code Snippets

## Official motion command / target computation

File: `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
Function/Class: `class .*Command|def .*command|def .*metrics`
Purpose: Loads reference motion and computes target/error signals for the tracking MDP.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Corresponds to Stage 1 motion tracking target and anchor-centered command logic.

```python
0053:     def body_lin_vel_w(self) -> torch.Tensor:
0054:         return self._body_lin_vel_w[:, self._body_indexes]
0055: 
0056:     @property
0057:     def body_ang_vel_w(self) -> torch.Tensor:
0058:         return self._body_ang_vel_w[:, self._body_indexes]
0059: 
0060: 
0061: class MotionCommand(CommandTerm):
0062:     cfg: MotionCommandCfg
0063: 
0064:     def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
0065:         super().__init__(cfg, env)
0066: 
0067:         self.robot: Articulation = env.scene[cfg.asset_name]
0068:         self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
0069:         self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
0070:         self.body_indexes = torch.tensor(
0071:             self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
0072:         )
0073: 
0074:         self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
0075:         self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
0076:         self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
0077:         self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
0078:         self.body_quat_relative_w[:, :, 0] = 1.0
0079: 
0080:         self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
0081:         self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
0082:         self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
0083:         self.kernel = torch.tensor(
0084:             [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
0085:         )
0086:         self.kernel = self.kernel / self.kernel.sum()
0087: 
0088:         self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
0089:         self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
0090:         self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
0091:         self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
```

## Official reward terms

File: `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
Function/Class: `def `
Purpose: Defines tracking reward and smoothing terms.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Corresponds to paper tracking reward implementation in the public Stage-1 repo.

```python
0008: from isaaclab.utils.math import quat_error_magnitude
0009: 
0010: from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
0011: 
0012: if TYPE_CHECKING:
0013:     from isaaclab.envs import ManagerBasedRLEnv
0014: 
0015: 
0016: def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
0017:     return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]
0018: 
0019: 
0020: def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
0021:     command: MotionCommand = env.command_manager.get_term(command_name)
0022:     error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
0023:     return torch.exp(-error / std**2)
0024: 
0025: 
0026: def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
0027:     command: MotionCommand = env.command_manager.get_term(command_name)
0028:     error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
0029:     return torch.exp(-error / std**2)
0030: 
0031: 
0032: def motion_relative_body_position_error_exp(
0033:     env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
0034: ) -> torch.Tensor:
0035:     command: MotionCommand = env.command_manager.get_term(command_name)
0036:     body_indexes = _get_body_indexes(command, body_names)
0037:     error = torch.sum(
0038:         torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
0039:     )
0040:     return torch.exp(-error.mean(-1) / std**2)
0041: 
0042: 
0043: def motion_relative_body_orientation_error_exp(
0044:     env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
0045: ) -> torch.Tensor:
0046:     command: MotionCommand = env.command_manager.get_term(command_name)
```

## Teacher rollout collection wrapper

File: `reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
Function/Class: `def main`
Purpose: Runs the selected teacher and writes rollout shards.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Provides local state-action data for downstream VAE/diffusion experiments.

```python
0537:         item["memory_total_mb"] = mem_total
0538:         item["utilization_samples"].append(util)
0539:     for item in per_gpu.values():
0540:         samples = item.pop("utilization_samples")
0541:         item["mean_utilization_gpu_percent"] = sum(samples) / len(samples) if samples else 0.0
0542:     return {"exists": True, "row_count": len(rows), "per_gpu": per_gpu}
0543: 
0544: 
0545: def main() -> None:
0546:     OUT.mkdir(parents=True, exist_ok=True)
0547:     LOG_DIR.mkdir(parents=True, exist_ok=True)
0548:     RUN_ROOT.mkdir(parents=True, exist_ok=True)
0549:     worker_path = OUT / "tracking_g1_resource_adjusted_teacher_rollout_worker.py"
0550:     worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
0551: 
0552:     timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
0553:     run_dir = RUN_ROOT / f"resource_adjusted_teacher_rollout_{timestamp}_seed{SEED}"
0554:     training_run = load_json(TRAINING_RUN_JSON)
0555:     checkpoint = select_checkpoint(training_run)
0556:     gpu_guard = write_gpu_guard(timestamp)
0557:     gpu_snapshot = query_gpus()
0558:     compute_processes = query_compute_processes()
0559:     selected_gpus = list(CANDIDATE_GPUS)
0560:     resource_ready = (
0561:         len([row for row in gpu_snapshot if row.get("index") in selected_gpus]) == 2
0562:         and all(
0563:             row.get("memory_free_mb", 0) >= MIN_FREE_MB and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
0564:             for row in gpu_snapshot
0565:             if row.get("index") in selected_gpus
0566:         )
0567:         and not [
0568:             proc for proc in compute_processes
0569:             if WANGJC_PATH_MARKER not in f"{proc.get('process_name', '')} {proc.get('cmdline', '')}"
0570:         ]
0571:     )
0572:     input_checks = {
0573:         "tracking_python_exists": TRACKING_PY.is_file(),
0574:         "training_run_completed": training_run.get("status")
0575:         in {
```

## Conditional action VAE

File: `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
Function/Class: `class ConditionalActionVAE`
Purpose: Encodes obs+action into a latent and decodes obs+z back to action.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Paper-faithful local reimplementation of latent action distillation, but not official checkpoint.

```python
0071: 
0072: def seed_everything(seed):
0073:     random.seed(seed)
0074:     np.random.seed(seed)
0075:     torch.manual_seed(seed)
0076:     torch.cuda.manual_seed_all(seed)
0077: 
0078: 
0079: class ConditionalActionVAE(nn.Module):
0080:     def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim):
0081:         super().__init__()
0082:         self.encoder = nn.Sequential(
0083:             nn.Linear(obs_dim + action_dim, hidden_dim),
0084:             nn.ELU(),
0085:             nn.Linear(hidden_dim, hidden_dim),
0086:             nn.ELU(),
0087:             nn.Linear(hidden_dim, latent_dim * 2),
0088:         )
0089:         self.decoder = nn.Sequential(
0090:             nn.Linear(obs_dim + latent_dim, hidden_dim),
0091:             nn.ELU(),
0092:             nn.Linear(hidden_dim, hidden_dim),
0093:             nn.ELU(),
0094:             nn.Linear(hidden_dim, action_dim),
0095:         )
0096: 
0097:     def forward(self, obs, action=None, deterministic=False):
0098:         if action is None:
0099:             raise ValueError("training forward requires action for posterior inference")
0100:         stats = self.encoder(torch.cat([obs, action], dim=-1))
0101:         mu, logvar = stats.chunk(2, dim=-1)
0102:         if deterministic:
0103:             z = mu
0104:         else:
0105:             z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
0106:         pred = self.decoder(torch.cat([obs, z], dim=-1))
0107:         return pred, mu, logvar
0108: 
0109:     def decode_mean(self, obs):
```

## State-latent denoiser

File: `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
Function/Class: `class StateLatentDenoiser`
Purpose: Predicts clean state-latent tokens from noisy tokens and timestep.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Local denoising objective corresponding to state-latent diffusion training.

```python
0095:         start = row["start"]
0096:         end = row["end_exclusive"]
0097:         obs = self.obs_by_rank[rank][start:end, env, :]
0098:         latent = self.latent_by_rank[rank][start:end, env, :]
0099:         token = np.concatenate([obs, latent], axis=-1).astype(np.float32)
0100:         return torch.from_numpy(token)
0101: 
0102: 
0103: class StateLatentDenoiser(nn.Module):
0104:     def __init__(self, token_dim, hidden_dim, steps):
0105:         super().__init__()
0106:         self.steps = steps
0107:         self.net = nn.Sequential(
0108:             nn.Linear(token_dim + steps, hidden_dim),
0109:             nn.SiLU(),
0110:             nn.Linear(hidden_dim, hidden_dim),
0111:             nn.SiLU(),
0112:             nn.Linear(hidden_dim, token_dim),
0113:         )
0114: 
0115:     def forward(self, noisy, step_idx):
0116:         onehot = F.one_hot(step_idx, num_classes=self.steps).to(noisy.dtype)
0117:         x = torch.cat([noisy, onehot], dim=-1)
0118:         return self.net(x)
0119: 
0120: 
0121: def read_window_index(path):
0122:     rows = []
0123:     with Path(path).open("r", encoding="utf-8", newline="") as f:
0124:         reader = csv.DictReader(f)
0125:         for row in reader:
0126:             rows.append(
0127:                 {
0128:                     "rank": int(row["rank"]),
0129:                     "env_index": int(row["env_index"]),
0130:                     "start": int(row["start"]),
0131:                     "end_exclusive": int(row["end_exclusive"]),
0132:                     "split": row["split"],
0133:                 }
```

## Diffusion noising and loss loop

File: `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
Function/Class: `noisy = torch.sqrt`
Purpose: Applies DDPM-style noising and computes token MSE.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Matches the clean-token denoising formulation used for local training.

```python
0154: def train_epoch(model, loader, optimizer, device, bars):
0155:     model.train()
0156:     losses = []
0157:     for clean in loader:
0158:         clean = clean.to(device, non_blocking=True)
0159:         step = torch.randint(0, DENOISING_STEPS, clean.shape[:2], device=device)
0160:         noise = torch.randn_like(clean)
0161:         alpha = bars.to(device)[step].unsqueeze(-1)
0162:         noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
0163:         pred = model(noisy, step)
0164:         loss = F.mse_loss(pred, clean)
0165:         optimizer.zero_grad(set_to_none=True)
0166:         loss.backward()
0167:         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
0168:         optimizer.step()
0169:         losses.append(float(loss.detach().cpu()))
0170:     return float(np.mean(losses))
0171: 
0172: 
0173: def evaluate(model, loader, device, bars):
0174:     model.eval()
0175:     pred_losses = []
0176:     noisy_losses = []
0177:     with torch.inference_mode():
0178:         for clean in loader:
0179:             clean = clean.to(device, non_blocking=True)
0180:             step = torch.randint(0, DENOISING_STEPS, clean.shape[:2], device=device)
0181:             noise = torch.randn_like(clean)
0182:             alpha = bars.to(device)[step].unsqueeze(-1)
0183:             noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
0184:             pred = model(noisy, step)
0185:             pred_losses.append(float(F.mse_loss(pred, clean).detach().cpu()))
0186:             noisy_losses.append(float(F.mse_loss(noisy, clean).detach().cpu()))
0187:     return {
0188:         "pred_token_mse": float(np.mean(pred_losses)),
0189:         "noisy_token_mse": float(np.mean(noisy_losses)),
0190:         "denoising_improvement_ratio": float(1.0 - (np.mean(pred_losses) / max(np.mean(noisy_losses), 1e-12))),
0191:     }
0192: 
```

## Offline state-latent guidance evaluation

File: `reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`
Function/Class: `def .*cost|def main|tasks`
Purpose: Evaluates task-cost gradients and scale response over denoiser outputs.
Input: see function signature and surrounding module context.
Output: see return values / written artifacts.
Paper relation: Implements local classifier/task guidance proxies for joystick/waypoint/smoothness-style costs.

```python
0151:         latent = latent_by_rank[rank][s:e, env, :]
0152:         action = action_by_rank[rank][s:e, env, :]
0153:         tokens.append(np.concatenate([obs, latent], axis=-1).astype(np.float32))
0154:         actions.append(action.astype(np.float32))
0155:         splits.append(row["split"])
0156:     return torch.from_numpy(np.stack(tokens, axis=0)), torch.from_numpy(np.stack(actions, axis=0)), splits
0157: 
0158: 
0159: def task_cost(task, tau, actions):
0160:     obs = tau[..., :160]
0161:     latent = tau[..., 160:]
0162:     root_vel = obs[..., :2]
0163:     command = torch.tensor([0.35, 0.0], dtype=tau.dtype, device=tau.device)
0164:     velocity = torch.mean((root_vel - command) ** 2, dim=(-2, -1))
0165:     latent_smooth = torch.mean((latent[:, 1:, :] - latent[:, :-1, :]) ** 2, dim=(-2, -1))
0166:     latent_mag = torch.mean(latent**2, dim=(-2, -1))
0167:     if task == "velocity_command":
0168:         return velocity
0169:     if task == "latent_smoothness":
0170:         return latent_smooth
0171:     if task == "latent_magnitude":
0172:         return latent_mag
0173:     if task == "composed":
0174:         return velocity + 0.25 * latent_smooth + 0.1 * latent_mag
0175:     raise ValueError(task)
0176: 
0177: 
0178: def direction(task):
0179:     return "lower_is_better"
0180: 
0181: 
0182: def main():
0183:     RUN_DIR.mkdir(parents=True, exist_ok=True)
0184:     random.seed(SEED)
0185:     np.random.seed(SEED)
0186:     torch.manual_seed(SEED)
0187:     start_time = time.time()
0188:     diffusion_summary = json.loads(DIFFUSION_JSON.read_text(encoding="utf-8"))
0189:     dataset_summary = json.loads(STATE_LATENT_JSON.read_text(encoding="utf-8"))
```

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
