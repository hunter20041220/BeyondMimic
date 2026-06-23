# Tracking Code Snippets

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
