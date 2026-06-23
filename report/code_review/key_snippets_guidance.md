# Guidance Code Snippets

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
