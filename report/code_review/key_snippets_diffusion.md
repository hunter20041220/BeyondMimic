# Diffusion Code Snippets

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
