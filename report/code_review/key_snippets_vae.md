# VAE Code Snippets

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
