# 论文和本项目对照

| 论文模块 | 本项目状态 | 证据 | 差距 | claim level |
| --- | --- | --- | --- | --- |
| Stage 1 motion tracking | 部分完成但质量弱 | 5/6 multi-source PPO checkpoint sweep | teacher reward 低、done/error 高；不是官方 teacher。 | local virtual partial |
| DAgger / teacher rollout | 部分完成 | 612352 rollout samples | 不是官方 DAgger 数据，且继承弱 teacher。 | local weak-teacher dataset |
| Conditional VAE | 离线复现 | test action MSE 0.00328968 | 没有 paper-level closed-loop VAE rollout。 | offline approximate |
| State-latent diffusion | 离线复现 | MSE 0.0728163 -> 0.0432214 | 没有官方 checkpoint / 完整 paper architecture strict eval。 | offline approximate |
| Classifier guidance | offline proxy | 8192 windows | 不是 Fig.5/Fig.6 closed-loop protocol。 | qualitative/proxy |
| MuJoCo/Isaac video | MuJoCo diagnostic | 六条连续 MuJoCo action-control MP4 | 控制质量差；H20 Isaac rendered MP4 blocked。 | failed/diagnostic local virtual |
| Real robot | 未做 | 无硬件 | 没有 Unitree G1 实机。 | requires_real_robot |

结论：本项目已经形成了可审计的 local reproduction chain，但 paper-level 完整复现仍未完成。
