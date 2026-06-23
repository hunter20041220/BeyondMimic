# Native PPO MuJoCo Adapter Comparison

## 结论

该结果比较 approximate native MuJoCo obs->PPO->action adapter 与旧的 open-loop stored-action replay。它用于排查 adapter，不是 paper-level 结果。

## Metrics

- Native root height min/max: `0.6801629089618233` / `0.7830244440424268`
- Native fall proxy count: `0`
- Native root position error mean/max: `0.031667729187286546` / `0.10622444597643989`
- Open-loop root height min/max: `0.643982828232582` / `0.7457355254466121`
- Open-loop fall proxy count: `0`
- Open-loop root position error mean/max: `0.1438565948581903` / `0.22297892348363033`

## Claim Boundary

这是本地 MuJoCo adapter probe，不是官方 BeyondMimic IsaacLab rollout，不是真实机器人结果。
