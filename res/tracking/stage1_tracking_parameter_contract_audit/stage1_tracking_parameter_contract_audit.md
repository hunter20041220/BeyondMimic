# Stage 1 Tracking Parameter Contract Audit

- Status: `blocked_stage1_teacher_contract_has_required_followups`
- Rows: `15`
- Status counts: `{"pass": 8, "pass_with_caution": 5, "fail_or_unverified": 2}`
- Severity counts: `{"info": 8, "medium": 5, "high": 2}`

## Immediate Conclusion

Official/public tracking code and the 4/7 paper-contract wrapper largely match the paper formulas, but current teacher checkpoints still fail quality gates. Two paper-vs-public-code cautions remain: adaptive sampling kernel K=1 in public code vs K=3 in supplement, and ankle-special joint offset described in text but not implemented in the public config. The 5/6 multisource summary also has a metadata flag mismatch for the USD source.

## Contract Matrix

| Component | Status | Severity | Required Action |
| --- | --- | --- | --- |
| PD natural frequency and damping ratio | `pass` | `info` | Keep this unchanged for any future teacher training. |
| Action scale formula | `pass` | `info` | MuJoCo adapter must use this exact exported scale/joint order, not hand-set scale. |
| G1 armature constants | `pass` | `info` | Do not zero or tune armature for stability; paper says wrong armature degrades tracking. |
| Tracking reward formula and weights | `pass` | `info` | If teacher remains weak, inspect motion/termination/curriculum before changing reward weights. |
| Termination contract | `pass` | `info` | Do not relax endpoint termination and call it paper-level; use relaxations only as diagnostics. |
| Default joint offset randomization | `pass` | `info` | Record as paper-vs-public-code ambiguity. If trying to improve ankle/single-leg stability, run a clearly labeled ankle-offset ablation, not an unmarked paper-contract run. |
| Adaptive sampling look-back kernel | `pass_with_caution` | `medium` | Likely important for hard segments. Add explicit K=3 ablation or patch only in a named non-official run. |
| Domain randomization | `pass_with_caution` | `medium` | Keep compact DR; do not add delay randomization unless labeled as ablation. |
| Actor/critic observation contract | `pass` | `info` | MuJoCo PPO adapter must recreate this exact 160-dim order and normalization. |
| RSL-RL PPO hyperparameters | `pass` | `info` | Checkpoint selector should choose best eval checkpoint, not last. |
| Robot asset physical contract | `pass_with_caution` | `medium` | This is a local asset adaptation. Keep claiming local public-resource training, not official checkpoint. |
| Local training metadata fidelity | `pass_with_caution` | `medium` | Patch multisource summary metadata in the next cleanup so report readers are not misled. |
| Teacher quality gate before VAE/diffusion | `fail_or_unverified` | `high` | Do not start final VAE/diffusion training or success videos until this gate passes. |
| Diagnostic termination patch guard | `pass_with_caution` | `medium` | Keep the env var unset for paper-contract training; if used, write a separate ablation row. |
| Known failure interpretation | `fail_or_unverified` | `high` | Next work should target why body/joint errors remain high: adaptive K=3/ankle offset ablations, motion scale/order, reset sampling, and eval per-motion failure bins. |

## Claim Boundary

This audit checks Stage-1 tracking implementation contracts. It does not certify a paper-level teacher, VAE, diffusion, guidance controller, Isaac rendered video, or real robot result.

当前不得声称完整复现 BeyondMimic；该审计只说明 Stage 1 参数/公式合同的对齐程度，不证明 teacher 已达到论文质量。
