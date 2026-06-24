# MuJoCo Torso Frame Offset Cross-Sample Audit

- Status: `blocked_fixed_torso_offset_not_stable_across_walk_and_dance_samples`
- Generated: `2026-06-24T05:31:56.142284+00:00`
- Scope: compares dance and walk IsaacLab samples; no training, no policy rollout, no video.
- 当前不得声称完整复现 BeyondMimic；该审计只判断 fixed torso offset 是否可作为 adapter 修正。

## Cross-Sample Offset

- Quaternion offset sign-invariant error: `0.16278608548427614`
- Position offset L2 difference: `0.0010926367946245451`
- Fixed offset stable: `False`

## Samples

- `dance_terminated` motion=`/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/dance1_subject1/motion.npz` terminated=`True` raw_ori_err=`0.3175156624836241` corrected_ori_err=`1.1436359326211232e-07` q_offset=`[0.981741168614011, 0.10665968775146022, -0.1357829358628406, -0.07981843888240901]`
- `walk_nonterminated` motion=`/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz` terminated=`False` raw_ori_err=`0.14584921251276103` corrected_ori_err=`4.4325126062616516e-08` q_offset=`[0.9956580662748654, -0.05612639773281594, -0.01978585107039479, 0.07157766856187599]`

## Failed / Blocking Checks

- `q_offset_stable_across_samples`
- `p_offset_stable_across_samples`
- `fixed_offset_adapter_patch_allowed`
- `success_video_claim_allowed`

## Interpretation

- walk 样本是 non-terminated 且 command metrics 为 0，因此它是更可信的低动态 adapter 对照。
- 两个样本各自都能被单独 fitted offset 修正，但 offset 不一致，说明固定 torso frame offset 不是充分修复。
- 后续应继续定位 IsaacLab/PhysX articulation body frame 与 MuJoCo MJCF body frame 的姿态表达差异，不能直接把单样本 offset 写入 rollout adapter。
