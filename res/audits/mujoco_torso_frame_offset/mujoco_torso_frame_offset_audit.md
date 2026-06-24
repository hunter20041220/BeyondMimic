# MuJoCo Torso Frame Offset Audit

- Status: `blocked_torso_frame_offset_hypothesis_single_terminated_sample_requires_walk_validation`
- Generated: `2026-06-24T05:15:54.855051+00:00`
- Scope: single-sample MuJoCo/IsaacLab torso frame offset hypothesis; no training, no rollout, no video.
- 当前不得声称完整复现 BeyondMimic；本审计不能放行 native MuJoCo PPO/VAE/diffusion 视频。

## Primary Result

- Primary model: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- Raw `motion_anchor_pos_b` error: `0.005219148958698112`
- Raw `motion_anchor_ori_b` error: `0.3175156624836241`
- Corrected `motion_anchor_pos_b` error: `1.4629287503620247e-09`
- Corrected `motion_anchor_ori_b` error: `1.1436359326211232e-07`
- Candidate right-multiplied quaternion offset: `[0.981741168614011, 0.10665968775146022, -0.1357829358628406, -0.07981843888240901]`
- Candidate world position offset: `[-7.13339349610459e-05, 0.0005856326823376577, -1.69969718941676e-05]`

## Sample Quality

- Motion file: `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/dance1_subject1/motion.npz`
- Motion time steps: `[232]`
- Reward after zero step: `0.03098168969154358`
- Terminated after zero step: `True`

## Failed / Blocking Checks

- `sample_is_nonterminated`
- `sample_motion_is_walk_or_low_dynamic`
- `independent_nonterminated_walk_sample_available`
- `offset_validated_across_independent_walk_sample`

## Interpretation

- 该审计支持一个很具体的失败假设：MuJoCo 的 `torso_link` frame 与 IsaacLab importer/exported `torso_link` frame 不一致。
- 在当前样本上，右乘候选四元数 offset 后 anchor orientation term 从大误差恢复到数值一致。
- 但当前样本来自 terminated dance state，因此该 offset 不能直接写入最终 adapter。
- 下一步必须抓取 non-terminated walk/single-leg IsaacLab observation sample，并验证同一个 offset 是否仍成立。
