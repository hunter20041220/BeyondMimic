# MuJoCo Torso Frame Offset Audit

- Status: `blocked_torso_frame_offset_hypothesis_single_nonterminated_sample_requires_cross_sample_validation`
- Generated: `2026-06-24T05:31:55.955191+00:00`
- Scope: single-sample MuJoCo/IsaacLab torso frame offset hypothesis; no training, no rollout, no video.
- 当前不得声称完整复现 BeyondMimic；本审计不能放行 native MuJoCo PPO/VAE/diffusion 视频。

## Primary Result

- Primary model: `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml`
- Raw `motion_anchor_pos_b` error: `0.005084645669435443`
- Raw `motion_anchor_ori_b` error: `0.14584921251276103`
- Corrected `motion_anchor_pos_b` error: `7.555605446851743e-10`
- Corrected `motion_anchor_ori_b` error: `4.4325126062616516e-08`
- Candidate right-multiplied quaternion offset: `[0.9956580662748654, -0.05612639773281594, -0.01978585107039479, 0.07157766856187599]`
- Candidate world position offset: `[-1.649246013607497e-05, -0.0005042594494395914, 3.761730001772268e-05]`

## Sample Quality

- Motion file: `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz`
- Motion time steps: `[157]`
- Reward after zero step: `0.0`
- Terminated after zero step: `False`

## Failed / Blocking Checks

- `independent_nonterminated_walk_sample_available`
- `offset_validated_across_independent_walk_sample`

## Interpretation

- 该审计支持一个很具体的失败假设：MuJoCo 的 `torso_link` frame 与 IsaacLab importer/exported `torso_link` frame 不一致。
- 在当前样本上，右乘候选四元数 offset 后 anchor orientation term 从大误差恢复到数值一致。
- 但当前样本来自 terminated dance state，因此该 offset 不能直接写入最终 adapter。
- 下一步必须抓取 non-terminated walk/single-leg IsaacLab observation sample，并验证同一个 offset 是否仍成立。
