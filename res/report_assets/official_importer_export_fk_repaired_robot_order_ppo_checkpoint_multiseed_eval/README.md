# Robot-Order FK-Repaired PPO Multi-Seed Evaluation Assets

These plots and tables summarize three full 2048-env x 299-step local virtual evaluations of the
iteration-999 robot-order FK-repaired PPO checkpoint.

## Source

- Multi-seed audit: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval.json`
- Rows CSV: `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval_rows.csv`
- Seeds: `[20260730, 20260731, 20260732]`
- Total env steps: `1837056`

## Key Aggregate Metrics

- reward_mean: `0.020480790998840676` +/- `0.0004249192220635496`
- done_rate: `0.1785340240036232` +/- `0.001763911381666986`
- body_pos_error_mean: `0.3597400628005382` +/- `0.002825779875965784`
- joint_pos_error_mean: `1.5772204704773731` +/- `0.0036281767104827073`

## Claim Level

local_virtual_robot_order_fk_multiseed_tracking_eval. This is a teacher-quality diagnostic for
the local virtual pipeline. It is not an official BeyondMimic teacher checkpoint, not DAgger,
not Fig.5/Fig.6 guided diffusion, and not real-robot validation.
