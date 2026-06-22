# Progress Update

## Goal

Diagnose why the current MuJoCo movement-control videos look poor before generating more PPO/VAE/guided MP4s. This round re-read the BeyondMimic paper method contract, official tracking code, local PPO eval results, MuJoCo video scripts, and current audit outputs. It did not start training.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/prompt06211658.txt`
- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_trace_mesh_video.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_ppo_adapter_gap_audit.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/configs/g1_joint_mapping.yaml`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/reward_termination_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/res/control_videos/mujoco_control_video_summary.json`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/beyondmimic_control_quality_diagnosis.py`
- Added this progress update under `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/beyondmimic_control_quality_diagnosis.py
python3 reproduction/scripts/beyondmimic_control_quality_diagnosis.py
```

## Results

New diagnostic outputs:

- `/mnt/infini-data/test/BeyondMimic/res/diagnostics/beyondmimic_control_quality_diagnosis/control_quality_diagnosis.json`
- `/mnt/infini-data/test/BeyondMimic/res/diagnostics/beyondmimic_control_quality_diagnosis/control_quality_diagnosis_findings.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/diagnostics/beyondmimic_control_quality_diagnosis/control_quality_diagnosis.md`

Main finding: the current poor MuJoCo movement-control videos are not primarily a video-rendering problem. They expose a weak local tracking teacher and a missing faithful native MuJoCo observation/action adapter.

Key teacher-quality signals:

- `scaled_ppo_iter999`: reward mean `0.02423080788881683`, done rate `0.9988405361622074`, dominant termination `ee_body_pos`.
- `fk_repaired_robot_order_iter999`: reward mean `0.02073384587805606`, done rate `0.1782798129180602`, action abs mean `0.5580962428879179`, body position error mean `0.4831680147145504`, joint position error mean `1.8487586905326332`.
- `endpoint_threshold_candidate_iter999`: done rate improves to `0.09407497648411371`, but reward mean drops to `0.005500174558917714`, action abs mean rises to `0.8344121457242647`, and joint position error mean rises to `3.2371604988806224`.

MuJoCo video boundary:

- Current `mujoco_mp4/res/control_videos/*` videos use `mj_step`, but they are PD target-tracking videos with root assist, not native PPO/VAE/guided MuJoCo controllers.
- `native_mujoco_ppo_obs_adapter=false` for current control-video rows.
- The videos are diagnostic/report media only, not paper-level BeyondMimic control reproduction.

## Verification

Immediate verification passed:

- Python compile passed.
- Diagnosis script completed and wrote JSON/TSV/Markdown outputs.

Full artifact/master-audit verification is run after this progress entry is added.

## Failed / Blocked Items

- MuJoCo PPO closed-loop rollout remains blocked by the missing faithful 160-D IsaacLab-compatible observation/action adapter and weak source PPO teacher.
- MuJoCo VAE/guided rollout videos should not be regenerated from the current weak teacher as paper-claim evidence.
- Isaac true rendered MP4 remains blocked on the H20 Isaac Sim rendering stack.
- Official BeyondMimic checkpoints, true DAgger logs, strict Fig.5/Fig.6 videos/metrics, TensorRT deployment, and real robot results remain unavailable.

## Effect on English Reading Report

This gives the report a stronger honesty boundary: the project can say that MuJoCo videos were attempted and diagnosed, but the current visual quality reflects upstream teacher/adapter limitations rather than a solved BeyondMimic controller. It also gives a concrete next-step roadmap for improving simulation evidence.

## Next Step

Repair tracking teacher quality before generating more controller videos. The next engineering target should be a tracking target-semantics audit that compares FK-repaired `body_pos_w/body_quat_w`, G1 body order, wrist endpoint z behavior, reset pose, target refresh, and action/default-offset semantics against the official `MotionCommand` and paper tracking contract.

## Git Commit

Pending.
