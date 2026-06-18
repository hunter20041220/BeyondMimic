# Progress Update

## Goal

Verify whether the recovered IsaacLab/whole_body_tracking environment can enter the official RSL-RL train stack after the resource-adjusted official-CSV task gate, without claiming formal PPO training or paper-level tracking performance.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`
- `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
- `envs/bm_tracking/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py`

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/RUNBOOK.md`
- `reproduction/PROGRESS.md`
- `.gitignore`

## Commands Run

```bash
git status --short && git log -1 --oneline
sed -n '1,180p' reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py
find reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1 -maxdepth 4 -type f -print
rg -n "class .*PPORunner|RslRl|OnPolicyRunner|RslRlVecEnvWrapper|max_iterations|num_steps_per_env" reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1 reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.py
```

The first diagnostic run reached `env_created`, `vec_env_wrapped`, and `runner_created`, then failed with `TypeError('expected str, bytes or os.PathLike object, not NoneType')` because RSL-RL tried to save code-state information with `log_dir=None`. The probe was corrected by setting `runner.disable_logs=True` after runner construction, then rerun successfully.

## Results

- New audit status: `ok_resource_adjusted_train_entry_diagnostic`.
- The probe constructs the official `Tracking-Flat-G1-v0` task, wraps it with `RslRlVecEnvWrapper`, instantiates `MotionOnPolicyRunner`, and completes one tiny PPO learning iteration.
- Metrics: `num_envs=1`, `configured_num_steps_per_env=4`, `requested_learning_iterations=1`, action dim `29`, policy obs dim `160`, privileged obs dim `286`, robot joints `29`, robot bodies `40`.
- No checkpoint is written and no formal training claim is made.
- The log records PhysX GPU narrowphase kernel launch warnings before the success sentinel.

## Verification

Full verification bundle is run after this progress file is committed into the audit chain:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/progress_report_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

- Official URDF/USD converter output is still missing.
- Official `csv_to_npz.py` and official `replay_npz.py` success are still missing.
- Formal PPO tracking training/evaluation has not been run.
- Teacher rollout data, DAgger rollout logs, trained tracking checkpoint, VAE/diffusion closed-loop evaluation, Fig. 5/Fig. 6 videos, and real-robot evidence remain unavailable.
- Current diagnostic logs contain PhysX GPU kernel warnings and therefore need follow-up before long training.

## Effect on English Reading Report

This gives the reading report a stronger, honest reproduction statement: the project now verifies not only IsaacLab task construction and full resource-adjusted task stepping, but also the RSL-RL train-entry wiring. It should still be described as a resource-adjusted diagnostic, not as a reproduced tracking teacher or official PPO result.

## Next Step

Choose between two next actions: continue debugging the official URDF/USD conversion path, or run a controlled short PPO training/evaluation attempt with GPU 5 and GPU 6 telemetry after documenting PhysX warnings and formal-training boundaries.

## Git Commit

Pending at the time this progress update was written.
