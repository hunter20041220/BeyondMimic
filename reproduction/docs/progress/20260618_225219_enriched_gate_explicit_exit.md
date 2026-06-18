# Progress Update

## Goal

Make the resource-adjusted enriched G1 USD replay preflight usable as a deterministic automation gate after the previous
run proved four IsaacLab render steps but timed out during Kit shutdown.

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
- `res/tracking/g1_enriched_usd_replay_preflight/tracking_g1_enriched_usd_replay_preflight_audit.json`
- `logs/tracking_g1_enriched_usd_replay_preflight/enriched_usd_replay_preflight.log`
- `reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py`
- `reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py`
- Isaac Sim `SimulationApp.close()` implementation under `envs/bm_tracking`

## Files Modified

- `reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py`
- `reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- refreshed small audit/report outputs under `res/artifact_manifest`, `res/comparison`, `res/final_report`,
  `res/docs`, `res/verification_command_*`, `res/progress_report_audit`, `res/master_audit`, and
  `res/tracking/g1_enriched_usd_replay_preflight`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py
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

## Results

- Added `--exit_after_success` to the bounded enriched USD replay entrypoint.
- The preflight now reaches `after_app`, `sim_created`, `scene_created`, `sim_reset`,
  `robot_contract=num_joints=29,num_bodies=40,device=cuda:6`, four render steps,
  `enriched_usd_replay_preflight_success`, and `explicit_exit_after_success`.
- The audit now returns `status=ok_resource_adjusted_step_gate_passed_with_explicit_exit` with return code `0`.
- The audit explicitly records `clean_kit_shutdown_verified=false`, so this does not hide the previous shutdown issue.

## Verification

- `artifact_manifest.py`: passed, `254` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid status count `0`.
- `verification_command_syntax_audit.py`: passed, `178` scripts.
- `verification_command_script_manifest.py`: passed, `178` scripts.
- `verification_command_coverage_audit.py`: passed, `186` commands.
- `progress_report_audit.py`: passed.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- Clean Kit shutdown is still not verified; the new deterministic gate uses explicit process exit after the success
  sentinel.
- Official `csv_to_npz.py` has still not produced official `motion.npz`.
- Official replay/evaluation, PPO, DAgger, VAE/diffusion closed-loop rollout, Fig. 5/Fig. 6 videos, TensorRT deployment,
  and real-robot execution remain incomplete or blocked.

## Effect on English Reading Report

This strengthens the reproduction section by showing that the virtual stack can now instantiate and step the generated
29-DoF/40-body G1 scaffold in IsaacLab deterministically enough for follow-up gate automation. The report must still
describe it as a resource-adjusted environment/articulation gate, not as an official BeyondMimic replay result.

## Next Step

Use this deterministic gate to attempt a longer resource-adjusted replay/eval diagnostic, while separately continuing
to repair the official `csv_to_npz.py` conversion path.

## Git Commit

Pending at progress-file creation time.
