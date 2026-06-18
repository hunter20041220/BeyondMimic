# Progress Update

## Goal

Check whether the structurally valid ASAP reference G1 USD found in the previous round can serve as a clearly labeled
resource-adjusted asset for the official whole_body_tracking replay gate.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json`
- `res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json`
- `res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json`
- `res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- Official G1 URDF:
  `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf`
- Reference USD:
  `download/reference_code/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.usd`

## Files Modified

- Added `reproduction/scripts/tracking_g1_reference_usd_compatibility_audit.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Updated `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- Updated `reproduction/scripts/reproduction_master_audit.py`
- Updated `reproduction/scripts/final_reproduction_report.py`
- Updated `reproduction/docs/known_limitations.md`
- Updated `reproduction/docs/experiment_protocol.md`
- Regenerated `reproduction/docs/final_reproduction_report.md`
- Regenerated small audit outputs under `res/artifact_manifest`, `res/final_report`, `res/master_audit`,
  `res/final_deliverables_audit`, `res/verification_command_script_manifest`,
  `res/verification_command_coverage`, and `res/tracking/official_replay_conversion`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_reference_usd_compatibility_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- New audit JSON:
  `res/tracking/g1_reference_usd_compatibility_audit/tracking_g1_reference_usd_compatibility_audit.json`
- Status: `ok_with_reference_usd_incompatible_or_partial`
- Official contract:
  - 40 URDF links
  - 29 non-fixed/action joints
  - 14 target bodies
  - anchor body `torso_link`
- Reference USD contract:
  - default prim `/g1_29dof`
  - 39 links
  - 38 total joints
  - 23 revolute joints
  - 39 rigid-body-like prims
  - 1 articulation root API
- All official target bodies are present in the reference USD.
- The six official wrist action joints are missing from the reference USD's revolute-joint set because they are fixed:
  `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`,
  `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`.
- Therefore the ASAP reference USD is not a drop-in 29-DoF BeyondMimic replay asset.

## Verification

- `artifact_manifest.py`: passed, 245 artifacts, missing count 0
- `paper_vs_reproduction_comparison.py`: passed, 122 comparison rows retained
- `final_reproduction_report.py`: passed
- `completion_matrix_status_audit.py`: passed, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: passed, 161 scripts, 0 failures
- `verification_command_script_manifest.py`: passed, 161 scripts
- `verification_command_coverage_audit.py`: passed, 169 commands, 10 smoke-pass commands
- `reproduction_master_audit.py`: passed

## Failed / Blocked Items

- Official 29-DoF tracking replay remains blocked.
- The ASAP USD may support a locked-wrist/resource-adjusted branch, but it cannot satisfy the official 29-DoF action
  contract without a separate contract, metrics boundary, and explicit labeling.
- No official `motion.npz`, replay video, PPO training, policy evaluation, DAgger rollout, VAE/diffusion closed-loop run,
  checkpoint, or robot result is claimed.

## Effect on English Reading Report

This adds an important reproducibility lesson: a visually and physically valid robot USD is not automatically compatible
with a paper's control/action contract. The reading report can use this as evidence of careful reproduction auditing
rather than simply reporting that an asset exists.

## Next Step

Either locate/build a full 29-DoF G1 USD whose six wrist joints remain revolute, or create a separate locked-wrist
resource-adjusted replay track with its own action dimension and comparison boundary. Do not use the ASAP USD as an
official BeyondMimic drop-in replacement.

## Git Commit

Pending at the time this progress note is written.
