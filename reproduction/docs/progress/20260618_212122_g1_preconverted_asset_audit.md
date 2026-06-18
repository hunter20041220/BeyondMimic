# Progress Update

## Goal

Continue the official whole_body_tracking replay gate recovery after GPU/renderer variants failed to produce a valid G1
USD. This round audits local preconverted G1 asset candidates to determine whether an existing full-robot USD can be
used to unblock `csv_to_npz.py` / `replay_npz.py`.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/tracking/g1_urdf_in_memory_variant_matrix/tracking_g1_urdf_in_memory_variant_matrix_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- Local asset trees under `download/`, `other/`, `reproduction/`, and `res/`

## Files Modified

- Added `reproduction/scripts/tracking_g1_preconverted_asset_audit.py`
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
find download other reproduction res -type f \( -iname '*g1*.usd' -o -iname '*g1*.usda' -o -iname '*g1*.usdc' -o -iname '*unitree*.usd' -o -iname '*unitree*.usda' -o -iname '*unitree*.usdc' -o -iname '*.mjcf' -o -iname '*.xml' -o -iname '*g1*.urdf' -o -iname '*.urdf' \)
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_preconverted_asset_audit.py
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
  `res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json`
- Status: `ok_with_reference_usd_candidate`
- Candidate count: `65`
- USD candidate count: `36`
- Official mesh-level G1 USD count: `35`
- Official full-robot preconverted G1 USD count: `0`
- Reference G1 USD count: `1`
- Validated reference robot-like USD count: `1`
- The validated reference USD is
  `download/reference_code/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.usd`.
- Kit read-only validation opened it successfully with default prim `/g1_29dof`, `161` prims, `38` joints,
  `39` rigid-body-like prims, and `1` articulation root API.

## Verification

- `artifact_manifest.py`: passed, 244 artifacts, missing count 0
- `paper_vs_reproduction_comparison.py`: passed, 122 comparison rows retained
- `final_reproduction_report.py`: passed
- `completion_matrix_status_audit.py`: passed, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: passed, 161 scripts, 0 failures
- `verification_command_script_manifest.py`: passed, 161 scripts
- `verification_command_coverage_audit.py`: passed, 169 commands, 10 smoke-pass commands
- `reproduction_master_audit.py`: passed

## Failed / Blocked Items

- No official full-robot preconverted G1 USD was found.
- The ASAP USD is structurally valid but comes from `download/reference_code`, not the official BeyondMimic
  `whole_body_tracking` repository or released dataset.
- It must not be reported as an official BeyondMimic replay asset.
- The official replay gate remains blocked until either an official full-robot USD is generated or a clearly labeled
  resource-adjusted replay path is built and audited.

## Effect on English Reading Report

This strengthens the reproduction limitations section: the project did not merely fail at URDF import; it also audited
local preconverted asset alternatives. The report can honestly state that a reference-code USD exists and is structurally
valid, while preserving the distinction between official-code reproduction and resource-adjusted workaround evidence.

## Next Step

Evaluate whether the ASAP reference USD can be used in a clearly labeled resource-adjusted conversion/replay gate, or
build a lower-level offline converter from the official URDF/MJCF. If used, the next audit must compare joint/body names
and kinematic compatibility before attempting `csv_to_npz.py` / `replay_npz.py`.

## Git Commit

Pending at the time this progress note is written.
