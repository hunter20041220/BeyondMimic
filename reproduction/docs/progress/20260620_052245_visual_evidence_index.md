# Progress Update

## Goal

Create a small, Git-trackable index for the existing report/PPT visual evidence so local MP4 videos, PNG plots, and
metric tables can be found and audited without committing large video files or overstating their claim level.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- Existing asset JSON files under `/mnt/infini-data/test/BeyondMimic/res/visualization/` and
  `/mnt/infini-data/test/BeyondMimic/res/report_assets/`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/visual_evidence_index.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/english_reading_report.md`

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/visual_evidence_index.py reproduction/scripts/artifact_manifest.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

The new visual evidence index is written to:

```text
res/report_assets/visual_evidence_index/
```

It records:

- `21` source asset JSON files
- `9` local MP4 files
- `47` PNG files
- `52` table or README assets
- total indexed MP4 size `5900210` bytes
- total indexed PNG size `12609371` bytes

Every indexed video row is marked with `github_commit_policy=do_not_commit_large_video`. The index checks also require
all rows to avoid paper-level overclaiming, real-robot overclaiming, and goal-complete claims.

## Verification

Local verification passed:

- `visual_evidence_index.py`: `status=ok`
- `artifact_manifest.py`: `633` artifacts
- `final_reproduction_report.py`: `status=ok`
- `reproduction_master_audit.py`: `status=ok`

The full required verification suite is run after this progress note.

## Failed / Blocked Items

No new experiment failed. This is a report/evidence-indexing step, not a new paper-level rollout. The existing local
videos remain resource-adjusted/local-virtual evidence and are not official Fig. 5/Fig. 6 or real-robot results.

## Effect on English Reading Report

The English reading report now points to `res/report_assets/visual_evidence_index/`, making it easier to choose
presentation videos and figures while preserving honest claim boundaries.

## Next Step

Continue from indexed visual evidence toward stronger full-bundle closed-loop guided evaluations, especially
guided-vs-unguided comparisons that can produce both quantitative curves and robot-motion videos.

## Git Commit

Pending at the time this progress file was written.
