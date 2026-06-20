# Initial Failures: Official Importer-Export Fig.5/Fig.6 Task-Protocol Proxy

## Scope

This note records two script-compatibility failures encountered while adding:

```text
reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py
```

The failures were analysis-script issues, not failed IsaacLab rollout experiments. The underlying source traces and MP4 paths were already present from earlier local closed-loop importer-export guidance runs.

## Failed Commands

```bash
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py 2>&1 | tee logs/official_importer_export_fig5_fig6_task_protocol_proxy.log
```

The command was run multiple times while repairing the new script. Because `tee` was used without `-a`, the final successful run overwrote the earlier traceback text in `logs/official_importer_export_fig5_fig6_task_protocol_proxy.log`. The failure reasons below are retained here for audit continuity.

## Failure 1: Missing Top-Level Seed

The first run failed because the legacy `seed_group_0_existing` rows in:

```text
res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json
```

have top-level `"seed": null`, while the true seed is present in each child task summary under `config.seed`.

Observed error:

```text
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

Resolution: the script now falls back from `source_row["seed"]` to `summary["config"]["seed"]`.

## Failure 2: NumPy Scalar JSON Encoding

The second repaired run produced metrics but failed while writing JSON because NumPy scalar values, including NumPy booleans, were still present in the payload.

Observed error:

```text
TypeError: Object of type bool is not JSON serializable
```

Resolution: the script now passes payloads through a `jsonable()` sanitizer that converts `np.generic` values to native Python scalars and converts paths/tuples recursively.

## Final Status

The final rerun completed successfully:

```text
status: ok_official_importer_export_fig5_fig6_task_protocol_proxy
rows: 20
seed groups: 5
trace NPZ count: 20
MP4 path count: 20
```

Final output:

```text
res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/fig5_fig6_task_protocol_proxy.json
```

## Claim Boundary

This asset is a local virtual task-protocol proxy over existing traces. It is not the official BeyondMimic Fig.5/Fig.6 protocol, not official checkpoints, not TensorRT deployment evidence, and not real-robot evidence.
