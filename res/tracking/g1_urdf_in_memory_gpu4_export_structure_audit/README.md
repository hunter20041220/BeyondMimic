# G1 URDF In-Memory GPU4 Export Structure Audit

This directory stores lightweight evidence for the local USDA exported by the official Isaac Sim
URDF importer in-memory GPU4 probe. The large USDA itself remains local and ignored by Git.

Status: `ok_with_physics_usd_export_but_vulkan_device_lost`
Latest blocker: `official_g1_importer_exports_physics_stage_but_vulkan_device_lost_before_payload_or_replay`

Claim boundary: this is an official importer structure audit only. It is not official replay,
not motion preprocessing success, not PPO training, and not paper-level tracking reproduction.
