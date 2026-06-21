# Unified Local Task Protocol

This table consolidates the local BeyondMimic-like guidance tasks used for the reading report. It is a proxy protocol table, not a paper-level Fig. 5/Fig. 6 reproduction.

- Status: `ok_unified_local_task_protocol`
- Task rows: `6`
- Paper-level reproduced rows: `0`

|task|local_protocol_status|main_metric|seed_count|local_proxy_pass_rate|reward_improved_vs_denoised_rate|tracking_error_not_worse_rate|claim_level|next_gap|
|---|---|---|---|---|---|---|---|---|
|joystick|implemented_multiseed_proxy|velocity tracking error, reward delta, tracking-error delta, done/fall proxy|5|0.8|0.6|0.6|local_virtual_proxy_not_paper_level|replace weak tracking teacher, use paper-style task thresholds, and rerun closed-loop guidance|
|waypoint|implemented_multiseed_proxy|final root distance, reward delta, tracking-error delta, success proxy|5|0.4|0.2|0.2|local_virtual_proxy_not_paper_level|replace weak tracking teacher, use paper-style task thresholds, and rerun closed-loop guidance|
|obstacle_avoidance|implemented_multiseed_proxy|minimum clearance/collision proxy, root distance, reward delta, tracking-error delta|5|0.8|0.6|0.8|local_virtual_proxy_not_paper_level|replace weak tracking teacher, use paper-style task thresholds, and rerun closed-loop guidance|
|composed|implemented_multiseed_proxy|combined joystick/waypoint/obstacle proxy score, reward delta, tracking-error delta|5|0.6|0.4|0.4|local_virtual_proxy_not_paper_level|replace weak tracking teacher, use paper-style task thresholds, and rerun closed-loop guidance|
|transition|implemented_single_seed_proxy|speed-ramp profile, transition smoothness, 299-step completion/fall proxy|1||||local_virtual_single_seed_proxy_not_paper_fig5b|upgrade to multi-seed paper-style walk-to-run transition metric and fall-rate gate|
|inpainting|implemented_single_seed_proxy|keyframe/body target error, guidance cost delta, 299-step completion/fall proxy|1||||local_virtual_single_seed_proxy_not_paper_inpainting|upgrade to multi-seed keyframe-error protocol with paper-facing success thresholds|

Interpretation: the current table is useful evidence for an auditable local virtual pipeline, but it should be read together with the tracking quality gate. The tracking teacher still has near-unit termination/done behavior in the latest FK-repaired PPO eval, so downstream results remain proxy evidence.

Current tracking dependency: the latest endpoint-group ablation identifies the wrist endpoints as the dominant `ee_body_pos` termination contributor. Before rerunning full PPO or rebuilding downstream VAE/diffusion/guidance, the next live probe should target wrist endpoint target/body order, FK height, reset velocity/action consistency, and termination semantics.
