# Official-importer-export Fig. 5/Fig. 6 proxy protocol matrix

This matrix maps the current local official-importer-export guidance evidence onto the six paper panels.
It is a report-facing planning aid, not a paper-level reproduction claim.

| Panel | Paper claim | Current local status | Local proxy tasks | Rollout rows | Proxy pass rate | Next virtual validation |
|---|---|---|---|---:|---:|---|
| Figure 5 A | Diffusion process visualization under joystick right-turn command | supporting_local_proxy_closed_loop_not_panel_protocol | joystick | 3 | 0.667 | Render a local denoising-trajectory visualization under a fixed joystick command and pair it with a paper-style velocity-tracking rollout metric in IsaacLab. |
| Figure 5 B | Transition from walking to running through latent diffusion | training_chain_available_no_transition_panel_protocol |  | 0 |  | Build a local walking-to-running transition protocol from teacher/VAE latents and evaluate the transition in closed-loop simulation. |
| Figure 5 C | Joystick teleoperation and recovery from disturbance | supporting_local_proxy_closed_loop_not_paper_success_protocol | joystick | 3 | 0.667 | Run a joystick-command IsaacLab protocol with velocity-tracking, survival, and disturbance-recovery metrics over multiple seeds. |
| Figure 5 D | t-SNE latent visualization from walking to running | latent_training_chain_available_no_tsne_panel_protocol |  | 0 |  | Generate a t-SNE or UMAP visualization from local teacher-rollout VAE latents and label walking, running, and transition clips. |
| Figure 6 A | Motion inpainting with future cartwheel keyframes and multi-round task switching | single_importer_export_keyframe_inpainting_proxy_closed_loop_diagnostic | inpainting | 1 |  | Turn the single diagnostic future-keyframe/root-path proxy into a multi-seed paper-style keyframe protocol with explicit fall, transition smoothness, and success/failure thresholds. |
| Figure 6 B | Real-world obstacle avoidance using waypoint plus SDF costs | supporting_local_proxy_closed_loop_but_real_world_panel_not_reproduced | waypoint,obstacle_avoidance,composed | 9 | 0.667 | Run a simulated waypoint plus SDF obstacle protocol with explicit collision, clearance, reach-target, fall, and timeout metrics. The exact paper panel still needs real robot or mocap evidence. |

Claim boundary: every row is local virtual evidence or a blocked/proxy mapping. None of the rows is an official BeyondMimic Fig. 5/Fig. 6 success, fall, collision, TensorRT, or real-robot result.
