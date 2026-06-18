# Failure Reason

- Gate: `isaaclab_kit_inotify`.
- The bounded official `whole_body_tracking` train-entry retry reproduced Kit watcher failures.
- Classification: `blocked_inotify`.
- Current inotify limits: `fs.inotify.max_user_watches = 8192
fs.inotify.max_user_instances = 128`.
- The retry did not reach a PPO training endpoint and produced no checkpoint, rollout, or video.
