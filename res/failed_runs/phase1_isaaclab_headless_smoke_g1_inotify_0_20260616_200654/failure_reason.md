# Failure Reason

- Gate: `isaaclab_kit_inotify`.
- Kit reports repeated `Failed to create change watch` errors with `errno=28`.
- Current inotify limits from evidence: `fs.inotify.max_user_watches = 8192
fs.inotify.max_user_instances = 128`.
- This blocks IsaacLab/Kit smoke, motion preprocessing, replay, PPO smoke, and live rollout evaluation.
