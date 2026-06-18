# Resolution Plan

- Ask an administrator to raise fs.inotify.max_user_watches to at least 524288 and fs.inotify.max_user_instances to at least 1024, then rerun the IsaacLab/Kit and tracking smoke commands.
- After limits are raised, rerun the IsaacLab/Kit smoke and then tracking preprocessing/replay/PPO smoke.
- Keep this failed run in `res/failed_runs`; do not delete or hide it.
