# Resolution Plan

- Ask an administrator to raise `fs.inotify.max_user_watches` to at least `524288`.
- Ask an administrator to raise `fs.inotify.max_user_instances` to at least `1024`.
- Rerun `tracking_official_train_entry_retry_audit.py` after the host limits change.
- If the retry reaches the training endpoint, only then schedule a longer official PPO smoke.
- Keep this failed run under `res/failed_runs`; do not delete or hide it.
