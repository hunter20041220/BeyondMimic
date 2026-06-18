#!/usr/bin/env python3
"""Generate local-smoke variants of official whole_body_tracking scripts.

The official scripts use WandB registry artifacts and `/tmp/motion.npz`.
For reproducible local smoke tests we keep the official files untouched and
write patched copies under `reproduction/generated/whole_body_tracking_local`.
Those copies still launch IsaacLab/Kit, so they are gated by the host inotify
limits documented in `docs/level_b_tracking_protocol.md`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OFFICIAL = ROOT / "reproduction/third_party/official/whole_body_tracking"
SCRIPTS = OFFICIAL / "scripts"
OUT = ROOT / "reproduction/generated/whole_body_tracking_local"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new)


def patch_csv_to_npz() -> tuple[str, str]:
    src = SCRIPTS / "csv_to_npz.py"
    text = src.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")\n',
        'parser.add_argument("--output_name", type=str, default="local_motion", help="The name of the motion npz file.")\n'
        'parser.add_argument("--output_file", type=str, required=True, help="Local output .npz path.")\n',
        "csv add output_file",
    )
    old = '''            np.savez("/tmp/motion.npz", **log)

            import wandb

            COLLECTION = args_cli.output_name
            run = wandb.init(project="csv_to_npz", name=COLLECTION)
            print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            REGISTRY = "motions"
            logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
'''
    new = '''            from pathlib import Path

            output_file = Path(args_cli.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_file, **log)
            print(f"[INFO]: Motion saved locally: {output_file}")
            simulation_app.close()
            return
'''
    text = replace_once(text, old, new, "csv local save")
    return src.name, text


def patch_replay_npz() -> tuple[str, str]:
    src = SCRIPTS / "replay_npz.py"
    text = src.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")\n',
        'parser.add_argument("--motion_file", type=str, required=True, help="Local motion .npz path.")\n'
        'parser.add_argument("--max_steps", type=int, default=200, help="Maximum render steps for smoke replay.")\n',
        "replay args",
    )
    old = '''    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
'''
    new = '''    motion_file = args_cli.motion_file
'''
    text = replace_once(text, old, new, "replay local motion")
    old = '''    while simulation_app.is_running():
        time_steps += 1
'''
    new = '''    smoke_steps = 0
    while simulation_app.is_running():
        smoke_steps += 1
        if smoke_steps > args_cli.max_steps:
            print(f"[INFO]: Replay smoke reached max_steps={args_cli.max_steps}")
            simulation_app.close()
            return
        time_steps += 1
'''
    text = replace_once(text, old, new, "replay max steps")
    return src.name, text


def patch_train() -> tuple[str, str]:
    src = SCRIPTS / "rsl_rl/train.py"
    text = src.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")\n',
        'parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")\n'
        'parser.add_argument("--motion_file", type=str, default=None, help="Local motion .npz path.")\n',
        "train args",
    )
    old = '''    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
'''
    new = '''    # load the motion file from a local smoke artifact or the WandB registry
    registry_name = args_cli.registry_name
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        registry_name = f"local:{args_cli.motion_file}"
        print(f"[INFO]: Using local motion file: {args_cli.motion_file}")
    else:
        if registry_name is None:
            raise ValueError("--registry_name or --motion_file is required")
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
'''
    text = replace_once(text, old, new, "train local motion")
    return "train_local.py", text


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = ["generated_file\tofficial_source_sha256\tgenerated_sha256"]
    for generated_name, text in [patch_csv_to_npz(), patch_replay_npz(), patch_train()]:
        if generated_name == "train_local.py":
            out_path = OUT / "rsl_rl" / generated_name
            src_path = SCRIPTS / "rsl_rl/train.py"
        else:
            out_path = OUT / generated_name.replace(".py", "_local.py")
            src_path = SCRIPTS / generated_name
        write_file(out_path, text)
        src_text = src_path.read_text(encoding="utf-8")
        manifest.append(f"{out_path}\t{sha256_text(src_text)}\t{sha256_text(text)}")
        print(out_path)
    cli_args_src = SCRIPTS / "rsl_rl/cli_args.py"
    cli_args_text = cli_args_src.read_text(encoding="utf-8")
    cli_args_out = OUT / "rsl_rl/cli_args.py"
    write_file(cli_args_out, cli_args_text)
    manifest.append(f"{cli_args_out}\t{sha256_text(cli_args_text)}\t{sha256_text(cli_args_text)}")
    print(cli_args_out)
    (OUT / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
