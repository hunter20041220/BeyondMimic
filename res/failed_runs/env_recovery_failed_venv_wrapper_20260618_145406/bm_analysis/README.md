# bm_analysis

This project-local prefix is a wrapper around the host Python at /usr/bin/python3.
It exists to keep environment metadata and lock files under ROOT/envs after migration.

A fully isolated venv could not be created because this host lacks ensurepip/python3.10-venv.
Do not treat this wrapper as a complete paper training environment or an IsaacLab runtime.
