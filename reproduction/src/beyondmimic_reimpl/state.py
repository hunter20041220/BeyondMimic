"""State projection helpers for paper-formula debug artifacts."""

from __future__ import annotations

import numpy as np

from beyondmimic_reimpl.validation import ensure_finite


def emphasis_projection(seed: int = 7, state_dim: int = 99, root_dim: int = 18, coefficient: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Construct paper yaw-centric state projection ``P`` and pseudoinverse.

    Shapes are ``P[(coefficient*root_dim+state_dim), state_dim]`` and
    ``P_inv[state_dim, (coefficient*root_dim+state_dim)]`` for 99-D
    yaw-centric trajectory state tokens.
    """
    if min(state_dim, root_dim, coefficient) <= 0 or root_dim > state_dim:
        raise ValueError("state_dim/root_dim/coefficient must be positive with root_dim <= state_dim")
    rng = np.random.default_rng(seed)
    b = np.zeros((root_dim, state_dim), dtype=np.float64)
    b[:, :root_dim] = np.eye(root_dim)
    a = rng.normal(scale=0.25, size=(coefficient * root_dim, root_dim))
    p = np.vstack([a @ b, np.eye(state_dim)])
    return p, np.linalg.pinv(p)


def smoothness_penalty(path: np.ndarray) -> float:
    """Mean squared second-difference penalty for trajectory path ``[T,D]``."""
    path = ensure_finite("path", path)
    if path.ndim < 2 or path.shape[0] < 3:
        raise ValueError(f"path must have shape [T,D] with T>=3, got {path.shape}")
    second = path[2:] - 2.0 * path[1:-1] + path[:-2]
    return float(np.mean(second**2))
