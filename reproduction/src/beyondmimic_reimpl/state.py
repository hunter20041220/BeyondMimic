"""State projection helpers for paper-formula debug artifacts."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from beyondmimic_reimpl.validation import ensure_finite


ROOT_STATE_DIM = 15
DEFAULT_TARGET_BODY_COUNT = 14
TARGET_BODY_FEATURE_DIM = DEFAULT_TARGET_BODY_COUNT * 6
HYBRID_STATE_DIM = ROOT_STATE_DIM + TARGET_BODY_FEATURE_DIM
EMPHASIS_COEFFICIENT = 6
DEFAULT_GAUSSIAN_ROWS = 64


@dataclass(frozen=True)
class HybridStateSchema:
    """Paper S3 hybrid state layout for one trajectory timestep.

    The default 99-D layout is root pose/twist in the current yaw-centric
    character frame plus target-body position/velocity in each local root
    yaw frame.  It intentionally omits reference-motion commands.
    """

    target_body_count: int = DEFAULT_TARGET_BODY_COUNT
    root_dim: int = ROOT_STATE_DIM
    coefficient: int = EMPHASIS_COEFFICIENT
    gaussian_rows: int = DEFAULT_GAUSSIAN_ROWS

    @property
    def body_feature_dim(self) -> int:
        return self.target_body_count * 6

    @property
    def state_dim(self) -> int:
        return self.root_dim + self.body_feature_dim

    @property
    def projected_dim(self) -> int:
        return self.state_dim + self.gaussian_rows

    @property
    def slices(self) -> dict[str, list[int]]:
        body_pos_start = self.root_dim
        body_vel_start = body_pos_start + 3 * self.target_body_count
        return {
            "root_pos_rel_current_frame": [0, 3],
            "root_rot6d_rel_current_frame": [3, 9],
            "root_lin_vel_rel_current_frame": [9, 12],
            "root_ang_vel_rel_current_frame": [12, 15],
            "body_pos_local_root_frame": [body_pos_start, body_vel_start],
            "body_lin_vel_local_root_frame": [body_vel_start, self.state_dim],
        }

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.update(
            {
                "body_feature_dim": self.body_feature_dim,
                "state_dim": self.state_dim,
                "projected_dim": self.projected_dim,
                "slices": self.slices,
            }
        )
        return data


def hybrid_state_schema(
    target_body_count: int = DEFAULT_TARGET_BODY_COUNT,
    coefficient: int = EMPHASIS_COEFFICIENT,
    gaussian_rows: int = DEFAULT_GAUSSIAN_ROWS,
) -> HybridStateSchema:
    """Return the default paper S3 99-D hybrid state schema."""
    if target_body_count <= 0 or coefficient <= 0 or gaussian_rows <= 0:
        raise ValueError("target_body_count, coefficient, and gaussian_rows must be positive")
    return HybridStateSchema(target_body_count=target_body_count, coefficient=coefficient, gaussian_rows=gaussian_rows)


def validate_hybrid_state(states: np.ndarray, schema: HybridStateSchema | None = None) -> np.ndarray:
    """Validate finite paper hybrid state tensors with last dimension 99."""
    schema = schema or hybrid_state_schema()
    arr = ensure_finite("hybrid_states", states)
    if arr.shape[-1] != schema.state_dim:
        raise ValueError(f"hybrid state last dim must be {schema.state_dim}, got {arr.shape}")
    return arr


def emphasis_projection(
    seed: int = 7,
    state_dim: int = HYBRID_STATE_DIM,
    root_dim: int = ROOT_STATE_DIM,
    coefficient: int = EMPHASIS_COEFFICIENT,
    gaussian_rows: int = DEFAULT_GAUSSIAN_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct paper yaw-centric state projection ``P`` and pseudoinverse.

    Shapes are ``P[(gaussian_rows+state_dim), state_dim]`` and
    ``P_inv[state_dim, (gaussian_rows+state_dim)]`` for 99-D yaw-centric
    trajectory state tokens.  The default root dimension is 15: 3-D relative
    root position, 6-D relative root orientation, 3-D relative linear velocity,
    and 3-D relative angular velocity.  The paper coefficient ``c=6`` scales
    the diagonal root-feature matrix ``B``; it is not the number of Gaussian
    rows in ``A``.
    """
    if min(state_dim, root_dim, coefficient, gaussian_rows) <= 0 or root_dim > state_dim:
        raise ValueError("state_dim/root_dim/coefficient/gaussian_rows must be positive with root_dim <= state_dim")
    rng = np.random.default_rng(seed)
    b = np.zeros((root_dim, state_dim), dtype=np.float64)
    b[:, :root_dim] = coefficient * np.eye(root_dim)
    a = rng.normal(size=(gaussian_rows, root_dim))
    p = np.vstack([a @ b, np.eye(state_dim)])
    return p, np.linalg.pinv(p)


def project_hybrid_state(
    states: np.ndarray,
    seed: int = 7,
    schema: HybridStateSchema | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project paper hybrid states with the emphasis projection matrix."""
    schema = schema or hybrid_state_schema()
    arr = validate_hybrid_state(states, schema)
    p, p_inv = emphasis_projection(
        seed=seed,
        state_dim=schema.state_dim,
        root_dim=schema.root_dim,
        coefficient=schema.coefficient,
        gaussian_rows=schema.gaussian_rows,
    )
    return arr @ p.T, p, p_inv


def unproject_hybrid_state(
    projected_states: np.ndarray,
    projection_inverse: np.ndarray,
    schema: HybridStateSchema | None = None,
) -> np.ndarray:
    """Recover 99-D hybrid states from projected-space predictions."""
    schema = schema or hybrid_state_schema()
    projected = ensure_finite("projected_states", projected_states)
    p_inv = ensure_finite("projection_inverse", projection_inverse)
    expected = schema.projected_dim
    if projected.shape[-1] != expected:
        raise ValueError(f"projected state last dim must be {expected}, got {projected.shape}")
    if p_inv.shape != (schema.state_dim, schema.projected_dim):
        raise ValueError(f"projection_inverse must have shape {(schema.state_dim, schema.projected_dim)}, got {p_inv.shape}")
    recovered = projected @ p_inv.T
    return validate_hybrid_state(recovered, schema)


def smoothness_penalty(path: np.ndarray) -> float:
    """Mean squared second-difference penalty for trajectory path ``[T,D]``."""
    path = ensure_finite("path", path)
    if path.ndim < 2 or path.shape[0] < 3:
        raise ValueError(f"path must have shape [T,D] with T>=3, got {path.shape}")
    second = path[2:] - 2.0 * path[1:-1] + path[:-2]
    return float(np.mean(second**2))


__all__ = [
    "DEFAULT_TARGET_BODY_COUNT",
    "DEFAULT_GAUSSIAN_ROWS",
    "EMPHASIS_COEFFICIENT",
    "HYBRID_STATE_DIM",
    "HybridStateSchema",
    "ROOT_STATE_DIM",
    "TARGET_BODY_FEATURE_DIM",
    "emphasis_projection",
    "hybrid_state_schema",
    "project_hybrid_state",
    "smoothness_penalty",
    "unproject_hybrid_state",
    "validate_hybrid_state",
]
