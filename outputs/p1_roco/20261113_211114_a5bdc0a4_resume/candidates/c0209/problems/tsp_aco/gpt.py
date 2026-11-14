import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    offdiag = ~np.eye(n, dtype=bool)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid = offdiag & finite & nonnegative
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    observed = symmetric[valid]

    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    spread = float(np.median(np.abs(observed - np.median(observed))))
    if not np.isfinite(spread) or spread <= 0.0:
        spread = scale

    safe_scale = max(scale, eps)
    temperature = max(
        0.35 * safe_scale + 0.65 * spread,
        np.sqrt(eps) * max(safe_scale, 1.0),
    )

    clean = np.where(valid, np.maximum(symmetric, 0.0), 0.0)
    centered = np.maximum(clean / temperature, 0.0)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-np.minimum(centered[valid], 60.0))

    row_total = affinity.sum(axis=1, keepdims=True)
    transition = affinity / np.maximum(row_total, eps)

    reverse_transition = transition.T
    reciprocal_flow = np.sqrt(
        np.maximum(transition * reverse_transition, 0.0)
    )

    two_step = transition @ transition
    three_step = two_step @ transition

    common_support = np.sqrt(
        np.maximum(two_step * reverse_transition, 0.0)
    )
    cyclic_support = np.sqrt(
        np.maximum(three_step * reverse_transition, 0.0)
    )

    row_entropy = -np.sum(
        transition * np.log(np.maximum(transition, eps)),
        axis=1,
    )
    row_entropy /= np.maximum(
        np.log(np.maximum(np.sum(valid, axis=1), 2.0)),
        1.0,
    )
    row_entropy = np.clip(row_entropy, 0.0, 1.0)

    node_mass = np.sqrt(
        np.maximum(
            transition.sum(axis=0) * transition.sum(axis=1),
            eps,
        )
    )
    node_mass /= max(float(np.max(node_mass)), eps)

    entropy_factor = np.sqrt(
        np.maximum(
            (1.0 - row_entropy[:, None])
            * (1.0 - row_entropy[None, :]),
            eps,
        )
    )

    center_penalty = 1.0 / np.sqrt(
        np.maximum(node_mass[:, None] + node_mass[None, :], eps)
    )
    center_penalty /= max(float(np.max(center_penalty)), eps)

    local_scale = np.median(
        np.where(valid, clean, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        safe_scale,
    )

    relative_distance = clean / np.maximum(
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        eps,
    )
    closeness = np.exp(-np.minimum(relative_distance, 30.0))
    closeness[~valid] = 0.0

    score = (
        0.30 * reciprocal_flow
        + 0.25 * common_support
        + 0.20 * cyclic_support
        + 0.15 * closeness * entropy_factor
        + 0.10 * center_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    valid_count = float(np.sum(valid))
    exploration = min(0.12, max(0.035, 2.0 / max(n, 2)))
    score = (
        (1.0 - exploration) * score
        + exploration * valid.astype(float) / max(valid_count, 1.0)
    )

    for _ in range(3):
        row_sum = score.sum(axis=1, keepdims=True)
        score /= np.maximum(row_sum, eps)
        score = 0.5 * (score + score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
