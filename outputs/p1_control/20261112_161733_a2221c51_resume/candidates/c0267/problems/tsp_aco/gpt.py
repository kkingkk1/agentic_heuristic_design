import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-supported edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric_distance = np.zeros((n, n), dtype=float)
    symmetric_distance[valid] = 0.5 * (
        d[valid] + d.T[valid]
    )

    observed = symmetric_distance[valid]
    scale = max(float(np.median(observed)), eps)

    normalized_distance = np.zeros_like(symmetric_distance)
    normalized_distance[valid] = symmetric_distance[valid] / scale

    with np.errstate(over="ignore", invalid="ignore"):
        direct_affinity = np.exp(
            -np.square(normalized_distance)
        )

    direct_affinity[~valid] = 0.0
    np.fill_diagonal(direct_affinity, 0.0)

    degree = np.sum(direct_affinity, axis=1)
    degree = np.maximum(degree, eps)

    transition = direct_affinity / degree[:, None]

    with np.errstate(invalid="ignore", over="ignore"):
        two_step_support = transition @ transition

    reciprocal_flow = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(transition.T, 0.0)
    )

    support_scale = np.sqrt(
        degree[:, None] * degree[None, :]
    )
    normalized_support = np.divide(
        two_step_support,
        support_scale,
        out=np.zeros_like(two_step_support),
        where=support_scale > 0.0,
    )

    degree_balance = np.minimum(
        degree[:, None],
        degree[None, :]
    ) / np.maximum(
        degree[:, None],
        degree[None, :],
    )

    score = (
        0.36 * direct_affinity
        + 0.34 * normalized_support
        + 0.22 * reciprocal_flow
        + 0.08 * degree_balance
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
