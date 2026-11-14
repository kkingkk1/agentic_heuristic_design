import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric_distance = np.zeros((n, n), dtype=float)
    symmetric_distance[valid] = 0.5 * (d[valid] + d.T[valid])
    symmetric_distance = np.maximum(symmetric_distance, 0.0)

    observed = symmetric_distance[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
        scale = max(scale, eps)
    else:
        scale = 1.0

    # Bounded similarity avoids exponential extinction and preserves zero-cost edges.
    similarity = np.zeros((n, n), dtype=float)
    similarity[valid] = scale / (scale + symmetric_distance[valid])
    similarity[valid] = np.sqrt(similarity[valid])

    # Remove isolated-node artifacts while keeping the valid graph structure.
    row_mass = similarity.sum(axis=1)
    active = row_mass > 0.0

    if not np.any(active):
        return np.zeros((n, n), dtype=float)

    # Symmetric iterative balancing approximates a degree-regular soft graph.
    balanced = similarity.copy()
    target = np.ones(n, dtype=float)
    target[~active] = 0.0

    for _ in range(48):
        mass = balanced.sum(axis=1)
        factors = np.ones(n, dtype=float)
        usable = mass > eps
        factors[usable] = np.sqrt(
            target[usable] / np.maximum(mass[usable], eps)
        )
        factors = np.clip(factors, 0.05, 20.0)
        factors[~active] = 0.0
        balanced *= factors[:, None] * factors[None, :]
        balanced[~valid] = 0.0

    balanced_mass = balanced.sum(axis=1)
    normalized = np.zeros_like(balanced)
    usable = balanced_mass > eps
    normalized[usable] = (
        balanced[usable] / balanced_mass[usable, None]
    )

    # Two-step support rewards edges embedded in compatible local cycles.
    two_step = normalized @ normalized
    two_step = np.maximum(two_step, 0.0)
    two_step[~valid] = 0.0

    support_scale = np.zeros(n, dtype=float)
    support_scale[usable] = np.maximum(
        np.median(two_step[usable], axis=1),
        eps,
    )

    relative_support = np.zeros_like(two_step)
    relative_support[usable] = (
        two_step[usable] / support_scale[usable, None]
    )
    relative_support = np.clip(relative_support, 0.0, 8.0)

    score = (
        0.35 * similarity
        + 0.50 * balanced
        + 0.15 * np.sqrt(relative_support)
    )
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
