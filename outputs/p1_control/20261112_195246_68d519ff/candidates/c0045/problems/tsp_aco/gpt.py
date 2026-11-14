import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    valid = np.isfinite(d) & (d > 0.0) & ~diagonal
    positive = d[valid]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = max(float(np.max(positive)), 1.0)
    safe = np.where(valid, d, fallback)
    np.fill_diagonal(safe, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    ordered = np.sort(safe, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = safe / (geometric_scale + eps)
    distance_term = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_term = np.exp(
        -0.5
        * (
            ranks / max(float(n - 1), 1.0)
            + ranks.T / max(float(n - 1), 1.0)
        )
    )

    nearest = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest):
        if i != j and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_score = distance_term * rank_term
    score = local_score * (1.0 + 0.25 * mutual)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
