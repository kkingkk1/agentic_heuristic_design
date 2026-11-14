import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d)

    positive = d[finite & (d > 0)]
    scale = np.median(positive) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse = 1.0 / (d + 0.035 * scale)
    inverse[~finite] = 0.0

    ranks = np.full((n, n), n, dtype=float)
    nearest = np.zeros((n, n), dtype=float)

    nearest_k = max(1, min(n - 1, int(np.ceil(0.9 * np.sqrt(n)))))

    for i in range(n):
        order = np.argsort(d[i], kind="stable")
        valid_order = order[np.isfinite(d[i, order])]

        ranks[i, valid_order] = np.arange(valid_order.size, dtype=float)
        nearest[i, valid_order[:nearest_k]] = 1.0

    rank_scale = max(1.0, 0.85 * np.sqrt(n))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~finite] = 0.0

    mutual = nearest * nearest.T
    local_support = (
        1.0
        + 0.95 * mutual
        + 0.20 * (nearest + nearest.T)
    )

    score = inverse * (0.68 + 0.22 * rank_score) * local_support
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    valid = finite & ~np.eye(n, dtype=bool)
    mean_score = score[valid].mean() if np.any(valid) else 0.0

    if mean_score > 0.0:
        score /= mean_score

    return score.astype(float, copy=False)
