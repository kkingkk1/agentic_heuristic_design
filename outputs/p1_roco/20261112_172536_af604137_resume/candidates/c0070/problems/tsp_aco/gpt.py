import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    safe_d = np.where(finite_positive, d, np.inf)

    positive_values = d[finite_positive]
    scale = np.median(positive_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive_values)
    scale = max(scale, np.finfo(float).eps)

    normalized = safe_d / scale

    inverse = np.where(
        finite_positive,
        1.0 / np.maximum(normalized, 1e-12),
        0.0,
    )

    ranks = np.full((n, n), n + 1.0, dtype=float)
    for i in range(n):
        order = np.argsort(safe_d[i], kind="stable")
        valid_order = order[np.isfinite(safe_d[i, order])]
        ranks[i, valid_order] = np.arange(1, valid_order.size + 1)

    rank_score = np.where(
        finite_positive,
        1.0 / np.sqrt(ranks),
        0.0,
    )

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if (
            j < n
            and i != j
            and np.isfinite(safe_d[i, j])
            and nearest[j] == i
        ):
            mutual[i, j] = 1.0

    mutual = np.maximum(mutual, mutual.T)

    row_sum = inverse.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, np.finfo(float).eps)
    local_preference = inverse / row_sum

    score = (
        0.52 * inverse
        + 0.28 * rank_score
        + 0.15 * local_preference
        + 0.05 * mutual
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~finite_positive] = 0.0

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
