import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    positive_values = finite_values[finite_values > 0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    positive_mask = np.isfinite(d) & (d > 0)
    safe_d = np.where(np.isfinite(d), d, 0.0)

    row_positive_counts = positive_mask.sum(axis=1, keepdims=True)
    row_scales = np.divide(
        safe_d.sum(axis=1, keepdims=True),
        np.maximum(row_positive_counts, 1),
    )
    row_scales = np.maximum(row_scales, eps)

    inverse_term = np.zeros_like(d)
    inverse_term[positive_mask] = 1.0 / np.maximum(d[positive_mask], eps)
    inverse_term /= np.maximum(inverse_term.sum(axis=1, keepdims=True), eps)

    adaptive_term = np.zeros_like(d)
    adaptive_term[positive_mask] = np.exp(
        -d[positive_mask] / row_scales[np.broadcast_to(
            np.arange(n)[:, None], (n, n)
        )[positive_mask]]
    )
    adaptive_term /= np.maximum(
        adaptive_term.sum(axis=1, keepdims=True),
        eps,
    )

    k = max(1, min(5, n - 1))
    nearest = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(d[i])
        selected = order[np.isfinite(d[i, order])][:k]
        if selected.size:
            nearest[i, selected] = 1.0

    mutual_affinity = nearest * nearest.T
    one_sided_affinity = 0.5 * (nearest + nearest.T)
    neighborhood_term = 0.7 * mutual_affinity + 0.3 * one_sided_affinity

    rank_term = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(d[i])
        valid = order[np.isfinite(d[i, order])]
        if valid.size:
            ranks = np.arange(valid.size, dtype=float)
            rank_term[i, valid] = 1.0 / (1.0 + ranks)

    rank_term /= np.maximum(rank_term.sum(axis=1, keepdims=True), eps)

    score = (
        0.50 * inverse_term
        + 0.25 * adaptive_term
        + 0.15 * rank_term
        + 0.10 * neighborhood_term
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if maximum > 0.0:
        score /= maximum

    return score
