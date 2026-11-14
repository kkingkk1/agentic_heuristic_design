import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    finite_offdiag = np.isfinite(d)
    np.fill_diagonal(finite_offdiag, False)

    if not np.any(finite_offdiag):
        return np.zeros_like(d, dtype=float)

    valid_values = d[finite_offdiag]
    replacement = np.max(valid_values) * 10.0 + 1.0
    d[~finite_offdiag] = replacement
    np.fill_diagonal(d, np.inf)

    # Robust inverse-distance attractiveness.
    positive = d[np.isfinite(d) & (d > 0)]
    scale = np.median(positive) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    inverse_score = np.exp(-d / scale)
    inverse_score[~np.isfinite(inverse_score)] = 0.0

    # Rank-based local preference: short edges receive larger scores.
    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(2.0, np.sqrt(n)))
    rank_score[~np.isfinite(d)] = 0.0
    np.fill_diagonal(rank_score, 0.0)

    # Mutual-nearest-neighbor preference.
    nearest = np.argmin(d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Penalize edges that are long relative to the local alternatives
    # available at either endpoint.
    local_scale = np.partition(d, 1, axis=1)[:, 1]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)
    relative_length = d / np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_score = np.exp(-np.minimum(relative_length, 50.0))
    relative_score[~np.isfinite(relative_score)] = 0.0
    np.fill_diagonal(relative_score, 0.0)

    score = (
        0.45 * inverse_score
        + 0.25 * rank_score
        + 0.20 * relative_score
        + 0.10 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~finite_offdiag] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
