import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array.")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    finite = np.isfinite(d)
    if not np.all(finite):
        finite_values = d[finite]
        replacement = (
            float(np.max(finite_values))
            if finite_values.size
            else 1.0
        )
        d[~finite] = replacement

    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, 0.0)

    positive = d[(d > 0) & ~np.eye(n, dtype=bool)]
    if positive.size == 0:
        return np.ones((n, n), dtype=float) - np.eye(n, dtype=float)

    scale = float(np.median(positive))
    scale = max(scale, np.finfo(float).eps)

    # Multi-scale inverse-distance preference.
    short_range = np.exp(-d / scale)
    long_range = 1.0 / (1.0 + d / scale)
    distance_score = 0.7 * short_range + 0.3 * long_range

    # Rank-based local preference: favor edges among each node's nearest neighbors.
    off_diag = ~np.eye(n, dtype=bool)
    ranks = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(d[i] + np.eye(n)[i] * np.inf)
        ranks[i, order] = np.arange(1, n + 1)

    rank_score = 1.0 / np.sqrt(np.maximum(ranks, 1.0))
    rank_score = 0.5 * (rank_score + rank_score.T)

    # Encourage edges that connect locally distinctive/sparse nodes.
    local_scale = np.median(
        np.where(off_diag, d, np.nan),
        axis=1
    )
    local_scale = np.nan_to_num(local_scale, nan=scale, posinf=scale, neginf=scale)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    sparsity = np.sqrt(local_scale[:, None] * local_scale[None, :])
    sparsity_score = 1.0 + 0.2 * np.clip(sparsity / scale, 0.0, 3.0)

    score = distance_score * (0.65 + 0.35 * rank_score) * sparsity_score
    score = 0.5 * (score + score.T)
    score[~off_diag] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
