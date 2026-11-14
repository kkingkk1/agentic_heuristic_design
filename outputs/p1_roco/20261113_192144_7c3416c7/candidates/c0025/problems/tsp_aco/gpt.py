import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = np.maximum(d, 0.0)

    finite_offdiag = np.isfinite(d)
    np.fill_diagonal(finite_offdiag, False)

    if not np.any(finite_offdiag):
        return np.zeros_like(d, dtype=float)

    finite_values = d[finite_offdiag]
    fallback = np.max(finite_values) * 2.0 + 1.0
    work_d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(work_d, fallback)

    scale = np.median(finite_values)
    if not np.isfinite(scale) or scale <= 0:
        positive = finite_values[finite_values > 0]
        scale = np.median(positive) if positive.size else 1.0

    inverse_distance = 1.0 / (work_d + 1e-12)
    inverse_distance /= np.max(inverse_distance[finite_offdiag]) + 1e-12

    rank_order = np.argsort(work_d, axis=1)
    ranks = np.empty_like(rank_order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, rank_order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))
    rank_score /= np.max(rank_score[finite_offdiag]) + 1e-12

    nearest = np.argmin(work_d + np.eye(n) * fallback, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i and i != j:
            mutual[i, j] = 1.0

    row_min = np.min(np.where(np.eye(n, dtype=bool), fallback, work_d), axis=1)
    col_min = np.min(np.where(np.eye(n, dtype=bool), fallback, work_d), axis=0)
    savings = row_min[:, None] + col_min[None, :] - work_d
    savings = np.maximum(savings, 0.0)
    savings /= np.max(savings[finite_offdiag]) + 1e-12

    compatibility = np.exp(-np.maximum(work_d - scale, 0.0) / (scale + 1e-12))
    compatibility /= np.max(compatibility[finite_offdiag]) + 1e-12

    heuristic = (
        0.45 * inverse_distance
        + 0.25 * rank_score
        + 0.15 * savings
        + 0.10 * compatibility
        + 0.05 * mutual
    )

    heuristic[~finite_offdiag] = 0.0
    np.fill_diagonal(heuristic, 0.0)

    maximum = np.max(heuristic)
    if maximum > 0:
        heuristic /= maximum

    return heuristic
