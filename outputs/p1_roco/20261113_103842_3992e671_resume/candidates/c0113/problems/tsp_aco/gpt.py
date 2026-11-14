import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_edges = d[np.isfinite(d)]

    if finite_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    global_scale = float(np.median(finite_edges))
    if not np.isfinite(global_scale) or global_scale <= eps:
        positive_edges = finite_edges[finite_edges > eps]
        global_scale = (
            float(np.median(positive_edges))
            if positive_edges.size
            else 1.0
        )
    global_scale = max(global_scale, eps)

    k = min(max(1, int(np.ceil(np.sqrt(n)))), n - 1)

    row_sorted = np.sort(d, axis=1)
    col_sorted = np.sort(d, axis=0)

    row_scale = row_sorted[:, k - 1]
    col_scale = col_sorted[k - 1, :]

    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > eps),
        row_scale,
        global_scale,
    )
    col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > eps),
        col_scale,
        global_scale,
    )

    row_ratio = d / row_scale[:, None]
    col_ratio = d / col_scale[None, :]

    row_affinity = np.exp(-np.minimum(row_ratio, 50.0))
    col_affinity = np.exp(-np.minimum(col_ratio, 50.0))

    relative_affinity = np.sqrt(row_affinity * col_affinity)

    row_rank = np.argsort(np.argsort(d, axis=1), axis=1)
    col_rank = np.argsort(np.argsort(d, axis=0), axis=0)

    row_rank = row_rank.astype(float)
    col_rank = col_rank.astype(float)

    row_rank_score = 1.0 / (1.0 + row_rank)
    col_rank_score = 1.0 / (1.0 + col_rank)
    rank_agreement = np.sqrt(row_rank_score * col_rank_score)

    row_competition = np.exp(-np.minimum(row_ratio, 50.0))
    col_competition = np.exp(-np.minimum(col_ratio, 50.0))

    row_scarcity = 1.0 / (
        eps + np.sum(row_competition, axis=1, keepdims=True)
    )
    col_scarcity = 1.0 / (
        eps + np.sum(col_competition, axis=0, keepdims=True)
    )

    scarcity = np.sqrt(row_scarcity * col_scarcity)
    scarcity /= np.max(scarcity) + eps

    global_affinity = np.exp(-np.minimum(d / global_scale, 50.0))

    score = (
        0.34 * relative_affinity
        + 0.24 * rank_agreement
        + 0.24 * scarcity * relative_affinity
        + 0.18 * global_affinity
    )

    score = np.where(np.isfinite(score), score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
