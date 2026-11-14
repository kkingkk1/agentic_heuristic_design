import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros_like(d, dtype=float)

    if n <= 1:
        return scores

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_edges = np.isfinite(d) & off_diagonal

    if not np.any(finite_edges):
        return scores

    finite_values = d[finite_edges]
    positive_values = finite_values[finite_values > 0]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.max(np.abs(finite_values))), 1.0)
    )
    upper_bound = max(float(np.max(finite_values)), fallback) * 2.5 + 1.0

    clean = np.where(
        finite_edges,
        np.maximum(d, 0.0),
        upper_bound,
    )
    np.fill_diagonal(clean, upper_bound)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    sorted_distances = np.sort(clean, axis=1)
    local_scale = np.mean(sorted_distances[:, :k], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = clean / np.maximum(pair_scale, np.finfo(float).eps)

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)

    proximity_score = np.exp(-1.15 * normalized)
    reciprocal_score = 1.0 / (1.0 + normalized)

    scores = (
        0.55 * proximity_score
        + 0.30 * reciprocal_score
        + 0.15 * rank_score
    )

    scores[~finite_edges] = 0.0
    np.fill_diagonal(scores, 0.0)

    if np.allclose(d, d.T, equal_nan=True):
        scores = 0.5 * (scores + scores.T)
        np.fill_diagonal(scores, 0.0)

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
