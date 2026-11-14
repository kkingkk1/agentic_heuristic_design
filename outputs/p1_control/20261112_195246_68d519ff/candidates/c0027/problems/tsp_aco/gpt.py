import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return prior indicators for promising TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores
    if n == 2:
        scores[0, 1] = scores[1, 0] = 1.0
        return scores

    work = distances.copy()
    np.fill_diagonal(work, np.nan)

    positive = work[np.isfinite(work) & (work > 0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    eps = max(global_scale * 1e-12, np.finfo(float).eps)

    row_scales = np.nanmedian(np.where(work > 0, work, np.nan), axis=1)
    row_scales = np.where(
        np.isfinite(row_scales) & (row_scales > 0),
        row_scales,
        global_scale,
    )

    normalized = work / np.sqrt(
        row_scales[:, None] * row_scales[None, :] + eps
    )

    order = np.argsort(np.where(np.isfinite(work), work, np.inf), axis=1)
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values

    ranks = ranks / max(n - 1, 1)
    endpoint_competition = np.exp(-0.5 * (ranks + ranks.T))

    distance_affinity = np.exp(-np.nan_to_num(normalized, nan=np.inf))
    relative_affinity = np.exp(
        -0.5
        * (
            work / (row_scales[:, None] + eps)
            + work / (row_scales[None, :] + eps)
        )
    )

    scores = (
        distance_affinity
        * relative_affinity
        * (0.5 + 0.5 * endpoint_competition)
    )

    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    row_max = scores.max(axis=1, keepdims=True)
    scores = np.divide(
        scores,
        row_max,
        out=np.zeros_like(scores),
        where=row_max > 0,
    )

    return scores.astype(float, copy=False)
