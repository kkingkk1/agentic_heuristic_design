import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    ranks[np.arange(n), np.arange(n)] = np.inf

    nearest = np.min(d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, scale)

    second_nearest = np.partition(d, 1, axis=1)[:, 1]
    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        nearest,
    )

    # Cheap distance quality: bounded inverse distance.
    inverse_distance = scale / (scale + d)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    # Favor edges that are simultaneously near neighbors of both endpoints.
    rank_quality = 1.0 / (1.0 + ranks)
    rank_quality[np.arange(n), np.arange(n)] = 0.0
    mutual_rank = np.sqrt(rank_quality * rank_quality.T)

    # Favor edges that improve over each endpoint's local alternatives.
    savings = nearest[:, None] + nearest[None, :] - d
    savings = np.maximum(savings, 0.0)
    savings_scale = float(np.median(savings[savings > 0])) if np.any(savings > 0) else scale
    savings_scale = max(savings_scale, np.finfo(float).eps)
    savings_score = savings / (savings + savings_scale)
    savings_score[~np.isfinite(savings_score)] = 0.0

    # Mild scarcity correction discourages repeatedly selecting one hub.
    local_budget = nearest + second_nearest
    scarcity = np.sqrt(
        local_budget[:, None] / (local_budget[:, None] + d + scale)
        * local_budget[None, :] / (local_budget[None, :] + d + scale)
    )
    scarcity[~np.isfinite(scarcity)] = 0.0

    score = (
        0.40 * mutual_rank
        + 0.30 * inverse_distance
        + 0.20 * savings_score
        + 0.10 * scarcity
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
