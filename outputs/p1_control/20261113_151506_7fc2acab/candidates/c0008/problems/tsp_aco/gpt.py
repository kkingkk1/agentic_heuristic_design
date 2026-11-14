import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    base_scale = max(base_scale, np.finfo(float).eps)

    sorted_d = np.sort(d, axis=1)
    local_scale = sorted_d[:, min(2, n - 2)]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        base_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    closeness = np.exp(-d / pair_scale)
    closeness[~np.isfinite(closeness)] = 0.0

    row_order = np.argsort(d, axis=1)
    ranks = np.empty_like(row_order, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[np.arange(n), np.arange(n)] = 0.0

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_closeness = np.sqrt(closeness * closeness.T)

    nearest = np.min(d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, base_scale)
    savings = nearest[:, None] + nearest[None, :] - d
    savings = np.maximum(savings, 0.0)
    savings_scale = max(float(np.median(savings[savings > 0])), base_scale)
    savings_score = np.tanh(savings / max(savings_scale, np.finfo(float).eps))

    score = (
        0.45 * mutual_closeness
        + 0.30 * mutual_rank
        + 0.15 * closeness
        + 0.10 * savings_score
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
