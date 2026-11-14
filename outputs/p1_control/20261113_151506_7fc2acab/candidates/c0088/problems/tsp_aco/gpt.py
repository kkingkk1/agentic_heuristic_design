import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric matrix of edge-promise scores for a TSP instance."""
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

    finite_positive = d[np.isfinite(d) & (d > 0.0)]
    base_scale = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )
    base_scale = max(base_scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(d, order, axis=1)

    nearest = sorted_d[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, base_scale)

    third_index = min(2, n - 2)
    local_scale = sorted_d[:, third_index]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        base_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    closeness = np.exp(-d / pair_scale)
    closeness[~np.isfinite(closeness)] = 0.0

    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[np.arange(n), np.arange(n)] = 0.0

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_closeness = np.sqrt(closeness * closeness.T)

    nearest_neighbor = np.zeros((n, n), dtype=float)
    nearest_neighbor[np.arange(n), order[:, 0]] = 1.0
    mutual_nearest = nearest_neighbor * nearest_neighbor.T

    third_rank_score = 1.0 / (1.0 + np.maximum(ranks - 2.0, 0.0))
    third_rank_score[ranks < 2.0] = 0.0
    third_support = np.sqrt(third_rank_score * third_rank_score.T)

    savings = nearest[:, None] + nearest[None, :] - d
    savings = np.maximum(savings, 0.0)
    positive_savings = savings[np.isfinite(savings) & (savings > 0.0)]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else base_scale
    )
    savings_scale = max(savings_scale, np.finfo(float).eps)
    savings_score = np.tanh(savings / savings_scale)
    savings_score[~np.isfinite(savings_score)] = 0.0

    score = (
        0.38 * mutual_closeness
        + 0.27 * mutual_rank
        + 0.12 * closeness
        + 0.10 * third_support
        + 0.08 * savings_score
        + 0.05 * mutual_nearest
    )

    implausible = ranks >= min(6.0, float(n - 1))
    suppression = np.ones_like(score)
    suppression[implausible] *= 0.72
    suppression[implausible.T] *= 0.72
    score *= suppression

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
