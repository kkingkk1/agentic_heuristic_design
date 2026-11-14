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

    distances = np.maximum(distances, 0.0)
    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive = d[np.isfinite(d) & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    base_scale = max(base_scale, eps)

    nearest_index = np.argmin(d, axis=1)
    nearest_distance = d[np.arange(n), nearest_index]
    nearest_distance = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0.0),
        nearest_distance,
        base_scale,
    )
    nearest_distance = np.maximum(nearest_distance, eps)

    pair_scale = np.sqrt(
        nearest_distance[:, None] * nearest_distance[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    local_closeness = np.exp(-d / pair_scale)
    local_closeness[~np.isfinite(local_closeness)] = 0.0

    nearest_choice = np.zeros((n, n), dtype=float)
    nearest_choice[np.arange(n), nearest_index] = 1.0
    mutual_nearest = nearest_choice * nearest_choice.T

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[np.arange(n), np.arange(n)] = 0.0
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    savings = (
        nearest_distance[:, None]
        + nearest_distance[None, :]
        - d
    )
    savings = np.maximum(savings, 0.0)
    positive_savings = savings[savings > 0.0]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else base_scale
    )
    savings_scale = max(savings_scale, eps)
    savings_score = np.tanh(savings / savings_scale)
    savings_score[~np.isfinite(savings_score)] = 0.0

    score = (
        0.65 * mutual_nearest
        + 0.25 * np.sqrt(local_closeness * local_closeness.T)
        + 0.08 * mutual_rank
        + 0.02 * savings_score
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1)
    pair_max = np.sqrt(row_max[:, None] * row_max[None, :])
    normalized_score = np.divide(
        score,
        pair_max,
        out=np.zeros_like(score),
        where=pair_max > 0.0,
    )
    normalized_score = 0.5 * (normalized_score + normalized_score.T)
    np.fill_diagonal(normalized_score, 0.0)

    score = 0.80 * score + 0.20 * normalized_score
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    global_max = float(np.max(score))
    if global_max > 0.0 and np.isfinite(global_max):
        score = score / global_max
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
