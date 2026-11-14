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

    d = np.maximum(distances, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    nearest = d[np.arange(n), order[:, 0]]
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        scale,
    )
    nearest = np.maximum(nearest, np.finfo(float).eps)

    pair_scale = np.sqrt(nearest[:, None] * nearest[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    distance_score = pair_scale / (pair_scale + d)
    distance_score[~np.isfinite(distance_score)] = 0.0

    rank_score = 1.0 / (1.0 + ranks)
    rank_score[np.arange(n), np.arange(n)] = 0.0
    reciprocal_rank = np.sqrt(rank_score * rank_score.T)
    reciprocal_rank[~np.isfinite(reciprocal_rank)] = 0.0

    nearest_choice = np.zeros((n, n), dtype=float)
    nearest_choice[np.arange(n), order[:, 0]] = 1.0
    mutual_near = nearest_choice * nearest_choice.T

    if n > 2:
        second_choice = np.zeros((n, n), dtype=float)
        second_choice[np.arange(n), order[:, 1]] = 1.0
        mutual_second = second_choice * second_choice.T
        mutual_near = np.maximum(mutual_near, 0.5 * mutual_second)

    score = (
        0.56 * distance_score
        + 0.32 * reciprocal_rank
        + 0.12 * mutual_near
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
