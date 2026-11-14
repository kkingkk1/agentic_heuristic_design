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

    distances = np.maximum(distances, 0.0)
    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0.0)]
    scale = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )
    scale = max(scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    nearest_distance = d[np.arange(n), order[:, 0]]
    nearest_distance = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0.0),
        nearest_distance,
        scale,
    )
    nearest_distance = np.maximum(nearest_distance, np.finfo(float).eps)

    pair_scale = np.sqrt(
        nearest_distance[:, None] * nearest_distance[None, :]
    )
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    relative_distance = d / pair_scale
    local_score = np.exp(-np.minimum(relative_distance, 30.0))
    local_score[~np.isfinite(local_score)] = 0.0

    rank_score = 1.0 / (1.0 + ranks)
    rank_score[np.arange(n), np.arange(n)] = 0.0
    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    nearest_choice = np.zeros((n, n), dtype=float)
    nearest_choice[np.arange(n), order[:, 0]] = 1.0
    mutual_nearest = nearest_choice * nearest_choice.T

    second_choice = np.zeros((n, n), dtype=float)
    if n > 2:
        second_choice[np.arange(n), order[:, 1]] = 1.0
    mutual_near = np.maximum(
        mutual_nearest,
        0.5 * (second_choice * second_choice.T),
    )

    row_median = np.median(
        np.where(np.isfinite(d), d, np.nan),
        axis=1,
    )
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        scale,
    )

    outlier_penalty = 1.0 / (
        1.0 + np.maximum(d / np.sqrt(row_median[:, None] * row_median[None, :]) - 1.0, 0.0)
    )
    outlier_penalty[~np.isfinite(outlier_penalty)] = 0.0

    score = (
        0.48 * local_score
        + 0.27 * reciprocal_rank
        + 0.20 * mutual_near
        + 0.05 * outlier_penalty
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
