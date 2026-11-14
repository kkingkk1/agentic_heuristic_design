import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & off_diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = np.maximum(distances, 0.0)
    positive = d[valid & (d > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    eps = np.finfo(float).eps * max(global_scale, 1.0)

    safe_d = np.where(valid, d, np.inf)
    positive_safe_d = np.where(
        valid & (d > 0),
        d,
        global_scale,
    )

    local_values = np.sort(positive_safe_d, axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n))))
    local_scale = np.mean(local_values[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    normalized_distance = np.where(
        valid,
        safe_d / endpoint_scale,
        np.inf,
    )

    short_edge = np.exp(-np.minimum(normalized_distance, 60.0))

    inverse_distance = np.where(
        valid,
        1.0 / np.maximum(safe_d, eps),
        0.0,
    )
    inverse_distance /= max(
        float(np.max(inverse_distance)),
        eps,
    )

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score, 0.0)
        * np.maximum(rank_score.T, 0.0)
    )

    competition = np.sum(
        np.where(
            valid,
            np.exp(-np.minimum(normalized_distance, 20.0)),
            0.0,
        ),
        axis=1,
    )
    competition = np.maximum(competition, eps)
    diversity = 1.0 / np.sqrt(
        competition[:, None] * competition[None, :]
    )

    score = (
        np.maximum(short_edge, eps) ** 0.45
        * np.maximum(inverse_distance, eps) ** 0.20
        * np.maximum(mutual_rank, eps) ** 0.25
        * np.maximum(diversity, eps) ** 0.10
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
