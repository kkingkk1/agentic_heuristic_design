import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    distances = 0.5 * (distances + distances.T)

    valid = (
        np.isfinite(distances)
        & (distances > 0.0)
        & finite_positive
        & finite_positive.T
    )
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, global_scale)

    work = np.where(valid, distances, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(3, n - 1)
    nearest = np.partition(work, neighbor_count - 1, axis=1)[
        :, :neighbor_count
    ]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / geometric_scale
    relative_length = np.minimum(relative_length, 1e6)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    normalized_ranks = ranks / max(n - 1, 1)
    mutual_rank = 0.5 * (
        np.log1p(normalized_ranks)
        + np.log1p(normalized_ranks.T)
    )

    scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_gap = np.minimum(scale_gap, 20.0)

    density = np.sum(
        valid * np.exp(-np.minimum(work / local_scale[:, None], 30.0)),
        axis=1,
    )
    crowding = np.log1p(density[:, None] + density[None, :])

    log_score = (
        -0.95 * np.log1p(relative_length)
        -0.30 * work / global_scale
        -1.35 * mutual_rank
        -0.22 * scale_gap
        -0.08 * crowding
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    log_score -= np.max(log_score[finite_scores])
    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
