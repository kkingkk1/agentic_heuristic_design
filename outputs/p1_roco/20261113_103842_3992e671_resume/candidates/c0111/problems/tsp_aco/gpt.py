import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    directed_valid = finite & (d > 0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid | directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d_t = d.T
    both = directed_valid & directed_valid.T

    symmetric_d = np.where(
        both,
        0.5 * (d + d_t),
        np.where(directed_valid, d, d_t),
    )
    symmetric_d = np.where(valid, symmetric_d, np.inf)
    np.fill_diagonal(symmetric_d, np.inf)

    positive_distances = symmetric_d[valid]
    scale = np.median(positive_distances)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized_d = symmetric_d / scale

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / (1.0 + normalized_d[valid])
    max_inverse = np.max(inverse_distance)
    if max_inverse > 0:
        inverse_distance /= max_inverse

    order = np.argsort(symmetric_d, axis=1, kind="stable")
    ordered_valid = np.take_along_axis(valid, order, axis=1)

    cumulative_rank = np.cumsum(ordered_valid, axis=1)
    ordered_rank_score = np.where(
        ordered_valid,
        1.0 / np.maximum(cumulative_rank, 1),
        0.0,
    )

    rank_score = np.zeros((n, n), dtype=float)
    np.put_along_axis(rank_score, order, ordered_rank_score, axis=1)

    row_max = np.max(rank_score, axis=1, keepdims=True)
    normalized_rank = rank_score / np.maximum(row_max, 1e-12)
    reciprocal_rank = np.sqrt(normalized_rank * normalized_rank.T)

    nearest = np.argmin(symmetric_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    source = np.arange(n)
    mutual[source, nearest] = (
        valid[source, nearest]
        & (nearest[nearest] == source)
    )
    mutual = np.maximum(mutual, mutual.T)

    score = (
        0.60 * inverse_distance
        + 0.25 * reciprocal_rank
        + 0.15 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score[valid] = np.maximum(score[valid], 1e-12)
    return score
