import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be square")

    if n == 0:
        return np.zeros((0, 0), dtype=float)

    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    positive = d[np.isfinite(d) & (d > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    d[d == 0] = fallback

    neighbor_count = max(1, min(7, n - 1))
    nearest = np.partition(
        d,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    local_scale = np.median(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / pair_scale

    proximity = np.exp(-1.15 * relative_distance)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_temperature = max(1.0, n / 3.5)
    row_rank = np.exp(-ranks / rank_temperature)
    bidirectional_rank = np.sqrt(row_rank * row_rank.T)

    ambiguity = np.sum(
        np.exp(-d / (1.10 * local_scale[:, None])),
        axis=1,
    )
    scarcity = 1.0 / np.sqrt(
        1.0 + ambiguity[:, None] + ambiguity[None, :]
    )

    score = proximity * bidirectional_rank * (1.0 + 1.25 * scarcity)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
