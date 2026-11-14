import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 85.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    symmetric_valid = valid & valid.T
    d = np.where(valid, distances, fallback)
    d = np.maximum((d + d.T) * 0.5, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_scale = float(np.median(positive))
    scale_floor = max(positive_scale * 1e-10, np.finfo(float).eps)

    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n) + 0.5))))
    top_indices = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    top_distances = np.take_along_axis(d, top_indices, axis=1)

    local_scale = np.mean(top_distances, axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    if n > 2:
        second_partition = np.partition(d, kth=1, axis=1)
        alternative = second_partition[:, 1]
    else:
        alternative = top_distances[:, 0]

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    exponent = (
        d / (pair_scale + np.finfo(float).eps)
        + 0.6 * d / (local_scale[:, None] + np.finfo(float).eps)
        + 0.6 * d / (local_scale[None, :] + np.finfo(float).eps)
    )
    affinity = np.exp(-np.clip(exponent, 0.0, 120.0))

    tail_rank = float(k)
    rank_score = np.full((n, n), 1.0 / np.sqrt(1.0 + tail_rank))

    local_order = np.argsort(top_distances, axis=1, kind="stable")
    local_ranks = np.empty_like(local_order, dtype=float)
    local_ranks[
        np.arange(n)[:, None],
        local_order
    ] = np.arange(k, dtype=float)[None, :]

    local_scores = 1.0 / np.sqrt(1.0 + local_ranks)
    rows = np.arange(n)[:, None]
    rank_score[rows, top_indices] = local_scores

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    margin = alternative_scale / (
        d + alternative_scale + scale_floor
    )
    margin = np.clip(margin, 0.0, 1.0)

    score = affinity * mutual_rank
    score *= 1.0 + 0.10 * margin
    score = 0.5 * (score + score.T)

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
