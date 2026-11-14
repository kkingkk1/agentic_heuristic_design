import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances, dtype=float)

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0:
        return np.zeros((n, n), dtype=float)

    working = np.where(valid, d, fallback)
    np.fill_diagonal(working, fallback)

    k = min(7, n - 1)
    nearest = np.partition(working, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = working / pair_scale
    relative = np.maximum(relative, np.finfo(float).eps)

    finite_relative = relative[valid]
    upper_clip = float(np.percentile(finite_relative, 95))
    upper_clip = max(upper_clip, 1.0)
    clipped_relative = np.minimum(relative, upper_clip)

    order = np.argsort(working, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = float(n)

    valid_ranks = ranks[valid]
    rank_temperature = max(
        1.0,
        float(np.median(valid_ranks) + 0.5 * np.sqrt(float(n))),
    )

    rank_score = np.exp(-ranks / rank_temperature)
    rank_score *= valid

    bidirectional_rank = np.sqrt(rank_score * rank_score.T)

    top_k = ranks < float(k)
    mutual_support = top_k * top_k.T
    mutual_support *= valid

    endpoint_support = 0.5 * (
        np.exp(-ranks / max(1.0, float(k)))
        + np.exp(-ranks.T / max(1.0, float(k)))
    )
    endpoint_support *= valid

    log_relative = np.log1p(clipped_relative)
    distance_score = np.exp(-1.35 * log_relative)

    contrast = np.exp(
        -0.65 * np.maximum(log_relative - np.log(2.0), 0.0)
    )

    score = (
        distance_score
        * contrast
        * (0.35 + 0.65 * bidirectional_rank)
        * (0.75 + 0.25 * endpoint_support)
        * (1.0 + 0.55 * mutual_support)
    )

    score *= valid
    score = 0.5 * (score + score.T)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    score = 0.5 * (score + score.T)
    score *= valid
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
