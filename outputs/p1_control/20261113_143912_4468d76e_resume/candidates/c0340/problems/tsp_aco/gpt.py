import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
        copy=True,
    )
    d = (d + d.T) * 0.5
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    invalid_distance = fallback_distance * 10.0
    work = np.where(valid, d, invalid_distance)
    np.fill_diagonal(work, invalid_distance)

    # Second-nearest local scale without full row sorting.
    kth = min(1, n - 2)
    local_scale = np.partition(work, kth=kth, axis=1)[:, kth]
    local_scale = np.maximum(local_scale, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        edge_scale = np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )
        relative_length = work / edge_scale
        global_length = work / global_scale

        # Smooth preference for edges short relative to both endpoints.
        endpoint_preference = np.exp(
            -1.10 * np.minimum(relative_length, 40.0)
        )

        # Stable bidirectional rank signal.
        order = np.argsort(work, axis=1, kind="stable")
        ranks = np.empty((n, n), dtype=float)
        rows = np.arange(n)[:, None]
        ranks[rows, order] = np.arange(n, dtype=float)[None, :]
        ranks[~valid] = float(n)

        normalized_rank = ranks / float(max(n - 1, 1))
        mutual_rank = normalized_rank + normalized_rank.T
        rank_preference = np.exp(-3.10 * np.minimum(mutual_rank, 40.0))

        # Nonzero floor preserves exploration and bridge edges.
        global_preference = 0.66 + 0.34 * np.exp(
            -0.26 * np.minimum(global_length, 80.0)
        )

        # Prefer edges whose endpoint neighborhoods have compatible scales.
        scale_difference = np.abs(
            local_scale[:, None] - local_scale[None, :]
        )
        scale_sum = local_scale[:, None] + local_scale[None, :]
        balance = 1.0 / (
            1.0 + 0.85 * scale_difference / (scale_sum + eps)
        )
        neighborhood_balance = 0.78 + 0.22 * balance

        score = (
            endpoint_preference
            * rank_preference
            * global_preference
            * neighborhood_balance
        )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
