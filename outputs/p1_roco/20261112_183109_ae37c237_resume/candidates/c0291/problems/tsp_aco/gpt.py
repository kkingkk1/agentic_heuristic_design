import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.max(valid_values))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    counts = valid.sum(axis=1)
    order = np.argsort(d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(d, order, axis=1)

    first = np.where(counts > 0, sorted_d[:, 0], scale)
    second = np.where(counts > 1, sorted_d[:, 1], first)

    first = np.where(np.isfinite(first), first, scale)
    second = np.where(np.isfinite(second), second, first)

    local_scale = np.maximum(0.5 * (first + second), eps)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    geo_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = np.divide(
        d,
        geo_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geo_scale > 0.0,
    )

    distance_quality = np.exp(-np.clip(1.15 * relative, 0.0, 60.0))

    rank_decay = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_quality = np.exp(-ranks / rank_decay)
    reciprocal_rank = np.sqrt(rank_quality * rank_quality.T)

    endpoint_quality = np.exp(
        -np.clip(
            0.5
            * (
                d / local_scale[:, None]
                + d / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - first,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second) & (second > 0.0),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.24 * (regret[:, None] + regret[None, :])

    nearest = order[:, 0]
    index = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == index[None, :])
        & (nearest[None, :] == index[:, None])
        & valid
    )

    mutual_bonus = np.where(mutual_nearest, 1.12, 1.0)

    neighborhood_width = min(4, max(1, n - 1))
    neighborhood = np.zeros((n, n), dtype=float)

    row_ids = np.repeat(index, neighborhood_width)
    col_ids = order[:, :neighborhood_width].reshape(-1)
    row_valid = valid[row_ids, col_ids]
    neighborhood[row_ids[row_valid], col_ids[row_valid]] = 1.0

    neighborhood_count = neighborhood.sum(axis=1)
    neighborhood_count = np.maximum(neighborhood_count, 1.0)

    overlap = neighborhood @ neighborhood.T
    overlap_limit = np.minimum(
        neighborhood_count[:, None],
        neighborhood_count[None, :],
    )
    overlap_ratio = np.divide(
        overlap,
        overlap_limit,
        out=np.zeros((n, n), dtype=float),
        where=overlap_limit > 0.0,
    )

    bridge = np.clip(1.0 - overlap_ratio, 0.0, 1.0)
    competitive = np.exp(-0.85 * np.maximum(relative - 1.0, 0.0))
    diversity_bonus = 1.0 + 0.09 * bridge * competitive

    nearest_mask = nearest[:, None] == index[None, :]
    endpoint_alternative = np.where(nearest_mask, second[:, None], first[:, None])

    reserve_quality = np.exp(
        -np.clip(
            0.5
            * (
                endpoint_alternative / local_scale[:, None]
                + endpoint_alternative.T / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    reserve_possible = (
        (counts[:, None] >= 2)
        & (counts[None, :] >= 2)
    )
    reserve_bonus = 1.0 + np.where(
        reserve_possible,
        0.07 * reserve_quality,
        0.0,
    )

    trap = mutual_nearest & (overlap_ratio > 0.66)
    trap_penalty = np.where(
        trap,
        1.0 - 0.07 * np.clip((overlap_ratio - 0.66) / 0.34, 0.0, 1.0),
        1.0,
    )

    smooth_long_edge_penalty = np.exp(
        -0.55 * np.maximum(relative - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (
            0.56
            + 0.26 * reciprocal_rank
            + 0.18 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * diversity_bonus
        * reserve_bonus
        * trap_penalty
        * smooth_long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
