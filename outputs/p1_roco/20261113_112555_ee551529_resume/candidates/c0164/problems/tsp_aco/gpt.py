import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-priority scores for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = raw.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    if not np.any(finite_nonnegative):
        return scores

    valid_values = raw[finite_nonnegative]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(valid_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    forward = np.where(finite_nonnegative, np.maximum(raw, 0.0), np.nan)
    backward = forward.T

    both_valid = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = ~np.isfinite(forward) & np.isfinite(backward)

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (
        forward[both_valid] + backward[both_valid]
    )
    distance[only_forward] = forward[only_forward]
    distance[only_backward] = backward[only_backward]

    valid = both_valid | only_forward | only_backward
    np.fill_diagonal(distance, np.inf)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    work = np.where(valid, distance, np.inf)
    valid_counts = valid.sum(axis=1)

    neighbor_count = min(4, n - 1)
    nearest_values = np.partition(
        work,
        kth=neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]

    finite_nearest = np.where(
        np.isfinite(nearest_values),
        nearest_values,
        np.nan,
    )

    local_scale = np.nanmedian(finite_nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = distance / pair_scale
        relative_distance = np.clip(relative_distance, 0.0, 60.0)
        shortness = np.exp(-relative_distance)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_weight = 1.0 / (1.0 + ranks)
    reciprocal_rank = np.sqrt(rank_weight * rank_weight.T)

    nearest = np.argmin(work, axis=1)
    nearest_is_valid = np.isfinite(work[np.arange(n), nearest])

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in np.flatnonzero(nearest_is_valid):
        j = nearest[i]
        if nearest_is_valid[j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts.astype(float), 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    scores = (
        shortness
        * (0.55 + 0.45 * reciprocal_rank)
        * (1.0 + 0.30 * mutual_nearest)
        * (0.90 + 0.20 * scarcity_pair)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
