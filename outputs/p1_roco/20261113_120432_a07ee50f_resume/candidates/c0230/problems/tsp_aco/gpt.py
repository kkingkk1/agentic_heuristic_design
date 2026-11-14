import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    clipped = np.maximum(raw, 0.0)

    forward = finite & ~diagonal
    backward = forward.T
    both = forward & backward

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (clipped[both] + clipped.T[both])

    only_forward = forward & ~backward
    only_backward = backward & ~forward
    distance[only_forward] = clipped[only_forward]
    distance[only_backward] = clipped.T[only_backward]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.isfinite(sorted_distance)

    valid_count = sorted_valid.sum(axis=1)

    first_distance = np.where(
        valid_count > 0,
        sorted_distance[:, 0],
        global_scale,
    )
    second_distance = np.where(
        valid_count > 1,
        sorted_distance[:, 1],
        first_distance,
    )
    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        global_scale,
    )

    first_four = sorted_distance[:, :4]
    first_four_valid = np.isfinite(first_four)

    positive_first_four = np.where(
        first_four_valid & (first_four > 0.0),
        first_four,
        np.nan,
    )
    local_positive = np.nanmedian(positive_first_four, axis=1)

    finite_first_four = np.where(
        first_four_valid,
        first_four,
        np.nan,
    )
    local_finite = np.nanmedian(finite_first_four, axis=1)

    local_scale = np.where(
        np.isfinite(local_positive),
        local_positive,
        local_finite,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    rows = np.arange(n)[:, None]
    ranks = np.zeros((n, n), dtype=float)
    rank_values = 1.0 / (np.arange(n, dtype=float) + 1.0)
    ranks[rows, order] = rank_values
    ranks *= sorted_valid

    neighborhood_size = min(
        n - 1,
        max(2, int(np.ceil(np.sqrt(n)))),
    )

    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[rows, order[:, :neighborhood_size]] = True
    top_mask &= valid

    nearest_mask = np.zeros((n, n), dtype=bool)
    nearest_mask[rows[:, 0], order[:, 0]] = valid_count > 0
    nearest_mask &= valid

    mutual_top = top_mask & top_mask.T
    mutual_nearest = nearest_mask & nearest_mask.T

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized_distance = distance / pair_scale
        distance_preference = np.exp(
            -np.clip(normalized_distance, 0.0, 60.0)
        )

        row_preference = np.exp(
            -np.clip(distance / local_scale[:, None], 0.0, 60.0)
        )
        column_preference = np.exp(
            -np.clip(distance / local_scale[None, :], 0.0, 60.0)
        )
        endpoint_preference = np.sqrt(
            row_preference * column_preference
        )

        reciprocal_rank = np.sqrt(ranks * ranks.T)

        gap = np.maximum(second_distance, global_scale)[:, None]
        ratio = gap / np.maximum(distance, eps)
        scarcity = 1.0 + 0.25 * np.clip(ratio - 1.0, 0.0, 4.0)

    scarcity = np.sqrt(scarcity * scarcity.T)

    scores = (
        distance_preference
        * endpoint_preference
        * (0.55 + 0.45 * reciprocal_rank)
        * (1.0 + 0.35 * mutual_top.astype(float))
        * (1.0 + 0.35 * mutual_nearest.astype(float))
        * scarcity
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
