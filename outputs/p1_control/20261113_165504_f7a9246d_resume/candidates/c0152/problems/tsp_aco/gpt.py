import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    directed_valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return result

    pair_distance = np.where(
        valid,
        0.5 * (d + d.T),
        np.inf,
    )

    positive = pair_distance[valid & (pair_distance > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
        q25 = float(np.percentile(positive, 25.0))
        q75 = float(np.percentile(positive, 75.0))
    else:
        global_scale = 1.0
        q25 = 1.0
        q75 = 1.0

    tiny = np.finfo(float).eps
    global_scale = max(global_scale, tiny)
    spread = max(q75 - q25, 0.25 * global_scale, tiny)

    safe = np.where(valid, pair_distance, np.inf)

    row_values = np.where(valid, pair_distance, np.nan)
    endpoint_median = np.nanmedian(row_values, axis=1)
    endpoint_median = np.where(
        np.isfinite(endpoint_median) & (endpoint_median > 0.0),
        endpoint_median,
        global_scale,
    )
    endpoint_median = np.maximum(endpoint_median, tiny)

    endpoint_q40 = np.nanpercentile(row_values, 40.0, axis=1)
    endpoint_q40 = np.where(
        np.isfinite(endpoint_q40) & (endpoint_q40 > 0.0),
        endpoint_q40,
        endpoint_median,
    )
    endpoint_q40 = np.maximum(endpoint_q40, tiny)

    pair_scale = np.sqrt(
        endpoint_median[:, None] * endpoint_median[None, :]
    )
    pair_scale = np.maximum(pair_scale, tiny)

    zero_edge = valid & (pair_distance == 0.0)

    global_shortness = 1.0 / (
        1.0 + np.maximum(pair_distance - q25, 0.0) / spread
    )
    endpoint_shortness = 1.0 / (
        1.0 + pair_distance / pair_scale
    )

    local_threshold = np.sqrt(
        endpoint_q40[:, None] * endpoint_q40[None, :]
    )
    local_threshold = np.maximum(local_threshold, tiny)

    local_margin = np.clip(
        (local_threshold - pair_distance) / local_threshold,
        -1.0,
        1.0,
    )
    local_shortness = 0.5 + 0.5 * local_margin

    local_shortness[zero_edge] = 1.0
    global_shortness[zero_edge] = 1.0

    endpoint_competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = safe[i]
        finite_row = np.isfinite(row)
        if np.any(finite_row):
            row_scale = max(
                float(np.nanpercentile(np.where(finite_row, row, np.nan), 60.0)),
                tiny,
            )
            endpoint_competition[i] = np.exp(
                -np.minimum(np.where(finite_row, row, 0.0) / row_scale, 40.0)
            )

    endpoint_competition[~valid] = 0.0
    competition = np.sqrt(
        endpoint_competition * endpoint_competition.T
    )

    closure_support = np.zeros((n, n), dtype=float)

    for k in range(n):
        through_k = safe[:, k, None] + safe[None, k, :]
        usable = valid & np.isfinite(through_k)

        tolerance = 0.12 * pair_scale + 0.05 * global_scale
        advantage = pair_distance + tolerance - through_k

        contribution = np.zeros((n, n), dtype=float)
        contribution[usable] = 1.0 / (
            1.0 + np.exp(
                -np.clip(advantage[usable] / (0.20 * spread), -40.0, 40.0)
            )
        )
        closure_support += contribution

    possible_intermediates = np.sum(
        np.isfinite(safe[:, None, :]) & np.isfinite(safe[None, :, :]),
        axis=2,
    )
    possible_intermediates = np.maximum(possible_intermediates, 1)
    closure_support /= possible_intermediates

    score = (
        0.30 * endpoint_shortness
        + 0.24 * global_shortness
        + 0.20 * local_shortness
        + 0.16 * competition
        + 0.10 * closure_support
    )

    score[zero_edge] = np.maximum(score[zero_edge], 0.95)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
