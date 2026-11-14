import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    observed = distances[finite_positive]
    fallback = float(np.median(observed))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    raw = np.where(finite_positive, distances, fallback)
    d = 0.5 * (raw + raw.T)
    np.fill_diagonal(d, np.inf)

    valid = finite_positive & finite_positive.T
    if not np.any(valid):
        valid = finite_positive | finite_positive.T

    finite_d = d[np.isfinite(d) & (d > 0)]
    if finite_d.size == 0:
        return np.zeros((n, n), dtype=float)

    global_scale = float(np.median(finite_d))
    global_scale = max(global_scale, np.finfo(float).eps)

    masked_d = np.where(valid, d, np.inf)
    k = min(6, n - 1)

    nearest = np.partition(masked_d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / np.maximum(pair_scale, np.finfo(float).eps)

    proximity = np.exp(-0.95 * relative_distance)
    visibility = 1.0 / np.maximum(relative_distance, 1e-12)

    order = np.argsort(masked_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_strength = np.exp(
        -ranks / max(1.0, 0.85 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    nearest_neighbor = order[:, 0]
    mutual_neighbor = (
        nearest_neighbor[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_neighbor[None, :] == np.arange(n)[:, None]
    )

    accessibility = np.exp(
        -np.minimum(masked_d / local_scale[:, None], 8.0)
    )
    opportunity = np.sum(
        np.where(valid, accessibility, 0.0),
        axis=1,
    )
    scarcity = 1.0 / np.sqrt(
        1.0 + opportunity[:, None] + opportunity[None, :]
    )

    reserve = np.full((n, n), global_scale, dtype=float)
    for i in range(n):
        row_order = order[i]
        row_values = masked_d[i, row_order]
        finite_values = row_values[np.isfinite(row_values)]
        if finite_values.size > 1:
            reserve[i, :] = finite_values[1]
        elif finite_values.size == 1:
            reserve[i, :] = finite_values[0]

    residual_ratio = (
        d + reserve + reserve.T
    ) / np.maximum(
        local_scale[:, None] + local_scale[None, :] + d,
        np.finfo(float).eps,
    )
    residual_penalty = np.exp(
        -0.22 * np.clip(residual_ratio - 1.0, 0.0, 3.0)
    )

    score = (
        visibility
        * (0.58 + 0.42 * proximity)
        * (0.62 + 0.38 * bidirectional_rank)
        * (1.0 + 0.22 * scarcity)
        * (0.88 + 0.12 * residual_penalty)
    )

    score *= np.where(mutual_neighbor, 1.45, 1.0)
    score *= valid
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score = score / maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
