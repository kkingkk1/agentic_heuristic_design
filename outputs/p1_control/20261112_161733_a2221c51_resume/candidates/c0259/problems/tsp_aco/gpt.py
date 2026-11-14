import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)

    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = work[i][np.isfinite(work[i])]
        positive_values = values[values > 0.0]

        if positive_values.size:
            ordered = np.sort(positive_values)
            local_scale[i] = max(
                float(ordered[min(1, ordered.size - 1)]),
                eps,
            )
        elif values.size:
            local_scale[i] = max(float(np.min(values)), eps)

    local_scale = np.maximum(local_scale, eps)

    normalized_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = np.where(valid, normalized_length, np.inf)
    normalized_length = np.minimum(normalized_length, 1e6)

    ranks = np.full((n, n), 1.0, dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            denominator = max(order.size - 1, 1)
            ranks[i, order] = np.arange(order.size, dtype=float) / denominator
            nearest[i] = order[0]

    bilateral_rank_quality = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank_quality = np.maximum(bilateral_rank_quality, eps)

    reciprocal = global_scale / np.maximum(d, eps)
    reciprocal = np.where(valid, np.minimum(reciprocal, 1e12), 0.0)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    density_radius = 1.5 * local_scale
    local_density = np.mean(
        valid & (work <= density_radius[:, None]),
        axis=1,
    )
    excess_density = np.maximum(local_density - 0.50, 0.0)

    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_mismatch = np.minimum(scale_mismatch, 12.0)

    log_score = (
        0.78 * np.log(np.maximum(reciprocal, eps))
        + 0.68 * np.log(bilateral_rank_quality)
        - 0.52 * normalized_length
        + 0.20 * mutual_nearest
        - 0.035 * scale_mismatch
        - 0.035 * density_penalty
    )

    base_score = np.exp(
        np.clip(log_score - np.nanmax(log_score[valid]), -745.0, 0.0)
    )

    cross_density_floor = np.exp(
        -0.30 * np.minimum(normalized_length, 5.0)
        -0.10 * np.minimum(scale_mismatch, 6.0)
    )

    score = base_score + 0.055 * cross_density_floor
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
