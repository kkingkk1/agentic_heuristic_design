import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw) & (raw >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = np.where(valid, 0.5 * (raw + raw.T), np.inf)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    global_scale = max(global_scale, eps)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[np.argsort(d[i, indices], kind="stable")]
        nearest[i] = int(order[0])

        if order.size > 1:
            second_nearest[i] = int(order[1])

        positive_row = d[i, order]
        positive_row = positive_row[positive_row > 0.0]

        if positive_row.size:
            # The second positive neighbor is less brittle than the nearest edge.
            local_scale[i] = max(
                float(positive_row[min(1, positive_row.size - 1)]),
                eps,
            )
        else:
            local_scale[i] = max(float(d[i, order[0]]), eps)

    local_scale = np.maximum(local_scale, eps)

    normalized_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = np.where(valid, normalized_length, np.inf)
    normalized_length = np.minimum(normalized_length, 1e6)

    # Continuous bilateral rank quality.
    ranks = np.ones((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            denominator = max(order.size - 1, 1)
            ranks[i, order] = (
                np.arange(order.size, dtype=float) / denominator
            )

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    # Explicit soft support for edges among the two closest neighbors.
    second_neighbor_signal = np.zeros((n, n), dtype=float)
    for i in range(n):
        if nearest[i] >= 0:
            second_neighbor_signal[i, nearest[i]] = 1.0
        if second_nearest[i] >= 0:
            second_neighbor_signal[i, second_nearest[i]] = 0.55

    second_neighbor_signal = np.maximum(
        second_neighbor_signal,
        second_neighbor_signal.T,
    )

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    # Local inverse distance is the main distance signal.
    local_inverse = 1.0 / np.maximum(normalized_length + 0.12, eps)
    local_inverse = np.where(valid, local_inverse, 0.0)

    # Retain a weak global contrast without letting it dominate local scale.
    global_inverse = global_scale / np.maximum(d, eps)
    global_inverse = np.clip(global_inverse, 1e-12, 1e12)
    global_inverse = np.where(valid, global_inverse, 0.0)

    # Mild density correction only; avoid suppressing useful cross-density edges.
    radius = 1.6 * local_scale
    local_density = np.mean(
        valid & (d <= radius[:, None]),
        axis=1,
    )
    density_excess = np.maximum(local_density - 0.45, 0.0)
    density_penalty = (
        density_excess[:, None] + density_excess[None, :]
    ) * np.minimum(normalized_length, 2.5)

    # Weak scale mismatch penalty, mainly preventing pathological pairings.
    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_penalty = np.minimum(scale_mismatch, 8.0)

    log_score = (
        0.78 * np.log(np.maximum(local_inverse, eps))
        + 0.18 * np.log(np.maximum(global_inverse, eps))
        + 0.92 * np.log(np.maximum(bilateral_rank, eps))
        - 0.62 * normalized_length
        + 0.22 * mutual_nearest
        + 0.14 * second_neighbor_signal
        - 0.035 * scale_penalty
        - 0.045 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
