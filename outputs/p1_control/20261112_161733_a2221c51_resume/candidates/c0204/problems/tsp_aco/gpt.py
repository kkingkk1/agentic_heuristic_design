import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    finite_positive = np.isfinite(d) & (d > 0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, global_scale, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    endpoint_rank = ranks + ranks.T
    rank_signal = 1.0 / (1.0 + endpoint_rank)

    neighbor_count = min(3, n - 1)
    nearest = np.take_along_axis(work, order[:, :neighbor_count], axis=1)

    if neighbor_count == 1:
        local_scale = nearest[:, 0]
    else:
        weights = np.linspace(1.0, 0.5, neighbor_count)
        weights /= np.sum(weights)
        local_scale = nearest @ weights

    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / geometric_scale

    absolute_length = work / global_scale

    local_density = global_scale / local_scale
    log_density = np.log(np.maximum(local_density, eps))
    density_gap = np.abs(log_density[:, None] - log_density[None, :])
    density_gap /= 1.0 + np.abs(log_density[:, None] + log_density[None, :])

    endpoint_competition = np.minimum(
        local_scale[:, None] / work,
        local_scale[None, :] / work,
    )
    endpoint_competition = np.clip(endpoint_competition, 0.0, 1.0)

    log_score = (
        -0.78 * normalized_length
        -0.16 * absolute_length
        + 2.10 * np.log(np.maximum(rank_signal, eps))
        + 0.28 * np.log(np.maximum(endpoint_competition, eps))
        - 0.30 * density_gap
    )

    raw_score = np.exp(np.clip(log_score, -745.0, 0.0))

    good_rank = np.exp(-0.45 * endpoint_rank)
    weak_cross_density_support = (
        0.06
        * density_gap
        * good_rank
        * np.exp(-0.45 * normalized_length)
    )

    score = raw_score * (1.0 + weak_cross_density_support)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
