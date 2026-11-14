import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive = valid & (symmetric_d > 0.0)
    positive_values = symmetric_d[positive]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    effective_d = np.where(
        valid,
        np.maximum(symmetric_d, eps * global_scale),
        np.inf,
    )

    positive_for_scale = np.where(
        positive,
        symmetric_d,
        np.inf,
    )
    sorted_positive = np.sort(positive_for_scale, axis=1)
    positive_count = np.sum(np.isfinite(positive_for_scale), axis=1)

    second_index = np.minimum(np.maximum(positive_count - 1, 0), 1)
    row_index = np.arange(n)

    local_scale = sorted_positive[row_index, second_index]
    row_minimum = np.min(
        np.where(valid, effective_d, np.inf),
        axis=1,
    )

    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        np.where(
            np.isfinite(row_minimum),
            row_minimum,
            global_scale,
        ),
    )
    local_scale = np.maximum(local_scale, eps * global_scale)

    scale_product = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = effective_d / np.maximum(scale_product, eps)
    relative_length = np.minimum(relative_length, 1e6)

    global_relative = effective_d / global_scale
    global_relative = np.minimum(global_relative, 1e6)

    order = np.argsort(
        np.where(valid, effective_d, np.inf),
        axis=1,
        kind="stable",
    )

    rank = np.empty((n, n), dtype=float)
    rank[row_index[:, None], order] = np.arange(n, dtype=float)
    rank_denominator = np.maximum(
        np.sum(valid, axis=1) - 1,
        1,
    )
    normalized_rank = rank / rank_denominator[:, None]
    normalized_rank = np.where(
        valid,
        np.minimum(normalized_rank, 1.0),
        1.0,
    )

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - normalized_rank, eps)
        * np.maximum(1.0 - normalized_rank.T, eps)
    )

    nearest = order[:, 0]
    has_neighbor = np.sum(valid, axis=1) > 0
    nearest[~has_neighbor] = -1

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    degree_fraction = np.sum(valid, axis=1) / max(n - 1, 1)
    sparsity = 1.0 / np.sqrt(
        np.maximum(degree_fraction[:, None], eps)
        * np.maximum(degree_fraction[None, :], eps)
    )
    sparsity = np.minimum(sparsity, 8.0)

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    scale_mismatch = np.minimum(
        scale_difference / np.maximum(scale_sum, eps),
        1.0,
    )

    candidate_limit = 1.8 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    mutual_candidate = (
        valid
        & (symmetric_d <= candidate_limit)
        & (symmetric_d <= candidate_limit.T)
    ).astype(float)

    reciprocal = global_scale / np.maximum(effective_d, eps)
    reciprocal = np.minimum(reciprocal, 1e12)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.95 * np.log(np.maximum(reciprocal, eps))
            + 1.20 * np.log(np.maximum(bilateral_rank, eps))
            - 0.95 * relative_length
            - 0.22 * np.minimum(global_relative, 24.0)
            - 0.18 * scale_mismatch
            + 0.36 * mutual_nearest
            + 0.24 * mutual_candidate
            + 0.10 * np.log(np.maximum(sparsity, 1.0))
        )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score = np.where(valid, score, 0.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
