import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, finite edge-promise matrix for stochastic TSP sampling."""
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

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive = valid & (d > 0.0)
    positive_values = d[positive]

    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
        fallback = max(float(np.max(positive_values)), global_scale)
    else:
        global_scale = 1.0
        fallback = 1.0

    work = np.where(valid, np.maximum(d, 0.0), np.inf)
    positive_work = np.where(valid, np.maximum(d, eps), np.inf)

    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = positive_work[i]
        finite_row = row[np.isfinite(row)]

        if finite_row.size:
            positive_row = finite_row[finite_row > eps]
            if positive_row.size:
                index = min(1, positive_row.size - 1)
                local_scale[i] = float(
                    np.partition(positive_row, index)[index]
                )
            else:
                local_scale[i] = eps

    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = positive_work / geometric_scale
    normalized_length = np.minimum(normalized_length, 1e6)

    relative_affinity = global_scale / np.maximum(positive_work, eps)
    relative_affinity = np.minimum(relative_affinity, 1e12)

    order = np.argsort(positive_work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, 0.0)
        * np.maximum(1.0 - ranks.T, 0.0)
    )
    bilateral_rank = np.maximum(bilateral_rank, eps)

    density_radius = 1.5 * local_scale
    local_density = np.mean(
        valid & (work <= density_radius[:, None]),
        axis=1,
    )

    excess_density = np.maximum(local_density - 0.35, 0.0)
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
        0.90 * np.log(np.maximum(relative_affinity, eps))
        + 0.95 * np.log(bilateral_rank)
        - 0.75 * normalized_length
        - 0.10 * scale_mismatch
        - 0.08 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
