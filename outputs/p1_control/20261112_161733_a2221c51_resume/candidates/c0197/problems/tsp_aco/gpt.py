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
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = d[i, finite_positive[i]]
        if row_values.size == 0:
            continue

        row_values = np.sort(row_values)
        if row_values.size == 1:
            local_scale[i] = max(float(row_values[0]), eps)
        else:
            local_scale[i] = max(
                0.8 * float(row_values[0])
                + 0.2 * float(row_values[1]),
                eps,
            )

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = work / geometric_scale
    absolute_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_support = 1.0 / (1.0 + ranks + ranks.T)

    log_density = np.log(global_scale / local_scale)
    density_gap = np.abs(
        log_density[:, None] - log_density[None, :]
    )
    density_gap /= (
        1.0
        + np.abs(log_density[:, None] + log_density[None, :])
    )

    log_score = (
        -1.35 * normalized_length
        -0.25 * absolute_length
        +0.55 * np.log(rank_support)
        -0.30 * density_gap
    )

    length_support = np.exp(
        np.clip(-0.55 * normalized_length, -745.0, 0.0)
    )
    density_support = np.exp(
        np.clip(-0.80 * density_gap, -745.0, 0.0)
    )

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score *= length_support * density_support

    score[~finite_positive] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))

    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
