import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    d = 0.5 * (d + d.T)
    finite = np.isfinite(d)

    valid = finite & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    maximum = (
        float(np.max(positive_values))
        if positive_values.size
        else global_scale
    )
    replacement = max(10.0 * maximum, 10.0 * global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), replacement)
    np.fill_diagonal(work, replacement)

    kth = min(1, n - 2)
    local_scale = np.partition(work, kth, axis=1)[:, kth]

    positive_local = local_scale > eps
    local_scale = np.where(
        positive_local,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / endpoint_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    mutual_rank = 1.0 / (1.0 + ranks + ranks.T)
    mutual_rank_log = np.log(np.maximum(mutual_rank, eps))

    local_log_scale = np.log(np.maximum(local_scale, eps))
    scale_gap = np.abs(
        local_log_scale[:, None] - local_log_scale[None, :]
    )
    scale_gap /= 1.0 + np.abs(
        local_log_scale[:, None] + local_log_scale[None, :]
    )

    local_density = np.mean(
        valid
        & (
            d
            <= 1.5
            * np.minimum(
                local_scale[:, None],
                local_scale[None, :],
            )
        ),
        axis=1,
    )
    density_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    log_score = (
        -0.85 * relative_length
        -0.25 * global_length
        + 1.75 * mutual_rank_log
        -0.22 * scale_gap
        -0.10 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - shift, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    peak = float(np.max(score))
    if peak > 0.0:
        score /= peak

    return score.astype(float, copy=False)
