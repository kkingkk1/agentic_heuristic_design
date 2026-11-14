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

    d = 0.5 * (d + d.T)
    positive = d[valid & (d > 0.0)]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    positive_fallback = (
        float(np.median(positive)) if positive.size else global_scale
    )
    positive_fallback = max(positive_fallback, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)
    rank_work = np.where(valid, np.maximum(d, eps), global_scale * 10.0)

    sorted_distances = np.sort(work, axis=1)
    local_scale = np.full(n, positive_fallback, dtype=float)

    for i in range(n):
        candidates = sorted_distances[i, : max(n - 1, 1)]
        finite_candidates = candidates[np.isfinite(candidates)]
        if finite_candidates.size:
            index = 1 if finite_candidates.size > 1 else 0
            candidate = finite_candidates[index]
            if candidate > 0.0:
                local_scale[i] = candidate

    local_scale = np.maximum(local_scale, eps)

    normalized_length = rank_work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = np.minimum(normalized_length, 1e6)

    order = np.argsort(rank_work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks = ranks / max(n - 1, 1)

    endpoint_attractiveness = np.sqrt(
        (1.0 - ranks) * (1.0 - ranks.T)
    )
    endpoint_attractiveness = np.maximum(endpoint_attractiveness, eps)

    reciprocal_distance = global_scale / np.maximum(rank_work, eps)
    reciprocal_distance = np.minimum(reciprocal_distance, 1e12)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_penalty = np.minimum(scale_mismatch, 20.0)

    local_density = np.mean(
        valid
        & (
            work
            <= 1.5 * np.minimum(
                local_scale[:, None],
                local_scale[None, :],
            )
        ),
        axis=1,
    )
    density_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(normalized_length, 3.0)

    log_score = (
        0.85 * np.log(np.maximum(reciprocal_distance, eps))
        + 0.75 * np.log(endpoint_attractiveness)
        - 0.95 * normalized_length
        - 0.30 * rank_work / global_scale
        - 0.20 * scale_penalty
        - 0.10 * density_penalty
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
