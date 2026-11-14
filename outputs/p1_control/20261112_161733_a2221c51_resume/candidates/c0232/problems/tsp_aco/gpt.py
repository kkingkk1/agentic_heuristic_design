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
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)
    positive_work = np.where(valid, np.maximum(d, eps), np.inf)

    sorted_distances = np.sort(positive_work, axis=1)
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = sorted_distances[i]
        finite = row[np.isfinite(row)]
        if finite.size:
            # Use the second available edge when possible to reduce
            # sensitivity to a single accidental near-zero edge.
            index = 1 if finite.size > 1 else 0
            local_scale[i] = max(float(finite[index]), eps)

    local_scale = np.maximum(local_scale, eps)

    # Stable per-node ranks: smaller distances receive larger attractiveness.
    order = np.argsort(positive_work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    rank_score = np.maximum(1.0 - ranks, eps)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    # Normalize each edge by the geometric mean of its endpoint scales.
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = positive_work / np.maximum(pair_scale, eps)
    relative_length = np.minimum(relative_length, 1e8)

    global_length = positive_work / global_scale
    global_length = np.minimum(global_length, 1e8)

    # Local compatibility rewards edges that are not anomalously long
    # for either endpoint.
    endpoint_ratio = np.maximum(
        positive_work / local_scale[:, None],
        positive_work / local_scale[None, :],
    )
    compatibility = np.exp(-np.minimum(endpoint_ratio, 30.0))

    # Reward mutual candidate edges. The threshold adapts to each endpoint.
    candidate_threshold = 1.75 * np.maximum(
        local_scale[:, None],
        local_scale[None, :],
    )
    mutual_candidate = (
        valid
        & (work <= candidate_threshold)
        & (work.T <= candidate_threshold)
    ).astype(float)

    # Slightly favor edges incident to sparse nodes so their connectivity
    # is not suppressed by dense neighborhoods.
    degree = np.sum(valid, axis=1).astype(float)
    degree_scale = np.maximum(degree / max(n - 1, 1), eps)
    sparsity_bonus = 1.0 / np.sqrt(
        degree_scale[:, None] * degree_scale[None, :]
    )
    sparsity_bonus = np.minimum(sparsity_bonus, 8.0)

    log_score = (
        1.20 * np.log(np.maximum(global_scale / positive_work, eps))
        + 1.05 * np.log(mutual_rank)
        + 0.85 * np.log(np.maximum(compatibility, eps))
        - 0.70 * np.minimum(relative_length, 12.0)
        - 0.22 * np.minimum(global_length, 20.0)
        + 0.30 * mutual_candidate
        + 0.10 * np.log(sparsity_bonus)
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
