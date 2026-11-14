import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    work = d.copy()
    finite_off_diag = np.isfinite(work)
    np.fill_diagonal(finite_off_diag, False)

    if not np.any(finite_off_diag):
        return np.zeros((n, n), dtype=float)

    valid_values = work[finite_off_diag]
    fallback = float(np.max(valid_values)) if valid_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    work[~np.isfinite(work)] = fallback
    work = np.maximum(work, 0.0)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    tiny = max(
        np.finfo(float).eps,
        float(np.min(positive)) * 1e-12 if positive.size else np.finfo(float).eps,
    )
    work = np.maximum(work, tiny)

    k = max(1, min(5, n - 1))
    sorted_distances = np.sort(work, axis=1)
    local_scale = np.mean(sorted_distances[:, :k], axis=1)
    local_scale = np.maximum(local_scale, tiny)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = work / np.maximum(pair_scale, tiny)

    attractiveness = np.exp(-np.minimum(relative_distance, 50.0))

    ranks = np.argsort(np.argsort(work, axis=1), axis=1)
    rank_score = 1.0 - ranks / max(1, n - 2)
    rank_score = np.clip(rank_score, 0.0, 1.0)

    nearest = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0

    endpoint_scarcity = 1.0 / np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], tiny)
    )
    scarcity_min = np.min(endpoint_scarcity[~np.eye(n, dtype=bool)])
    scarcity_max = np.max(endpoint_scarcity[~np.eye(n, dtype=bool)])
    if scarcity_max > scarcity_min:
        endpoint_scarcity = (
            endpoint_scarcity - scarcity_min
        ) / (scarcity_max - scarcity_min)
    else:
        endpoint_scarcity.fill(0.5)

    score = (
        0.55 * attractiveness
        + 0.25 * np.sqrt(np.maximum(rank_score * rank_score.T, 0.0))
        + 0.15 * mutual
        + 0.05 * endpoint_scarcity
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
