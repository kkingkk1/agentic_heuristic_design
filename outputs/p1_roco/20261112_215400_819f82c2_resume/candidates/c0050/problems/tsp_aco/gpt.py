import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric positive edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_off_diagonal = np.isfinite(d) & off_diagonal

    if not np.any(finite_off_diagonal):
        return result

    valid = d[finite_off_diagonal]
    positive = valid[valid > 0]

    if positive.size:
        fallback = float(np.median(positive))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    finite_values = d[finite_off_diagonal & np.isfinite(d)]
    finite_max = float(np.max(finite_values)) if finite_values.size else fallback
    large_distance = max(finite_max, fallback) * 1e6

    clean = np.array(d, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = large_distance
    clean[clean < 0] = large_distance
    np.fill_diagonal(clean, np.inf)

    eps = np.finfo(float).eps

    k = min(max(1, int(np.sqrt(n)) + 1), n - 1)
    nearest_values = np.partition(clean, kth=k - 1, axis=1)[:, :k]

    local_scale = np.median(nearest_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    relative_shortness = np.exp(-clean / (pair_scale + eps))
    relative_shortness[~np.isfinite(relative_shortness)] = 0.0

    ranks = np.argsort(np.argsort(clean, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )
    rank_consensus[~np.isfinite(rank_consensus)] = 0.0

    nearest = np.argmin(clean, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i and np.isfinite(d[i, j]):
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    local_compatibility = np.exp(
        -0.5
        * (
            np.abs(np.log((clean + eps) / (local_scale[:, None] + eps)))
            + np.abs(np.log((clean + eps) / (local_scale[None, :] + eps)))
        )
    )
    local_compatibility[~np.isfinite(local_compatibility)] = 0.0

    score = (
        0.50 * relative_shortness
        + 0.25 * rank_consensus
        + 0.15 * local_compatibility
        + 0.10 * mutual_bonus
    )

    score = 0.5 * (score + score.T)
    score[~finite_off_diagonal] = 0.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    score[score > 0.0] += 1e-12
    return score.astype(float, copy=False)
