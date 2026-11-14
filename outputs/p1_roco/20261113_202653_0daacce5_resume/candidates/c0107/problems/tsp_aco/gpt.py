import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    valid = finite & nonnegative
    valid = valid & valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(valid, 0.5 * (distances + distances.T), np.inf)
    np.fill_diagonal(cleaned, np.inf)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(observed)) if observed.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / (cleaned[valid] + eps)

    row_max = np.maximum(
        inverse_distance.max(axis=1, keepdims=True),
        eps,
    )
    row_preference = inverse_distance / row_max

    order = np.argsort(cleaned, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = rank_values

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    neighborhood_size = min(3, n - 1)
    local_scale = np.full(n, scale, dtype=float)

    for i in range(n):
        candidates = cleaned[i, valid[i]]
        if candidates.size:
            nearest = np.sort(candidates)[:neighborhood_size]
            local_scale[i] = float(np.mean(nearest))

    local_scale = np.maximum(local_scale, 0.05 * scale)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    contrast = np.zeros((n, n), dtype=float)
    contrast[valid] = np.exp(
        -cleaned[valid] / (pair_scale[valid] + eps)
    )

    row_mass = np.maximum(inverse_distance.sum(axis=1), eps)
    endpoint_scarcity = 1.0 / row_mass
    endpoint_scarcity /= max(
        float(np.mean(endpoint_scarcity)),
        eps,
    )

    scarcity_support = np.sqrt(
        endpoint_scarcity[:, None] * endpoint_scarcity[None, :]
    )

    score = (
        0.48 * row_preference
        + 0.24 * mutual_rank
        + 0.18 * contrast
        + 0.10 * scarcity_support * row_preference
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
