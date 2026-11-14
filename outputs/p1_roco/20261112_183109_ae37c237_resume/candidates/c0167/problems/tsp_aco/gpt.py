import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, locally scaled edge-priority matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = (
        np.isfinite(distances)
        & (distances > 0)
        & np.isfinite(distances.T)
        & (distances.T > 0)
    )
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_mask = np.isfinite(d)
    degree = finite_mask.sum(axis=1)

    sorted_d = np.sort(d, axis=1)
    k = min(7, max(1, n - 1))
    neighbors = sorted_d[:, :k]
    neighbor_mask = np.isfinite(neighbors)

    local_scale = np.full(n, 1.0, dtype=float)
    for i in range(n):
        values = neighbors[i][neighbor_mask[i]]
        if values.size:
            local_scale[i] = np.median(values)

    positive_scales = local_scale[np.isfinite(local_scale) & (local_scale > 0)]
    fallback_scale = (
        float(np.median(positive_scales))
        if positive_scales.size
        else 1.0
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback_scale,
    )

    best = sorted_d[:, 0]
    second = sorted_d[:, 1] if n > 2 else best.copy()

    best = np.where(np.isfinite(best), best, local_scale)
    second = np.where(np.isfinite(second), second, best)

    regret = np.divide(
        second - best,
        np.maximum(second, np.finfo(float).eps),
    )
    regret = np.clip(np.nan_to_num(regret, nan=0.0), 0.0, 1.0)
    regret *= (degree > 0)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = np.divide(
        d,
        np.maximum(pair_scale, np.finfo(float).eps),
        out=np.full_like(d, np.inf),
        where=np.isfinite(d),
    )

    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    distance_term = np.exp(-np.power(relative_distance, 1.25))

    logits = np.divide(
        -d,
        np.maximum(local_scale[:, None], np.finfo(float).eps),
        out=np.full_like(d, -np.inf),
        where=np.isfinite(d),
    )
    logits = np.clip(logits, -50.0, 0.0)

    weights = np.exp(logits)
    weights[~valid] = 0.0

    normalizer = weights.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        weights,
        normalizer,
        out=np.zeros_like(weights),
        where=normalizer > 0,
    )

    mutual_preference = np.sqrt(
        probabilities * probabilities.T
    )

    order = np.argsort(d, axis=1)
    ranks = np.full((n, n), np.inf, dtype=float)

    row_ids = np.arange(n)[:, None]
    rank_values = np.arange(n, dtype=float)[None, :]
    ranks[row_ids, order] = rank_values

    ranks[~valid] = np.inf
    rank_scale = max(1.0, np.sqrt(float(n)))

    rank_term = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )
    rank_term[~valid] = 0.0

    regret_term = 1.0 + 0.45 * (
        regret[:, None] + regret[None, :]
    )

    mutual_term = 0.85 + 0.30 * mutual_preference
    rank_support_term = 0.70 + 0.30 * rank_term

    long_edge_penalty = np.exp(
        -0.20 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        distance_term
        * mutual_term
        * rank_support_term
        * regret_term
        * long_edge_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
