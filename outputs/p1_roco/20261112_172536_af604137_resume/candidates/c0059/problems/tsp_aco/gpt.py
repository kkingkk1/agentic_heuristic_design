import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = distances.copy()
    d[~np.isfinite(d)] = np.inf
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0]
    fallback = float(np.median(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)
    d[~np.isfinite(d)] = 10.0 * fallback

    positive = d[d > 0]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    row_sorted = np.sort(d, axis=1)
    local_k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    local_scale = np.median(row_sorted[:, :local_k], axis=1)
    local_scale = np.maximum(local_scale, 0.1 * global_scale)

    # Smooth local distance preference.
    local_inverse = 1.0 / (d + 0.05 * local_scale[:, None])
    inverse_norm = local_inverse / (
        np.mean(local_inverse, axis=1, keepdims=True) + 1e-12
    )

    # Rank-based proximity, with a slower decay for larger instances.
    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_decay = max(1.5, 0.30 * np.sqrt(n))
    rank_score = np.exp(-ranks / rank_decay)

    # Endpoint-relative competitiveness and regret.
    nearest = row_sorted[:, 0]
    second_nearest = row_sorted[:, 1] if n > 2 else row_sorted[:, 0]
    regret_scale = np.maximum(second_nearest - nearest, 0.05 * global_scale)
    endpoint_quality = 1.0 / (
        1.0 + np.maximum(0.0, d - nearest[:, None]) /
        (local_scale[:, None] + 1e-12)
    )
    regret_bonus = np.exp(
        -np.maximum(0.0, d - nearest[:, None]) /
        (regret_scale[:, None] + local_scale[:, None] + 1e-12)
    )

    # Mutual-nearest-neighbor and shared-locality preference.
    relative = d / (local_scale[:, None] + 1e-12)
    mutuality = np.exp(-0.5 * (relative + relative.T))
    nearest_cutoff = row_sorted[:, min(local_k - 1, n - 2)]
    candidate_gate = 1.0 / (
        1.0
        + np.maximum(0.0, d - nearest_cutoff[:, None])
        / (local_scale[:, None] + global_scale + 1e-12)
    )

    score = (
        0.34 * inverse_norm
        + 0.24 * rank_score
        + 0.18 * endpoint_quality
        + 0.12 * regret_bonus
        + 0.08 * mutuality
        + 0.04 * candidate_gate
    )

    score = 0.5 * (score + score.T)

    # Retain exploration mass on every finite off-diagonal edge.
    finite_mask = np.isfinite(distances)
    off_diagonal = ~np.eye(n, dtype=bool)
    score = np.where(finite_mask & off_diagonal, score, 0.0)
    positive_scores = score[score > 0]
    if positive_scores.size:
        floor = 0.02 * float(np.median(positive_scores))
        score = np.where(score > 0, score + floor, 0.0)

    np.fill_diagonal(score, 0.0)
    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
