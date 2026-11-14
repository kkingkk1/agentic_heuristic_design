import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0)
    positive_values = distances[finite_nonnegative & ~diagonal &
                                (distances > 0)]

    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
        invalid_replacement = max(float(np.max(positive_values)) * 10.0,
                                  global_scale)
    else:
        global_scale = 1.0
        invalid_replacement = 1.0

    cleaned = np.where(finite_nonnegative, distances, invalid_replacement)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    valid = np.isfinite(cleaned) & (cleaned >= 0) & ~diagonal
    zero_edges = valid & (cleaned == 0.0)

    work = np.where(valid, cleaned, invalid_replacement)
    np.fill_diagonal(work, invalid_replacement)

    positive = work[valid & (work > 0)]
    fallback_scale = max(
        float(np.median(positive)) if positive.size else global_scale,
        eps,
    )

    # Use the second-nearest valid distance when possible.
    sorted_work = np.sort(work, axis=1)
    local_index = min(1, n - 2)
    local_scale = sorted_work[:, local_index]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Absolute closeness, with zero-length edges explicitly preferred.
    inverse = np.zeros((n, n), dtype=float)
    positive_mask = valid & (cleaned > 0)
    inverse[positive_mask] = 1.0 / np.maximum(cleaned[positive_mask], eps)
    inverse[zero_edges] = 1.0
    inverse_max = float(np.max(inverse))
    if inverse_max > 0:
        inverse /= inverse_max

    # Compatibility with both endpoints' local distance scales.
    normalized_length = cleaned / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_proximity = np.exp(-np.minimum(normalized_length, 50.0))
    relative_proximity[~valid] = 0.0
    relative_proximity[zero_edges] = 1.0

    # Reciprocal endpoint ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    rank_quality = 1.0 / (1.0 + ranks)
    rank_quality /= np.maximum(
        np.max(rank_quality, axis=1, keepdims=True),
        eps,
    )
    mutual_rank = np.sqrt(rank_quality * rank_quality.T)
    mutual_rank[~valid] = 0.0

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )
    scale_compatibility = np.exp(-np.minimum(scale_difference, 50.0))
    scale_compatibility[~valid] = 0.0

    # Mildly down-weight edges incident to unusually crowded neighborhoods.
    neighborhood_strength = np.sum(
        np.exp(-np.minimum(work / local_scale[:, None], 50.0)) * valid,
        axis=1,
    )
    crowding = 1.0 / np.sqrt(
        (1.0 + neighborhood_strength[:, None])
        * (1.0 + neighborhood_strength[None, :])
    )
    crowding /= max(float(np.max(crowding[valid])), eps)

    score = (
        0.45 * inverse
        + 0.25 * relative_proximity
        + 0.20 * mutual_rank
        + 0.10 * scale_compatibility
    )
    score *= 0.90 + 0.10 * crowding
    score[zero_edges] = np.maximum(score[zero_edges], 1.0)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
