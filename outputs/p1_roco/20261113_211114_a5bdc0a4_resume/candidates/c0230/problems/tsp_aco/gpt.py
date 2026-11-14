import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    # Only retain edges that are finite in both directions before symmetrizing.
    valid = np.isfinite(raw) & np.isfinite(raw.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw), raw, np.nan)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = np.where(
        np.isfinite(cleaned) & np.isfinite(cleaned.T),
        0.5 * (np.nan_to_num(cleaned, nan=0.0)
               + np.nan_to_num(cleaned.T, nan=0.0)),
        0.0,
    )

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = max(maximum + 8.0 * scale, scale)

    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid_values = distances[valid]
    median_distance = (
        float(np.median(valid_values))
        if valid_values.size
        else scale
    )
    temperature = max(
        0.24 * median_distance,
        0.08 * scale,
        np.finfo(float).eps,
    )

    # Global distance affinity.
    affinity = np.exp(-np.minimum(distances, penalty) / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0.0):
        result = np.ones((n, n), dtype=float)
        result[~valid] = 0.0
        np.fill_diagonal(result, 0.0)
        return result

    # Row-wise local competition and candidate ranking.
    order = np.argsort(distances, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(distances, order, axis=1)

    rank = np.empty_like(order, dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    nearest = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 1 else nearest
    third = sorted_distances[:, 2] if n > 2 else second

    local_scale = np.maximum(
        0.5 * (second - nearest) + 0.5 * (third - second),
        0.12 * scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    row_gap = np.maximum(
        distances - nearest[:, None],
        0.0,
    )
    scarcity = np.exp(
        -np.minimum(row_gap / local_scale[:, None], 24.0)
    )

    # Reciprocal rank favors edges that are competitive at both endpoints.
    reciprocal_rank = 1.0 / (1.0 + rank)
    reciprocal_rank[~valid] = 0.0
    mutual_rank = np.sqrt(reciprocal_rank * reciprocal_rank.T)

    # Normalize local distances relative to each endpoint's neighborhood.
    row_median = np.median(
        np.where(valid, distances, np.nan),
        axis=1,
    )
    row_median = np.nan_to_num(row_median, nan=median_distance)
    row_median = np.maximum(row_median, 0.12 * scale)

    relative_distance = np.exp(
        -distances / np.sqrt(row_median[:, None] * row_median[None, :])
    )
    relative_distance[~valid] = 0.0

    # Compute the best two-hop bypass for every edge.
    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        via_k = distances[:, k, None] + distances[None, k, :]
        bypass = np.minimum(bypass, via_k)

    np.fill_diagonal(bypass, penalty)

    # A large bypass cost relative to the direct edge indicates an important edge.
    detour_gain = np.maximum(bypass - distances, 0.0)
    detour_scale = max(0.65 * scale, 0.12 * median_distance)
    irreplaceability = 1.0 - np.exp(
        -detour_gain / max(detour_scale, np.finfo(float).eps)
    )
    irreplaceability[~valid] = 0.0

    # Symmetric endpoint scarcity.
    symmetric_scarcity = np.sqrt(scarcity * scarcity.T)
    symmetric_scarcity[~valid] = 0.0

    base = affinity / max(float(np.max(affinity)), np.finfo(float).eps)

    score = (
        0.38 * base
        + 0.22 * relative_distance
        + 0.18 * mutual_rank
        + 0.12 * symmetric_scarcity
        + 0.10 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    # A small number of symmetric balancing passes reduces hub bias.
    balanced = score.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum_result = float(np.max(result))
    if np.isfinite(maximum_result) and maximum_result > 0.0:
        result /= maximum_result

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
