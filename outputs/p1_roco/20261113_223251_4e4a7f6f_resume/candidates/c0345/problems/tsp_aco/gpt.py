import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return efficient structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = finite & (raw >= 0.0) & finite.T & (raw.T >= 0.0)
    valid &= ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    valid_values = distances[valid]
    positive = valid_values[valid_values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(positive, [0.25, 0.50, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    eps = np.finfo(float).eps
    scale = max(scale if np.isfinite(scale) else 1.0, eps)

    upper = max(
        float(np.quantile(valid_values, 0.95)),
        float(np.quantile(valid_values, 0.75)),
        scale,
    )
    replacement = upper + 12.0 * scale

    filled = np.where(valid, distances, replacement)
    np.fill_diagonal(filled, replacement)

    normalized = np.minimum(filled / scale, 60.0)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid])

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + scale * 1e-9)
    inverse /= max(float(np.max(inverse)), eps)

    neighbor_order = np.argsort(filled, axis=1, kind="stable")
    neighbor_count = min(n - 1, max(4, min(12, int(np.sqrt(n)) + 2)))
    nearest = neighbor_order[:, :neighbor_count]

    neighbor_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    neighbor_mask[rows, nearest] = True
    neighbor_mask &= valid

    rank = np.full((n, n), float(n), dtype=float)
    rank[rows, neighbor_order] = np.arange(n, dtype=float)
    rank_score = np.exp(-rank / max(1.0, 0.75 * np.sqrt(float(n))))
    rank_score[~valid] = 0.0

    row_mass = affinity.sum(axis=1)
    preference = affinity / np.maximum(row_mass[:, None], eps)
    mutual_preference = np.sqrt(
        np.maximum(preference * preference.T, 0.0)
    )
    mutual_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )
    mutual_rank[~valid] = 0.0

    sorted_distances = np.sort(filled, axis=1)
    first = sorted_distances[:, 0]
    second = sorted_distances[:, 1]
    gap = np.maximum(second - first, 0.18 * scale)
    gap = np.maximum(gap, eps)

    competition = 1.0 - np.exp(
        -np.maximum(second[:, None] - filled, 0.0)
        / gap[:, None]
    )
    competition[~valid] = 0.0

    mutual_competition = np.sqrt(
        np.maximum(competition * competition.T, 0.0)
    )
    mutual_competition[~valid] = 0.0

    bypass = np.full((n, n), replacement, dtype=float)
    node_ids = np.arange(n)

    for column in range(neighbor_count):
        intermediates = nearest[:, column]
        two_hop = (
            filled[node_ids, intermediates][:, None]
            + filled[intermediates, :]
        )
        bypass = np.minimum(bypass, two_hop)

    np.fill_diagonal(bypass, replacement)

    detour_gain = np.maximum(bypass - filled, 0.0)
    irreplaceability = 1.0 - np.exp(
        -detour_gain / max(0.75 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    local_degree = neighbor_mask.sum(axis=1).astype(float)
    local_degree /= max(float(neighbor_count), 1.0)

    mutual_neighbor = neighbor_mask & neighbor_mask.T
    mutual_support = mutual_neighbor.astype(float)

    endpoint_support = np.sqrt(
        np.maximum(
            local_degree[:, None] * local_degree[None, :],
            0.0,
        )
    )
    connectivity = (
        0.55 * mutual_support
        + 0.45 * endpoint_support
    )
    connectivity[~valid] = 0.0

    structural = (
        0.56 * mutual_rank
        + 0.44 * mutual_preference
    )

    necessity = (
        0.58 * mutual_competition
        + 0.34 * irreplaceability
        + 0.08
    )

    signals = (
        np.maximum(affinity, eps)
        * np.maximum(inverse, eps)
        * np.maximum(structural, eps)
        * np.maximum(necessity, eps)
        * np.maximum(0.82 + 0.36 * connectivity, eps)
    )

    signals[~valid] = 0.0
    score = 0.5 * (signals + signals.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    balanced = score.copy()
    for _ in range(8):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.70 * score + 0.30 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
