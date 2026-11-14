import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return nonnegative edge-priority indicators for stochastic TSP sampling.

    The score combines:
    1. distance normalized by local node scales;
    2. endpoint-wise relative attractiveness;
    3. a mutual-preference bonus for edges favored by both endpoints;
    4. a mild scarcity bonus for nodes with fewer competitive neighbors.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a two-dimensional array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)
    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    finite_values = distance_matrix[np.isfinite(distance_matrix)]
    if finite_values.size == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    positive_values = finite_values[finite_values > 0]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    d = np.array(distance_matrix, dtype=float, copy=True)
    d[~np.isfinite(d)] = np.inf
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    local_scale = np.nanmedian(positive, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )

    scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    scale = np.maximum(scale, np.finfo(float).eps)

    normalized_distance = d / scale
    closeness = np.exp(-np.minimum(normalized_distance, 50.0))
    closeness[~np.isfinite(closeness)] = 0.0
    np.fill_diagonal(closeness, 0.0)

    row_totals = closeness.sum(axis=1, keepdims=True)
    relative_attractiveness = closeness / np.maximum(
        row_totals,
        np.finfo(float).eps,
    )

    mutual_preference = np.sqrt(
        relative_attractiveness * relative_attractiveness.T
    )

    scarcity = 1.0 / np.maximum(row_totals.ravel(), np.finfo(float).eps)
    scarcity = scarcity / max(float(np.max(scarcity)), np.finfo(float).eps)
    scarcity_bonus = np.sqrt(scarcity[:, None] * scarcity[None, :])

    score = (
        0.45 * closeness
        + 0.35 * mutual_preference
        + 0.20 * scarcity_bonus * closeness
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score
