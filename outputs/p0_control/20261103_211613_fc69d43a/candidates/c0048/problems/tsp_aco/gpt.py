import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric adaptive edge-score matrix for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & off_diagonal
    positive = valid & (distances > 0)

    positive_values = distances[positive]
    if positive_values.size == 0:
        return scores

    fallback = float(np.median(positive_values))
    eps = max(fallback * 1e-12, np.finfo(float).eps)

    # Adaptive local scale from the nearest positive neighbors.
    local_scale = np.empty(n, dtype=float)
    neighbor_count = max(1, int(np.ceil(np.sqrt(n - 1))))

    for i in range(n):
        candidates = distances[i, positive[i]]
        if candidates.size == 0:
            local_scale[i] = fallback
        else:
            candidates = np.sort(candidates)
            local_scale[i] = float(
                np.mean(candidates[: min(neighbor_count, candidates.size)])
            )

    local_scale = np.maximum(local_scale, eps)
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    safe_distances = np.where(positive, distances, np.inf)
    normalized = safe_distances / np.maximum(endpoint_scale, eps)

    # Direct local closeness.
    direct = np.exp(-np.minimum(normalized, 700.0))

    # Bidirectional local competition.
    row_competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_values = normalized[i]
        finite_row = np.isfinite(row_values)
        if np.any(finite_row):
            baseline = float(np.median(row_values[finite_row]))
            row_competition[i, finite_row] = (
                baseline / np.maximum(row_values[finite_row], eps)
            )

    competition = np.sqrt(
        row_competition * row_competition.T
    )
    competition = competition / (1.0 + competition)

    # Best two-hop continuity through a common intermediary.
    two_hop = safe_distances[:, :, None] + safe_distances[None, :, :]
    bridge = np.min(two_hop, axis=1)
    bridge_ratio = bridge / np.maximum(safe_distances, eps)
    continuity = 1.0 / (1.0 + np.maximum(bridge_ratio, 0.0))

    affinity = (
        direct
        * (0.55 + 0.45 * competition)
        * (0.65 + 0.35 * continuity)
    )

    raw = np.zeros_like(distances, dtype=float)
    raw[positive] = affinity[positive] / np.maximum(
        distances[positive],
        eps,
    )

    values = raw[positive]
    if values.size == 0:
        return scores

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if maximum > minimum:
        scores[positive] = (raw[positive] - minimum) / (maximum - minimum)
    else:
        scores[positive] = 1.0

    scores = 0.5 * (scores + scores.T)
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
