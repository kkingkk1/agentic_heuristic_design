import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    safe_distances = np.where(valid, distances, np.inf)

    row_medians = np.full(n, 1.0, dtype=float)
    for i in range(n):
        row_values = safe_distances[i, valid[i]]
        if row_values.size:
            row_medians[i] = np.median(row_values)

    scale = np.sqrt(row_medians[:, None] * row_medians[None, :])
    normalized_distance = np.divide(
        distances,
        scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=scale > 0,
    )

    shortness = np.exp(-normalized_distance)

    endpoint_scarcity = np.sqrt(
        np.divide(
            row_medians[:, None],
            distances,
            out=np.zeros_like(distances, dtype=float),
            where=valid,
        )
        * np.divide(
            row_medians[None, :],
            distances,
            out=np.zeros_like(distances, dtype=float),
            where=valid,
        )
    )
    endpoint_scarcity = np.clip(endpoint_scarcity, 0.0, 3.0)

    if n > 2:
        two_hop = np.full((n, n), np.inf, dtype=float)
        for k in range(n):
            via_k = safe_distances[:, k, None] + safe_distances[None, k, :]
            via_k[k, :] = np.inf
            via_k[:, k] = np.inf
            two_hop = np.minimum(two_hop, via_k)

        bridge_indispensability = np.divide(
            two_hop - distances,
            two_hop + distances,
            out=np.zeros_like(distances, dtype=float),
            where=valid & np.isfinite(two_hop) & ((two_hop + distances) > 0),
        )
        bridge_indispensability = np.clip(bridge_indispensability, 0.0, 1.0)
    else:
        bridge_indispensability = np.zeros_like(distances, dtype=float)

    scores = (
        shortness
        * (1.0 + 0.5 * endpoint_scarcity)
        * (1.0 + bridge_indispensability)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
