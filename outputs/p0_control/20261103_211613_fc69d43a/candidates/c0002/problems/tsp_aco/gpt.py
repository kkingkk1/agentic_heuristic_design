import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return mutual-scarcity edge scores for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square NumPy array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros_like(distances, dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, 0.0)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    positive = d[~np.eye(n, dtype=bool)]
    global_scale = np.median(positive[positive > 0]) if np.any(positive > 0) else 1.0

    endpoint_scales = np.empty(n, dtype=float)
    for i in range(n):
        row = d[i, np.arange(n) != i]
        positive_row = row[row > 0]
        endpoint_scales[i] = (
            np.median(positive_row) if positive_row.size else global_scale
        )

    pair_scale = np.sqrt(
        np.maximum(endpoint_scales[:, None] * endpoint_scales[None, :], 1e-12)
    )
    pair_scale = np.maximum(pair_scale, global_scale * 1e-6)

    shortness = np.exp(-d / pair_scale)
    np.fill_diagonal(shortness, 0.0)

    row_mass = shortness.sum(axis=1, keepdims=True)
    preference = shortness / np.maximum(row_mass, 1e-12)
    mutual_scarcity = np.sqrt(preference * preference.T)

    scores = shortness * (1.0 + mutual_scarcity)
    np.fill_diagonal(scores, 0.0)

    return np.asarray(scores, dtype=float)
