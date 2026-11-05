import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative values")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    off_diagonal = ~diagonal

    positive_values = distances[off_diagonal & (distances > 0)]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    epsilon = max(np.finfo(float).eps * scale, 1e-12)

    masked_distances = distances.copy()
    masked_distances[diagonal] = np.nan

    row_scale = np.nanmedian(masked_distances, axis=1)
    column_scale = np.nanmedian(masked_distances, axis=0)

    local_scale = 0.5 * (
        row_scale[:, None] + column_scale[None, :]
    )

    relative_closeness = local_scale / (distances + epsilon)
    relative_closeness = np.maximum(relative_closeness, 0.0)

    tolerance = 1.25
    row_competitors = (
        distances[:, :, None] <= tolerance * distances[:, None, :]
    ).sum(axis=2)
    column_competitors = (
        distances[:, :, None] <= tolerance * distances[None, :, :]
    ).sum(axis=1)

    competitor_count = 0.5 * (
        row_competitors + column_competitors
    )
    scarcity = 1.0 / np.sqrt(competitor_count + 1.0)

    scores = relative_closeness * scarcity
    scores[diagonal] = 0.0

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
