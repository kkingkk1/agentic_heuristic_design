import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric local-compatibility heuristic for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances)
    valid = finite & ~diagonal

    if np.any(distances[valid] < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    positive = valid & (distances > 0)
    positive_values = distances[positive]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    row_values = np.where(positive, distances, np.nan)
    row_scales = np.nanmedian(row_values, axis=1)
    row_scales = np.where(
        np.isfinite(row_scales) & (row_scales > 0),
        row_scales,
        global_scale,
    )
    row_scales = np.maximum(row_scales, np.finfo(float).tiny)

    pair_scale = np.sqrt(row_scales[:, None] * row_scales[None, :])
    normalized = distances / pair_scale

    affinity = np.zeros_like(distances, dtype=float)
    affinity[valid] = 1.0 / (1.0 + np.maximum(normalized[valid], 0.0))

    row_mass = affinity.sum(axis=1)
    mass_floor = np.finfo(float).eps
    endpoint_competition = affinity / (row_mass[:, None] + mass_floor)

    reverse_competition = endpoint_competition.T
    compatibility = np.sqrt(
        np.maximum(endpoint_competition * reverse_competition, 0.0)
    )

    direct = affinity
    raw_scores = direct * (0.5 + 0.5 * compatibility)
    raw_scores[~valid] = 0.0

    raw_scores = 0.5 * (raw_scores + raw_scores.T)
    valid_undirected = valid | valid.T
    values = raw_scores[valid_undirected]

    if values.size:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum > minimum:
            scores = (raw_scores - minimum) / (maximum - minimum)
        else:
            scores[valid_undirected] = 1.0

    scores[~valid_undirected] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    return scores.astype(float, copy=False)
