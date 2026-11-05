import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge scores for a fixed ant-colony TSP solver.

    Higher scores indicate edges that are short and locally competitive
    relative to both incident nodes.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0)
    usable = distances[finite_positive]

    if usable.size == 0:
        return scores

    fallback = float(np.median(usable))
    safe_distances = np.where(finite_positive, distances, fallback)

    candidates = safe_distances.copy()
    np.fill_diagonal(candidates, np.nan)

    row_baseline = np.nanmedian(candidates, axis=1)
    row_baseline = np.where(
        np.isfinite(row_baseline) & (row_baseline > 0),
        row_baseline,
        fallback,
    )

    endpoint_competitiveness = (
        0.5 * row_baseline[:, None] / safe_distances
        + 0.5 * row_baseline[None, :] / safe_distances
    )

    competitiveness = np.power(
        np.maximum(endpoint_competitiveness, 0.0),
        0.75,
    )
    inverse_distance = 1.0 / safe_distances
    raw_scores = inverse_distance * (1.0 + 0.8 * competitiveness)

    valid_edges = finite_positive.copy()
    np.fill_diagonal(valid_edges, False)

    raw_scores = np.where(valid_edges, raw_scores, 0.0)
    np.fill_diagonal(raw_scores, 0.0)

    valid_values = raw_scores[valid_edges]
    if valid_values.size == 0:
        return scores

    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))

    if maximum > minimum:
        scores = (raw_scores - minimum) / (maximum - minimum)
    else:
        scores = np.where(valid_edges, 1.0, 0.0)

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
