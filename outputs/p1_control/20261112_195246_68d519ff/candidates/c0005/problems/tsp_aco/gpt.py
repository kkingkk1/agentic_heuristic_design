import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return prior indicators for promising TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite_distances = np.where(np.isfinite(distances), distances, 0.0)
    finite_distances = np.maximum(finite_distances, 0.0)

    diagonal = np.eye(n, dtype=bool)
    off_diagonal = ~diagonal

    positive_edges = finite_distances[off_diagonal]
    positive_edges = positive_edges[positive_edges > 0]

    if positive_edges.size == 0:
        scores = np.zeros((n, n), dtype=float)
        np.fill_diagonal(scores, 0.0)
        return scores

    fallback = float(np.max(positive_edges))
    distances = np.where(
        np.isfinite(distances) & (distances > 0),
        distances,
        fallback,
    )
    np.fill_diagonal(distances, fallback)

    row_scale = np.median(
        np.where(off_diagonal, distances, np.nan),
        axis=1,
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )

    local_scale = np.sqrt(row_scale[:, None] * row_scale[None, :])
    relative_distance = distances / np.maximum(local_scale, np.finfo(float).eps)

    scale_score = 1.0 / (1.0 + relative_distance)

    ranks = np.argsort(np.argsort(distances, axis=1), axis=1).astype(float)
    rank_denominator = max(n - 1, 1)
    rank_score = 1.0 - ranks / rank_denominator

    scores = 0.65 * scale_score + 0.35 * rank_score
    scores = 0.5 * (scores + scores.T)
    scores[~np.isfinite(scores)] = 0.0
    scores[diagonal] = 0.0

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
