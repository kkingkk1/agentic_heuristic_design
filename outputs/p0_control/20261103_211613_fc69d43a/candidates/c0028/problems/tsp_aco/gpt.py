import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be two-dimensional")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe_distances = np.maximum(distances, eps)

    inverse_distance = 1.0 / safe_distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_reference = np.median(
        np.where(np.isfinite(distances), safe_distances, np.nan),
        axis=1
    )
    row_reference = np.where(
        np.isfinite(row_reference) & (row_reference > 0),
        row_reference,
        scale
    )

    column_reference = np.median(
        np.where(np.isfinite(distances), safe_distances, np.nan),
        axis=0
    )
    column_reference = np.where(
        np.isfinite(column_reference) & (column_reference > 0),
        column_reference,
        scale
    )

    source_selectivity = row_reference[:, None] / safe_distances
    destination_selectivity = column_reference[None, :] / safe_distances

    scores = np.power(
        np.maximum(inverse_distance, eps)
        * np.maximum(source_selectivity, eps)
        * np.maximum(destination_selectivity, eps),
        1.0 / 3.0
    )

    finite_scores = scores[np.isfinite(scores) & (scores > 0)]
    if finite_scores.size:
        scores /= np.median(finite_scores)

    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0
    return scores
