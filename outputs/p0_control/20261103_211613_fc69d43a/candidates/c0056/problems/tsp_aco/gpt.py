import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty_like(distance_matrix, dtype=float)

    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    distances = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(distances, np.inf)

    if n == 1:
        return np.zeros_like(distance_matrix, dtype=float)

    finite_positive = distances[
        np.isfinite(distances) & (distances > 0.0)
    ]
    scale = float(np.median(finite_positive)) if finite_positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe_distances = np.maximum(distances, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    kth = k - 1

    row_nearest = np.partition(safe_distances, kth=kth, axis=1)[:, :k]
    col_nearest = np.partition(safe_distances, kth=kth, axis=0)[:k, :]

    row_reference = np.mean(row_nearest, axis=1, keepdims=True)
    col_reference = np.mean(col_nearest, axis=0, keepdims=True)

    row_surprise = row_reference / safe_distances
    col_surprise = col_reference / safe_distances

    denominator = row_surprise + col_surprise + eps
    scarcity = (2.0 * row_surprise * col_surprise) / denominator

    scores = (scale / safe_distances) * np.sqrt(
        np.maximum(scarcity, 0.0)
    )

    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
