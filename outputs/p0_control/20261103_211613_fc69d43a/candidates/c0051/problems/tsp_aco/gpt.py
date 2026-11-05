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
        return np.zeros_like(distances, dtype=float)

    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
        scale = max(scale, np.finfo(float).tiny)
    else:
        scale = 1.0

    normalized = distances / scale
    normalized = np.nan_to_num(
        normalized,
        nan=np.finfo(float).max,
        posinf=np.finfo(float).max,
        neginf=0.0,
    )

    positive_floor = np.finfo(float).eps
    safe_distances = np.maximum(normalized, positive_floor)
    np.fill_diagonal(safe_distances, np.finfo(float).max)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    row_neighbors = np.partition(safe_distances, kth=k - 1, axis=1)[:, :k]
    col_neighbors = np.partition(safe_distances, kth=k - 1, axis=0)[:k, :]

    row_reference = np.mean(row_neighbors, axis=1, keepdims=True)
    col_reference = np.mean(col_neighbors, axis=0, keepdims=True)

    row_shortness = np.divide(
        row_reference,
        safe_distances,
        out=np.zeros_like(safe_distances),
        where=safe_distances > 0.0,
    )
    col_shortness = np.divide(
        col_reference,
        safe_distances,
        out=np.zeros_like(safe_distances),
        where=safe_distances > 0.0,
    )

    scarcity = np.divide(
        2.0 * row_shortness * col_shortness,
        row_shortness + col_shortness + positive_floor,
        out=np.zeros_like(safe_distances),
        where=(row_shortness + col_shortness) > 0.0,
    )

    inverse_distance = np.divide(
        1.0,
        safe_distances,
        out=np.zeros_like(safe_distances),
        where=safe_distances > 0.0,
    )

    scores = inverse_distance * np.sqrt(np.maximum(scarcity, 0.0))

    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
