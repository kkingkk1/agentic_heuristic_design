import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0) & off_diagonal

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    valid_off_diagonal = d[off_diagonal]
    finite_off_diagonal = valid_off_diagonal[
        np.isfinite(valid_off_diagonal)
    ]

    if finite_off_diagonal.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_off_diagonal))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, 0.0)

    positive_values = d[off_diagonal & (d > 0.0)]
    if positive_values.size:
        center = float(np.median(positive_values))
        deviation = float(
            np.median(np.abs(positive_values - center))
        )
    else:
        center = fallback
        deviation = 0.0

    eps = np.finfo(float).eps
    temperature = max(
        center,
        1.4826 * deviation,
        eps,
    )

    scaled = np.clip(d / temperature, 0.0, 60.0)
    kernel = np.exp(-scaled)
    np.fill_diagonal(kernel, 0.0)

    if not np.any(kernel > 0.0):
        return np.zeros((n, n), dtype=float)

    balanced = kernel.copy()

    for _ in range(60):
        row_sum = np.sum(balanced, axis=1)
        row_sum = np.maximum(row_sum, eps)
        balanced /= row_sum[:, None]
        np.fill_diagonal(balanced, 0.0)

        column_sum = np.sum(balanced, axis=0)
        column_sum = np.maximum(column_sum, eps)
        balanced /= column_sum[None, :]
        np.fill_diagonal(balanced, 0.0)

    score = 0.5 * (balanced + balanced.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    score /= maximum
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
