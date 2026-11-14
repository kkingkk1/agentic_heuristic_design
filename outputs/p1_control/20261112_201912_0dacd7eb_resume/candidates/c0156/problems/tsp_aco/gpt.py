import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid_values = distances[finite & (distances >= 0.0)]

    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(valid_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive = d[off_diagonal & (d > 0.0)]

    eps = np.finfo(float).eps
    if positive.size:
        median = float(np.median(positive))
        q25, q75 = np.percentile(positive, [25.0, 75.0])
        spread = max(float(q75 - q25), eps)
        temperature = max(median + 0.5 * spread, eps)
    else:
        temperature = max(fallback, 1.0)

    scaled = np.clip(d / temperature, 0.0, 45.0)

    kernel = np.exp(-scaled)
    kernel[~off_diagonal] = 0.0

    row_mass = np.sum(kernel, axis=1)
    if np.any(row_mass <= 0.0) or not np.all(np.isfinite(row_mass)):
        return np.zeros((n, n), dtype=float)

    balanced = kernel.copy()

    for _ in range(80):
        row_sum = np.sum(balanced, axis=1)
        balanced /= np.maximum(row_sum[:, None], eps)

        column_sum = np.sum(balanced, axis=0)
        balanced /= np.maximum(column_sum[None, :], eps)

    score = np.sqrt(np.maximum(balanced, 0.0) * np.maximum(balanced.T, 0.0))
    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
