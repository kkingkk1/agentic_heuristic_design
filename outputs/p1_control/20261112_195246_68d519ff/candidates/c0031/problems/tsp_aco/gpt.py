import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    positive_values = d[finite_positive]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    q25, q50, q75 = np.percentile(positive_values, [25.0, 50.0, 75.0])
    spread = max(float(q75 - q25), np.finfo(float).eps)
    fallback = float(q75 + 4.0 * spread)

    work = np.array(d, copy=True)
    invalid = ~np.isfinite(work) | (work <= 0)
    work[invalid] = fallback
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, fallback)

    center = float(np.median(work[~np.eye(n, dtype=bool)]))
    scale = max(float(np.percentile(work[~np.eye(n, dtype=bool)], 75.0) - center), spread)
    scale = max(scale, np.finfo(float).eps)

    standardized = (work - center) / scale
    affinity = np.exp(-0.5 * standardized * standardized)
    affinity = np.clip(affinity, np.finfo(float).tiny, 1.0)
    np.fill_diagonal(affinity, 0.0)

    # Symmetric Sinkhorn balancing: equalize the total prior mass of each node.
    for _ in range(80):
        row_sum = affinity.sum(axis=1)
        row_sum = np.maximum(row_sum, np.finfo(float).tiny)
        factors = 1.0 / np.sqrt(row_sum)
        affinity *= factors[:, None] * factors[None, :]
        affinity = 0.5 * (affinity + affinity.T)
        np.fill_diagonal(affinity, 0.0)

    maximum = float(np.max(affinity))
    if maximum > 0.0:
        affinity /= maximum

    np.fill_diagonal(affinity, 0.0)
    return affinity.astype(float, copy=False)
