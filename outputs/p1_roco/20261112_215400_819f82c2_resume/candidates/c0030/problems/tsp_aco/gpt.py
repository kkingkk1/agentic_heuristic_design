import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances)

    d = distances.copy()
    diagonal = np.eye(n, dtype=bool)

    finite_off_diagonal = np.isfinite(d) & ~diagonal
    if not np.any(finite_off_diagonal):
        return np.zeros_like(d)

    positive = d[finite_off_diagonal & (d > 0)]
    fallback = np.max(positive) if positive.size else 1.0

    d[~np.isfinite(d)] = fallback * 10.0
    d[d < 0] = 0.0
    np.fill_diagonal(d, np.inf)

    row_scale = np.median(
        np.where(np.isfinite(d), d, np.nan),
        axis=1
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback
    )

    inverse_distance = 1.0 / (d + 1e-12)
    inverse_distance /= np.max(inverse_distance, axis=1, keepdims=True) + 1e-12

    nearest = np.argmin(d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if 0 <= j < n and nearest[j] == i and i != j:
            mutual[i, j] = 1.0

    local_sparsity = row_scale[:, None] / (d + row_scale[:, None] + 1e-12)
    scores = inverse_distance * (0.65 + 0.25 * mutual + 0.10 * local_sparsity)

    scores[diagonal] = 0.0
    scores[~np.isfinite(scores)] = 0.0

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
