import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distance_matrix)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid & np.isfinite(distances)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size == 0:
        scale = 1.0
        low_scale = 1.0
        high_scale = 1.0
    else:
        low_scale = float(np.percentile(positive_values, 20.0))
        high_scale = float(np.percentile(positive_values, 80.0))
        median_scale = float(np.percentile(positive_values, 50.0))
        low_scale = max(low_scale, np.finfo(float).eps)
        high_scale = max(high_scale, low_scale)
        scale = max(median_scale, np.finfo(float).eps)

    spread = high_scale / low_scale
    decay = float(np.clip(0.85 + 0.10 * np.log1p(spread), 0.75, 1.35))
    exponent = float(np.clip(1.15 - 0.08 * np.log1p(spread), 0.85, 1.25))

    normalized = np.divide(
        distances,
        scale,
        out=np.full_like(distances, np.inf),
        where=np.isfinite(distances),
    )

    attraction = np.zeros((n, n), dtype=float)
    finite_distances = np.isfinite(normalized)
    attraction[finite_distances] = 1.0 / (
        1.0 + decay * np.power(np.maximum(normalized[finite_distances], 0.0), exponent)
    )

    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

    row_min = np.min(distances, axis=1)
    col_min = np.min(distances, axis=0)

    row_min = np.where(np.isfinite(row_min), row_min, scale)
    col_min = np.where(np.isfinite(col_min), col_min, scale)

    row_potential = 1.0 / (1.0 + row_min / scale)
    col_potential = 1.0 / (1.0 + col_min / scale)

    outgoing = np.broadcast_to(row_potential[:, None], (n, n))
    incoming = np.broadcast_to(col_potential[None, :], (n, n))

    continuation = outgoing * incoming
    continuation = 0.55 + 0.45 * continuation

    row_rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue
        values = distances[i, indices]
        order = np.argsort(values, kind="stable")
        ranks = np.empty(indices.size, dtype=float)
        ranks[order] = np.arange(indices.size, dtype=float)
        row_rank[i, indices] = np.exp(-ranks / max(1.0, 0.30 * indices.size))

    valid_fraction = np.sqrt(
        np.divide(
            row_counts[:, None] * col_counts[None, :],
            max(1.0, float(n - 1) ** 2),
        )
    )
    coverage = 0.75 + 0.25 * np.clip(valid_fraction, 0.0, 1.0)

    nearest_index = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest_index[i])
        if valid[i, j] and valid[j, i] and nearest_index[j] == i:
            mutual[i, j] = 1.0

    scores = (
        attraction
        * (0.52 + 0.30 * row_rank + 0.18 * coverage)
        * continuation
        * (1.0 + 0.16 * mutual)
    )

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
