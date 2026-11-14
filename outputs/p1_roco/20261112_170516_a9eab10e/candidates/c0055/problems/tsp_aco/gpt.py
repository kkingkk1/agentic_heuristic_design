import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d > 0)
    valid_values = d[valid]

    if valid_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.median(valid_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(valid, d, fallback)
    np.fill_diagonal(clean, fallback)

    # Convert possibly asymmetric input into a symmetric edge-length matrix.
    distance = 0.5 * (clean + clean.T)
    np.fill_diagonal(distance, np.inf)

    edge_mask = ~np.eye(n, dtype=bool)
    finite_edges = distance[edge_mask & np.isfinite(distance)]
    global_scale = (
        float(np.median(finite_edges))
        if finite_edges.size
        else fallback
    )
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = fallback

    row_min = np.min(distance, axis=1, keepdims=True)
    row_min = np.where(
        np.isfinite(row_min) & (row_min > 0),
        row_min,
        global_scale,
    )

    finite_distance = np.where(np.isfinite(distance), distance, global_scale)

    row_median = np.median(
        np.where(edge_mask, finite_distance, np.nan),
        axis=1,
        keepdims=True,
    )
    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0),
        row_median,
        global_scale,
    )

    inverse = 1.0 / (finite_distance + 1e-12)
    nearest_affinity = np.exp(
        -0.5 * finite_distance / (row_min + 1e-12)
    )
    local_affinity = np.exp(
        -finite_distance / (row_median + 1e-12)
    )

    def normalize_rows(signal: np.ndarray) -> np.ndarray:
        signal = np.maximum(signal, 0.0)
        signal = np.where(edge_mask, signal, 0.0)

        lo = np.min(
            np.where(edge_mask, signal, np.inf),
            axis=1,
            keepdims=True,
        )
        hi = np.max(signal, axis=1, keepdims=True)
        span = hi - lo

        normalized = np.divide(
            signal - lo,
            span,
            out=np.zeros_like(signal),
            where=span > 1e-12,
        )

        flat_rows = span <= 1e-12
        normalized = np.where(
            flat_rows,
            np.where(edge_mask, 1.0, 0.0),
            normalized,
        )
        return normalized

    score = (
        0.50 * normalize_rows(inverse)
        + 0.30 * normalize_rows(nearest_affinity)
        + 0.20 * normalize_rows(local_affinity)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    positive = score[score > 0]
    if positive.size:
        temperature = float(np.median(positive))
        if np.isfinite(temperature) and temperature > 0:
            score = np.power(score / temperature, 1.15)

    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
