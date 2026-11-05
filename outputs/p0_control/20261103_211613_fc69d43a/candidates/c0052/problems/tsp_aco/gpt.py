import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    d = distance_matrix.copy()
    np.fill_diagonal(d, np.nan)

    finite = np.isfinite(d)
    if not np.any(finite):
        return scores

    finite_values = d[finite]
    largest = np.max(finite_values)
    if not np.isfinite(largest):
        return scores

    fallback = max(float(largest), 0.0) * 10.0 + 1.0
    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)

    kth = min(2, n - 1)
    row_scale = np.partition(d, kth, axis=1)[:, kth]
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    kernel = np.exp(-d / row_scale[:, None])
    kernel[~finite] = 0.0
    np.fill_diagonal(kernel, 0.0)

    direct = kernel
    mutual = np.exp(-d / row_scale[None, :])
    mutual[~finite] = 0.0
    affinity = direct * mutual

    k = min(8, n - 1)
    neighbor_idx = np.argpartition(-kernel, kth=k - 1, axis=1)[:, :k]
    neighbor_weights = np.take_along_axis(kernel, neighbor_idx, axis=1)

    overlap = np.zeros((n, n), dtype=float)
    source_rows = np.arange(n)

    for neighbor in range(n):
        selected = np.flatnonzero(np.any(neighbor_idx == neighbor, axis=1))
        if selected.size == 0:
            continue

        positions = np.argmax(neighbor_idx[selected] == neighbor, axis=1)
        weights = neighbor_weights[selected, positions]
        overlap[np.ix_(selected, selected)] += weights[:, None] * weights[None, :]

    overlap_scale = np.maximum(
        np.max(overlap, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    compatibility = overlap / overlap_scale

    scores = affinity * (0.5 + 0.5 * compatibility)
    scores[~finite] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
