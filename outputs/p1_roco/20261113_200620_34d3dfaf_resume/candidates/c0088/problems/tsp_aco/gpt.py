import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~np.eye(n, dtype=bool)
    )

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    finite_distances = distances[valid]
    positive_distances = finite_distances[finite_distances > 0.0]
    scale = (
        float(np.median(positive_distances))
        if positive_distances.size
        else 1.0
    )
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe_distances = np.where(valid, distances, np.inf)
    inverse = np.zeros_like(distances, dtype=float)
    inverse[valid] = 1.0 / (safe_distances[valid] + eps)

    inverse_values = inverse[valid]
    cap = float(np.quantile(inverse_values, 0.995))
    if not np.isfinite(cap) or cap <= 0.0:
        cap = float(np.max(inverse_values))

    inverse = np.minimum(inverse, cap)
    inverse /= max(cap, eps)

    row_counts = valid.sum(axis=1, keepdims=True)
    col_counts = valid.sum(axis=0, keepdims=True)

    row_rank = (
        (valid & (distances <= distances[:, :, None])).sum(axis=1)
        if False
        else np.zeros_like(distances, dtype=float)
    )

    for i in range(n):
        row_valid = valid[i]
        if np.any(row_valid):
            row_rank[i, row_valid] = (
                (distances[i, row_valid, None] >= distances[i, row_valid])
                .sum(axis=1)
                / max(int(row_counts[i, 0]), 1)
            )

    col_rank = np.zeros_like(distances, dtype=float)
    for j in range(n):
        col_valid = valid[:, j]
        if np.any(col_valid):
            col_rank[col_valid, j] = (
                (distances[col_valid, j, None] >= distances[col_valid, j])
                .sum(axis=1)
                / max(int(col_counts[0, j]), 1)
            )

    bidirectional_rank = 0.5 * (row_rank + col_rank)

    row_order = np.argsort(-inverse, axis=1)
    col_order = np.argsort(-inverse, axis=0)

    row_best = inverse[row_order[:, 0], np.arange(n)]
    row_second = np.zeros(n, dtype=float)
    col_best = inverse[np.arange(n), col_order[0, :]]
    col_second = np.zeros(n, dtype=float)

    if n > 1:
        row_second = inverse[row_order[:, 1], np.arange(n)]
        col_second = inverse[np.arange(n), col_order[1, :]]

    continuation = np.zeros_like(inverse, dtype=float)

    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue

            outgoing = row_best[j]
            if row_order[j, 0] == i:
                outgoing = row_second[j]

            incoming = col_best[i]
            if col_order[0, i] == j:
                incoming = col_second[i]

            continuation[i, j] = np.sqrt(max(outgoing, 0.0) * max(incoming, 0.0))

    continuation_max = float(np.max(continuation))
    if continuation_max > 0.0:
        continuation /= continuation_max

    score = (
        0.55 * inverse
        + 0.25 * bidirectional_rank
        + 0.20 * continuation
    )
    score[~valid] = -np.inf

    row_max = np.max(score, axis=1, keepdims=True)
    temperature = 0.30 + 0.05 * np.log1p(max(n, 1))

    centered = np.zeros_like(score)
    finite_rows = np.isfinite(row_max[:, 0])
    centered[finite_rows] = (
        score[finite_rows] - row_max[finite_rows]
    ) / temperature

    weights = np.zeros_like(score)
    weights[finite_rows] = np.exp(
        np.clip(centered[finite_rows], -60.0, 0.0)
    )

    valid_count = max(int(valid.sum()), 1)
    exploration_floor = min(0.05, 1.0 / np.sqrt(valid_count))
    weights[valid] = (
        exploration_floor
        + (1.0 - exploration_floor) * weights[valid]
    )

    weights[~valid] = 0.0

    maximum = float(np.max(weights))
    if maximum > 0.0:
        weights /= maximum

    return weights
