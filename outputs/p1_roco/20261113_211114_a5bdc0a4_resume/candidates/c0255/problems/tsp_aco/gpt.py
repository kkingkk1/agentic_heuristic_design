import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.maximum(np.where(np.isfinite(raw), raw, 0.0), 0.0)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = distances[valid]
    fallback = float(np.max(finite_values)) if finite_values.size else scale
    invalid_penalty = fallback + 8.0 * scale

    safe_distances = distances.copy()
    safe_distances[~valid] = invalid_penalty
    np.fill_diagonal(safe_distances, invalid_penalty)

    temperature = max(0.85 * scale, np.finfo(float).eps)

    distance_signal = np.exp(
        -np.clip(safe_distances / temperature, 0.0, 30.0)
    )
    distance_signal[~valid] = 0.0
    np.fill_diagonal(distance_signal, 0.0)

    row_ranks = np.zeros((n, n), dtype=float)
    col_ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        order = np.argsort(safe_distances[i], kind="stable")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        row_ranks[i] = rank

    for j in range(n):
        order = np.argsort(safe_distances[:, j], kind="stable")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        col_ranks[:, j] = rank

    rank_scale = max(0.38 * max(n - 1, 1), 1.0)
    row_rank_signal = np.exp(-row_ranks / rank_scale)
    col_rank_signal = np.exp(-col_ranks / rank_scale)

    row_max = np.maximum(
        np.max(distance_signal, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    col_max = np.maximum(
        np.max(distance_signal, axis=0, keepdims=True),
        np.finfo(float).eps,
    )

    row_support = distance_signal / row_max
    col_support = distance_signal / col_max
    local_support = np.sqrt(row_support * col_support)

    score = (
        0.58 * distance_signal
        + 0.24 * np.sqrt(row_rank_signal * col_rank_signal)
        + 0.13 * local_support
    )

    score[valid] += 0.05
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
