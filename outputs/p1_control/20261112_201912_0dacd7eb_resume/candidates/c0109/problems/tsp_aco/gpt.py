import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    finite_positive &= ~np.eye(n, dtype=bool)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[finite_positive]
    fallback = float(np.max(valid_values))
    fallback = max(fallback, np.finfo(float).eps)

    work = np.where(finite_positive, distances, fallback)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work) & (work > 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = work[valid]
    fallback = float(np.max(positive))
    safe = np.where(valid, work, fallback)
    np.fill_diagonal(safe, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = np.partition(safe, k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe / (pair_scale + eps)

    distance_signal = 1.0 / (1.0 + normalized_distance)
    distance_signal[~valid] = 0.0

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]

    rank_signal = 1.0 / (1.0 + ranks)
    rank_signal[~valid] = 0.0
    rank_signal = np.sqrt(rank_signal * rank_signal.T)

    nearest_index = order[:, 0]
    mutual_neighbor = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest_index):
        if i != j and nearest_index[j] == i and valid[i, j]:
            mutual_neighbor[i, j] = 1.0
            mutual_neighbor[j, i] = 1.0

    two_hop = safe[:, None, :] + safe[None, :, :]
    two_hop_valid = valid[:, None, :] & valid[None, :, :]
    two_hop_valid &= ~np.eye(n, dtype=bool)[:, :, None]
    two_hop_valid &= ~np.eye(n, dtype=bool)[None, :, :]
    two_hop[~two_hop_valid] = np.inf

    best_alternative = np.min(two_hop, axis=2)
    best_alternative[~valid] = 0.0

    relative_alternative = best_alternative / (safe + eps)
    indispensability = relative_alternative / (1.0 + relative_alternative)
    indispensability[~valid] = 0.0

    score = (
        0.48 * distance_signal
        + 0.24 * rank_signal
        + 0.14 * mutual_neighbor
        + 0.14 * indispensability
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    competitive = score / row_max

    score = 0.72 * score + 0.28 * np.sqrt(np.maximum(competitive, 0.0))
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
