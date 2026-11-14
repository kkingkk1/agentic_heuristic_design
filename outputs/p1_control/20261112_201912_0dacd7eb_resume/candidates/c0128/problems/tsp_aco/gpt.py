import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    positive = distances[valid]
    if positive.size == 0:
        return np.zeros_like(distances, dtype=float)

    fallback = float(np.max(positive))
    work = np.where(valid, distances, fallback)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work) & (work > 0)
    np.fill_diagonal(valid, False)

    positive_work = work[valid]
    fallback = float(np.max(positive_work)) if positive_work.size else 1.0
    safe = np.where(valid, work, fallback)
    np.fill_diagonal(safe, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) / 1.5))))
    nearest_values = np.partition(safe, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest_values, axis=1), eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_length = safe / (pair_scale + eps)

    closeness = np.exp(-np.minimum(relative_length, 40.0))
    closeness[~valid] = 0.0

    ranks = np.argsort(np.argsort(safe, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + 0.85 * ranks)
    rank_score[~valid] = 0.0
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    nearest = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if i != j and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    detour = safe[:, None, :] + safe[None, :, :] - safe[:, :, None]

    intermediate_valid = np.isfinite(safe)
    detour_valid = (
        intermediate_valid[:, None, :]
        & intermediate_valid[None, :, :]
    )
    detour_valid &= ~np.eye(n, dtype=bool)[:, :, None]
    detour_valid &= ~np.eye(n, dtype=bool)[None, :, :]
    detour[~detour_valid] = np.inf

    replacement = np.min(detour, axis=2)
    replacement[~valid] = 0.0
    replacement = np.maximum(replacement, 0.0)

    indispensable = 1.0 - np.exp(
        -np.minimum(replacement / (pair_scale + eps), 40.0)
    )
    indispensable[~valid] = 0.0

    score = (
        0.50 * closeness
        + 0.27 * mutual_rank
        + 0.16 * mutual
        + 0.07 * indispensable
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
