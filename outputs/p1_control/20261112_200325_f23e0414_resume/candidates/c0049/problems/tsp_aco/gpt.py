import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw > 0) & off_diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = raw[valid]
    fallback = float(np.max(positive))
    fallback = max(fallback, np.finfo(float).eps)

    clean = np.where(valid, raw, fallback)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    eps = np.finfo(float).eps * max(fallback, 1.0)

    inverse_distance = 1.0 / (clean + eps)
    inverse_distance[~off_diagonal] = 0.0

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest_values = np.partition(clean, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_affinity = np.exp(-clean / (pair_scale + eps))
    relative_affinity[~off_diagonal] = 0.0

    ranks = np.argsort(np.argsort(clean, axis=1), axis=1).astype(float)
    rank_affinity = 1.0 / (1.0 + ranks)
    rank_affinity[~off_diagonal] = 0.0
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    nearest = np.argmin(clean, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if i != j and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    detour = clean[:, :, None] + clean[None, :, :]
    candidate_mask = np.ones((n, n, n), dtype=bool)
    indices = np.arange(n)
    candidate_mask[indices, indices, :] = False
    candidate_mask[:, :, indices] &= True
    candidate_mask[indices[:, None], indices[None, :], :] &= True

    for k_idx in range(n):
        candidate_mask[k_idx, :, k_idx] = False
        candidate_mask[:, k_idx, k_idx] = False

    detour = np.where(candidate_mask, detour, np.inf)
    best_detour = np.min(detour, axis=2)
    best_detour[~off_diagonal] = np.inf

    replacement_gap = np.maximum(best_detour - clean, 0.0)
    indispensability = 1.0 - np.exp(
        -replacement_gap / (pair_scale + eps)
    )
    indispensability[~off_diagonal] = 0.0
    indispensability[~np.isfinite(indispensability)] = 0.0

    inv_norm = inverse_distance / (
        np.max(inverse_distance, axis=1, keepdims=True) + eps
    )
    inv_norm = 0.5 * (inv_norm + inv_norm.T)

    score = (
        0.45 * inv_norm
        + 0.25 * relative_affinity
        + 0.15 * mutual_rank
        + 0.10 * mutual_nearest
        + 0.05 * indispensability
    )

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    values = score[off_diagonal]
    maximum = float(np.max(values)) if values.size else 0.0

    if maximum > 0.0:
        score /= maximum
    else:
        score = off_diagonal.astype(float)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
