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
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(positive_values))
    fallback = max(fallback, np.finfo(float).eps)

    clean = np.array(d, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = fallback
    clean[clean <= 0] = fallback
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, np.inf)

    eps = np.finfo(float).eps

    finite_clean = clean[np.isfinite(clean)]
    global_scale = float(np.median(finite_clean)) if finite_clean.size else 1.0
    global_scale = max(global_scale, eps)

    nearest = np.min(clean, axis=1)
    nearest[~np.isfinite(nearest)] = global_scale

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_neighbors = np.partition(clean, k - 1, axis=1)[:, :k]
    local_scale = np.mean(local_neighbors, axis=1)
    local_scale = np.maximum(local_scale, eps)

    relative_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) + eps

    local_affinity = np.exp(-clean / relative_scale)
    local_affinity[~np.isfinite(local_affinity)] = 0.0

    ranks = np.argsort(np.argsort(clean, axis=1), axis=1).astype(float)
    rank_affinity = np.exp(-ranks / max(1.0, np.sqrt(n)))
    mutual_affinity = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    inverse_distance = 1.0 / (clean + eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_distance /= np.max(inverse_distance) + eps

    detours = clean[:, :, None] + clean[None, :, :] - clean[:, :, None]
    detours = clean[:, None, :] + clean[None, :, :] - clean[:, :, None]

    candidate_mask = np.ones((n, n, n), dtype=bool)
    indices = np.arange(n)
    candidate_mask[indices, indices, :] = False
    candidate_mask[:, :, indices] &= (
        indices[None, None, :] != indices[:, None, :]
    )

    detours = np.where(candidate_mask, detours, np.inf)
    replacement = np.min(detours, axis=2)
    replacement[~np.isfinite(replacement)] = 0.0
    replacement = np.maximum(replacement, 0.0)

    regret_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) + eps
    regret_bonus = 1.0 - np.exp(-replacement / regret_scale)

    mst_bonus = np.zeros((n, n), dtype=float)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0

    for _ in range(n):
        candidates = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(candidates))

        if in_tree[u] or not np.isfinite(best[u]):
            break

        in_tree[u] = True

        if parent[u] >= 0:
            v = parent[u]
            mst_bonus[u, v] = 1.0
            mst_bonus[v, u] = 1.0

        for v in range(n):
            if not in_tree[v] and clean[u, v] < best[v]:
                best[v] = clean[u, v]
                parent[v] = u

    score = (
        0.35 * inverse_distance
        + 0.25 * local_affinity
        + 0.20 * mutual_affinity
        + 0.12 * regret_bonus
        + 0.08 * mst_bonus
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros_like(score, dtype=float)

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
