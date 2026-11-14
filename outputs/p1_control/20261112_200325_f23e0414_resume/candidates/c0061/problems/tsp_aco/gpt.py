import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    fallback = float(np.max(finite_positive)) if finite_positive.size else 1.0

    clean = np.array(d, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = fallback
    clean[clean <= 0] = fallback
    np.fill_diagonal(clean, np.inf)

    positive = clean[np.isfinite(clean) & (clean > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0

    inverse = 1.0 / (clean + 1e-12)
    inverse[~np.isfinite(inverse)] = 0.0
    inverse_max = float(np.max(inverse))
    if inverse_max > 0:
        inverse /= inverse_max

    nearest = np.argmin(clean, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    mst = np.zeros((n, n), dtype=float)

    if n > 1:
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
                mst[u, v] = 1.0
                mst[v, u] = 1.0

            for v in range(n):
                if not in_tree[v] and clean[u, v] < best[v]:
                    best[v] = clean[u, v]
                    parent[v] = u

    local_scale = np.exp(-clean / (0.90 * scale + 1e-12))
    local_scale[~np.isfinite(local_scale)] = 0.0
    local_max = float(np.max(local_scale))
    if local_max > 0:
        local_scale /= local_max

    score = (
        0.60 * inverse
        + 0.15 * local_scale
        + 0.17 * mutual
        + 0.08 * mst
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
