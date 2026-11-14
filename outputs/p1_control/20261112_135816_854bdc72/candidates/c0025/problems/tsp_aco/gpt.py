import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for a symmetric TSP distance matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)
    valid_values = d[positive]

    if valid_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(valid_values) * 10.0 + 1.0)
    clean = np.where(positive, d, fallback)
    clean = np.minimum(clean, clean.T)
    np.fill_diagonal(clean, 0.0)

    off_diag = ~np.eye(n, dtype=bool)
    positive_edges = clean[off_diag]
    scale = float(np.median(positive_edges))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    # Robust inverse-distance affinity.
    inv_distance = 1.0 / (1.0 + clean / scale)
    np.fill_diagonal(inv_distance, 0.0)

    # Rank-based local affinity: closer edges receive larger scores.
    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n, dtype=float)[None, :]
    local_rank = 1.0 / (1.0 + ranks)
    np.fill_diagonal(local_rank, 0.0)

    # Mutual-neighbor bonus, especially useful for preserving compact tour segments.
    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.zeros((n, n), dtype=bool)
    for i in range(n):
        nearest[i, order[i, 1 : k + 1]] = True
    mutual = nearest & nearest.T

    # Clarke-Wright-style savings relative to each node's nearest reference edge.
    nearest_distance = np.min(
        np.where(off_diag, clean, np.inf),
        axis=1,
    )
    savings = nearest_distance[:, None] + nearest_distance[None, :] - clean
    savings = np.maximum(savings, 0.0)
    savings_scale = np.percentile(savings[off_diag], 90)
    if not np.isfinite(savings_scale) or savings_scale <= 0:
        savings_scale = 1.0
    savings = np.minimum(savings / savings_scale, 1.0)
    np.fill_diagonal(savings, 0.0)

    # Balance endpoint support so one node is not assigned too many weak edges.
    endpoint_support = local_rank + local_rank.T
    support_scale = np.max(endpoint_support[off_diag])
    if not np.isfinite(support_scale) or support_scale <= 0:
        support_scale = 1.0
    endpoint_support /= support_scale

    score = (
        0.42 * inv_distance
        + 0.24 * local_rank
        + 0.16 * local_rank.T
        + 0.12 * savings
        + 0.06 * mutual.astype(float)
    )
    score *= 0.85 + 0.15 * endpoint_support
    score = 0.5 * (score + score.T)
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
