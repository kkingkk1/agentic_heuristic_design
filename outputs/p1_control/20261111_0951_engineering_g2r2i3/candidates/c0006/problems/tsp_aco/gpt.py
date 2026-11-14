import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    eps = np.finfo(float).eps

    safe_d = np.where(valid, d, np.inf)
    safe_d = np.where(safe_d > 0.0, safe_d, global_scale)
    np.fill_diagonal(safe_d, np.inf)

    positive_rows = np.where(
        np.isfinite(safe_d) & (safe_d > 0.0),
        safe_d,
        np.nan,
    )
    row_scale = np.nanmedian(positive_rows, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        global_scale,
    )
    row_scale = np.maximum(row_scale, eps)

    normalized_distance = safe_d / row_scale[:, None]

    short_edge = 1.0 / (1.0 + normalized_distance)
    exponential_affinity = np.exp(-normalized_distance)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + ranks.astype(float))
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )

    nearest = np.argmin(safe_d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0

    nearest_distance = np.min(safe_d, axis=1)
    second_nearest_distance = np.partition(safe_d, 1, axis=1)[:, 1]
    savings = (
        second_nearest_distance[:, None]
        + second_nearest_distance[None, :]
        - safe_d
    )
    savings = np.maximum(savings, 0.0)
    savings_max = np.max(savings, axis=1, keepdims=True)
    savings_score = np.divide(
        savings,
        np.maximum(savings_max, eps),
        out=np.zeros_like(savings),
        where=savings_max > 0.0,
    )

    sparsity = row_scale / max(float(np.mean(row_scale)), eps)
    bridge_bonus = np.sqrt(sparsity[:, None] * sparsity[None, :])
    bridge_bonus = bridge_bonus / np.maximum(
        np.max(bridge_bonus, axis=1, keepdims=True),
        eps,
    )

    heuristic = (
        0.35 * short_edge
        + 0.20 * exponential_affinity
        + 0.15 * rank_score
        + 0.15 * mutual
        + 0.10 * savings_score
        + 0.05 * bridge_bonus
    )

    heuristic[~valid] = 0.0
    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    row_max = np.max(heuristic, axis=1, keepdims=True)
    heuristic = np.divide(
        heuristic,
        row_max,
        out=np.zeros_like(heuristic),
        where=row_max > 0.0,
    )

    return heuristic.astype(float, copy=False)
