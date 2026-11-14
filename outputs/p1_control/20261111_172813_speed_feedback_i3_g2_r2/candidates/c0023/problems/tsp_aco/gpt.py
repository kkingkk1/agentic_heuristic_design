import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Larger values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    finite_positive = np.isfinite(d) & (d > 0)
    positive = d[finite_positive]
    if positive.size == 0:
        return score

    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    work = np.where(
        np.isfinite(d),
        np.maximum(d, 0.0),
        global_scale * 1e6,
    )
    np.fill_diagonal(work, np.inf)

    valid = np.isfinite(work) & ~np.eye(n, dtype=bool)
    if not np.any(valid):
        return score

    # Stable scale-free inverse-distance preference.
    positive_work = work[valid]
    local_scale = np.median(positive_work)
    local_scale = max(float(local_scale), np.finfo(float).eps)

    inv = np.zeros_like(work)
    inv[valid] = 1.0 / (work[valid] + local_scale * 1e-12)
    inv /= np.max(inv) + np.finfo(float).eps

    # Vectorized nearest-neighbor and second-nearest-neighbor structure.
    nearest = np.argmin(work, axis=1)
    nearest_distance = work[np.arange(n), nearest]

    second_nearest_distance = np.partition(work, 1, axis=1)[:, 1]
    alternative = np.minimum(second_nearest_distance, global_scale * 1e6)

    mutual = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
        & valid
    ).astype(float)

    # Directed local savings, clipped to avoid rewarding invalid edges.
    savings = (
        alternative[:, None]
        + alternative[None, :]
        - work
    )
    savings = np.maximum(savings, 0.0)
    savings[~valid] = 0.0

    max_savings = float(np.max(savings))
    if max_savings > 0:
        savings /= max_savings

    # Reciprocal compatibility helps both symmetric and mildly asymmetric inputs.
    reciprocal = 0.5 * (inv + inv.T)
    reciprocal[~valid] = 0.0

    # Row-wise preference preserves local alternatives for exploration.
    row_max = np.max(inv, axis=1, keepdims=True)
    row_preference = inv / (row_max + np.finfo(float).eps)
    row_preference[~valid] = 0.0

    # A small normalized floor prevents premature collapse of sampling support.
    exploration = np.zeros_like(work)
    exploration[valid] = 1.0
    exploration /= np.maximum(np.sum(exploration, axis=1, keepdims=True), 1.0)

    score = (
        0.42 * inv
        + 0.20 * reciprocal
        + 0.16 * savings
        + 0.12 * mutual
        + 0.08 * row_preference
    )

    score += 0.015 * exploration
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
