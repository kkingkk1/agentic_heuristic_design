import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.where(np.isfinite(d) & (d >= 0.0), d, np.inf)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = valid & (d > 0.0)

    positive_values = d[positive]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        finite_values = d[valid]
        global_scale = float(np.median(finite_values)) if finite_values.size else 1.0
    global_scale = max(global_scale, eps)

    row_values = np.where(valid, d, np.nan)
    row_scale = np.nanmedian(row_values, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        global_scale,
    )
    row_scale = np.maximum(row_scale, eps)

    pair_scale = np.sqrt(row_scale[:, None] * row_scale[None, :])
    normalized_distance = d / np.maximum(pair_scale, eps)
    normalized_distance = np.where(valid, normalized_distance, np.inf)
    normalized_distance = np.minimum(normalized_distance, 1e6)

    affinity = np.exp(-np.minimum(normalized_distance, 50.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    # Symmetric Sinkhorn-like balancing:
    # reduce the dominance of nodes connected to many unusually short edges.
    balancing = np.ones(n, dtype=float)
    for _ in range(32):
        degree = affinity @ balancing
        degree = np.maximum(degree, eps)
        balancing = 1.0 / degree
        balancing /= np.sqrt(np.maximum(np.mean(balancing**2), eps))

    balanced = (
        balancing[:, None]
        * affinity
        * balancing[None, :]
    )
    balanced = np.maximum(balanced, 0.0)

    # Triangle detour benefit:
    # a large value means that replacing edge (i, j) by a route through k
    # would incur a substantial additional distance.
    safe_d = np.where(valid, d, np.inf)
    detour = (
        safe_d[:, :, None]
        + safe_d[None, :, :]
        - safe_d[:, None, :]
    )

    intermediary_valid = (
        valid[:, :, None]
        & valid[None, :, :]
        & valid[:, None, :]
    )
    detour = np.where(intermediary_valid, detour, np.nan)

    detour_benefit = np.nanmedian(detour, axis=2)
    detour_benefit = np.where(
        np.isfinite(detour_benefit),
        detour_benefit,
        0.0,
    )
    detour_benefit = np.maximum(detour_benefit, 0.0)

    robust_detour_scale = np.nanmedian(
        np.where(valid, detour_benefit, np.nan)
    )
    if not np.isfinite(robust_detour_scale) or robust_detour_scale <= 0.0:
        robust_detour_scale = global_scale
    robust_detour_scale = max(float(robust_detour_scale), eps)

    detour_signal = 1.0 - np.exp(
        -detour_benefit / robust_detour_scale
    )

    endpoint_capacity = np.sqrt(
        np.maximum(balancing[:, None] * balancing[None, :], eps)
    )
    endpoint_capacity /= max(float(np.max(endpoint_capacity)), eps)

    score = (
        balanced
        * (0.35 + 0.65 * detour_signal)
        * (0.45 + 0.55 * endpoint_capacity)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
