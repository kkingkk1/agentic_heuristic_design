import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    if n == 1:
        return np.zeros((1, 1), dtype=float)

    work = 0.5 * (d + d.T)
    np.fill_diagonal(work, np.inf)

    finite_edges = work[np.isfinite(work)]
    if finite_edges.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    positive = finite_edges[finite_edges > 0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    base_scale = max(base_scale, np.finfo(float).eps)

    # Robust global temperature based on pairwise distance dispersion.
    center = float(np.median(finite_edges))
    mad = float(np.median(np.abs(finite_edges - center)))
    temperature = max(1.4826 * mad, 0.25 * base_scale, np.finfo(float).eps)

    centered = (work - center) / temperature
    kernel = np.exp(-np.clip(centered, -60.0, 60.0))
    kernel[~np.isfinite(kernel)] = 0.0
    np.fill_diagonal(kernel, 0.0)

    # Best admissible two-hop alternative for each edge.
    two_hop = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        left = work[:, k][:, None]
        right = work[k, :][None, :]
        candidate = left + right
        candidate[k, :] = np.inf
        candidate[:, k] = np.inf
        two_hop = np.minimum(two_hop, candidate)

    valid_detour = np.isfinite(two_hop) & np.isfinite(work)
    savings = np.zeros((n, n), dtype=float)
    savings[valid_detour] = two_hop[valid_detour] - work[valid_detour]

    savings_scale = float(np.median(np.abs(savings[valid_detour])))
    if not np.isfinite(savings_scale) or savings_scale <= 0:
        savings_scale = base_scale

    # Saturating bonus for edges that beat their best two-hop substitute.
    shortcut_bonus = np.tanh(np.maximum(savings, 0.0) / savings_scale)
    shortcut_bonus[~valid_detour] = 0.0

    score = kernel * (1.0 + 0.8 * shortcut_bonus)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    # Symmetric Sinkhorn-style balancing without changing edge symmetry.
    for _ in range(24):
        row_sum = np.sum(score, axis=1)
        positive_rows = row_sum > np.finfo(float).eps
        factors = np.ones(n, dtype=float)
        factors[positive_rows] = 1.0 / np.sqrt(row_sum[positive_rows])
        score *= factors[:, None] * factors[None, :]
        score = 0.5 * (score + score.T)
        np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0:
        score = np.ones((n, n), dtype=float)
        np.fill_diagonal(score, 0.0)
    else:
        score /= maximum

    return score.astype(float, copy=False)
