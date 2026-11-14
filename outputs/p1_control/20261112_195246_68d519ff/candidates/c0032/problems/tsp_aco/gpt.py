import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-based edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0)
    positive_values = distances[positive]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive_values))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    safe = np.where(positive, distances, fallback)
    safe = 0.5 * (safe + safe.T)

    valid = positive & positive.T
    np.fill_diagonal(valid, False)
    np.fill_diagonal(safe, np.inf)

    finite_safe = safe[np.isfinite(safe) & (safe > 0.0)]
    if finite_safe.size:
        scale = float(np.median(finite_safe))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

    normalized = np.where(np.isfinite(safe), safe / scale, 50.0)
    kernel = np.exp(-np.clip(normalized * normalized, 0.0, 50.0))
    kernel *= valid
    np.fill_diagonal(kernel, 0.0)

    degree = np.sum(kernel, axis=1)
    degree_scale = np.median(degree[degree > 0.0]) if np.any(degree > 0.0) else 1.0
    degree_scale = max(float(degree_scale), np.finfo(float).eps)

    two_hop = kernel @ kernel
    support_scale = np.max(two_hop) if two_hop.size else 0.0
    if np.isfinite(support_scale) and support_scale > 0.0:
        support = two_hop / support_scale
    else:
        support = np.zeros_like(two_hop)

    degree_factor = 1.0 / np.sqrt(
        (1.0 + degree / degree_scale)[:, None]
        * (1.0 + degree / degree_scale)[None, :]
    )

    score = kernel * (0.65 + 0.35 * support) * degree_factor
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)
