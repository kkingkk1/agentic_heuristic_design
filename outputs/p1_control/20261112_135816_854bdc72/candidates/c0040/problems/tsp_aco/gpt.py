import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric diffusion-based prior for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d)

    valid_values = d[finite & offdiag]
    if valid_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = valid_values[valid_values > 0]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    replacement = (
        float(np.max(valid_values[valid_values >= 0]))
        if np.any(valid_values >= 0)
        else global_scale
    )
    replacement = max(replacement, global_scale)

    safe = np.where(finite, np.maximum(d, 0.0), replacement)
    safe = 0.5 * (safe + safe.T)
    np.fill_diagonal(safe, np.inf)

    row_positive = np.where(
        np.isfinite(safe) & (safe > 0),
        safe,
        np.nan,
    )
    local_scale = np.nanmedian(row_positive, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    normalized = safe / endpoint_scale
    normalized = np.where(np.isfinite(normalized), normalized, 60.0)

    scales = np.array([0.5, 1.0, 2.0], dtype=float)
    kernels = []
    for temperature in scales:
        kernel = np.exp(-np.minimum(normalized / temperature, 60.0))
        kernel[~offdiag] = 0.0
        kernels.append(kernel)

    affinity = (
        0.25 * kernels[0]
        + 0.50 * kernels[1]
        + 0.25 * kernels[2]
    )
    affinity = 0.5 * (affinity + affinity.T)

    degree = np.sum(affinity, axis=1)
    degree = np.maximum(degree, eps)

    normalized_affinity = affinity / np.sqrt(
        np.outer(degree, degree)
    )

    two_hop = normalized_affinity @ normalized_affinity
    two_hop[~np.isfinite(two_hop)] = 0.0
    two_hop[~offdiag] = 0.0

    triangle_support = np.sum(
        affinity[:, :, None] * affinity[None, :, :],
        axis=1,
    )
    triangle_support = triangle_support / np.sqrt(
        np.outer(degree, degree)
    )
    triangle_support[~np.isfinite(triangle_support)] = 0.0
    triangle_support[~offdiag] = 0.0

    endpoint_similarity = np.exp(
        -np.minimum(
            np.abs(local_scale[:, None] - local_scale[None, :])
            / (local_scale[:, None] + local_scale[None, :] + eps),
            60.0,
        )
    )
    endpoint_similarity[~offdiag] = 0.0

    concentration_penalty = 1.0 / np.sqrt(
        (1.0 + degree[:, None]) * (1.0 + degree[None, :])
    )

    score = (
        0.30 * affinity
        + 0.35 * two_hop
        + 0.20 * triangle_support
        + 0.15 * endpoint_similarity
    )
    score *= 0.75 + 0.25 * concentration_penalty

    score[~np.isfinite(score)] = 0.0
    score[~offdiag] = 0.0
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
