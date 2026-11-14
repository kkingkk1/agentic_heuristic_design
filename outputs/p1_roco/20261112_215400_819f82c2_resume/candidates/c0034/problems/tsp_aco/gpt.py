import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, entropy-balanced edge-prior matrix for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d[~valid] = np.inf

    positive = symmetric_d[valid & (symmetric_d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    row_values = np.where(valid, symmetric_d, np.nan)
    local_scale = np.nanmedian(row_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    temperature = np.sqrt(np.outer(local_scale, local_scale))
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(symmetric_d[valid] / temperature[valid], 0.0, 60.0)
    )

    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)

    # Symmetric matrix balancing: A <- diag(x) A diag(x),
    # with approximately equal weighted degree at every node.
    scaling = np.ones(n, dtype=float)
    for _ in range(80):
        degree = affinity @ scaling
        updated = np.zeros_like(scaling)
        positive_degree = degree > np.finfo(float).eps
        updated[positive_degree] = 1.0 / degree[positive_degree]

        # Damping prevents oscillation on sparse or highly uneven graphs.
        updated = np.sqrt(np.maximum(updated, 0.0) * scaling)
        if np.max(np.abs(updated - scaling)) < 1e-10:
            scaling = updated
            break
        scaling = updated

    balanced = affinity * scaling[:, None] * scaling[None, :]
    balanced[~valid] = 0.0

    # Reward edges whose endpoints have dissimilar neighborhood profiles.
    row_norm = np.linalg.norm(affinity, axis=1)
    profile_cosine = np.zeros((n, n), dtype=float)
    usable = row_norm > np.finfo(float).eps
    if np.any(usable):
        normalized_profiles = np.zeros_like(affinity)
        normalized_profiles[usable] = (
            affinity[usable] / row_norm[usable, None]
        )
        profile_cosine = normalized_profiles @ normalized_profiles.T
        profile_cosine = np.clip(profile_cosine, 0.0, 1.0)

    bridge_factor = 0.75 + 0.25 * (1.0 - profile_cosine)
    score = balanced * bridge_factor
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
