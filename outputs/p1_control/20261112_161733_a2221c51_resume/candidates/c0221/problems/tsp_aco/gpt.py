import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback = max(float(np.max(positive_values)) * 10.0, global_scale)

    work = np.where(valid, d, fallback)
    np.fill_diagonal(work, fallback)

    local_index = min(2, n - 2)
    local_scale = np.partition(work, local_index, axis=1)[:, local_index]
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = work / geometric_scale
    absolute_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    endpoint_preference = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    reciprocal_distance = global_scale / np.maximum(work, eps)
    reciprocal_distance = np.minimum(reciprocal_distance, 1e12)

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_mismatch = np.minimum(scale_mismatch, 20.0)

    local_neighbor_mask = (
        valid
        & (
            d
            <= 1.5
            * np.minimum(local_scale[:, None], local_scale[None, :])
        )
    )
    local_density = local_neighbor_mask.mean(axis=1)

    density_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    nearest_k = min(3, n - 1)
    mutual_nearest = (
        valid
        & (ranks <= nearest_k / max(n - 1, 1))
        & (ranks.T <= nearest_k / max(n - 1, 1))
    )

    log_score = (
        0.95 * np.log(np.maximum(reciprocal_distance, eps))
        + 1.05 * np.log(np.maximum(endpoint_preference, eps))
        - 0.90 * relative_length
        - 0.22 * absolute_length
        - 0.28 * scale_mismatch
        - 0.10 * density_penalty
        + 0.14 * mutual_nearest.astype(float)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
