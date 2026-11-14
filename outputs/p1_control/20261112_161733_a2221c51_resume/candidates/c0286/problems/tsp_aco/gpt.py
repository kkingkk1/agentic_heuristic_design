import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = (
        np.isfinite(d)
        & (d > 0.0)
        & np.isfinite(d.T)
        & (d.T > 0.0)
    )
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros((n, n), dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    edge_values = symmetric_d[valid]
    global_scale = max(float(np.median(edge_values)), eps)
    maximum_distance = max(float(np.max(edge_values)), global_scale)

    work = np.where(valid, symmetric_d, np.inf)
    np.fill_diagonal(work, np.inf)

    # Select the second available neighbor without a full row-wise sort.
    neighbor_count = np.sum(valid, axis=1)
    kth = 1 if n > 2 else 0
    local_scale = np.partition(work, kth, axis=1)[:, kth]

    fallback_scale = max(global_scale, 0.5 * maximum_distance)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        fallback_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_work = np.where(valid, symmetric_d, fallback_scale)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = safe_work / endpoint_scale
    global_length = safe_work / global_scale

    relative_i = safe_work / local_scale[:, None]
    relative_j = safe_work / local_scale[None, :]

    # Continuous approximation of mutual nearest-neighbor preference.
    local_preference = np.exp(
        -0.55 * (relative_i + relative_j)
    )

    # Prefer edges whose endpoints have comparable local geometry.
    log_scale_ratio = np.abs(
        np.log(local_scale[:, None])
        - np.log(local_scale[None, :])
    )
    scale_compatibility = np.exp(
        -0.50 * np.minimum(log_scale_ratio, 50.0)
    )

    # Softly suppress edges that are long for either endpoint.
    endpoint_longness = np.maximum(relative_i, relative_j)
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    log_score = (
        0.95 * np.log(np.maximum(local_preference, eps))
        + 0.55 * np.log(np.maximum(scale_compatibility, eps))
        - 0.70 * relative_length
        - 0.12 * global_length
        - 0.30 * excess_longness
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))

    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
